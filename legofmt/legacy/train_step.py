"""Training-step paths the recipe no longer takes, kept so older configs and checkpoints still run unchanged:

    inverse events   ``mask_conf.p_forward < 1``: per event a coin flip generates the incoming particle from the
                     outgoing set instead (``_sample_mask``)
    t distribution   ``t_dist: sm_norm`` (logit-normal) and ``t_dist_shift != 1`` (``_sample_t_options``)
    reflow           a frozen teacher's transport of the base replaces the data target: ``reflow_path`` from step 0,
                     ``reflow_start_epoch > 0`` the student's end-of-epoch snapshot from that epoch on

``LegacyTrainStep`` follows ``TrainStep`` in ``LEGOLtng``'s bases; ``TrainStep._step`` dispatches to it in one line
each. ``LEGOLtngDirect`` (``legacy/reflow.py``) reuses ``_sample_mask`` and ``_teacher_transport``.
"""

import torch
from torch import Tensor

from legofmt.data.struct import DataStruct


class LegacyTrainStep:
    """Legacy training-step mixin for :class:`~legofmt.main.modules.LEGOLtng`."""

    @torch.no_grad()
    def _sample_mask(self, ds_t: DataStruct) -> Tensor:
        """Samples each event's mask: the data mask or its inverse, a coin flip with ``mask_conf.p_forward``.

        The inverse generates the incoming from the outgoing set; scalar tokens stay conditioning.
        """
        if not self.rc.mask_conf:
            return ds_t.m.full
        fwd = ds_t.m.full
        inv = (fwd == 0) & (ds_t.am.full == 1)
        inv[:, :self.rc.n_prefix] = 0  # all conditioning scalars stay conditioning
        pick = torch.rand(fwd.shape[0], device=fwd.device) < self.rc.mask_conf.get("p_forward", 0.5)
        return torch.where(pick.unsqueeze(-1), fwd, inv)

    def _sample_t_options(self, ds_t: DataStruct) -> Tensor:
        """Samples the training-time ``t`` per event.

        Logit-normal (``sm_norm``) or SD3 mode sampling (``sd3``; scale 0 is uniform); ``t_dist_shift`` applies ``t **
        (1/shift)``.
        """
        if self.rc.t_dist == "sm_norm":
            t = torch.sigmoid(self.rc.t_dist_scale * torch.randn_like(ds_t.f.d))
        elif self.rc.t_dist == "sd3":
            u = torch.rand_like(ds_t.f.d)
            t = 1 - u + self.rc.t_dist_scale / 3 * ((torch.pi / 2 * u).sin() ** 2 - u)
        else:
            raise ValueError(f"unknown t_dist: {self.rc.t_dist!r}")
        if self.rc.t_dist_shift != 1.0:
            t = t.clamp(min=0) ** (1.0 / self.rc.t_dist_shift)
        return t

    # ---- reflow -------------------------------------------------------------------------------------------------

    def _init_reflow(self) -> None:
        """Loads the ``reflow_path`` teacher, or sets none; not a submodule, so outside state_dict and DDP."""
        teacher = None
        if self.rc.reflow_path is not None:
            from legofmt.legacy.reflow import _build_reflow_teacher  # avoids import cycle
            teacher = _build_reflow_teacher(self.rc.reflow_path)
        object.__setattr__(self, "reflow_teacher", teacher)

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        """Makes the student's snapshot the teacher from ``reflow_start_epoch`` on (self-reflow, or a refresh)."""
        from legofmt.cfm.project_model import uncompiled
        start = self.rc.reflow_start_epoch
        if start <= 0 or self.current_epoch + 1 < start:
            return
        vf = uncompiled(self.model).vf
        if self.reflow_teacher is None:  # self-reflow: teacher = snapshot of the student
            from legofmt.legacy.reflow import _teacher_from_state  # avoids import cycle
            teacher = _teacher_from_state(self.rc.config, vf.state_dict())
            object.__setattr__(self, "reflow_teacher", teacher.to(self.device))
        else:  # refresh: teacher = last epoch's student
            uncompiled(self.reflow_teacher.model).vf.load_state_dict(vf.state_dict())

    @torch.no_grad()
    def _teacher_transport(self, ds_t: DataStruct, base: Tensor, **solve_kwargs) -> Tensor:
        """Returns the reflow teacher's transport of ``base`` on the generated slots, the data elsewhere.

        The E_dep token is generated but its direction columns are the (1,1,1) filler; transporting them gave that token
        an off-manifold target.
        """
        tgt = self.reflow_teacher.solve(ds_t, x_init=base, **solve_kwargs)
        tgt = torch.where((ds_t.m.full == 1).unsqueeze(-1), tgt, ds_t.f.model_in)
        tgt[:, :self.rc.n_prefix, 1:] = ds_t.f.non_cc[..., 1:]
        return tgt

    @torch.no_grad()
    def _reflow_coupling(self, ds_t: DataStruct, base: Tensor) -> DataStruct:
        """Replaces the batch's target by the teacher's transport of ``base`` on the steps reflow fires."""
        if self.reflow_teacher is None or not self.training or self.global_step % self.rc.reflow_every != 0:
            return ds_t
        tgt = self._teacher_transport(ds_t, base, **self.rc.reflow_kwargs)
        return DataStruct(torch.cat((tgt, ds_t.f.pdgids), dim=-1), ds_t.m.full, ds_t.am.full)
