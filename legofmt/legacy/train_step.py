"""Training-step paths that are not the recipe's, kept so that older configs and checkpoints still run unchanged:

    inverse events   ``mask_conf.p_forward < 1``: per event a coin flip generates the incoming particle from the
                     outgoing set instead (``_sample_mask``)
    t distribution   ``t_dist: sm_norm`` (logit-normal) and ``t_dist_shift != 1`` (``_sample_t_options``)
    reflow           a frozen teacher's transport of the base replaces the data target: ``reflow_path`` from step 0,
                     ``reflow_start_epoch > 0`` the student's end-of-epoch snapshot from that epoch on
    overflow_delta   an outgoing energy of exactly 0 (the particle keeps the full incoming energy) and an E_dep of
                     exactly 0 (no deposit) move to ``-overflow_delta``, outside [0, 1] (``_shift_overflow_targets``)

``LegacyTrainStep`` follows ``TrainStep`` in ``LEGOLtng``'s bases; ``TrainStep`` reaches each path in one line
(``_flow_frame``, ``_sample_t``, ``_step``). ``LEGOLtngDirect`` (``legacy/reflow.py``) reuses ``_sample_mask`` and
``_teacher_transport``.
"""

import torch
from torch import Tensor

from legofmt.data.struct import DataStruct, _F


class LegacyTrainStep:
    """Legacy training-step mixin for :class:`~legofmt.main.modules.LEGOLtng`."""

    def _shift_overflow_targets(self, ds_t: DataStruct) -> DataStruct:
        """Returns a copy of the batch with its exact zeros at ``-overflow_delta``, outside [0, 1].

        The zeros are an outgoing energy 0 on a real outgoing token, and an E_dep 0.
        """
        f = ds_t.f.full.clone()
        out_e, edep = _F(f).out_energy, _F(f).edep
        out_e.masked_fill_((out_e == 0) & ds_t.am.out_p.bool(), -self.rc.overflow_delta)
        edep.masked_fill_(edep == 0, -self.rc.overflow_delta)
        return DataStruct(f, ds_t.m.full, ds_t.am.full)

    @torch.no_grad()
    def _sample_mask(self, ds_t: DataStruct) -> Tensor:
        """Samples each event's mask (B, L): the data's with probability ``mask_conf.p_forward``, else its inverse.

        The inverse mask generates the incoming particle from the outgoing set and E_dep; the conditioning scalar
        tokens stay conditioning tokens. Without ``mask_conf`` every event keeps the data's mask.
        """
        if not self.rc.mask_conf:
            return ds_t.m.full
        fwd = ds_t.m.full
        inv = (fwd == 0) & (ds_t.am.full == 1)
        inv[:, :self.rc.n_prefix] = 0  # the conditioning scalar tokens stay conditioning tokens
        pick = torch.rand(fwd.shape[0], device=fwd.device) < self.rc.mask_conf.get("p_forward", 0.5)
        return torch.where(pick.unsqueeze(-1), fwd, inv)

    def _sample_t_options(self, ds_t: DataStruct) -> Tensor:
        """Samples each event's time ``t`` (B,) for a ``t_dist`` or ``t_dist_shift`` that is not the recipe's.

        ```
        sm_norm   t = sigmoid(t_dist_scale z),   z ~ N(0, 1)      (logit-normal)
        sd3       t = sd3_time(u, t_dist_scale),   u ~ U(0, 1)    (``main/train_step.py``)
        shift     t = t ** (1 / t_dist_shift)                     applied to either
        ```

        Raises:
            ValueError: an unknown ``t_dist``.
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
        """Sets ``reflow_teacher`` to the ``reflow_path`` teacher, or to None without a path.

        It is set past ``nn.Module``'s registration: the teacher is no submodule, so it is in no state dict and DDP
        does not see it.
        """
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
        """Returns the reflow teacher's transport of ``base`` on the generated tokens, the data elsewhere.

        The prefix tokens keep the data in every channel but the energy: the E_dep token is generated, but its
        direction channels hold the (1, 1, 1) filler, and their transport would be an off-manifold target.
        """
        tgt = self.reflow_teacher.solve(ds_t, x_init=base, **solve_kwargs)
        tgt = torch.where((ds_t.m.full == 1).unsqueeze(-1), tgt, ds_t.f.model_in)
        tgt[:, :self.rc.n_prefix, 1:] = ds_t.f.non_cc[..., 1:]
        return tgt

    @torch.no_grad()
    def _reflow_coupling(self, ds_t: DataStruct, base: Tensor) -> DataStruct:
        """Replaces the batch's target by the teacher's transport of ``base`` on the steps reflow fires.

        Reflow fires on every ``reflow_every``-th training step of a model that has a teacher.
        """
        if self.reflow_teacher is None or not self.training or self.global_step % self.rc.reflow_every != 0:
            return ds_t
        tgt = self._teacher_transport(ds_t, base, **self.rc.reflow_kwargs)
        return DataStruct(torch.cat((tgt, ds_t.f.pdgids), dim=-1), ds_t.m.full, ds_t.am.full)
