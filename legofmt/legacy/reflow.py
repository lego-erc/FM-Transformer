"""Reflow's teacher and the one-step models, both superseded by the flow-map term (``cfm/flow_map.py``).

Reflow re-couples the base to a frozen teacher's transport of it. A ``reflow_path`` teacher is used from step 0;
``reflow_start_epoch > 0`` makes the student's end-of-epoch snapshot the teacher from that epoch on (self-reflow
without a path, a refresh with one). Neither set: no reflow. The training-step side is
:mod:`legofmt.legacy.train_step`, which imports the builders here lazily (this module subclasses the production
classes, so nothing in production may import it at module level).

The one-step models ``LEGOLtngDirect`` / ``GenerateOutDirect`` have a field without time (``time_cond: false``) that
predicts the residual ``target - base`` in a single forward.
"""

import copy
import warnings
from pathlib import Path

import torch
from torch import Tensor, nn

from legofmt.cfm.cfm_trafo_x import CFMTrafo_x
from legofmt.data.struct import DataStruct
from legofmt.main.generate import GenerateOut
from legofmt.cfm.project_model import ProjectModel
from legofmt.main.modules import LEGOLtng


def _teacher_from_state(config: dict, state_dict: dict) -> nn.Module:
    """Constructs the frozen, compiled teacher (a ``LEGOLtng``) from a config and the field's state dict."""
    conf = copy.deepcopy(config)
    conf["model_conf"]["reflow_path"] = None  # the teacher must not load a teacher of its own
    teacher = LEGOLtng({"config": conf, "state_dict": state_dict})
    teacher.eval().requires_grad_(False)
    teacher.model = torch.compile(teacher.model, dynamic=False)
    return teacher


def _build_reflow_teacher(reflow_path: str | None) -> nn.Module | None:
    """Loads the teacher from the checkpoint ``reflow_path``.

    Returns None without a path, and with a warning when the file is missing.
    """
    if reflow_path is None:
        return None
    if not Path(reflow_path).is_file():
        warnings.warn(f"reflow_path={reflow_path!r} not found; reflow disabled.", stacklevel=2)
        return None
    ckpt = torch.load(reflow_path, map_location="cpu", weights_only=False)
    return _teacher_from_state(ckpt["config"], ckpt["state_dict"])


class ProjectModelDirect(ProjectModel):
    """The ``ProjectModel`` of the one-step model: it returns ``x + field(x)``, projected onto the manifold.

    The field takes no time and predicts a residual in place of a velocity. Only the generated tokens are projected;
    a conditioning token, whose filler ``projx`` cannot take, is returned as it is.
    """

    def forward(self, x: Tensor, mask: Tensor, attn_mask: Tensor, types: Tensor,
                pdgids: Tensor | None = None) -> Tensor:
        _, x_att = self._prep_x(x, attn_mask)
        out_raw  = x_att + self.vf(x_att, mask, attn_mask, types, pdgids)
        gen = (mask == 1).unsqueeze(-1)
        ref = torch.zeros_like(out_raw)
        ref[..., [1, 4]] = 1.0
        safe = torch.where(gen, out_raw, ref)  # on conditioning tokens a filler that projx can take
        return torch.where(gen, self.manifold.projx(safe), out_raw)


class LEGOLtngDirect(LEGOLtng):
    """The one-step model: a field without time that maps a sample of the base to its target in one forward.

    With ``model_conf.reflow_path`` a frozen teacher's ``solve`` of the base replaces the data as the target. The
    teacher is built and moved to the device as in ``LEGOLtng``.
    """

    def _build_model(self, rc) -> nn.Module:
        return ProjectModelDirect(CFMTrafo_x(**rc.model_args, time_cond=False), rc.manifold)

    def _reflow_target(self, ds_t: DataStruct, base: Tensor) -> Tensor:
        """Returns the teacher's transport of ``base``, or the data target without a teacher."""
        if self.reflow_teacher is None:
            return ds_t.f.model_in
        return self._teacher_transport(ds_t, base, **self.rc.reflow_kwargs)

    def _step(self, ds_t: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        with torch.no_grad():
            ds_t = DataStruct(ds_t.f.full, self._sample_mask(ds_t), ds_t.am.full)
            base = self.gen_base_wrapper(ds_t)
            pdgid_idx = self.convert_pdgids(ds_t.f.pdgids)
            target = self._reflow_target(ds_t, base)
        pred = self.model(base, mask=ds_t.m.full, attn_mask=ds_t.am.full, types=self.types_embd, pdgids=pdgid_idx)
        loss, logs = self.reduce_loss((pred - target) ** 2, ds_t)
        if logs:
            self.log_dict(logs, on_step=True, on_epoch=False, logger=True, sync_dist=False)
        return loss

    def _solve(self, ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]", x_init: Tensor | None = None,
               split_size: int | None = None, **_kw) -> Tensor:
        """Solves with one forward of the field; the solver arguments in ``_kw`` are not read.

        The inherited ``solve`` (under ``torch.no_grad``) and ``_solve_by_length`` both call this.
        """
        ds_t, pdgids_idx = self._prep_solve(ds_t)
        if x_init is None:
            x_init = self.gen_base_wrapper(ds_t)
        fwd = lambda x, m, a, pi: self.model(x, mask=m, attn_mask=a, types=self.types_embd, pdgids=pi)
        return self.chunked(fwd, x_init, ds_t.m.full, ds_t.am.full, pdgids_idx, split_size=split_size)


class GenerateOutDirect(GenerateOut):
    """``GenerateOut`` with ``LEGOLtngDirect`` as its flow."""
    flow_cls = LEGOLtngDirect
