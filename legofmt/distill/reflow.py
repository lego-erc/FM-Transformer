import copy
import warnings
from pathlib import Path

import torch
from torch import Tensor, nn

from legofmt.cfm.cfm_trafo_x import CFMTrafo_x
from legofmt.data.struct import DataStruct
from legofmt.main.generate import GenerateOut
from legofmt.main.modules import LEGOLtng, ProjectModel


def _teacher_from_state(config: dict, state_dict: dict) -> nn.Module:
    """Frozen, ``torch.compile``'d velocity teacher from a config + vf state dict."""
    conf = copy.deepcopy(config)
    conf["model_conf"]["reflow_path"] = None  # teacher must not chain-load its own teacher
    teacher = LEGOLtng({"config": conf, "state_dict": state_dict})
    teacher.eval().requires_grad_(False)
    teacher.model = torch.compile(teacher.model, dynamic=False)
    return teacher


def _build_reflow_teacher(reflow_path: str | None) -> nn.Module | None:
    """Teacher loaded from a checkpoint path; ``None`` if path unset/missing."""
    if reflow_path is None:
        return None
    if not Path(reflow_path).is_file():
        warnings.warn(f"reflow_path={reflow_path!r} not found; reflow disabled.", stacklevel=2)
        return None
    ckpt = torch.load(reflow_path, map_location="cpu", weights_only=False)
    return _teacher_from_state(ckpt["config"], ckpt["state_dict"])


class ProjectModelDirect(ProjectModel):
    """Residual-prediction wrapper for the no-time direct model. Mirror
    of :class:`ProjectModel`; only the forward differs (no time argument,
    residual instead of velocity, single Euler step + safe sphere snap)."""

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        attn_mask: Tensor,
        types: Tensor,
        pdgids: Tensor | None = None,
    ) -> Tensor:
        _, x_att = self._prep_x(x, attn_mask)
        out_raw  = x_att + self.vf(x_att, mask, attn_mask, types, pdgids)
        gen = (mask == 1).unsqueeze(-1)
        ref = torch.zeros_like(out_raw)
        ref[..., [1, 4]] = 1.0
        safe = torch.where(gen, out_raw, ref)  # projx-safe filler at conditioning slots
        return torch.where(gen, self.manifold.projx(safe), out_raw)


class LEGOLtngDirect(LEGOLtng):
    """Direct residual model: predicts target - base in one step. With
    model_conf.reflow_path set, a frozen velocity teacher's solve(base)
    replaces the data target (fixed base->target reflow coupling).
    Teacher construction/device placement is inherited from LEGOLtng."""

    def _build_model(self, rc) -> nn.Module:
        return ProjectModelDirect(
            CFMTrafo_x(**rc.model_args, time_cond=False),
            rc.manifold,
            cond_cube=rc.cond_cube,
        )

    def _step(self, ds_t: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        with torch.no_grad():
            ds_t = DataStruct(ds_t.f.full, self._sample_mask(ds_t), ds_t.am.full)
            base = self.gen_base_wrapper(ds_t)
            pdgid_idx = self.convert_pdgids(ds_t.f.pdgids)
            if self.reflow_teacher is not None:
                solve_kwargs = dict(self.rc.reflow_kwargs)
                if solve_kwargs.get("method", "midpoint") == "midpoint":
                    solve_kwargs.setdefault("time_grid", base.new_tensor([0.0, 1.0]))
                target = self.reflow_teacher.solve(ds_t, x_init=base, **solve_kwargs)
            else:
                target = ds_t.f.model_in
        pred = self.model(
            base,
            mask=ds_t.m.full, attn_mask=ds_t.am.full,
            types=self.types_embd, pdgids=pdgid_idx,
        )
        if self.rc.loss_sc_fac > 0:
            m_gen = (ds_t.m.full == 1).to(pred.dtype)
            loss_sc = self.loss_fn(pred[..., 0] * m_gen, target[..., 0] * m_gen)
        else:
            loss_sc = 0.0
        sq = (pred - target) ** 2
        return self._reduce_and_log(sq, ds_t, loss_sc)

    @torch.no_grad()
    def solve(
        self,
        ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]",
        x_init: Tensor | None = None,
        split_size: int | None = None,
        **_kw,
    ) -> Tensor:
        ds_t, pdgids_idx = self._prep_solve(ds_t)
        if x_init is None:
            x_init = self.gen_base_wrapper(ds_t)

        def _fwd(x, m, a, pi):
            return self.model(
                x, mask=m, attn_mask=a, types=self.types_embd, pdgids=pi,
            )

        return self.chunked(
            _fwd, x_init, ds_t.m.full, ds_t.am.full, pdgids_idx,
            split_size=split_size,
        )


class GenerateOutDirect(GenerateOut):
    """Direct variant -- uses LEGOLtngDirect as the flow component."""
    flow_cls = LEGOLtngDirect
