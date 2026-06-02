"""Direct (residual-prediction) mirror of :class:`legofmt.main.modules.LEGOLtng`.

The transformer predicts the residual ``target - base`` at the generated
slots; the wrapper applies one Euler step ``final = base + residual``
(plus a manifold snap on the sphere half) and returns the predicted
target. The loss is MSE between that predicted target and the per-batch
training target.

When ``model_conf.reflow_path`` is set to a velocity-model checkpoint,
this module loads it as a frozen teacher and uses ``teacher.solve(base)``
as the training target. That replaces the data target with the velocity
model's deterministic ODE map, giving the direct student a fixed
``base -> target`` coupling per batch (the prerequisite for one-step
Euler to reach the target). When the path is unset or missing, training
falls back to ``MSE(pred, data target)`` as before.
"""

import warnings
from pathlib import Path

import torch
from torch import Tensor, nn

from legofmt.data.struct import DataStruct
from legofmt.cfm.cfm_trafo_direct import CFMTrafo_x
from legofmt.main.modules import (
    LEGOLtng as LEGOLtngVelocity,
    ProjectModel as ProjectModelVelocity,
)


class ProjectModel(ProjectModelVelocity):
    """Residual-prediction wrapper for the no-time direct model.

    The transformer predicts the offset ``target - base`` at gen slots; the
    wrapper applies one Euler step ``final = base + residual`` and snaps the
    sphere half back to the manifold, returning the predicted target. Init
    is inherited from :class:`legofmt.main.modules.ProjectModel`; only the
    forward differs (no time argument, residual instead of velocity).
    """

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        attn_mask: torch.Tensor,
        types: torch.Tensor,
        pdgids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, x_attended, x_surr = self._prep_x(x, attn_mask)
        residual = self.vf(x_surr, mask, attn_mask, types, pdgids)
        out_raw = x_attended + residual
        gen = (mask == 1).unsqueeze(-1)
        ref = torch.zeros_like(out_raw)
        ref[..., 3] = 1.0
        safe = torch.where(gen, out_raw, ref)
        out_proj = self.manifold.projx(safe)
        return torch.where(gen, out_proj, out_raw)


class LEGOLtng(LEGOLtngVelocity):
    def __init__(self, full_config: dict) -> None:
        super().__init__(full_config)
        object.__setattr__(self.rc, "ot_coupling", False)
        object.__setattr__(
            self, "reflow_teacher",
            self._maybe_build_reflow_teacher(self.rc.reflow_path),
        )

    def _build_model(self, rc) -> nn.Module:
        return ProjectModel(
            CFMTrafo_x(**rc.model_args),
            rc.manifold,
            cond_cube=rc.cond_cube,
        )

    @staticmethod
    def _maybe_build_reflow_teacher(reflow_path: str | None) -> nn.Module | None:
        """Load the velocity-model checkpoint at ``reflow_path`` as a frozen,
        ``torch.compile``'d teacher whose ``solve`` provides the reflow
        target. ``None`` when unset or missing (so a saved direct checkpoint
        can be reloaded after the teacher file is gone)."""
        if reflow_path is None:
            return None
        if not Path(reflow_path).is_file():
            warnings.warn(f"reflow_path={reflow_path!r} not found; reflow disabled.", stacklevel=2)
            return None
        teacher = LEGOLtngVelocity(
            torch.load(reflow_path, map_location="cpu", weights_only=False)
        )
        teacher.eval()
        teacher.requires_grad_(False)
        try:
            teacher.model = torch.compile(teacher.model, dynamic=False)
        except Exception as exc:
            warnings.warn(
                f"torch.compile of reflow teacher failed ({exc!r}); eager fallback.",
                stacklevel=2,
            )
        return teacher

    @torch.no_grad()
    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.reflow_teacher is not None:
            self.reflow_teacher.to(self.device)

    def _step(self, ds_t: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        with torch.no_grad():
            base = self.gen_base_wrapper(ds_t)
            pdgid_idx = self.convert_pdgids(ds_t.f.pdgids)
            if self.reflow_teacher is not None:
                solve_kwargs = dict(self.rc.reflow_kwargs)
                if (
                    "time_grid" not in solve_kwargs
                    and solve_kwargs.get("method", "midpoint") == "midpoint"
                ):
                    solve_kwargs["time_grid"] = base.new_tensor([0.0, 1.0])
                target = self.reflow_teacher.solve(
                    ds_t, x_init=base, **solve_kwargs,
                )
            else:
                target = ds_t.f.model_in
        pred = self.model(
            base,
            mask=ds_t.m.full, attn_mask=ds_t.am.full,
            types=self.types_embd, pdgids=pdgid_idx,
        )
        if self.rc.loss_sc_fac > 0:
            m_gen = (ds_t.m.full == 1).to(pred.dtype)
            loss_sc = self.loss_fn(
                pred[..., :3].norm(dim=-1) * m_gen,
                target[..., :3].norm(dim=-1) * m_gen,
            )
        else:
            loss_sc = pred.new_zeros(())
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
