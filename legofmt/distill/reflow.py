"""Reflow, and the one-step ``Direct`` variants.

Reflow re-couples the base to a frozen teacher's transport of it; with no
``reflow_path`` the student snapshots itself as its own teacher each epoch.
It is gated solely by ``reflow_start_epoch > 0`` -- there is no on/off flag.
``LEGOLtngDirect`` / ``GenerateOutDirect`` drop time conditioning and predict
the residual ``target - base`` in a single forward.
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
        return_species: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        _, x_att = self._prep_x(x, attn_mask)
        res = self.vf(x_att, mask, attn_mask, types, pdgids, return_species=return_species)
        delta, species = res if return_species else (res, None)
        out_raw  = x_att + delta
        gen = (mask == 1).unsqueeze(-1)
        ref = torch.zeros_like(out_raw)
        ref[..., [1, 4]] = 1.0
        safe = torch.where(gen, out_raw, ref)  # projx-safe filler at conditioning slots
        out = torch.where(gen, self.manifold.projx(safe), out_raw)
        return (out, species) if return_species else out


class LEGOLtngDirect(LEGOLtng):
    """Direct residual model: predicts target - base in one step. With
    model_conf.reflow_path set, a frozen velocity teacher's solve(base)
    replaces the data target (fixed base->target reflow coupling).
    Teacher construction/device placement is inherited from LEGOLtng."""

    def _build_model(self, rc) -> nn.Module:
        return ProjectModelDirect(
            CFMTrafo_x(**rc.model_args, time_cond=False),
            rc.manifold,
        )

    def _reflow_target(self, ds_t: DataStruct, base: Tensor) -> Tensor:
        """The teacher's transport of ``base``, or the data target without one."""
        if self.reflow_teacher is None:
            return ds_t.f.model_in
        solve_kwargs = dict(self.rc.reflow_kwargs)
        if solve_kwargs.get("method", "midpoint") == "midpoint":
            solve_kwargs.setdefault("time_grid", base.new_tensor([0.0, 1.0]))
        return self.reflow_teacher.solve(ds_t, x_init=base, **solve_kwargs)

    def _step(self, ds_t: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        """Mirrors ``TrainStep._step``'s layout -- every outgoing slot a candidate,
        pad slots targeting their own base draw, pass-through events weighted out --
        but the class comes in at the uniform source rather than a ``t``-corruption:
        there is no trajectory, so the head predicts ``x_1``'s class in one shot."""
        n0 = self.rc.n_prefix + 1
        with torch.no_grad():
            m      = self._sample_mask(ds_t)
            gen_sp = (m[:, self.rc.n_prefix] == 0).unsqueeze(-1)
            m      = m.clone()
            m[:, n0:] |= gen_sp
            ds_t   = DataStruct(ds_t.f.full, m, ds_t.am.full)
            pt_tgt = ((ds_t.f.edep <= 0) & (ds_t.am.out_p.sum(-1) == 1)).float()
            pt_idx = self._in_species(ds_t)
            pt_ok  = self.pt_allowed(ds_t, pt_idx) & gen_sp.squeeze(-1)
            w      = 1.0 - pt_tgt * pt_ok
            base   = self.gen_base_wrapper(ds_t)
            s1     = self.convert_pdgids(ds_t.f.pdgids).squeeze(-1)
            rows   = (torch.arange(m.shape[1], device=m.device) >= n0) & gen_sp
            s_0    = torch.where(
                rows, torch.randint_like(s1, self.rc.model_args["npdgids"]), s1)
            pad    = (~ds_t.am.full & rows).unsqueeze(-1)
            ds_t   = DataStruct(
                torch.cat((torch.where(pad, base, ds_t.f.model_in), ds_t.f.pdgids), dim=-1),
                m, ds_t.am.full | rows,
            )
            target = self._reflow_target(ds_t, base)
        pred, sp_logits = self.model(
            base,
            mask=ds_t.m.full, attn_mask=ds_t.am.full,
            types=self.types_embd, pdgids=s_0, return_species=True,
        )
        sq = (pred - target) ** 2
        loss, logs = self.reduce_loss(sq, ds_t, w)
        if pt_ok.any():
            loss = loss + self.rc.passthrough_fac * nn.functional.binary_cross_entropy_with_logits(
                self.pt_logits(ds_t, pt_idx)[pt_ok], pt_tgt[pt_ok])
        rows = rows & (w > 0).unsqueeze(-1)
        if rows.any():
            loss = loss + self.rc.species_fac * nn.functional.cross_entropy(
                sp_logits[rows], s1[rows])
        if logs:
            self.log_dict(logs, on_step=True, on_epoch=False, logger=True, sync_dist=False)
        return loss

    @torch.no_grad()
    def solve(
        self,
        ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]",
        x_init: Tensor | None = None,
        split_size: int | None = None,
        return_species: bool = False,
        **_kw,
    ) -> Tensor | tuple[Tensor, Tensor]:
        ds_t, pdgids_idx = self._prep_solve(ds_t)
        if x_init is None:
            x_init = self.gen_base_wrapper(ds_t)

        sp  = pdgids_idx.reshape(*ds_t.m.full.shape).long()
        gen = self._species_gen(ds_t.m.full, sp)
        sp  = torch.where(gen, torch.randint_like(sp, self.rc.model_args["npdgids"]), sp)

        def _fwd(x, m, a, pi, g):
            v, lg = self.model(
                x, mask=m, attn_mask=a, types=self.types_embd, pdgids=pi,
                return_species=True,
            )
            return torch.cat(
                (v, torch.where(g, lg.argmax(-1), pi).unsqueeze(-1).to(v.dtype)), dim=-1)

        res = self.chunked(
            _fwd, x_init, ds_t.m.full, ds_t.am.full, sp, gen,
            split_size=split_size,
        )
        out, sp_out = res[..., :-1], res[..., -1].long()
        return (out, sp_out) if return_species else out


class GenerateOutDirect(GenerateOut):
    """Direct variant -- uses LEGOLtngDirect as the flow component."""
    flow_cls = LEGOLtngDirect
