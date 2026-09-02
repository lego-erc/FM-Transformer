from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.utils.data import (
    BatchSampler, DataLoader, RandomSampler, SequentialSampler, random_split,
)

import lightning as ltng

from flow_matching.solver import ODESolver
from flow_matching.utils.manifolds import Sphere

try:
    from torch_lap_cuda_lib import solve_lap as slap
except ImportError:
    slap = None

_OT_COUPLING_REQUIRES_LAP = (
    "ot_coupling=True requires `torch_lap_cuda_lib`. "
    "Install it or set model_conf.ot_coupling=False."
)

# Conditioning scalars base_head_params reads; no base_head without them.
_BASE_HEAD_SCALARS = ("Z", "A", "Size")

from legofmt.cfm.cfm_trafo_x import CFMTrafo_x

from legofmt.data.dataloaders import LEGODataset
from legofmt.data.prep import DataPrep
from legofmt.data.struct import DataStruct, _F

from legofmt.distill.distill import curvature_loss, one_step_euler_loss

from legofmt.geometry.geom_trafos import GeomTrafos
from legofmt.geometry.gen_base import GenerateBase
from legofmt.geometry.path_sample_mult import ProductPathSampler, ProductManifold
from legofmt.geometry.raytracing_proj import CubeTrace
from legofmt.geometry.symmetry_projections import CubeSymmetry

from legofmt.mod_comps.config import resolve_legoltng_config
from legofmt.mod_comps.optimizers import build_optimizer

from legofmt.log_metrics.val_metrics import ShowerValMetrics


class ProjectModel(nn.Module):
    """Projection Wrapper for Riemannian FM Model."""

    def __init__(self, vf: nn.Module, manifold: ProductManifold, **kwargs) -> None:
        super().__init__()
        self.vf          = vf
        self.manifold    = manifold
        self.geom_trafos = GeomTrafos()
        self.cond_cube   = kwargs.get("cond_cube", False)
        self.no_detach   = kwargs.get("no_detach", False)

    def _prep_x(self, x: Tensor, attn_mask: Tensor) -> tuple[Tensor, Tensor]:
        x_proj = self.manifold.projx(x)
        x_att  = torch.where(attn_mask.unsqueeze(-1), x_proj, x)
        if not self.no_detach:
            x_att.detach_()
        if self.cond_cube:
            x_att = x_att.clone()
            in_p  = _F(x_att).in_p
            in_p.copy_(self.geom_trafos.to_cube(in_p))
        return x_proj, x_att

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        mask: Tensor,
        attn_mask: Tensor,
        types: Tensor,
        pdgids: Tensor | None = None,
        d: Tensor | None = None,
    ) -> Tensor:
        x_proj, x_att = self._prep_x(x, attn_mask)
        t = torch.atleast_2d(t).expand_as(attn_mask)
        t = torch.where(mask == 1, t, 1.0)  # conditions get t=1
        if getattr(self.vf, "step_cond", False):
            d = torch.zeros_like(t) if d is None else torch.atleast_2d(d).expand_as(attn_mask)
            d = torch.where(mask == 1, d, 0.0)  # conditions are not transported
        v = self.vf(x_att, mask, attn_mask, types, pdgids, t=t, d=d)
        v_proj = self.manifold.proju(x_proj, v)
        return torch.where(attn_mask.unsqueeze(-1), v_proj, v)


class LEGOLtng(ltng.LightningModule):

    def __init__(self, full_config: dict) -> None:
        super().__init__()
        self.rc = resolve_legoltng_config(full_config)

        types_embd = torch.arange(self.rc.max_seq_l, dtype=torch.int64).clamp_max(self.rc.n_prefix + 1).view(1, -1)

        self.register_buffer("pdgids_template", self.rc.pdgids_template)
        self.register_buffer("types_embd", types_embd)

        self.model       = self._build_model(self.rc)
        self.gen_base    = GenerateBase(self.rc.config)
        self.sym         = CubeSymmetry() if self.rc.canon_sym else None
        self.val_metrics = ShowerValMetrics()
        self.loss_fn     = nn.MSELoss()
        self.ps          = ProductPathSampler(self.rc.manifold)

        self._base_dist_loss = None
        params = list(self.model.parameters())
        bc = self.rc.config.get("base_conf") or {}
        # base_head_params reads Z/A/Size, so there is no head without them: an
        # explicit ask (base_dist_loss, or a saved head that would otherwise
        # vanish silently) is an error, the base_pretrain_batches default is not.
        asked = self.rc.base_dist_loss > 0 or bc.get("base_head") is not None
        want = ((asked or self.rc.base_pretrain_batches > 0)
                and self.gen_base.scale_dist == "sm_norm")
        if want and not all(s in self.rc.cond_scalars for s in _BASE_HEAD_SCALARS):
            if asked:
                raise ValueError(
                    f"base_head needs the {_BASE_HEAD_SCALARS} conditioning scalars; "
                    f"got cond_scalars={self.rc.cond_scalars}."
                )
            want = False
        if want:
            self.base_head = nn.Sequential(
                nn.Linear(4 + len(self.rc.pdgids_template), 16), nn.Mish(),
                nn.Linear(16, 4))
            with torch.no_grad():
                self.base_head[-1].weight.zero_()
                self.base_head[-1].bias.copy_(torch.tensor(
                    [float(self.gen_base.sm_scale), 1.0, 1.0,
                     float(self.gen_base.kappa)]).log())
                hs = bc.get("base_head")
                if hs is not None and hs.keys() == self.base_head.state_dict().keys():
                    if hs["2.weight"].shape == self.base_head[-1].weight.shape:
                        self.base_head.load_state_dict(hs)
                    else:

                        n_old = hs["2.weight"].shape[0]
                        self.base_head[0].load_state_dict(
                            {"weight": hs["0.weight"], "bias": hs["0.bias"]})
                        self.base_head[-1].weight[:n_old].copy_(hs["2.weight"])
                        self.base_head[-1].bias[:n_old].copy_(hs["2.bias"])
                    self.base_head.requires_grad_(not bc.get("base_head_frozen", False))
            params += [p for p in self.base_head.parameters() if p.requires_grad]
        if self.rc.uncert_weighting:
            self.lv = nn.Parameter(torch.zeros(self.rc.uncert_bins * 3))
            params += [self.lv]
            if self.rc.one_step_euler_fac > 0:
                self.lv_flow = nn.Parameter(torch.zeros(1))
                params += [self.lv_flow]

        self.opt, self._lr_sched = build_optimizer(params, self.rc.opt_conf)
        self._opt_is_sf = callable(getattr(self.opt, "train", None))

        if self.rc.state_dict is not None:
            self.model.vf.load_state_dict(self.rc.state_dict, strict=False)

        teacher = None
        if self.rc.reflow_path is not None:
            from legofmt.distill.reflow import _build_reflow_teacher  # avoids import cycle
            teacher = _build_reflow_teacher(self.rc.reflow_path)
        object.__setattr__(self, "reflow_teacher", teacher)  # not a submodule: no state_dict/DDP

    def _build_model(self, rc) -> nn.Module:
        return ProjectModel(
            CFMTrafo_x(**rc.model_args),
            rc.manifold,
            cond_cube=rc.cond_cube,
        )

    def _opt_train(self) -> None:
        if getattr(self, "_opt_is_sf", False):
            self.opt.train()

    def _opt_eval(self) -> None:
        if getattr(self, "_opt_is_sf", False):
            self.opt.eval()

    def on_validation_model_eval(self) -> None:
        super().on_validation_model_eval()
        self._opt_eval()

    def on_validation_model_train(self) -> None:
        super().on_validation_model_train()
        self._opt_train()

    @torch.no_grad()
    def on_fit_start(self) -> None:
        if self.rc.ot_coupling and slap is None:
            raise RuntimeError(_OT_COUPLING_REQUIRES_LAP)
        if self.reflow_teacher is not None:
            self.reflow_teacher.to(self.device)
        self.model.train()
        self._opt_train()

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        start = self.rc.reflow_start_epoch
        if start <= 0 or self.current_epoch + 1 < start:
            return
        vf = (self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model).vf
        if self.reflow_teacher is None:  # self-reflow: teacher = snapshot of the student
            from legofmt.distill.reflow import _teacher_from_state  # avoids import cycle
            teacher = _teacher_from_state(self.rc.config, vf.state_dict())
            object.__setattr__(self, "reflow_teacher", teacher.to(self.device))
        else:  # refresh: teacher = last epoch's student
            t_m = self.reflow_teacher.model
            (t_m._orig_mod if hasattr(t_m, "_orig_mod") else t_m).vf.load_state_dict(vf.state_dict())

    @torch.no_grad()
    def convert_pdgids(self, pdgids: Tensor) -> Tensor:
        cond = torch.isnan(pdgids) | (pdgids == 0) | (pdgids >= 1e8)
        pdgid_idx = torch.searchsorted(
            self.pdgids_template.to(pdgids.device), pdgids.contiguous()) + 1
        return pdgid_idx.masked_fill_(cond, 0)

    def _shift_overflow_targets(self, ds_t: DataStruct) -> DataStruct:
        f = ds_t.f.full.clone()
        if self.rc.overflow_delta > 0:
            op = _F(f).out_p
            op[..., 0].masked_fill_((op[..., 0] == 0) & ds_t.am.out_p.bool(),
                                    -self.rc.overflow_delta)
        if self.rc.edep_overflow_delta > 0:
            edep = _F(f).edep
            edep.masked_fill_(edep == 0, -self.rc.edep_overflow_delta)
        return DataStruct(f, ds_t.m.full, ds_t.am.full)

    def _canon_dirs(self, x: Tensor, face: Tensor, fwd: Tensor, inverse: bool = False) -> Tensor:
        rot  = self.sym.uncanonicalize if inverse else self.sym.canonicalize
        dirs = rot(torch.stack((x[..., 1:4], x[..., 4:7]), dim=-2), face)
        new  = torch.cat((x[..., 0:1], dirs[..., 0, :], dirs[..., 1, :], x[..., 7:]), dim=-1)
        rows = torch.arange(x.shape[-2], device=x.device) >= self.rc.n_prefix
        new  = torch.where(rows.view(1, -1, 1), new, x)
        return torch.where(fwd[:, None, None], new, x)

    def base_head_params(self, ds_t: DataStruct) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pid     = ds_t.f.in_p[..., 0, -1]
        idx     = pid.long() if self.rc.pdgid_is_idx else self.convert_pdgids(pid)
        species = nn.functional.one_hot(
            idx.long().clamp(0, len(self.pdgids_template)),
            len(self.pdgids_template) + 1).float()
        Z, A    = ds_t.f.cond("Z"), ds_t.f.cond("A")
        x0      = 716.4 * A / (Z * (Z + 1) * (287.0 / Z.sqrt()).log())
        t       = ds_t.f.cond("Size") * ds_t.f.cond("Density") / x0
        # cord length: full chord through the cube along the incoming
        # ray (edge lengths); fwd+bwd -> entry/exit-storage invariant
        inc     = ds_t.f.in_cc[..., 0, 1:7].nan_to_num(1.0)
        u, pos  = inc[..., :3], inc[..., 3:]
        p       = pos / pos.abs().amax(-1, keepdim=True).clamp_min(1e-8)
        ok_u    = u.abs() > 1e-6
        tf      = torch.where(ok_u, (u.sign() - p) / u, torch.full_like(u, 4.0))
        tb      = torch.where(ok_u, (p + u.sign()) / u, torch.full_like(u, 4.0))
        chord   = ((tf.amin(-1) + tb.amin(-1)).clamp(0.0, 3.5) / 2).unsqueeze(-1)
        x       = torch.cat((ds_t.f.in_cc[..., 0], species, chord,
                             t.log().unsqueeze(-1)), dim=-1)
        out     = self.base_head(x)
        return (out[..., 0:1].exp(), out[..., 1:2], out[..., 2:3].exp(),
                out[..., 3:4].exp().clamp_min(1e-3))  # kap: divisor in sample()

    def gen_base_wrapper(self, ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]") -> Tensor:
        if not isinstance(ds_t, DataStruct):
            ds_t = DataStruct(*ds_t)
        self._base_dist_loss = None
        data  = ds_t.f.model_in
        m     = ds_t.m.full
        fwd   = m[:, self.rc.n_prefix] == 0  # incoming slot conditions => forward event
        noise = None if fwd.all() else self.gen_base.iso(m.shape, data.device)
        if fwd.any():
            if hasattr(self, "base_head"):
                learn = self.model.training and self.base_head[-1].weight.requires_grad
                with torch.set_grad_enabled(learn):
                    s, mu, sig, kap = self.base_head_params(ds_t)
                    if learn:
                        z   = 2 ** 0.5 * torch.erfinv(torch.linspace(
                            -0.995, 0.995, 100, device=s.device))  # N(0,1) quantile nodes
                        u_b = 1 - (z.abs() * s).tanh().mean(-1)
                        v   = (ds_t.am.out_p & fwd.unsqueeze(-1)).float()
                        n   = v.sum(-1).clamp(min=1)
                        u_t = (ds_t.f.out_cc[..., 0] * v).sum(-1) / n
                        ev  = (v.sum(-1) > 0).float()
                        l_scale = ((u_b - u_t) ** 2 * ev).sum() / ev.sum().clamp(min=1)
                        ed_b = self.gen_base.e_dep_max * torch.sigmoid(mu + sig * z)
                        ed_t = ds_t.f.edep
                        w    = fwd.float()
                        l_edep = (((ed_b.mean(-1) - ed_t) ** 2
                                   + (ed_b.pow(2).mean(-1) - ed_t ** 2) ** 2) * w
                                  ).sum() / w.sum().clamp(min=1)
                        l_kappa = 0.0
                        if self.gen_base.tanh_theta:
                            th_b = (z.abs() / kap).tanh().mean(-1)
                            u_i  = nn.functional.normalize(
                                ds_t.f.in_cc[..., 0, 1:4], dim=-1).unsqueeze(1)
                            p_i  = nn.functional.normalize(
                                ds_t.f.in_cc[..., 0, 4:7], dim=-1).unsqueeze(1)
                            a_m  = (ds_t.f.out_cc[..., 1:4] * u_i).sum(-1).clamp(-1, 1).acos()
                            a_p  = (ds_t.f.out_cc[..., 4:7] * p_i).sum(-1).clamp(-1, 1).acos()
                            th_t = (((a_m + a_p) / 2) * v).sum(-1) / n / torch.pi
                            l_kappa = ((th_b - th_t) ** 2 * ev).sum() / ev.sum().clamp(min=1)
                        self._base_dist_loss = l_scale + l_edep + l_kappa
                self.gen_base.sm_scale = s.detach().unsqueeze(-1)
                self.gen_base.edep_mu  = mu.detach()
                self.gen_base.edep_sig = sig.detach()
                self.gen_base.kappa    = kap.detach()
            base = torch.cat(
                (ds_t.f.non_cc, self.gen_base(ds_t.m.out_p.shape, ds_t.f.in_cc)), dim=1,
            )
            if self.rc.ot_coupling and self.model.training:
                if slap is None:
                    raise RuntimeError(_OT_COUPLING_REQUIRES_LAP)
                base = base.where(ds_t.am.full.unsqueeze(-1), data)
                inf_cond = ds_t.am.out_p.unsqueeze(-1).logical_xor(ds_t.am.out_p.unsqueeze(-2))
                if self.rc.ot_same_pdgid:
                    pid = ds_t.f.out_p[..., -1]
                    inf_cond = inf_cond | (pid.unsqueeze(-1) != pid.unsqueeze(-2))
                out = _F(base).out_p
                if self.rc.ot_e_only:
                    nt = ds_t.f.out_cc[..., 0:1]
                    nb = out[..., 0].unsqueeze(-2)
                    cost = (nt - nb).abs() + inf_cond * 1e6
                else:
                    man = self.rc.manifold
                    tgt = ds_t.f.out_cc.unsqueeze(-2).split(man.ambient_dims, dim=-1)
                    ref = out.unsqueeze(-3).split(man.ambient_dims, dim=-1)
                    cost = sum(
                        ((a * b).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6).acos()
                         if isinstance(mf, Sphere) else (a - b).norm(dim=-1)) ** 2
                        for mf, a, b in zip(man.manifolds, tgt, ref)
                    ) + inf_cond * 1e6
                assign = slap(cost, cost.device).long()
                out[:] = torch.take_along_dim(out, assign.unsqueeze(-1), dim=1)
            base = self.gen_base.insert_add(base)
            noise = base if noise is None else torch.where(fwd.view(-1, 1, 1), base, noise)
        return torch.where((m == 1).unsqueeze(-1), noise, data)

    def _reduce_and_log(
        self, sq: Tensor, ds_t: DataStruct, loss_sc, t: Tensor | None = None,
    ) -> Tensor:
        g        = ((ds_t.m.full == 1) & (ds_t.am.full == 1)).unsqueeze(-1)
        denom    = g.sum().clamp(min=1)
        out      = sq * g
        loss_e   = out[..., 0:1].sum() / denom
        loss_dir = out[..., 1:4].sum() / (denom * 3)
        loss_x   = out[..., 4:7].sum() / (denom * 3)
        logs     = {}
        if not self.rc.uncert_weighting:
            total = loss_e + loss_dir + loss_x
        else:
            if t is None:
                raise ValueError(
                    "uncert_weighting=True needs the per-event t, which the "
                    "time-independent (LEGOLtngDirect) path does not have."
                )
            nb  = self.rc.uncert_bins
            idx = (t.detach().clamp(0, 1) * nb).long().clamp(max=nb - 1)
            lv  = self.lv.view(nb, 3)[idx].clamp(min=self.rc.uncert_min)  # (B, 3)
            w   = torch.exp(-lv)
            total = (
                (out[..., 0:1] * w[:, 0].view(-1, 1, 1)).sum() / denom
                + (out[..., 1:4] * w[:, 1].view(-1, 1, 1)).sum() / (denom * 3)
                + (out[..., 4:7] * w[:, 2].view(-1, 1, 1)).sum() / (denom * 3)
                + lv.mean(0).sum()  # the +log(sigma^2) barrier; zero at init
            )
            v = lv.detach().exp().mean(0)
            logs = {"uncert/var_energy": v[0], "uncert/var_dir": v[1], "uncert/var_pos": v[2]}
        if self.training:
            log_sc = loss_sc.detach() if torch.is_tensor(loss_sc) else loss_sc
            self.log_dict(
                {
                    # unweighted, so these stay comparable across uncert settings
                    "loss/energy": loss_e.detach(),
                    "loss/out_dir": loss_dir.detach(),
                    "loss/out_pos": loss_x.detach(),
                    "loss/raw": (loss_e + loss_dir + loss_x).detach(),
                    "loss/sc": log_sc,
                    **logs,
                },
                on_step=True, on_epoch=False, logger=True, sync_dist=False,
            )
        return total + self.rc.loss_sc_fac * loss_sc

    @torch.no_grad()
    def _sample_mask(self, ds_t: DataStruct) -> Tensor:
        if not self.rc.mask_conf:
            return ds_t.m.full
        fwd = ds_t.m.full
        inv = (fwd == 0) & (ds_t.am.full == 1)
        inv[:, :self.rc.n_prefix] = 0  # all conditioning scalars stay conditioning
        pick = torch.rand(fwd.shape[0], device=fwd.device) < self.rc.mask_conf.get("p_forward", 0.5)
        return torch.where(pick.unsqueeze(-1), fwd, inv.long())

    def _step(self, ds_t: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        with torch.no_grad():
            ds_t = DataStruct(ds_t.f.full, self._sample_mask(ds_t), ds_t.am.full)
            if self.rc.overflow_delta > 0 or self.rc.edep_overflow_delta > 0:
                ds_t = self._shift_overflow_targets(ds_t)
            if self.sym is not None:
                fwd  = ds_t.m.full[:, self.rc.n_prefix] == 0
                face = self.sym.face_of(ds_t.f.in_cc[..., 0, 4:7])
                ds_t = DataStruct(self._canon_dirs(ds_t.f.full, face, fwd), ds_t.m.full, ds_t.am.full)
            base      = self.gen_base_wrapper(ds_t)
            pdgid_idx = self.convert_pdgids(ds_t.f.pdgids)
            if self.rc.t_dist == "sm_norm":
                t = torch.sigmoid(self.rc.t_dist_scale * torch.randn_like(ds_t.f.d))
            elif self.rc.t_dist == "sd3":
                u = torch.rand_like(ds_t.f.d)
                t = 1 - u + self.rc.t_dist_scale / 3 * ((torch.pi / 2 * u).sin() ** 2 - u)
            else:
                raise ValueError(f"unknown t_dist: {self.rc.t_dist!r}")
            if self.rc.t_zero_frac > 0:
                t = t * (torch.rand_like(t) >= self.rc.t_zero_frac)
            if self.rc.t_dist_shift != 1.0:
                t = t.clamp(min=0) ** (1.0 / self.rc.t_dist_shift)
            if (
                self.reflow_teacher is not None and self.training
                and self.global_step % self.rc.reflow_every == 0
            ):  # reflow: couple base to the teacher's transport of it
                tgt = self.reflow_teacher.solve(ds_t, x_init=base, **self.rc.reflow_kwargs)
                tgt = torch.where((ds_t.m.full == 1).unsqueeze(-1), tgt, ds_t.f.model_in)
                ds_t = DataStruct(
                    torch.cat((tgt, ds_t.f.pdgids), dim=-1), ds_t.m.full, ds_t.am.full,
                )
            ps_ = self.ps.sample(base, ds_t.f.model_in, t)
        v_out = self.model(
            ps_.x_t, ps_.t,
            mask=ds_t.m.full, attn_mask=ds_t.am.full,
            types=self.types_embd, pdgids=pdgid_idx,
        )
        if self.rc.loss_sc_fac > 0:
            am = ds_t.am.full
            pred = ((1 - ps_.t)[..., None] * v_out[..., 0:1] + ps_.x_t[..., 0:1]).squeeze(-1)
            loss_sc = self.loss_fn(pred * am, ds_t.f.energy.squeeze(-1) * am)
        else:
            loss_sc = 0.0
        sq = (v_out - ps_.dx_t) ** 2
        loss = self._reduce_and_log(sq, ds_t, loss_sc, t=t)

        every = self.rc.one_step_euler_every
        fac   = self.rc.one_step_euler_fac
        lvf   = None
        if fac > 0 and hasattr(self, "lv_flow"):
            # Barrier every step, data term only when the gate fires: the
            # expectations still meet at exp(lv_flow) = E[L_flow], and lv_flow
            # never leaves the graph (DDP find_unused_parameters=False).
            lvf  = self.lv_flow.clamp(min=self.rc.uncert_min).squeeze()
            loss = loss + fac * lvf
        if fac > 0 and (not self.training or self.global_step % every == 0):
            sc = one_step_euler_loss(self, base, ds_t, pdgid_idx)
            w = fac * (every if self.training else 1)
            if lvf is not None:
                w = w * torch.exp(-lvf)
            if self.training:
                logs = {"loss/one_step_euler": sc.detach()}
                if lvf is not None:
                    logs["uncert/var_flow"] = lvf.detach().exp()
                self.log_dict(logs, on_step=True, on_epoch=False, logger=True, sync_dist=False)
            loss = loss + w * sc

        if self.rc.curv_fac > 0 and self.training and self.global_step % self.rc.curv_every == 0:
            w    = self.rc.curv_warmup
            ramp = 1.0 if w <= 0 else min(1.0, self.global_step / w)
            if ramp > 0:
                cv = curvature_loss(self, ps_.x_t, ps_.t, v_out, ds_t, pdgid_idx)
                self.log("loss/curvature", cv.detach(), on_step=True, on_epoch=False, logger=True, sync_dist=False)
                loss = loss + self.rc.curv_fac * self.rc.curv_every * ramp * cv

        if (ot := self._base_dist_loss) is not None:
            loss = loss + self.rc.base_dist_loss * ot
            if self.training:
                self.log_dict(
                    {"loss/base_dist": ot.detach(),
                     "base/sm_scale": self.gen_base.sm_scale.detach().mean()},
                    on_step=True, on_epoch=False, logger=True, sync_dist=False,
                )
        return loss

    def pretrain_base(self, batches, lr: float = 1e-2) -> float:
        opt          = torch.optim.Adam(self.base_head.parameters(), lr=lr)
        was_training = self.model.training
        rc           = self.rc
        self.rc      = replace(rc, ot_coupling=False)
        self.model.train()
        ot = float("nan")
        for ds_t in batches:
            opt.zero_grad()
            self.gen_base_wrapper(ds_t)
            if self._base_dist_loss is None:
                continue
            self._base_dist_loss.backward()
            opt.step()
            ot = self._base_dist_loss.item()
        self.model.train(was_training)
        self.rc = rc
        self.base_head.requires_grad_(False)
        self._base_dist_loss = None
        return ot

    def training_step(self, batch: tuple, _batch_idx: int | Tensor) -> Tensor:
        return self._step(batch, _batch_idx)

    @torch.no_grad()
    def validation_step(self, batch: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        bs = len(batch)
        loss = self._step(batch, _batch_idx)
        self.log("Validation Loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        for name, val in self.val_metrics(self, batch).items():
            self.log(name, val, on_epoch=True, sync_dist=True, batch_size=bs)
        return loss

    def configure_optimizers(self):
        if self._lr_sched is None:
            return self.opt
        return {"optimizer": self.opt, "lr_scheduler": self._lr_sched}

    def setup(self, stage: str | None = None) -> None:
        if getattr(self, "_val_ds", None) is not None:
            return
        full  = LEGODataset(**self.rc.dl_conf["lds_args"], prep=DataPrep(self.rc.config))
        n_val = max(1, int(len(full) * self.rc.val_conf.get("val_frac", 0.01)))
        gen   = torch.Generator().manual_seed(self.rc.val_conf.get("seed", 0))
        self._train_ds, self._val_ds = random_split(
            full, [len(full) - n_val, n_val], generator=gen,
        )
        if (self.rc.base_pretrain_batches and getattr(self, "_trainer", None)
                and hasattr(self, "base_head") and self.base_head[-1].weight.requires_grad):
            if self.trainer.is_global_zero:
                dev = self.trainer.strategy.root_device
                self.base_head.to(dev)
                loader = self._make_loader(
                    self._train_ds, shuffle=True, bs=self.rc.base_pretrain_bs
                )
                final = self.pretrain_base(b.to(dev) for _, b in zip(
                    range(self.rc.base_pretrain_batches), loader))
                self.base_head.cpu()
                print(f"base_head: pretrained on {self.rc.base_pretrain_batches} "
                      f"batches (final moment loss {final:.4f}), frozen")
            sd = self.trainer.strategy.broadcast(
                {k: v.cpu() for k, v in self.base_head.state_dict().items()}, src=0)
            self.base_head.load_state_dict(sd)
            self.base_head.requires_grad_(False)

    def _make_loader(self, dataset, *, shuffle: bool, bs: int | None = None) -> DataLoader:
        num_workers = self.rc.dl_conf.get("num_workers", 4)
        sampler = (RandomSampler if shuffle else SequentialSampler)(dataset)
        return DataLoader(
            dataset,
            sampler=BatchSampler(
                sampler,
                bs if bs is not None else self.rc.dl_conf.get("bs", 2**12),
                drop_last=False,
            ),
            batch_size=None,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            multiprocessing_context="fork" if num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val_ds, shuffle=False)

    def chunked(self, fn, *tensors, split_size=None, dim=0, cat_dim=None):
        if split_size is None or split_size >= tensors[0].shape[dim]:
            return fn(*tensors)
        if cat_dim is None:
            cat_dim = dim
        out = [fn(*chunk) for chunk in zip(*(t.split(split_size, dim) for t in tensors))]
        return torch.cat(out, dim=cat_dim)

    def _prep_solve(
        self, ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]",
    ) -> tuple[DataStruct, Tensor]:
        if not isinstance(ds_t, DataStruct):
            ds_t = DataStruct(*ds_t)
        if self.model.training:
            self.model.eval()
            self._opt_eval()
        pdgids     = ds_t.f.pdgids
        pdgids_idx = pdgids.int() if self.rc.pdgid_is_idx else self.convert_pdgids(pdgids)
        return ds_t, pdgids_idx

    @torch.no_grad()
    def log_likelihood(
        self,
        ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]",
        log_p0,
        step_size: float = 0.04,
        method: str = "midpoint",
    ) -> Tensor:
        ds_t, pdgids_idx = self._prep_solve(ds_t)
        am = ds_t.am.full.unsqueeze(-1)
        cc = ds_t.f.model_in.where(am, ds_t.f.in_cc)

        if method == "rk4":
            def vm(*args, **kwargs):
                return self.model(*args, **kwargs).clone()
        else:
            vm = self.model
        solver = ODESolver(velocity_model=vm)

        self.model.no_detach = True
        try:
            _, log_ll = solver.compute_likelihood(
                x_1=cc, log_p0=log_p0,
                mask=ds_t.m.full, attn_mask=ds_t.am.full,
                pdgids=pdgids_idx, types=self.types_embd,
                step_size=step_size, method=method,
            )
        finally:
            self.model.no_detach = False
        return log_ll

    @torch.no_grad()
    def solve(
        self,
        ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]",
        x_init: Tensor | None = None,
        reverse: bool = False,
        split_size: int | None = None,
        step_size: float = 0.04,
        method: str = "midpoint",
        time_grid: Tensor | None = None,
        return_intermediates: bool = False,
    ) -> Tensor:
        ds_t, pdgids_idx = self._prep_solve(ds_t)
        am = ds_t.am.full.unsqueeze(-1)
        cc = ds_t.f.model_in.where(am, ds_t.f.in_cc)
        if x_init is not None and x_init.shape == cc.shape:
            x_init = x_init.where(am, _F(x_init).in_p)
        if x_init is None:
            x_init = self.gen_base_wrapper(ds_t)

        if time_grid is None:
            if method in ("euler", "midpoint"):
                n = max(round(1.0 / step_size), 1)
                time_grid = torch.linspace(0.0, 1.0, n + 1, device=x_init.device, dtype=x_init.dtype)
            else:
                time_grid = x_init.new_tensor([0.0, 1.0])
            if reverse:
                time_grid = time_grid.flip(0)

        if method == "rk4":
            def vm(*args, **kwargs):
                return self.model(*args, **kwargs).clone()
        else:
            vm = self.model
        solver = ODESolver(velocity_model=vm)

        def _sample(x_init, mask, attn_mask, pdgids_idx):
            extras = dict(mask=mask, attn_mask=attn_mask, types=self.types_embd, pdgids=pdgids_idx)
            if method == "midpoint":
                return self._midpoint_steps(
                    x_init, time_grid, return_intermediates=return_intermediates, **extras,
                )
            if method == "euler":
                return self._euler_steps(
                    x_init, time_grid, return_intermediates=return_intermediates, **extras,
                )
            return solver.sample(
                x_init=x_init, time_grid=time_grid,
                step_size=step_size, method=method,
                return_intermediates=return_intermediates, **extras,
            )

        return self.chunked(
            _sample, x_init, ds_t.m.full, ds_t.am.full, pdgids_idx,
            split_size=split_size, cat_dim=-3,
        )

    def _midpoint_steps(
        self, x: Tensor, time_grid: Tensor,
        return_intermediates: bool = False, **extras,
    ) -> Tensor:
        xs  = [x]
        man = self.model.manifold
        for t_a, t_b in zip(time_grid[:-1], time_grid[1:]):
            dt     = t_b - t_a
            v1     = self.model(x, t_a, **extras)
            x_half = man.expmap(x, dt / 2 * v1)
            v2     = self.model(x_half, t_a + dt / 2, **extras)
            x      = man.expmap(x, dt * man.proju(x, v2))
            if return_intermediates:
                xs.append(x)
        return torch.stack(xs) if return_intermediates else x

    def _euler_steps(
        self, x: Tensor, time_grid: Tensor,
        return_intermediates: bool = False, **extras,
    ) -> Tensor:
        xs  = [x]
        gen = (extras["mask"] == 1).unsqueeze(-1)
        for t_a, t_b in zip(time_grid[:-1], time_grid[1:]):
            dt = t_b - t_a
            s  = self.model(x, t_a, d=dt, **extras)
            x  = torch.where(gen, self.model.manifold.expmap(x, dt * s), x)
            if return_intermediates:
                xs.append(x)
        return torch.stack(xs) if return_intermediates else x

    @torch.no_grad()
    def forward(self, batch: DataStruct | tuple, _batch_idx: int | Tensor | None = None) -> tuple:
        if self.model.training:
            self.model.eval()
            self._opt_eval()

        cfg = self.rc.odeint_conf
        if cfg.get("fwd_compile", False) and not (
            hasattr(self.model, "_orig_mod") or hasattr(self.model.vf, "_orig_mod")
        ):
            self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=False)

        ds_t = DataStruct(*batch) if isinstance(batch, tuple) else batch
        if self.sym is not None:
            fwd  = ds_t.m.full[:, self.rc.n_prefix] == 0
            face = self.sym.face_of(ds_t.f.in_cc[..., 0, 4:7])
            ds_t = DataStruct(self._canon_dirs(ds_t.f.full, face, fwd), ds_t.m.full, ds_t.am.full)
        base = self.gen_base_wrapper(ds_t)

        pdgids = ds_t.f.pdgids
        am     = ds_t.am.full.unsqueeze(-1)

        if cfg.get("return_base", False):
            sols = base.masked_fill(~am, torch.nan)
        else:
            step_size = cfg.get("step_size", 0.04)
            time_grid = cfg.get("time_grid")
            if time_grid is None:
                time_grid = torch.arange(
                    0, 1 + step_size, step=step_size, device=self.device
                ).clamp_max(1)
            sols = self.solve(
                ds_t, x_init=base,
                split_size=cfg.get("split_size"),
                step_size=step_size,
                method=cfg.get("method", "midpoint"),
                time_grid=time_grid,
                return_intermediates=cfg.get("return_timesteps", False),
            )
            sols = sols.masked_fill_(~am, torch.nan)
            filter_pdgid = cfg.get("filter_pdgid")
            if filter_pdgid is not None:
                pdgids_idx = pdgids.int() if self.rc.pdgid_is_idx else self.convert_pdgids(pdgids)
                keep = torch.isin(pdgids_idx, self.convert_pdgids(filter_pdgid)) | (pdgids_idx == 0)
                sols.masked_fill_(~keep, torch.nan)
                pdgids = pdgids.masked_fill(~keep, 0)

        if self.sym is not None:
            if sols.dim() == 4:
                sols = torch.stack([self._canon_dirs(s, face, fwd, inverse=True) for s in sols])
            else:
                sols = self._canon_dirs(sols, face, fwd, inverse=True)

        if sols.dim() == 4:
            pdgids = pdgids.unsqueeze(0).expand(sols.shape[0], -1, -1, -1)
        return torch.cat((sols, pdgids), dim=-1), ds_t.m.full, ds_t.am.full
