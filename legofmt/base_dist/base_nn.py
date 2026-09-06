"""The flow's base (prior) distribution: the learned per-event head and the
wrapper that draws ``x_0``.

Split out of ``main/modules.py``. Like ``cfm/solvers.py`` these are mixin
methods rather than a standalone object, because ``lego_eval`` reaches for
them on the LightningModule itself: ``lego.gen_base``, ``lego.gen_base_wrapper``
and ``hasattr(lego, "base_head")``. ``base_head`` is also a state_dict key, so
it must stay a direct attribute of ``LEGOLtng`` under that exact name -- which
is why ``build_base_head`` returns the module for ``__init__`` to assign rather
than owning it here.

Members resolved through ``self`` and owned by ``LEGOLtng``: ``rc``, ``model``,
``gen_base``, ``base_head``, ``pdgids_template``, ``convert_pdgids``,
``_base_dist_loss``, ``_make_loader``, ``_train_ds``.
"""

from dataclasses import replace

import torch
from torch import Tensor, nn

from flow_matching.utils.manifolds import Sphere

from legofmt.compat import load_legacy_base_head
from legofmt.data.struct import DataStruct, _F

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


def build_base_head(rc, gen_base) -> nn.Sequential | None:
    """The learned per-event base head, or ``None`` when the config wants none.

    The caller assigns the result as ``LEGOLtng.base_head``; the absence of the
    attribute is what ``gen_base_wrapper`` and ``setup`` test with ``hasattr``.
    """
    bc = rc.config.get("base_conf") or {}
    # base_head_params reads Z/A/Size, so there is no head without them: an
    # explicit ask (base_dist_loss, or a saved head that would otherwise
    # vanish silently) is an error, the base_pretrain_batches default is not.
    asked = rc.base_dist_loss > 0 or bc.get("base_head") is not None
    want = ((asked or rc.base_pretrain_batches > 0)
            and gen_base.scale_dist == "sm_norm")
    if want and not all(s in rc.cond_scalars for s in _BASE_HEAD_SCALARS):
        if asked:
            raise ValueError(
                f"base_head needs the {_BASE_HEAD_SCALARS} conditioning scalars; "
                f"got cond_scalars={rc.cond_scalars}."
            )
        want = False
    if not want:
        return None
    head = nn.Sequential(
        nn.Linear(4 + len(rc.pdgids_template), 16), nn.Mish(),
        nn.Linear(16, 4))
    with torch.no_grad():
        head[-1].weight.zero_()
        head[-1].bias.copy_(torch.tensor(
            [float(gen_base.sm_scale), 1.0, 1.0,
             float(gen_base.kappa)]).log())
        hs = bc.get("base_head")
        if hs is not None and hs.keys() == head.state_dict().keys():
            if hs["2.weight"].shape == head[-1].weight.shape:
                head.load_state_dict(hs)
            else:
                load_legacy_base_head(head, hs)
            head.requires_grad_(not bc.get("base_head_frozen", False))
    return head


class BaseDist:
    """Base-distribution mixin for :class:`~legofmt.main.modules.LEGOLtng`."""

    def base_head_params(self, ds_t: DataStruct) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pid     = ds_t.f.in_p[..., 0, -1]
        idx     = pid.long() if self.rc.pdgid_is_idx else self.convert_pdgids(pid)
        species = nn.functional.one_hot(
            idx.long().clamp(0, len(self.pdgids_template)),
            len(self.pdgids_template) + 1).float()
        Z, A    = ds_t.f.cond("Z"), ds_t.f.cond("A")
        x0      = 716.4 * A / (Z * (Z + 1) * (287.0 / Z.sqrt()).log())
        t       = ds_t.f.cond("Size") * ds_t.f.cond("Density") / x0
        # chord length: full chord through the cube along the incoming
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
                        ed_t = ds_t.f.edep
                        w    = fwd.float() * (ed_t > 0)
                        ly   = torch.logit(
                            (ed_t / self.gen_base.e_dep_max).clamp(1e-4, 1 - 1e-4))
                        mu1, sig1 = mu.squeeze(-1), sig.squeeze(-1)
                        l_edep = ((sig1.log() + (ly - mu1) ** 2 / (2 * sig1 ** 2))
                                  * w).sum() / w.sum().clamp(min=1)
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
                pid = ds_t.f.out_p[..., -1]
                inf_cond = (
                    ds_t.am.out_p.unsqueeze(-1).logical_xor(ds_t.am.out_p.unsqueeze(-2))
                    | (pid.unsqueeze(-1) != pid.unsqueeze(-2))
                )
                out = _F(base).out_p
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

    def _pretrain_base_if_needed(self) -> None:
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
