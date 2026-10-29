"""The flow's base (prior) distribution: the learned per-event head and the wrapper that draws ``x_0``.

Mixin methods rather than a standalone object, because ``lego_eval`` reaches for them on the LightningModule itself:
``lego.gen_base``, ``lego.gen_base_wrapper`` and ``hasattr(lego, "base_head")``. ``base_head`` is also a state_dict
key, so it must stay a direct attribute of ``LEGOLtng`` under that exact name -- which is why ``build_base_head``
returns the module for ``__init__`` to assign rather than owning it here.

Members resolved through ``self`` and owned by ``LEGOLtng``: ``rc``, ``model``, ``gen_base``, ``base_head``,
``pdgids_template``, ``convert_pdgids``, ``_base_head_loss``, ``_make_loader``, ``_train_ds``.
"""

from dataclasses import replace
from itertools import chain, repeat

import torch
from torch import Tensor, nn

from flow_matching.utils.manifolds import Sphere

from legofmt.cfm.geom_encoder import x0_gcm2
from legofmt.compat import load_legacy_base_head
from legofmt.data.struct import DataStruct, _F

try:
    from torch_linear_assignment import batch_linear_assignment
except ImportError:
    batch_linear_assignment = None
try:  # skips TLA's NaN / infeasibility .item() checks: our costs are finite and square
    from torch_linear_assignment._triton import batch_linear_assignment as _lap_cuda_unchecked
except ImportError:
    _lap_cuda_unchecked = None

_OT_COUPLING_REQUIRES_LAP = (
    "ot_coupling=True requires `torch-linear-assignment` (a pixi dependency). "
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
    hs = bc.get("base_head")
    # base_head_params reads Z/A/Size, so there is no head without them: an
    # explicit ask (a saved head that would otherwise vanish silently) is an
    # error, the base_pretrain_batches default is not.
    asked = hs is not None
    if not ((asked or rc.base_pretrain_batches > 0) and gen_base.scale_dist == "sm_norm"):
        return None
    if not all(s in rc.cond_scalars for s in _BASE_HEAD_SCALARS):
        if asked:
            raise ValueError(f"base_head needs the {_BASE_HEAD_SCALARS} conditioning scalars; "
                             f"got cond_scalars={rc.cond_scalars}.")
        return None

    n_in = 4 + len(rc.pdgids_template)
    if bc.get("base_head_nout", False):
        n_in += len(rc.pdgids_template) + 2
    n_ang = len(rc.pdgids_template) + 1 if bc.get("ang_species", False) else 1
    n_out = 3 + n_ang * (1 + int(bc.get("ang_shape", False)))
    dims = [n_in, *bc.get("base_head_hidden", [16])]  # hidden widths; 16 under-fits E_dep's prior
    head = nn.Sequential(*(m for a, b in zip(dims, dims[1:]) for m in (nn.Linear(a, b), nn.Mish())),
                         nn.Linear(dims[-1], n_out))
    last = f"{len(head) - 1}.weight"
    with torch.no_grad():
        head[-1].weight.zero_()
        head[-1].bias.copy_(torch.tensor(
            [float(gen_base.sm_scale), 1.0, 1.0]
            + [float(gen_base.kappa)] * n_ang + [1.0] * (n_out - 3 - n_ang)).log())
        if hs is None or hs.keys() != head.state_dict().keys():
            return head  # fresh head: zero-init output layer, priors in the bias
        if any(hs[k].shape != v.shape for k, v in head[:-1].state_dict().items()):
            return head   # base_head_nout / base_head_hidden toggled: re-pretrain, do not mis-load
        if hs[last].shape[0] < head[-1].weight.shape[0]:
            return head   # ang_shape / ang_species toggled on: re-pretrain, do not freeze at pow=1
        if hs[last].shape == head[-1].weight.shape:
            head.load_state_dict(hs)
        else:
            load_legacy_base_head(head, hs)
        head.requires_grad_(False)  # already fitted; the flow never trains the head
    return head


class BaseDist:
    """Base-distribution mixin for :class:`~legofmt.main.modules.LEGOLtng`."""

    def base_head_params(self, ds_t: DataStruct):
        """Per-event ``(sm_scale, edep_mu, edep_sig, kappa, ang_pow)`` from the head's
        inputs: incoming kinematics, species one-hot, cuboid chord,
        ``log(Size*Density/X0)``. ``ang_pow`` is None unless base_conf.ang_shape."""
        bc      = self.rc.config.get("base_conf") or {}
        n_cls   = len(self.pdgids_template) + 1
        species = nn.functional.one_hot(self._species_idx(ds_t.f.in_p[..., 0, -1], n_cls), n_cls).float()
        t       = ds_t.f.cond("Size") * ds_t.f.cond("Density") / x0_gcm2(ds_t.f.cond("Z"), ds_t.f.cond("A"))
        # chord length: full chord through the cuboid along the incoming ray
        # (longest-edge lengths); fwd+bwd -> entry/exit-storage invariant
        inc     = ds_t.f.in_cc[..., 0, 1:7].nan_to_num(1.0)
        u, pos  = inc[..., :3], inc[..., 3:]
        d       = self.cuboid_unit.to(u)
        p       = pos / pos.abs().amax(-1, keepdim=True).clamp_min(1e-8) * d  # pos is dir(x / d)
        ok_u    = u.abs() > 1e-6
        u_safe  = torch.where(ok_u, u, torch.ones_like(u))   # the discarded branch must not divide by 0 (backward)
        tf      = torch.where(ok_u, (u.sign() * d - p) / u_safe, torch.full_like(u, 4.0))
        tb      = torch.where(ok_u, (p + u.sign() * d) / u_safe, torch.full_like(u, 4.0))
        chord   = ((tf.amin(-1) + tb.amin(-1)).clamp(0.0, 3.5) / 2).unsqueeze(-1)
        x       = torch.cat((ds_t.f.in_cc[..., 0], species, chord, t.log().unsqueeze(-1)), dim=-1)
        if bc.get("base_head_nout", False):
            # Only E_dep's (mu, sig) see the outgoing set. sm_scale and kappa are
            # read from a second pass with those inputs zeroed -- a constant, so
            # the NLL fits them to the marginal over n_out, as without the flag.
            # Conditioning them too moved kappa 5.6..44.7 across branches and cost
            # 2x on direction losses; conditioning E_dep alone kept the gain.
            extra   = torch.cat(self._out_set_feats(ds_t), dim=-1)
            both    = torch.cat((torch.cat((x, extra), -1), torch.cat((x, torch.zeros_like(extra)), -1)))
            with torch.autocast(x.device.type, enabled=False):  # fp32 as at inference and in pretraining
                out_e, out = self.base_head(both).chunk(2)  # one forward for both passes
            ang     = out_e if bc.get("ang_nout", False) else out
            out     = torch.cat((out[..., 0:1], out_e[..., 1:3], ang[..., 3:]), dim=-1)
        else:
            with torch.autocast(x.device.type, enabled=False):
                out = self.base_head(x)
        n_ang   = n_cls if bc.get("ang_species", False) else 1
        return (out[..., 0:1].exp(), out[..., 1:2], out[..., 2:3].exp(),
                out[..., 3:3 + n_ang].exp().clamp_min(1e-3),  # kap: divisor in sample()
                out[..., 3 + n_ang:].exp().clamp(0.1, 10.0) if out.shape[-1] > 3 + n_ang else None)

    def _edep_whiten(self, f: Tensor, ds_t: DataStruct, inverse: bool = False) -> Tensor:
        """``model_conf.edep_whiten``: the generated E_dep row in units of the event's base prior (``inverse``: back).

            y = (logit(u / e_dep_max) - mu) / sig      a draw of the prior becomes N(0, 1)

        No deposit (u <= 0) sits 6 prior sigmas below mu and decodes back to 0 from below 4.5; a real deposit is held
        at 4.4, so it never turns into a pass-through. (mu, sig) come from the frozen head on ``ds_t``, so any
        sub-batch is consistent. Options: ``edep_whiten_global`` (c, s) is one map for every event,
        ``edep_whiten_max_sig`` caps the unit, ``edep_whiten_feed`` writes (mu, unit) on the row's direction spheres.
        Returns a copy; ``f`` may carry a leading time axis."""
        _, mu, sig, _, _ = self.base_head_params(ds_t)
        mu, sig = mu.reshape(-1), sig.reshape(-1)
        if self.rc.edep_whiten_global is not None:
            c, s = self.rc.edep_whiten_global
            mu, sig = torch.full_like(mu, c), torch.full_like(sig, s)
        unit = sig if self.rc.edep_whiten_max_sig is None else sig.clamp(max=self.rc.edep_whiten_max_sig)
        sd = sig / unit                            # one prior sigma in whitened units
        emax, e = self.gen_base.e_dep_max, _F(f).edep
        if inverse:
            new = torch.where(e > -4.5 * sd, emax * torch.sigmoid(mu + unit * e), torch.zeros_like(e))
        else:
            y = (torch.logit((e / emax).clamp(1e-6, 1 - 1e-6)) - mu) / unit
            new = torch.where(e > 0, torch.maximum(y, -4.4 * sd), -6.0 * sd)
        gen = ds_t.m.full[:, 1] == 1               # only generated E_dep rows change
        f = f.clone()                              # e still reads the caller's tensor
        _F(f).edep.copy_(torch.where(gen, new, e))
        if self.rc.edep_whiten_feed:
            ones, row = torch.ones_like(mu), f[..., 1, 1:7]
            code = torch.cat([nn.functional.normalize(torch.stack((a, ones, 0 * ones), -1), dim=-1)
                              for a in (mu, unit.log())], -1)
            row.copy_(torch.where(gen.unsqueeze(-1), torch.ones_like(row) if inverse else code, row))
        return f

    def _species_idx(self, pid: Tensor, n_cls: int) -> Tensor:
        """Class index in ``[0, n_cls)`` of raw pdgids (of indices with ``pdgid_is_idx``)."""
        return (pid.long() if self.rc.pdgid_is_idx else self.convert_pdgids(pid)).long().clamp(0, n_cls - 1)

    def _out_set_feats(self, ds_t: DataStruct) -> list[Tensor]:
        """Outgoing multiplicity and per-pdgid composition, both fixed before the
        flow runs. E_dep is bimodal in them, which one logit-normal cannot hold."""
        valid = ds_t.am.out_p
        n_cls = len(self.pdgids_template) + 1
        oh    = nn.functional.one_hot(self._species_idx(ds_t.f.out_p[..., -1], n_cls), n_cls)
        cnt   = (oh * valid.unsqueeze(-1)).sum(-2).float()
        return [valid.sum(-1, keepdim=True).float().log1p(), cnt.log1p()]

    def _base_moment_loss(self, ds_t, fwd, s, mu, sig, kap, pw=None) -> Tensor:
        """Moment-match the prior to the batch: quantile-node mean of the energy scale,
        logit-normal NLL for E_dep, and (``tanh_theta``) the mean opening angle for kappa."""
        z  = 2 ** 0.5 * torch.erfinv(torch.linspace(
            -0.995, 0.995, 100, device=s.device))  # N(0,1) quantile nodes
        v  = (ds_t.am.out_p & fwd.unsqueeze(-1)).float()
        n  = v.sum(-1).clamp(min=1)
        ev = (v.sum(-1) > 0).float()

        u_b = 1 - (z.abs() * s).tanh().mean(-1)
        u_t = (ds_t.f.out_cc[..., 0] * v).sum(-1) / n
        l_scale = ((u_b - u_t) ** 2 * ev).sum() / ev.sum().clamp(min=1)

        ed_t = ds_t.f.edep
        w    = fwd.float() * (ed_t > 0)
        ly   = torch.logit((ed_t.float() / self.gen_base.e_dep_max).clamp(1e-4, 1 - 1e-4))
        mu1, sig1 = mu.squeeze(-1), sig.squeeze(-1)
        l_edep = ((sig1.log() + (ly - mu1) ** 2 / (2 * sig1 ** 2))
                  * w).sum() / w.sum().clamp(min=1)

        if not self.gen_base.tanh_theta:
            return l_scale + l_edep

        tb   = (z.abs() / kap.unsqueeze(-1)).tanh()          # (B, 1, nodes) or (B, K, nodes) per slot
        if pw is not None:
            tb = tb.pow(pw.unsqueeze(-1))
        m1, m2 = tb.mean(-1), tb.square().mean(-1)
        u_i  = nn.functional.normalize(ds_t.f.in_cc[..., 0, 1:4], dim=-1).unsqueeze(1)
        p_i  = nn.functional.normalize(ds_t.f.in_cc[..., 0, 4:7], dim=-1).unsqueeze(1)
        a_m  = (ds_t.f.out_cc[..., 1:4] * u_i).sum(-1).clamp(-1, 1).acos()
        a_p  = (ds_t.f.out_cc[..., 4:7] * p_i).sum(-1).clamp(-1, 1).acos()
        ang  = (a_m + a_p) / 2 / torch.pi
        den  = ev.sum().clamp(min=1)
        if kap.shape[-1] > 1:   # per-slot prior: match each slot's own moments (ang_species)
            vs = v.sum().clamp(min=1)
            l_kappa = (((m1 - ang) ** 2) * v).sum() / vs
            if pw is not None:
                l_kappa = l_kappa + (((m2 - ang.square()) ** 2) * v).sum() / vs
        else:
            l_kappa = ((m1.squeeze(-1) - (ang * v).sum(-1) / n) ** 2 * ev).sum() / den
            if pw is not None:
                l_kappa = l_kappa + (
                    (m2.squeeze(-1) - (ang.square() * v).sum(-1) / n) ** 2 * ev).sum() / den
        l_kappa = l_kappa * (self.rc.config.get("base_conf") or {}).get("ang_weight", 1.0)
        return l_scale + l_edep + l_kappa

    def _fit_base_head(self, ds_t, fwd) -> None:
        """Run the head, hand its outputs to the sampler, and cache the moment loss
        while the head is trainable. With grad mode on (legofmt.design) the frozen head keeps its graph to the
        inputs, so the draw is reparameterized in them; only a head being fitted hands the sampler detached values."""
        learn = self.model.training and self.base_head[-1].weight.requires_grad
        keep = (lambda v: v.detach()) if learn else (lambda v: v)
        with torch.set_grad_enabled(learn or torch.is_grad_enabled()):
            s, mu, sig, kap, pw = self.base_head_params(ds_t)
            if kap.shape[-1] > 1:   # ang_species: one column per species -> one value per slot
                idx = self._species_idx(ds_t.f.out_p[..., -1], kap.shape[-1])
                kap = kap.gather(1, idx)
                pw  = None if pw is None else pw.gather(1, idx)
            if learn:
                self._base_head_loss = self._base_moment_loss(ds_t, fwd, s, mu, sig, kap, pw)
        self.gen_base.sm_scale = keep(s).unsqueeze(-1)
        self.gen_base.edep_mu  = keep(mu)
        self.gen_base.edep_sig = keep(sig)
        self.gen_base.kappa    = keep(kap)
        self.gen_base.ang_pow  = None if pw is None else keep(pw)

    def _ot_couple(self, base: Tensor, ds_t, data: Tensor) -> Tensor:
        """Per-event Hungarian assignment of base slots to data slots, blocked across
        pdgids and valid/pad pairs; base samples are pdgid-independent so permuting is free."""
        if batch_linear_assignment is None:
            raise RuntimeError(_OT_COUPLING_REQUIRES_LAP)
        base = base.where(ds_t.am.full.unsqueeze(-1), data)
        pid  = ds_t.f.out_p[..., -1]
        blocked = (
            ds_t.am.out_p.unsqueeze(-1).logical_xor(ds_t.am.out_p.unsqueeze(-2))
            | (pid.unsqueeze(-1) != pid.unsqueeze(-2))
        )
        out = _F(base).out_p
        man = self.rc.manifold
        tgt = ds_t.f.out_cc.unsqueeze(-2).split(man.ambient_dims, dim=-1)
        ref = out.unsqueeze(-3).split(man.ambient_dims, dim=-1)
        cost = sum(
            (((a * b).sum(-1).float().clamp(-1 + 1e-6, 1 - 1e-6).acos()
              if isinstance(mf, Sphere) else (a - b).norm(dim=-1).float()) ** 2)
            for mf, a, b in zip(man.manifolds, tgt, ref)
        ) + blocked * 1e6
        assign = (_lap_cuda_unchecked(cost, validation="off") if cost.is_cuda and _lap_cuda_unchecked is not None
                  else batch_linear_assignment(cost)).long()
        out[:] = torch.take_along_dim(out, assign.unsqueeze(-1), dim=1)
        return base

    def gen_base_wrapper(self, ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]", all_forward: bool = False) -> Tensor:
        """Draw ``x_0``: data on conditioning cells, the (learned) prior on generated
        ones, an isotropic sample for inverse-mask events. ``lego_eval`` calls this."""
        if not isinstance(ds_t, DataStruct):
            ds_t = DataStruct(*ds_t)
        self._base_head_loss = None
        data  = ds_t.f.model_in
        m     = ds_t.m.full
        fwd   = m[:, self.rc.n_prefix] == 0  # incoming slot conditions => forward event
        noise = None if all_forward or fwd.all() else self.gen_base.iso(m.shape, data.device)
        if not all_forward and not fwd.any():
            return torch.where((m == 1).unsqueeze(-1), noise, data)

        if hasattr(self, "base_head"):
            self._fit_base_head(ds_t, fwd)
        base = torch.cat(
            (ds_t.f.non_cc, self.gen_base(ds_t.m.out_p.shape, ds_t.f.in_cc)), dim=1,
        )
        if self.rc.ot_coupling and self.model.training:
            base = self._ot_couple(base, ds_t, data)
        base = self.gen_base.insert_add(base)
        noise = base if noise is None else torch.where(fwd.view(-1, 1, 1), base, noise)
        return torch.where((m == 1).unsqueeze(-1), noise, data)

    def pretrain_base(self, batches, lr: float = 1e-2, steps: int | None = None) -> float:
        """Fit the base head alone on ``batches`` with Adam, then freeze it: the head
        is never trained alongside the flow. ``steps`` cosine-anneals the lr to lr/100
        over that many steps. Returns the final moment loss."""
        opt          = torch.optim.Adam(self.base_head.parameters(), lr=lr)
        sched        = None if steps is None else torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, lr / 100)
        was_training = self.model.training
        rc           = self.rc
        self.rc      = replace(rc, ot_coupling=False)
        self.model.train()
        ot = float("nan")
        try:
            for ds_t in batches:
                if rc.dir_zoom:
                    fwd_z = ds_t.m.full[:, rc.n_prefix] == 0
                    ds_t = DataStruct(self._zoom_dirs(ds_t.f.full, ds_t.am.full, fwd_z, pdgids=ds_t.f.pdgids),
                                      ds_t.m.full, ds_t.am.full)
                opt.zero_grad()
                self.gen_base_wrapper(ds_t)
                if self._base_head_loss is None:
                    continue
                self._base_head_loss.backward()
                opt.step()
                if sched is not None:
                    sched.step()
                ot = self._base_head_loss.item()
        finally:
            self.model.train(was_training)
            self.rc = rc
        self.base_head.requires_grad_(False)
        self._base_head_loss = None
        return ot

    def _pretrain_base_if_needed(self) -> None:
        """Rank 0 pretrains on CPU-resident data moved to its device and broadcasts the
        weights; every rank ends with the head frozen."""
        if (self.rc.base_pretrain_batches and getattr(self, "_trainer", None)
                and hasattr(self, "base_head") and self.base_head[-1].weight.requires_grad):
            if self.trainer.is_global_zero:
                dev = self.trainer.strategy.root_device
                self.base_head.to(dev)
                loader = self._make_loader(
                    self._train_ds, shuffle=True, bs=self.rc.base_pretrain_bs
                )
                final = self.pretrain_base((b.to(dev) for _, b in zip(  # one reshuffled epoch after another
                    range(self.rc.base_pretrain_batches), chain.from_iterable(repeat(loader)))),
                    steps=self.rc.base_pretrain_batches)
                self.base_head.cpu()
                print(f"base_head: pretrained on {self.rc.base_pretrain_batches} "
                      f"batches (final moment loss {final:.4f}), frozen")
            sd = self.trainer.strategy.broadcast(
                {k: v.cpu() for k, v in self.base_head.state_dict().items()}, src=0)
            self.base_head.load_state_dict(sd)
            self.base_head.requires_grad_(False)
