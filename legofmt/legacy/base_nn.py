"""Base-distribution paths superseded by ``base_conf.base_head_lik`` (:mod:`legofmt.base_dist.base_nn`), kept so older
configs and checkpoints still run unchanged: the moment-matched MLP head and no learned head at all.

``build_legacy_base_head`` is reached from ``base_nn.build_base_head`` for every config without ``base_head_lik``.
``LegacyBaseDist`` precedes ``BaseDist`` in ``LEGOLtng``'s bases; on a ``base_head_lik`` model each override hands
straight to ``BaseDist``.
"""

from dataclasses import replace
from itertools import chain, repeat

import torch
from torch import Tensor, nn

from legofmt.compat import load_legacy_base_head
from legofmt.data.struct import DataStruct

# Conditioning scalars base_head_params reads; no base_head without them.
_BASE_HEAD_SCALARS = ("Z", "A", "Size")


def _lik(rc) -> bool:
    return (rc.config.get("base_conf") or {}).get("base_head_lik", False)


def build_legacy_base_head(rc, gen_base) -> nn.Module | None:
    """The moment head, or ``None`` when the config wants no head.

    The caller assigns the result as ``LEGOLtng.base_head``; the absence of the
    attribute is what ``_fit_base_head`` and ``setup`` test with ``hasattr``.
    """
    bc = rc.config.get("base_conf") or {}
    hs = bc.get("base_head")
    if hs is not None and "k0.weight" in hs:
        raise ValueError("base_conf.base_head holds a base_head_lik head; set base_conf.base_head_lik: true.")
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


class LegacyBaseDist:
    """``BaseDist`` methods for the moment head and for no head; a ``base_head_lik`` model passes through."""

    def base_head_params(self, ds_t: DataStruct):
        """Per-event ``(sm_scale, edep_mu, edep_sig, kappa, ang_pow, None)`` from the head's
        inputs: incoming kinematics, species one-hot, cuboid chord,
        ``log(Size*Density/X0)``. ``ang_pow`` is None unless base_conf.ang_shape."""
        if _lik(self.rc):
            return super().base_head_params(ds_t)
        bc      = self.rc.config.get("base_conf") or {}
        n_cls   = len(self.pdgids_template) + 1
        x       = self._base_head_x(ds_t)
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
            out     = torch.cat((out[..., :1], out_e[..., 1:3], ang[..., 3:]), dim=-1)
        else:
            with torch.autocast(x.device.type, enabled=False):
                out = self.base_head(x)
        n_ang   = n_cls if bc.get("ang_species", False) else 1
        return (out[..., 0:1].exp(), out[..., 1:2], out[..., 2:3].exp(),
                out[..., 3:3 + n_ang].exp().clamp_min(1e-3),  # kap: divisor in sample()
                out[..., 3 + n_ang:].exp().clamp(0.1, 10.0) if out.shape[-1] > 3 + n_ang else None, None)

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
        """No head: nothing to fit. Moment head: as ``BaseDist``'s, with the moment loss, and one ``(kappa, pw)`` per
        slot only under ``ang_species``."""
        if not hasattr(self, "base_head"):
            return
        if _lik(self.rc):
            return super()._fit_base_head(ds_t, fwd)
        learn = self.model.training and next(self.base_head.parameters()).requires_grad
        keep = (lambda v: v.detach()) if learn else (lambda v: v)
        with torch.set_grad_enabled(learn or torch.is_grad_enabled()):
            s, mu, sig, kap, pw, _ = self.base_head_params(ds_t)
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
        self.gen_base.sm_pow   = 1.0

    def pretrain_base(self, batches, lr: float | None = None, steps: int | None = None) -> float:
        """Fit the moment head alone on ``batches`` with Adam, then freeze it. ``steps``
        cosine-anneals the lr to lr/100 over that many steps. Returns the final moment loss;
        ``lr`` defaults to 1e-2."""
        if _lik(self.rc):
            return super().pretrain_base(batches, lr=lr, steps=steps)
        lr           = 1e-2 if lr is None else lr
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
        if _lik(self.rc):
            return super()._pretrain_base_if_needed()
        if (self.rc.base_pretrain_batches and getattr(self, "_trainer", None)
                and hasattr(self, "base_head") and next(self.base_head.parameters()).requires_grad):
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
