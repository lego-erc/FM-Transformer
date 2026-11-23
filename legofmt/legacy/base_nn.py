"""Base-distribution paths superseded by ``base_conf.base_head_lik`` (:mod:`legofmt.base.interface`), kept so older
configs and checkpoints still run unchanged: the moment-matched MLP head and no learned head at all.

``build_legacy_base_head`` is reached from ``interface.build_base_head`` for every config without ``base_head_lik``.
``LegacyBaseDist`` precedes ``BaseDist`` in ``LEGOLtng``'s bases; on a ``base_head_lik`` model each override defers
to ``BaseDist``.
"""

from dataclasses import replace
from itertools import chain, repeat

import torch
from torch import Tensor, nn

from legofmt.data.struct import DataStruct

# the conditioning scalars the head's inputs need (``BaseDist._head_inputs``): no moment head without them
_BASE_HEAD_SCALARS = ("Z", "A", "Size")


def _lik(rc) -> bool:
    return (rc.config.get("base_conf") or {}).get("base_head_lik", False)


def build_legacy_base_head(rc, gen_base) -> nn.Module | None:
    """Constructs the moment head, or returns None for a config that gets no head.

    A config gets a head when ``scale_dist`` is ``sm_norm``, ``cond_scalars`` hold Z, A and Size, and it has a saved
    ``base_conf.base_head`` or ``base_pretrain_batches > 0``. A saved head whose layers match the config's is loaded
    and frozen; otherwise the head is fresh, for ``pretrain_base`` to fit. ``LEGOLtng.__init__`` assigns a head as
    ``base_head`` and leaves the attribute unset for None: ``_set_prior`` and ``_pretrain_base_if_needed`` test for
    it with ``hasattr``.

    Args:
        rc:       the resolved config.
        gen_base: the sampler; a fresh head starts at its constants ``sm_scale`` and ``kappa``.

    Returns:
        The head (an ``nn.Sequential``), or None.

    Raises:
        ValueError: ``model_conf.edep_whiten`` without a head, a ``base_head_lik`` head in ``base_conf.base_head``, or
            a saved head without the Z, A and Size conditioning scalars.
    """
    head = _moment_head(rc, gen_base)
    if head is None and rc.edep_whiten:
        raise ValueError("model_conf.edep_whiten needs the base head's per-event E_dep prior (base_pretrain_"
                         "batches > 0 or a saved base_conf.base_head, scale_dist sm_norm, Z/A/Size conditioning).")
    return head


def _moment_head(rc, gen_base) -> nn.Module | None:
    bc = rc.config.get("base_conf") or {}
    hs = bc.get("base_head")
    if hs is not None and "k0.weight" in hs:
        raise ValueError("base_conf.base_head holds a base_head_lik head; set base_conf.base_head_lik: true.")
    # The head's inputs need Z, A and Size. Without them a saved head is an error (it would vanish silently);
    # the base_pretrain_batches default alone gives no head.
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
    dims = [n_in, *bc.get("base_head_hidden", [16])]  # hidden widths; the default 16 under-fits E_dep's prior
    head = nn.Sequential(*(m for a, b in zip(dims, dims[1:]) for m in (nn.Linear(a, b), nn.Mish())),
                         nn.Linear(dims[-1], n_out))
    last = f"{len(head) - 1}.weight"
    with torch.no_grad():
        head[-1].weight.zero_()
        head[-1].bias.copy_(torch.tensor(
            [float(gen_base.sm_scale), 1.0, 1.0]
            + [float(gen_base.kappa)] * n_ang + [1.0] * (n_out - 3 - n_ang)).log())
        if hs is None or hs.keys() != head.state_dict().keys():
            return head  # nothing saved, or a head of another depth: a fresh one, at the sampler's constants
        if any(hs[k].shape != v.shape for k, v in head[:-1].state_dict().items()):
            return head   # base_head_nout / base_head_hidden changed since it was saved: a fresh head
        if hs[last].shape[0] < head[-1].weight.shape[0]:
            return head   # ang_shape / ang_species switched on since: fresh, not the saved one frozen at ang_pow = 1
        if hs[last].shape == head[-1].weight.shape:
            head.load_state_dict(hs)
        else:
            load_legacy_base_head(head, hs)
        head.requires_grad_(False)  # already fitted; the flow never trains the head
    return head


class LegacyBaseDist:
    """``BaseDist`` methods for the moment head and for a model without a head.

    On a ``base_head_lik`` model each method defers to ``BaseDist``.
    """

    def base_head_params(self, ds_t: DataStruct):
        """Evaluates the moment head on the events of ``ds_t``.

        The head reads ``[E, species one-hot, chord, log t]`` of the incoming particle and the block
        (``BaseDist._head_inputs``). Under ``base_head_nout`` it also reads the outgoing set, which then conditions
        E_dep's ``(mu, sig)`` only; with ``ang_nout``, ``kappa`` and ``ang_pow`` too.

        Returns:
            ``(sm_scale, edep_mu, edep_sig, kappa, ang_pow, None)`` in the order of ``BaseParams``: ``sm_scale``
            (B, 1), E_dep's ``mu`` and ``sig`` (B,), ``kappa`` and ``ang_pow`` (B, 1), under ``ang_species``
            (B, n_cls). ``ang_pow`` is None without ``ang_shape``; the last entry is ``sm_pow``, which this head
            does not predict.
        """
        if _lik(self.rc):
            return super().base_head_params(ds_t)
        bc      = self.rc.config.get("base_conf") or {}
        n_cls   = len(self.pdgids_template) + 1
        cont, species, extra = self._head_inputs(ds_t)
        x       = torch.cat((cont[:, :1], species, cont[:, 1:]), dim=-1)   # [E, species one-hot, chord, log t]
        if bc.get("base_head_nout", False):
            # Only E_dep's (mu, sig) see the outgoing set. sm_scale and kappa come from a second pass with the
            # outgoing set zeroed, so they are fitted to the marginal over it, as without base_head_nout
            # (ang_nout: kappa and ang_pow see it too). Conditioned on it, kappa moved 5.6..44.7 between outgoing
            # sets and the direction losses doubled; conditioning E_dep alone kept the gain.
            both    = torch.cat((torch.cat((x, extra), -1), torch.cat((x, torch.zeros_like(extra)), -1)))
            with torch.autocast(x.device.type, enabled=False):  # fp32 as at inference and in pretraining
                out_e, out = self.base_head(both).chunk(2)  # one forward for both passes
            ang     = out_e if bc.get("ang_nout", False) else out
            out     = torch.cat((out[..., :1], out_e[..., 1:3], ang[..., 3:]), dim=-1)
        else:
            with torch.autocast(x.device.type, enabled=False):
                out = self.base_head(x)
        n_ang   = n_cls if bc.get("ang_species", False) else 1
        return (out[..., 0:1].exp(), out[..., 1], out[..., 2].exp(),
                out[..., 3:3 + n_ang].exp().clamp_min(1e-3),  # kappa: a divisor in the sampler
                out[..., 3 + n_ang:].exp().clamp(0.1, 10.0) if out.shape[-1] > 3 + n_ang else None, None)

    def _base_moment_loss(self, ds_t, fwd, s, mu, sig, kap, pw=None) -> Tensor:
        """Returns the moment head's loss on a batch: the prior's moments matched to the forward events' data.

        ```
        energy   (E_z[1 - tanh(|z| s)] - mean u) ** 2                          u: an outgoing slot's energy channel
        E_dep    log(sig) + (logit(E / e_dep_max) - mu) ** 2 / (2 sig ** 2)    the logit-normal NLL, events with E > 0
        angles   (E_z[w] - mean a) ** 2,   w = tanh(|z| / kappa) ** pw         a: the mean of a slot's momentum and
                 + (E_z[w ** 2] - mean a ** 2) ** 2   with ``pw``                position angle to the incoming, / pi
        ```

        ``E_z`` is the mean over 100 quantile nodes of N(0, 1); ``mean`` runs over an event's outgoing slots (under
        ``ang_species`` every slot is matched on its own). The angle term needs ``tanh_theta`` and is weighted by
        ``base_conf.ang_weight``.

        ``fwd`` (B,) marks the forward events. ``s`` is (B, 1), ``mu`` and ``sig`` are (B,); ``kap`` is (B, 1) per
        event or (B, n_out) per outgoing slot, and ``pw`` has its shape or is None.
        """
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
        l_edep = ((sig.log() + (ly - mu) ** 2 / (2 * sig ** 2))
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

    def _set_prior(self, ds_t) -> None:
        """Sets the sampler's parameters to this batch's prior from the moment head; without a head it sets nothing.

        ``kappa`` and ``ang_pow`` are one per event, under ``ang_species`` one per outgoing slot. While the head is
        being fitted (trainable, the field in training mode), the batch's moment loss is left in ``_base_head_loss``
        for ``pretrain_base``, and the sampler gets detached values.
        """
        if not hasattr(self, "base_head"):
            return
        if _lik(self.rc):
            return super()._set_prior(ds_t)
        self._base_head_loss = None
        fwd   = ds_t.m.full[:, self.rc.n_prefix] == 0
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
        self.gen_base.sm_scale = keep(s)
        self.gen_base.edep_mu  = keep(mu)
        self.gen_base.edep_sig = keep(sig)
        self.gen_base.kappa    = keep(kap)
        self.gen_base.ang_pow  = None if pw is None else keep(pw)
        self.gen_base.sm_pow   = 1.0

    def pretrain_base(self, batches, lr: float | None = None, steps: int | None = None) -> float:
        """Fits the moment head alone on ``batches`` with Adam, then freezes it.

        Each batch goes through ``gen_base_wrapper`` with the OT coupling off; ``_set_prior`` leaves its moment loss
        in ``_base_head_loss``.

        Args:
            batches: the training batches (``DataStruct``), one step each.
            lr:      the learning rate; None: 1e-2.
            steps:   the number of steps over which the learning rate cosine-anneals to ``lr / 100``; None: constant.

        Returns:
            The last step's moment loss; nan when no batch held a forward event.
        """
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
                self._base_head_loss = None   # stays None when the batch has no forward event
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
        """Fits the moment head before the flow trains, unless there is none or it is frozen already.

        ``setup`` calls this. Rank 0 fits the head on its device (the data stays on the CPU, moved per batch) and
        broadcasts the weights; every rank ends with the head frozen.
        """
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


def load_legacy_base_head(head: nn.Sequential, hs: dict) -> None:
    """Loads the state dict ``hs`` of a one-hidden-layer ``base_head`` with fewer outputs than ``head``.

    Copies the output rows that ``hs`` has and leaves the others at their initial values. Call under
    ``torch.no_grad()``: it copies into parameter slices in place.
    """
    n_old = hs["2.weight"].shape[0]
    head[0].load_state_dict({"weight": hs["0.weight"], "bias": hs["0.bias"]})
    head[-1].weight[:n_old].copy_(hs["2.weight"])
    head[-1].bias[:n_old].copy_(hs["2.bias"])
