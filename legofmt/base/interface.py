"""How the base (prior) distribution plugs into ``LEGOLtng``: the ``BaseDist`` mixin and the head's construction.

``BaseDist.gen_base_wrapper`` draws ``x_0``: the data on conditioning cells, a draw of the prior on generated ones.
An event's prior comes from the base head, a small network fitted by likelihood once before the flow trains and
frozen from then on. The pieces live next door:

    base_dists.py   the prior's distributions (``Energy``, ``Angles``, ``Edep``) and their parameters (``BaseParams``)
    nn.py           the head's network (``BaseHead``)
    nn_fit.py       its loss and fitting recipe
    gen_base.py     the sampler (``GenerateBase``), ot_coupling.py the training-time coupling to the data
    edep_whiten.py  the unit the flow carries the E_dep token in

``BaseDist`` is a mixin rather than an object because ``lego_eval`` reaches for ``lego.gen_base``,
``lego.gen_base_wrapper`` and ``hasattr(lego, "base_head")`` on the LightningModule itself, and ``base_head`` is a
state_dict key: it must stay a direct attribute of ``LEGOLtng`` under that name, so ``build_base_head`` returns the
module for ``__init__`` to assign. The mixin reads from ``LEGOLtng``: ``rc``, ``model``, ``gen_base``, ``base_head``,
``pdgids_template``, ``cuboid_unit``, ``convert_pdgids``, ``_zoom_dirs``, ``_make_loader``, ``_train_ds``, ``trainer``.

Configs without ``base_conf.base_head_lik`` (moment head, no head) run through :mod:`legofmt.legacy.base_nn`, an
E_dep token without the plain ``model_conf.edep_whiten`` through :mod:`legofmt.legacy.edep_token`, sampler options
other than the recipe's through :mod:`legofmt.legacy.gen_base`.
"""

from itertools import chain, islice, repeat

import torch
from torch import Tensor, nn

from legofmt.base_dist.base_dists import BaseParams
from legofmt.base_dist.edep_whiten import whiten_edep
from legofmt.base_dist.nn import BaseHead
from legofmt.base_dist.nn_fit import fit_head, head_nll
from legofmt.base_dist.ot_coupling import ot_couple
from legofmt.cfm.geom_encoder import x0_gcm2
from legofmt.data.struct import DataStruct
from legofmt.geometry.geom_trafos import cuboid_chord
from legofmt.legacy.base_nn import build_legacy_base_head
from legofmt.rng import species_rank_ids


def build_base_head(rc, gen_base) -> nn.Module | None:
    """The base head for ``LEGOLtng.__init__`` to assign as ``base_head``: fresh (fitted in ``setup``), or loaded from
    ``base_conf.base_head`` and frozen. ``resolve_legoltng_config`` has checked the config. Configs without
    ``base_head_lik`` get the legacy moment head, or ``None`` (no head)."""
    base_conf = rc.config.get("base_conf") or {}
    if not base_conf.get("base_head_lik", False): return build_legacy_base_head(rc, gen_base)
    head  = BaseHead(len(rc.pdgids_template) + 1, base_conf.get("base_head_hidden", [256, 256]),
                        float(gen_base.sm_scale), float(gen_base.kappa))
    saved = base_conf.get("base_head")
    if saved is None:
        return head
    try:   # a mismatch raises, never silently re-pretrains
        head.load_state_dict(saved)
    except RuntimeError as err:
        raise ValueError("base_conf.base_head does not load into the base_head_lik head (base_head_hidden, "
                         "pdgids?).") from err
    return head.requires_grad_(False)


class BaseDist:
    """Base-distribution mixin for :class:`~legofmt.main.modules.LEGOLtng`: draw ``x_0``, fit the head, whiten E_dep."""

    # ---- drawing x_0 ------------------------------------------------------

    def gen_base_wrapper(self, ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]", all_forward: bool = False,
                         rng=None) -> Tensor:
        """Draw ``x_0``: the data on conditioning cells; on generated cells the head's prior for forward events and
        an isotropic sample for inverse-mask events (``all_forward``: treat every event as forward). The E_dep cell
        is in the flow's unit (``edep_whiten.py``); ``solve`` starts from it as it is. ``lego_eval`` calls this.
        ``rng`` (legofmt.rng.EventRNG) keys every draw on the event and an outgoing slot's (species, rank)."""
        if not isinstance(ds_t, DataStruct):
            ds_t = DataStruct(*ds_t)
        data, m, fwd = ds_t.f.model_in, ds_t.m.full, ds_t.m.fwd   # fwd: the incoming slot conditions
        if all_forward or fwd.all():
            x0 = self._draw_prior(ds_t, data, rng)
        else:   # inverse-mask events get the isotropic sample, drawn first
            x0 = self.gen_base.iso(m.shape, data.device, rng=rng)
            if fwd.any():
                x0 = torch.where(fwd.view(-1, 1, 1), self._draw_prior(ds_t, data, rng), x0)
        return torch.where((m == 1).unsqueeze(-1), x0, data)

    def _draw_prior(self, ds_t: DataStruct, data: Tensor, rng) -> Tensor:
        """A draw of the head's prior for every event: the outgoing slots, in training their OT coupling to the
        data, then the E_dep token (a standard normal)."""
        self._set_prior(ds_t)
        elem  = None if rng is None else species_rank_ids(ds_t.f.out_pid.long())
        prior = torch.cat((ds_t.f.non_cc, self.gen_base(ds_t.m.out_p.shape, ds_t.f.in_cc, rng=rng, elem=elem)), dim=1)
        if self.rc.ot_coupling and self.model.training:
            prior = ot_couple(prior, ds_t, data, self.rc.manifold)
        return self.gen_base.insert_add(prior, rng=rng)

    def _set_prior(self, ds_t: DataStruct) -> None:
        """Hand the sampler this batch's prior (``lego_eval`` reads ``gen_base.edep_mu`` / ``edep_sig`` back). With
        grad mode on (density design) the frozen head keeps its graph to the inputs, so the draw is reparameterized
        in them."""
        p = self._slot_prior(ds_t)
        if self.model.training and next(self.base_head.parameters()).requires_grad:
            p = BaseParams(*(v.detach() for v in p))   # a head not fitted yet: the flow's loss does not train it
        self.gen_base.sm_scale = p.sm_scale.unsqueeze(-1)
        self.gen_base.edep_mu  = p.edep_mu
        self.gen_base.edep_sig = p.edep_sig
        self.gen_base.kappa    = p.kappa
        self.gen_base.ang_pow  = p.ang_pow
        self.gen_base.sm_pow   = p.sm_pow.unsqueeze(-1)

    def _slot_prior(self, ds_t: DataStruct) -> BaseParams:
        """``base_head_params`` with each per-class column gathered onto the outgoing slots, ``(B, n_out)``."""
        p   = self.base_head_params(ds_t)
        idx = self._species_idx(ds_t.f.out_pid, p.kappa.shape[-1])
        return BaseParams(p.sm_scale.gather(1, idx), p.edep_mu, p.edep_sig,
                          p.kappa.gather(1, idx), p.ang_pow.gather(1, idx), p.sm_pow.gather(1, idx))

    def base_head_params(self, ds_t: DataStruct) -> BaseParams:
        """The head's prior for each event of ``ds_t``, one column per outgoing class."""
        return self.base_head(*self._head_inputs(ds_t))

    def _head_inputs(self, ds_t: DataStruct) -> tuple[Tensor, Tensor, Tensor]:
        """What the head reads off an event, ``(cont, species, out_set)``: ``cont`` (B, 3) = (E, chord, log t) of the
        incoming particle and the block (t = ``Size*Density/X0``), the primary's species one-hot (B, n_cls), and the
        outgoing set (B, 1 + n_cls) = log1p of (multiplicity, count per class). The outgoing set is fixed before the
        flow runs; E_dep is bimodal in it, which one logit-normal cannot hold."""
        f, n_cls = ds_t.f, len(self.pdgids_template) + 1
        species  = nn.functional.one_hot(self._species_idx(f.in_pid, n_cls), n_cls).float()
        t        = f.cond("Size") * f.cond("Density") / x0_gcm2(f.cond("Z"), f.cond("A"))
        chord    = cuboid_chord(f.in_mom.nan_to_num(1.0), f.in_pos.nan_to_num(1.0), self.cuboid_unit)
        valid    = ds_t.am.out_p
        out_oh   = nn.functional.one_hot(self._species_idx(f.out_pid, n_cls), n_cls)
        count    = (out_oh * valid.unsqueeze(-1)).sum(-2).float()
        n_out    = valid.sum(-1, keepdim=True).float()
        out_set  = torch.cat((torch.log1p(n_out), torch.log1p(count)), dim=-1)
        return torch.cat((f.in_energy, chord, torch.log(t).unsqueeze(-1)), dim=-1), species, out_set

    def _species_idx(self, pid: Tensor, n_cls: int) -> Tensor:
        """Class index in ``[0, n_cls)`` of raw pdgids (of indices with ``pdgid_is_idx``)."""
        return (pid.long() if self.rc.pdgid_is_idx else self.convert_pdgids(pid)).long().clamp(0, n_cls - 1)

    # ---- fitting the head: once, before the flow trains --------------------

    def pretrain_base(self, batches, lr: float | None = None, steps: int | None = None) -> float:
        """Fit the head alone on ``batches`` by likelihood (``nn_fit.fit_head``; ``steps`` is its schedule length,
        required), then it is frozen: the flow never trains it. Nothing is drawn, so the random stream stays where
        it was. Returns the final NLL."""
        if steps is None:
            raise ValueError("base_head_lik pretraining needs steps (its warmup and cosine).")

        def nll(ds_t: DataStruct) -> Tensor | None:
            fwd = ds_t.m.fwd
            if not fwd.any():   # the loss is over forward events
                return None
            if self.rc.dir_zoom:
                ds_t = DataStruct(self._zoom_dirs(ds_t.f.full, ds_t.am.full, fwd, pdgids=ds_t.f.pdgids),
                                  ds_t.m.full, ds_t.am.full)
            return head_nll(ds_t, fwd, self._slot_prior(ds_t), self.gen_base.e_dep_max).sum().float()

        return fit_head(self.base_head, nll, batches, steps, lr)

    def _pretrain_base_if_needed(self) -> None:
        """``setup`` calls this. Rank 0 fits the head on its device (the data stays on the CPU, moved per batch) and
        broadcasts the weights; every rank ends with the head frozen. A head that is frozen already (loaded from the
        config, or a resumed run) is left alone."""
        head, n = self.base_head, self.rc.base_pretrain_batches
        if not (n and getattr(self, "_trainer", None) and next(head.parameters()).requires_grad):
            return
        if self.trainer.is_global_zero:
            dev = self.trainer.strategy.root_device
            head.to(dev)
            cont = []   # standardise (E, chord, log t) on the forward training events
            with torch.no_grad():
                for b in self._make_loader(self._train_ds, shuffle=False, bs=2**17):
                    b = b.to(dev)
                    cont.append(self._head_inputs(b)[0][b.m.fwd])
            cont = torch.cat(cont)
            std  = cont.std(0)
            head.mean.copy_(cont.mean(0))
            head.std.copy_(std.where(std > 1e-6, torch.inf))
            loader = self._make_loader(self._train_ds, shuffle=True, bs=self.rc.base_pretrain_bs)
            epochs = chain.from_iterable(repeat(loader))   # one reshuffled epoch after another
            final  = self.pretrain_base((b.to(dev) for b in islice(epochs, n)), steps=n)
            head.cpu()
            print(f"base_head: pretrained on {n} batches (final NLL {final:.4f}), frozen")
        weights = self.trainer.strategy.broadcast({k: v.cpu() for k, v in head.state_dict().items()}, src=0)
        head.load_state_dict(weights)
        head.requires_grad_(False)

    # ---- the flow's E_dep token in units of the prior -------------------------

    def _edep_whiten(self, f: Tensor, ds_t: DataStruct, inverse: bool = False) -> Tensor:
        """A copy of ``f`` with the generated E_dep token in units of each event's prior (``inverse``: back to a
        deposit), see ``edep_whiten.py``. The prior comes from the frozen head on ``ds_t``, so any sub-batch is
        consistent."""
        _, mu, sig, _, _, _ = self.base_head_params(ds_t)   # by position: a legacy head returns a plain tuple
        return whiten_edep(f, ds_t.m.full[:, 1] == 1, mu.reshape(-1), sig.reshape(-1), self.gen_base.e_dep_max,
                           inverse)
