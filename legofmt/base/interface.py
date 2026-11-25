"""How the base plugs into ``LEGOLtng``: the ``BaseDist`` mixin and the base head's construction.

``BaseDist.gen_base_wrapper`` samples ``x_0``: the data on the conditioning tokens, a sample of the event's prior on
the generated ones. An event's prior comes from the base head, a small network that is fitted by likelihood once,
before the flow trains, and frozen from then on. The pieces are in the files next to this one:

    base_dists.py    the prior's distributions (``Energy``, ``Angles``, ``Edep``) and their parameters (``BaseParams``)
    nn.py            the head's network (``BaseHead``)
    nn_fit.py        its loss and how it is fitted
    gen_base.py      the sampler (``GenerateBase``)
    ot_coupling.py   the coupling of the sample to the data, in training
    edep_whiten.py   the unit the flow carries the E_dep token in

``BaseDist`` is a mixin rather than an object for two reasons. ``lego_eval`` reads ``lego.gen_base``,
``lego.gen_base_wrapper`` and ``hasattr(lego, "base_head")`` on the LightningModule itself. And ``base_head`` is a
state_dict key, so the head must stay a direct attribute of ``LEGOLtng`` under that name: ``build_base_head`` returns
the module and ``LEGOLtng.__init__`` assigns it. The mixin reads from ``LEGOLtng``: ``rc``, ``model``, ``gen_base``,
``base_head``, ``pdgids_template``, ``cuboid_unit``, ``species_idx``, ``_zoom_dirs``, ``_make_loader``, ``_train_ds``,
``trainer``.

Legacy paths:

    no ``base_conf.base_head_lik`` (moment head, or no head)             :mod:`legofmt.legacy.base_nn`
    an E_dep token without the plain ``model_conf.edep_whiten``          :mod:`legofmt.legacy.edep_token`
    sampler options other than the recipe's                              :mod:`legofmt.legacy.gen_base`
    inverse-mask events (``mask_conf.p_forward < 1``)                    :mod:`legofmt.legacy.gen_base`
"""

from itertools import chain, islice, repeat

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from legofmt.base.base_dists import BaseParams
from legofmt.base.edep_whiten import whiten_edep
from legofmt.base.nn import BaseHead
from legofmt.base.nn_fit import fit_head, head_nll
from legofmt.base.ot_coupling import ot_couple
from legofmt.data.struct import DataStruct
from legofmt.geometry.geom_trafos import cuboid_chord
from legofmt.geometry.material import x0_gcm2
from legofmt.legacy.base_nn import build_legacy_base_head
from legofmt.legacy.gen_base import x0_inverse_mask


def build_base_head(rc, gen_base) -> nn.Module | None:
    """Constructs the base head that ``LEGOLtng.__init__`` assigns as ``base_head``.

    The head is fresh (``setup`` fits it), or loaded from ``base_conf.base_head`` and frozen. Configs without
    ``base_head_lik`` get the legacy moment head, or no head. ``resolve_legoltng_config`` has checked the config.

    Args:
        rc:       the resolved config.
        gen_base: the sampler; a fresh head starts at its constants ``sm_scale`` and ``kappa``.

    Returns:
        The head, or None (no head).

    Raises:
        ValueError: ``base_conf.base_head`` does not load into the head.
    """
    base_conf = rc.config.get("base_conf") or {}
    if not base_conf.get("base_head_lik", False): return build_legacy_base_head(rc, gen_base)
    head  = BaseHead(len(rc.pdgids_template) + 1, base_conf.get("base_head_hidden", [256, 256]),
                     float(gen_base.sm_scale), float(gen_base.kappa))
    saved = base_conf.get("base_head")
    if saved is None:
        return head
    try:   # a saved head that does not load raises: it is never silently replaced by a fresh one
        head.load_state_dict(saved)
    except RuntimeError as err:
        raise ValueError("base_conf.base_head does not load into the base_head_lik head (base_head_hidden, "
                         "pdgids?).") from err
    return head.requires_grad_(False)


class BaseDist:
    """The base's mixin for :class:`~legofmt.main.modules.LEGOLtng`: samples ``x_0``, fits the head, whitens E_dep."""

    # ---- sampling x_0 -----------------------------------------------------

    def gen_base_wrapper(self, ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]", all_forward: bool = False,
                         rng=None) -> Tensor:
        """Samples ``x_0`` for a batch: the data on the conditioning tokens, a sample of the base on the generated ones.

        Every event is sampled from its own prior, which the base head predicts. The E_dep token is in the flow's
        unit (``edep_whiten.py``), which is what ``solve`` starts from. ``lego_eval`` calls this method.

        Args:
            ds_t:        the batch, a ``DataStruct`` or the three tensors it is built from.
            all_forward: treat every event as forward, whatever its mask.
            rng:         keys the noise on the event and on an outgoing slot's (species, rank)
                (``legofmt.rng.EventRNG``); None: torch's noise.

        Returns:
            ``x_0`` (B, L, 7).
        """
        if not isinstance(ds_t, DataStruct):
            ds_t = DataStruct(*ds_t)
        if not (all_forward or ds_t.m.fwd.all()): return x0_inverse_mask(self, ds_t, rng)   # legacy: p_forward < 1
        x0 = self._draw_prior(ds_t, rng)
        return torch.where((ds_t.m.full == 1).unsqueeze(-1), x0, ds_t.f.model_in)

    def _draw_prior(self, ds_t: DataStruct, rng) -> Tensor:
        """Samples every event of ``ds_t`` from its prior.

        In training the outgoing slots are OT-coupled to the data (``model_conf.ot_coupling``). Returns ``x_0``
        (B, L, 7); ``rng`` as in ``gen_base_wrapper``.
        """
        self._set_prior(ds_t)
        prior = self.gen_base(ds_t, rng)
        if self.rc.ot_coupling and self.model.training:
            prior = ot_couple(prior, ds_t, self.rc.manifold)
        return prior

    def _set_prior(self, ds_t: DataStruct) -> None:
        """Sets the sampler's parameters to this batch's prior, per outgoing slot.

        ``lego_eval`` reads ``gen_base.edep_mu`` / ``edep_sig`` back. With grad mode on (density design) the frozen
        head keeps its graph to the inputs, so the sample is reparameterised in them.
        """
        p = self._slot_prior(ds_t)
        if self.model.training and next(self.base_head.parameters()).requires_grad:
            p = BaseParams(*(v.detach() for v in p))   # a head not frozen yet: the flow's loss must not train it
        self.gen_base.sm_scale = p.sm_scale
        self.gen_base.edep_mu  = p.edep_mu
        self.gen_base.edep_sig = p.edep_sig
        self.gen_base.kappa    = p.kappa
        self.gen_base.ang_pow  = p.ang_pow
        self.gen_base.sm_pow   = p.sm_pow

    def _slot_prior(self, ds_t: DataStruct) -> BaseParams:
        """Gathers the head's per-species parameters onto the outgoing slots of ``ds_t``, by each slot's species.

        Returns ``BaseParams`` with the energy and angle parameters (B, n_out); E_dep's stay (B,).
        """
        p   = self.base_head_params(ds_t)
        idx = self._species_idx(ds_t.f.out_pid, p.kappa.shape[-1])
        return BaseParams(p.sm_scale.gather(1, idx), p.edep_mu, p.edep_sig,
                          p.kappa.gather(1, idx), p.ang_pow.gather(1, idx), p.sm_pow.gather(1, idx))

    def base_head_params(self, ds_t: DataStruct) -> BaseParams:
        """Evaluates the base head on the events of ``ds_t``.

        Returns ``BaseParams`` with the energy and angle parameters (B, n_cls), one column per outgoing species, and
        E_dep's (B,).
        """
        return self.base_head(*self._head_inputs(ds_t))

    def _head_inputs(self, ds_t: DataStruct) -> tuple[Tensor, Tensor, Tensor]:
        """Builds the head's inputs from the events of ``ds_t``.

        The outgoing set is an input because it is fixed before the flow runs. Without it E_dep is bimodal, which one
        logit-normal cannot hold.

        Returns:
            cont:    (B, 3), ``(E, chord, log t)``: the incoming energy scalar, the chord of the incoming particle's
                line through the cuboid (``cuboid_chord``), and ``t = Size * Density / X0``.
            species: (B, n_cls), the incoming species' one-hot.
            out_set: (B, 1 + n_cls), log1p of (the number of outgoing particles, their count per species).
        """
        f, n_cls, valid = ds_t.f, len(self.pdgids_template) + 1, ds_t.am.out_p
        species = F.one_hot(self._species_idx(f.in_pid, n_cls), n_cls).float()
        t       = f.cond("Size") * f.cond("Density") / x0_gcm2(f.cond("Z"), f.cond("A"))
        chord   = cuboid_chord(f.in_mom.nan_to_num(1.0), f.in_pos.nan_to_num(1.0), self.cuboid_unit)
        count   = _count_per_class(self._species_idx(f.out_pid, n_cls), valid, n_cls)
        n_out   = valid.sum(-1, keepdim=True).float()
        out_set = torch.log1p(torch.cat((n_out, count), dim=-1))
        cont    = torch.cat((f.in_energy, chord, torch.log(t).unsqueeze(-1)), dim=-1)   # (E, chord, log t)
        return cont, species, out_set

    def _species_idx(self, pid: Tensor, n_cls: int) -> Tensor:
        """Maps pdgids (with ``rc.pdgid_is_idx``: species indices) to the head's species indices in ``[0, n_cls)``."""
        return self.species_idx(pid).clamp(0, n_cls - 1)

    # ---- fitting the head: once, before the flow trains --------------------

    def pretrain_base(self, batches, lr: float | None = None, steps: int | None = None) -> float:
        """Fits the head alone on ``batches`` by likelihood, then freezes it.

        The flow never trains the head. The fit samples nothing, so torch's random generator stays where it was. The
        loss is the sum of ``nn_fit.head_nll``'s three terms over the forward events; the optimiser and the schedule
        are ``nn_fit.fit_head``'s.

        Args:
            batches: the training batches (``DataStruct``), one step each.
            lr:      the peak learning rate; None: ``fit_head``'s default.
            steps:   the schedule's length (its warmup and cosine), required.

        Returns:
            The last batch's loss.

        Raises:
            ValueError: ``steps`` is None.
        """
        if steps is None:
            raise ValueError("base_head_lik pretraining needs steps (its warmup and cosine).")

        def nll(ds_t: DataStruct) -> Tensor | None:
            fwd = ds_t.m.fwd
            if not fwd.any():   # no forward event to score: the batch is skipped
                return None
            if self.rc.dir_zoom:
                ds_t = DataStruct(self._zoom_dirs(ds_t.f.full, ds_t.am.full, fwd, pdgids=ds_t.f.pdgids),
                                  ds_t.m.full, ds_t.am.full)
            return head_nll(ds_t, fwd, self._slot_prior(ds_t), self.gen_base.e_dep_max).sum().float()

        return fit_head(self.base_head, nll, batches, steps, lr)

    def _pretrain_base_if_needed(self) -> None:
        """Fits the head before the flow trains, unless it is frozen already.

        ``setup`` calls this. Rank 0 standardises the head's ``cont`` inputs on the forward training events, fits the
        head on its device (the data stays on the CPU and is moved per batch) and broadcasts the weights; every rank
        ends with the head frozen. Nothing happens without a trainer, with ``base_pretrain_batches: 0``, or for a head
        that is frozen already (loaded from the config, or a resumed run).
        """
        head, n = self.base_head, self.rc.base_pretrain_batches
        if not (n and getattr(self, "_trainer", None) and next(head.parameters()).requires_grad):
            return
        if self.trainer.is_global_zero:
            dev = self.trainer.strategy.root_device
            head.to(dev)
            cont = _forward_cont(self, dev)   # standardise (E, chord, log t) on the forward training events
            std  = cont.std(0)
            head.mean.copy_(cont.mean(0))
            head.std.copy_(torch.where(std > 1e-6, std, torch.inf))   # inf: a constant input is dropped
            final = self.pretrain_base(_reshuffled_batches(self, n, dev), steps=n)
            head.cpu()
            print(f"base_head: pretrained on {n} batches (final NLL {final:.4f}), frozen")
        _share_frozen(head, self.trainer.strategy)

    # ---- the flow's E_dep token in units of the prior -------------------------

    def _edep_whiten(self, f: Tensor, ds_t: DataStruct, inverse: bool = False, drawn: bool = False) -> Tensor:
        """Whitens the generated E_dep token of ``f`` into units of each event's prior (``inverse``: back to a deposit).

        The map is ``edep_whiten.py``. The prior is the frozen head's on ``ds_t``, a function of the event alone, so a
        sub-batch gets the same unit as the full batch.

        Args:
            f:       (B, L, D), the tokens to map; may carry a leading time axis.
            ds_t:    the batch ``f`` belongs to: the head's inputs, and its ``m`` marks the generated E_dep tokens.
            inverse: map the token back to a deposit.
            drawn:   the base of exactly this batch was just sampled (``gen_base_wrapper``), so its prior is on the
                sampler and the head is not evaluated again.

        Returns:
            A copy of ``f`` with the E_dep token mapped.
        """
        if drawn:
            mu, sig = self.gen_base.edep_mu, self.gen_base.edep_sig
        else:
            _, mu, sig, _, _, _ = self.base_head_params(ds_t)   # by position: a legacy head returns a plain tuple
        return whiten_edep(f, ds_t.m.edep == 1, mu, sig, self.gen_base.e_dep_max, inverse)


# ---- plumbing --------------------------------------------------------------------------------------------------------

def _count_per_class(cls: Tensor, valid: Tensor, n_cls: int) -> Tensor:
    """Counts each event's valid slots per species: (B, n) species indices -> (B, n_cls), float.

    A scatter-add, not a sum over a (B, n, n_cls) one-hot: with nuclei as species (n_cls ~ 200) and events of ~1000
    tokens that one-hot takes hundreds of GB.
    """
    return torch.zeros(*cls.shape[:-1], n_cls, device=cls.device).scatter_add_(-1, cls, valid.float())


def _forward_cont(lego: BaseDist, dev) -> Tensor:
    """Returns the head's ``cont`` inputs (n, 3) of every forward training event, one unshuffled pass on ``dev``."""
    cont = []
    with torch.no_grad():
        for b in lego._make_loader(lego._train_ds, shuffle=False, bs=2**17):
            b = b.to(dev)
            cont.append(lego._head_inputs(b)[0][b.m.fwd])
    return torch.cat(cont)


def _reshuffled_batches(lego: BaseDist, n: int, dev):
    """Yields ``n`` training batches on ``dev``, one reshuffled epoch after another."""
    loader = lego._make_loader(lego._train_ds, shuffle=True, bs=lego.rc.base_pretrain_bs)
    epochs = chain.from_iterable(repeat(loader))
    return (b.to(dev) for b in islice(epochs, n))


def _share_frozen(head: nn.Module, strategy) -> None:
    """Broadcasts rank 0's weights of ``head`` to every rank and freezes it."""
    weights = strategy.broadcast({k: v.cpu() for k, v in head.state_dict().items()}, src=0)
    head.load_state_dict(weights)
    head.requires_grad_(False)
