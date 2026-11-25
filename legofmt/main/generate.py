"""``GenerateOut``: a multiplicity checkpoint and a flow checkpoint put together into a generator.

One call, in the order it runs:

```
cond      [*cond_scalars, mom * E_MeV (3), pos (3), pdgid] per incoming particle, as given;
          [*cond_scalars, e, dir (3), pos (3), pdgid] on the model's scales: the position ray-traced onto the cube,
          the energy split off the momentum (_model_cond); after the scalars it is the incoming token
counts    the multiplicity model samples the pass-through flag and the counts per species (sample_counts); a
          given gt_mult takes the counts' place, and then no event is a pass-through (gen_batch)
slots     n_s = floor(c_s min(1, max_particles / C)),  C = sum_s c_s;  the slots the rounding frees take species
          sampled ~ c_s (fit_counts); the slots fill species by species (slot_species)
tokens    [prefix, incoming, slot_1 .. slot_max_particles], the prefix being the conditioning scalar tokens and the
          E_dep token (struct.py): each slot a copy of the incoming token with its species index, 0 when empty
          (build_batch)
flow      LEGOLtng.forward flows the base to the outgoing particles and E_dep; pass-through events skip it, and with
          model_conf.exclude_absorbed so do absorbed events (no outgoing particle)
output    the species stay vocabulary indices (pdgids_of: the PDG ids); the E_dep token capped at
          norm_edep_layer(E_in + excess(species))
          (_clamp_edep); an absorbed event's E_dep = E_in, + 2 m_e for an e+ (absorbed_deposit); an exit point just
          across an edge from the face its momentum leaves through mirrored onto that face (CubeTrace.exit_across_edge);
          with model_conf.edep_ae the E_dep layers [MeV], left in self.edep_layers
```

The momentum in ``cond`` must be energy-scaled, not unit-norm, or the event is conditioned on a ~1 MeV particle. The
output stays on the model's scales; ``dp.denorm_out_energy`` decodes an outgoing energy to MeV, with ``pdgids_of``'s
ids.
``gen_model_w_g4_args`` takes Geant4-shaped arguments instead. The energy balance and exit consistency
(``odeint_conf``) and ``GenerateIn``, the reverse direction, are :mod:`legofmt.legacy.generate`.
"""

import warnings
from dataclasses import replace

import torch
import torch.nn.functional as F

from legofmt.cfm.project_model import is_compiled
from legofmt.compat import dynamic_shape_attention
from legofmt.data.write.norm import DataPrep
from legofmt.data.struct import _F, _M, cond_slot, loss_mask, set_layout, species_values, vocab_index
from legofmt.legacy.generate import balance_passthrough, close_balance, generate_in
from legofmt.legacy.norm import HAND_SET_EXCESS_MEV
from legofmt.main.modules import LEGOLtng
from legofmt.multiplicity.model import MultModel
from legofmt.rng import REFILL, categorical, rng_on, rng_sub

# an absorbed e+ annihilates at rest, and its two photons' 2 m_e [MeV] are deposited too
_REST_DEPOSIT_MEV = {-11: 1.022}


def absorbed_deposit(e_in: torch.Tensor, pid_in: torch.Tensor) -> torch.Tensor:
    """Returns the E_dep [MeV] of absorbed events (no outgoing particle): ``E_in``, plus 2 m_e for an e+.

    ``e_in`` [MeV] and ``pid_in`` (raw pdgids) are (B,).
    """
    rest, _ = species_values(_REST_DEPOSIT_MEV, pid_in, e_in)
    return e_in + rest


def fit_counts(counts: torch.Tensor, max_particles: int, refill) -> torch.Tensor:
    """Fits each event's per-species counts into ``max_particles`` slots (module docstring, ``slots``).

    ```
    n_s = floor(c_s min(1, M / C)),   C = sum_s c_s,   M = max_particles
    when C > M, the M - sum_s n_s slots the rounding frees take species sampled ~ c_s
    ```

    An event without particles (C = 0) hands ``refill`` uniform weights, only so that it has weights to sample
    from: none of its samples count.

    Args:
        counts:        (B, S), the sampled counts.
        max_particles: the slots per event.
        refill:        a function that samples (B, max_particles) species indices from (B, S) weights.

    Returns:
        (B, S), the fitted counts.
    """
    total   = counts.sum(-1, keepdim=True)                                 # C
    weights = counts.float() + (total == 0)                                # ~ c_s; uniform when C = 0
    fitted  = (counts * (max_particles / total).clamp(max=1.0)).long()     # n_s
    freed   = ((total > max_particles) * (max_particles - fitted.sum(-1, keepdim=True))).clamp(min=0)
    slots   = torch.arange(max_particles, device=counts.device)
    return fitted.scatter_add_(-1, refill(weights), (slots < freed).long())   # the first `freed` samples count


def slot_species(counts: torch.Tensor, max_particles: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns which slots hold a particle and each slot's species: the slots fill species by species.

    Args:
        counts:        (B, S), the fitted counts.
        max_particles: the slots per event.

    Returns:
        ``(occupied, species)``, both (B, max_particles). Slot j's species, an index into the S species, is the
        number of cumulative counts ``<= j``; on an empty slot it means nothing.
    """
    slots    = torch.arange(max_particles, device=counts.device)
    occupied = slots < counts.sum(-1, keepdim=True)
    bounds   = counts.cumsum(-1)[..., :-1].contiguous()
    return occupied, torch.searchsorted(bounds, slots.expand(len(counts), -1).contiguous(), right=True)


def mult_energy_scale(mult_conf: dict, rc) -> float:
    """Checks a multiplicity config against the flow's, and returns the factor between their energy scalars.

    ```
    scale = log(max_energy / cutoff) / log(mm_max_energy / cutoff)       1 when the two max_energy agree
    ```

    The flow's energy scalar times ``scale`` is the same energy on the multiplicity model's scale. Warns when the two
    ``cutoff_mev`` differ: the factor assumes one cutoff.

    Args:
        mult_conf: the multiplicity checkpoint, ``{"config": ..., "state_dict": ...}``.
        rc:        the flow's resolved config.

    Raises:
        ValueError: the multiplicity model's ``cond_scalars`` are not the first of the flow's, in order.
    """
    mm = mult_conf["config"]["mm_conf"]
    mm_cond = mm.get("cond_scalars")
    if mm_cond and tuple(mm_cond) != tuple(rc.cond_scalars)[:len(mm_cond)]:
        raise ValueError(f"mult cond_scalars {tuple(mm_cond)} is not a prefix of the flow's "
                         f"{tuple(rc.cond_scalars)}; cond carries only the flow's scalars.")
    mm_cut = mult_conf["config"]["dl_conf"]["lds_args"].get("cutoff_mev")
    if mm_cut is not None and mm_cut != rc.cutoff_mev:
        warnings.warn(f"mult cutoff_mev {mm_cut} != flow {rc.cutoff_mev}: "
                      "the two energy scales disagree and mult_e_scale assumes they do not", stacklevel=3)
    mm_max = mm.get("max_energy")
    if not mm_max or mm_max == rc.max_energy:
        return 1.0
    cut = rc.cutoff_mev
    return (torch.log(torch.tensor(rc.max_energy / cut)) / torch.log(torch.tensor(mm_max / cut))).item()


class GenerateOut(torch.nn.Module):
    """The generator: incoming particles in, outgoing particles and E_dep out.

    Args:
        flow_conf_path:       the path of the flow checkpoint.
        mult_conf_path:       the path of the multiplicity checkpoint.
        device:               where both models run.
        couple_in_out_pdgids: keep only the outgoing particles whose species is one of the multiplicity model's
            incoming species ``ptypes_in``; the others come back NaN with pdgid 0 (``odeint_conf.filter_pdgid``).
    """

    flow_cls = LEGOLtng

    def __init__(self, flow_conf_path: str, mult_conf_path: str, device="cpu", couple_in_out_pdgids=False):
        super().__init__()
        # the flow; it reads species as vocabulary indices from here on
        flow_conf  = torch.load(flow_conf_path, map_location=device, weights_only=False)
        self.model = self.flow_cls(flow_conf).to(device)
        self.model.rc = rc = replace(self.model.rc, pdgid_is_idx=True)
        mc = flow_conf["config"]["model_conf"]
        self.cond_names = tuple(rc.cond_scalars)
        self.n_cond     = len(self.cond_names)   # = n_prefix - 1: every prefix token but E_dep
        self.n_prefix   = rc.n_prefix
        self.max_seq_l  = mc["model_args"]["max_seq_l"]
        self.pdgids     = mc["pdgids"].to(device)

        # the multiplicity model; its conditioning scalars are the first of the flow's, its energy scalar is rescaled
        mult_conf     = torch.load(mult_conf_path, map_location=device, weights_only=False)
        self.gen_mult = MultModel(mult_conf).to(device).eval()   # in training mode its dropout enters the samples
        mm = mult_conf["config"]["mm_conf"]
        self._compile_mult(mm)
        self.pdgid_in     = mm["ptypes_in"].to(device)
        self.ptypes       = mm["ptypes"].to(device)
        # per species of the flow: its column in the sampled counts, and whether the multiplicity model counts it
        self.ptype_idx, self.ptype_in_mask = vocab_index(self.ptypes, self.pdgids)
        self.n_mult_cond  = (len(mm["cond_scalars"]) if mm.get("cond_scalars")
                             else self.gen_mult.proj_in_.in_features - 7)
        self.mult_e_scale = mult_energy_scale(mult_conf, rc)

        set_layout(self.cond_names)   # back to the flow's layout: resolving the multiplicity config set its own
        if couple_in_out_pdgids:
            rc.odeint_conf["filter_pdgid"] = self.pdgid_in

        # the flow's DataPrep: the E_dep scales (dp), the energy scalars (pen), the incoming position's ray trace
        # (proj_ray)
        self.dp         = DataPrep(flow_conf)
        self.pen        = self.dp.pen
        self.proj_ray   = self.dp.ppa
        # cuboid_dim is kept apart from proj_ray: for a model trained without the ray trace, callers stub proj_ray
        self.cuboid_dim = self.proj_ray.cuboid_dim

    def __call__(self, cond: torch.Tensor, prepped: bool = False, passthrough: bool = True, rng=None,
                 gt_mult: torch.Tensor | None = None):
        """Generates the outgoing particles and E_dep of a batch of incoming particles.

        Args:
            cond:        (B, n_cond + 7), ``[*cond_scalars, mom * E_MeV (3), pos (3), pdgid]``; the momentum must be
                energy-scaled.
            prepped:     ``cond`` is on the model's scales already, (B, n_cond + 8) (``_model_cond``): no ray trace
                and no energy split.
            passthrough: False: no event is sampled as a pass-through. For a caller that sampled the pass-through
                flag itself (``passthrough_prob``) and generates the interacting events.
            rng:         makes every random draw a function of the event's key: independent of the batch, and shared by
                two models run on the same keys (``legofmt.rng.EventRNG``, one unique key per event).
            gt_mult:     (B, S), each event's counts per species of the multiplicity model's ``ptypes``, e.g. the
                data's: the flow generates them, and the multiplicity model is not sampled (``gen_batch``). None: the
                sampled counts.

        Returns:
            ``(sols, mask, attn_mask)``: the tokens (B, L, 8) on the model's scales, the species index in the last
            channel (``pdgids_of``: its PDG id);
            the loss mask (B, L), 1 on the generated tokens; the attention mask (B, L), True on the real tokens.
            With ``model_conf.edep_ae`` the E_dep layers [MeV] are left in ``self.edep_layers`` (None without).

        Raises:
            ValueError: ``rng`` does not hold one key per event, or ``gt_mult`` is not (B, S).
        """
        rng  = rng_on(rng, cond)
        cond = self._model_cond(cond, prepped)
        cond_fm, mask, attn_mask = self.gen_batch(cond, passthrough=passthrough, rng=rng, gt_mult=gt_mult)
        e_in, pid_in = self._incoming_energy_pid(cond)

        pt_mask  = mask.sum(-1) == 0                                             # nothing to generate
        pt_mask  = balance_passthrough(self, pt_mask, attn_mask, e_in, pid_in)   # legacy: energy_balance
        absorbed = self.model.rc.exclude_absorbed & ~_M(attn_mask).out_p.any(-1) & ~pt_mask   # no outgoing particle
        sols, mask, attn_mask = self._flow(cond_fm, mask, attn_mask, pt_mask | absorbed, rng)

        out = _F(sols)
        out.edep.copy_(self._clamp_edep(out.edep, e_in, pid_in))
        e_absorbed = self.dp.norm_edep_margin(absorbed_deposit(e_in, pid_in), e_in, pid_in).to(out.edep)
        out.edep.copy_(torch.where(absorbed, e_absorbed, out.edep))   # an absorbed event deposits everything
        out.out_pos.copy_(self.proj_ray.exit_across_edge(out.out_mom, out.out_pos))   # no exit point across an edge
        close_balance(self, sols, e_in, pid_in)   # legacy: energy_balance, exit_consistency
        self.edep_layers = self._edep_layers(sols, pt_mask, e_in, pid_in, rng)
        return sols, mask, attn_mask

    def _model_cond(self, cond: torch.Tensor, prepped: bool) -> torch.Tensor:
        """Returns ``cond`` on the model's scales (module docstring); a ``prepped`` one is only copied."""
        if prepped:
            return cond.clone()
        scalars, mom, pos, pid = cond.split((self.n_cond, 3, 3, 1), dim=-1)
        pos     = self._on_cube(mom, pos)
        dir_, e = self.pen.to_scalar(mom)
        return torch.cat((scalars, e, dir_, pos, pid), dim=-1)

    def _clamp_edep(self, edep: torch.Tensor, e_in: torch.Tensor, pid_in: torch.Tensor) -> torch.Tensor:
        """Caps the E_dep token at ``norm_edep_layer(E_in + excess(species))``, out of place.

        Only a token on the layer scale is capped; a margin token is capped by its scale. The excess is the training
        data's largest ``edep - e_in`` per incoming species (``model_conf.edep_excess_mev``); a checkpoint from before
        that key takes the hand-set ``legacy.norm.HAND_SET_EXCESS_MEV``. A species without an excess stays uncapped.
        Out of place, so that the density design can differentiate through it.

        ``edep`` is (B,), the generated E_dep token; ``e_in`` [MeV] and ``pid_in`` are (B,), the incoming energy and
        pdgid.
        """
        excess, capped = self._edep_excess(pid_in, e_in)
        cap = self.dp.norm_edep_layer(e_in + excess).to(edep)
        return torch.where(capped, torch.minimum(edep, cap), edep)

    def _edep_layers(self, sols, pt_mask, e_in, pid_in, rng):
        """Returns the E_dep layers [MeV], (B, n_layers); None without an edep_ae.

        The edep_ae samples them from the latents the flow generated (``model._edep_latents``) and the event's total
        E_dep. A pass-through event deposits nothing.
        """
        ae = self.model.edep_ae
        if ae is None:
            return None
        layers, run = sols.new_zeros(sols.shape[0], ae.n_layers), ~pt_mask
        if run.any():
            edep_mev    = self.dp.denorm_edep_margin(_F(sols).edep[run].float(), e_in[run].float(), pid_in[run])
            layers[run] = ae.sample(self.model._edep_latents, edep_mev, rng=rng_sub(rng, run))
        return layers

    # ---- the multiplicity model -------------------------------------------------------------------------------------

    def gen_batch(self, cond: torch.Tensor, passthrough: bool = True, rng=None, gt_mult: torch.Tensor | None = None):
        """Builds the flow's padded ``(cond_fm, mask, attn_mask)`` from the sampled counts, or from a given ``gt_mult``.

        ``gt_mult`` is (B, S), each event's counts per species of the multiplicity model's ``ptypes``. It takes the
        place of the sample, so that the flow is evaluated without the multiplicity model: no event is a
        pass-through then, whatever ``passthrough`` says.

        Raises:
            ValueError: ``gt_mult`` is not (B, S).
        """
        if gt_mult is None:
            pt_mask, counts = self.sample_counts(cond, passthrough, rng=rng)
        else:
            pt_mask, counts = self._given_counts(cond, gt_mult)
        return self.build_batch(cond, pt_mask, counts, rng=rng)

    def sample_counts(self, cond: torch.Tensor, passthrough: bool = True,
                      rng=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples the multiplicity model for ``cond``, (B, n_cond + 8), on the model's scales.

        Returns:
            ``(pt_mask, counts)``: (B,), the pass-through flags (all False with ``passthrough=False``); (B, S), the
            counts per species of the multiplicity model's ``ptypes``, as ``MultModel.forward`` returns and
            ``MultModel.log_prob`` scores them.
        """
        mult_in, pdgid_in_idx = self._mult_in(cond)
        if passthrough:
            pt_mask = self.gen_mult.sample_passthrough(mult_in, pdgid_in_idx, rng=rng)
        else:   # the interacting branch: nothing passes through
            pt_mask = torch.zeros(len(cond), dtype=torch.bool, device=cond.device)
        counts = self.gen_mult((mult_in, None, pdgid_in_idx), pt_mask=pt_mask, rng=rng)
        return pt_mask, counts

    def passthrough_prob(self, cond: torch.Tensor, prepped: bool = False) -> torch.Tensor:
        """Returns each event's probability, (B,), that the incoming particle passes through unchanged.

        The pass-through flag is sampled with this probability; it is 0 for a species the pass-through head does not
        cover. ``cond`` and ``prepped`` as for ``__call__``.
        """
        return self.gen_mult.passthrough_prob(*self._mult_in(self._model_cond(cond, prepped)))

    def _mult_in(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the multiplicity model's input for ``cond`` and the incoming species' index into ``ptypes_in``.

        The input, (B, n_mult_cond + 7), is the multiplicity model's conditioning scalars and the incoming token without
        its pdgid; the energy scalar is rescaled to the multiplicity model's ``max_energy`` (``mult_e_scale``).

        Raises:
            ValueError: an incoming species the multiplicity model was not trained on.
        """
        scalars, e, dirs, pid = cond.split((self.n_cond, 1, 6, 1), dim=-1)
        pdgid_in_idx = self._mult_species(pid[:, 0].long())
        if self.mult_e_scale != 1.0:
            e = (e * self.mult_e_scale).clamp(0.0, 1.0)
        return torch.cat((scalars[:, :self.n_mult_cond], e, dirs), dim=-1), pdgid_in_idx

    # ---- the flow's tokens ------------------------------------------------------------------------------------------

    def build_batch(self, cond: torch.Tensor, pt_mask: torch.Tensor, counts: torch.Tensor, rng=None):
        """Builds the flow's padded tokens for the sampled counts (module docstring: slots, tokens).

        ``mask`` is 1 on the generated tokens (E_dep and the occupied slots); ``attn_mask`` is True on the prefix, the
        incoming token and the occupied slots. A pass-through event generates nothing and deposits nothing, and its
        slots hold the outgoing energy 0, which stands for the full incoming energy.

        Args:
            cond:    (B, n_cond + 8), ``cond`` on the model's scales.
            pt_mask: (B,), the pass-through events.
            counts:  (B, S), the counts per species of the multiplicity model's ``ptypes`` (``sample_counts``).
            rng:     keys ``fit_counts``' refill on the event (``legofmt.rng.EventRNG``), or None.

        Returns:
            ``(cond_fm, mask, attn_mask)``: the tokens (B, max_seq_l, 8) and the two masks (B, max_seq_l).
        """
        n_slots = self.max_seq_l - (self.n_prefix + 1)
        slots   = torch.arange(n_slots, device=counts.device)
        counts  = fit_counts(counts[:, self.ptype_idx] * self.ptype_in_mask,      # per species of the flow
                             n_slots, lambda w: categorical(w, rng, REFILL, slots))
        occupied, species = slot_species(counts, n_slots)

        tok = _F(self._tokens(cond, n_slots))   # [prefix, incoming, a copy of the incoming token per slot]
        tok.out_pid.copy_(occupied * (species + 1))                        # 0: an empty slot
        attn_mask = F.pad(occupied, (self.n_prefix + 1, 0), value=True)   # the prefix, the incoming, occupied slots
        mask      = loss_mask(attn_mask, self.n_prefix)                    # E_dep and the occupied slots

        tok.edep.masked_fill_(pt_mask, 0.0)                      # a pass-through event deposits nothing,
        tok.out_energy.masked_fill_(pt_mask.unsqueeze(1), 0.0)   # its slots hold the full incoming energy (0),
        mask.masked_fill_(pt_mask.unsqueeze(1), 0)               # and nothing is generated
        return tok.full, mask, attn_mask

    # ---- Geant4-shaped arguments ------------------------------------------------------------------------------------

    def gen_model_w_g4_args(self, n, pos, mom, energy, density, size, pdgids, Z=None, A=None, passthrough=True,
                            rng=None, gt_mult=None):
        """Generates events from Geant4-shaped arguments.

        Each argument has 1 or B rows. Row b is generated ``n`` times: ``B * n`` events, ordered by row
        (``repeat_interleave``).

        Args:
            n:           the events per row.
            pos:         (1 or B, 3), the incoming position.
            mom:         (1 or B, 3), the incoming direction; its length is ignored.
            energy:      (1 or B,), the incoming energy [MeV].
            density:     (1 or B,), the conditioning scalar ``Density``.
            size:        (1 or B,), the conditioning scalar ``Size``.
            pdgids:      (1 or B,), the incoming species.
            Z, A:        (1 or B,), the conditioning scalars ``Z`` and ``A``; None unless the checkpoint's
                ``cond_scalars`` name them.
            passthrough: as in ``__call__``.
            rng:         as in ``__call__``, with ``B * n`` keys in the events' order.
            gt_mult:     as in ``__call__``, with ``B * n`` rows in the events' order.

        Returns:
            ``{per_event, per_particle, per_voxel}`` on the model's scales:

            ```
            per_event      E_dep, and each conditioning scalar by its name, (B * n,) each
            per_particle   Incoming (B * n, 1, 8) and Outgoing (B * n, n_slots, 8), the tokens
            per_voxel      E_dep, empty: (B * n, 0, 4)
            per_slab       E_dep (B * n, n_layers), the E_dep layers (only with model_conf.edep_ae)
            ```

            ``per_event["E_dep"]`` is the normalised token, not MeV, and always on the ``norm_edep_layer`` scale,
            which callers decode with ``edep_log_min`` (a ``model_conf.edep_margin`` token is re-encoded on it here);
            the E_dep layers are on the same scale. The outgoing energy decodes to MeV with ``dp.denorm_out_energy``
            and the event's incoming energy and pdgid.
        """
        cond = self._g4_cond(n, pos, mom, energy, density, size, pdgids, Z, A)
        sols, _, _ = self(cond, passthrough=passthrough, rng=rng, gt_mult=gt_mult)
        s = _F(sols)
        out = {
            "per_event": {"E_dep": self._edep_on_layer_scale(s), **{k: s.cond(k) for k in self.cond_names}},
            "per_particle": {"Incoming": s.in_p, "Outgoing": s.out_p},
            "per_voxel": {"E_dep": sols.new_empty(sols.shape[0], 0, 4)},
        }
        if self.edep_layers is not None:  # the slab geometry's own group, on the layer scale like E_dep
            out["per_slab"] = {"E_dep": self.dp.norm_edep_layer(self.edep_layers)}
        return out

    def passthrough_prob_w_g4_args(self, n, pos, mom, energy, density, size, pdgids, Z=None, A=None):
        """Returns ``passthrough_prob`` for the arguments of ``gen_model_w_g4_args``."""
        return self.passthrough_prob(self._g4_cond(n, pos, mom, energy, density, size, pdgids, Z, A))

    # ---- plumbing ---------------------------------------------------------------------------------------------------

    def _compile_mult(self, mm: dict) -> None:
        """Compiles the multiplicity model's trunk when ``mm_conf.fwd_compile`` asks for it.

        With ``dynamic=True``: the KV-cache decode grows the cache at every slot, so a static graph recompiles per slot
        until the recompile limit sends it to eager (115 ms per 8192 events, against 68 ms for the dynamic graph).
        """
        if mm.get("fwd_compile", False) and not is_compiled(self.gen_mult.model):
            dynamic_shape_attention()
            self.gen_mult.model = torch.compile(self.gen_mult.model, dynamic=True)

    def _incoming_energy_pid(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the incoming particle's energy [MeV] and pdgid, (B,) each, from ``cond`` on the model's scales."""
        _, e, dir_, _, pid = cond.split((self.n_cond, 1, 3, 3, 1), dim=-1)
        return self.pen.from_scalar(dir_, e).norm(dim=-1), pid[:, 0].long()

    def _on_cube(self, mom: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Returns the incoming position ray-traced onto the cube along ``mom``, as the direction of x / cuboid_dim."""
        traced = self.proj_ray(torch.cat((mom, pos), dim=-1))[..., 3:]
        return F.normalize(traced / self.cuboid_dim.to(pos), dim=-1)

    def _mult_species(self, pdgid_in: torch.Tensor) -> torch.Tensor:
        """Returns the incoming species' index into the multiplicity model's ``ptypes_in``.

        Raises:
            ValueError: a species that is not in ``ptypes_in``.
        """
        pdgid_in_idx, known = vocab_index(self.pdgid_in, pdgid_in)
        if not known.all():
            raise ValueError(f"incoming pdgid(s) not in the mult model's ptypes_in={self.pdgid_in.tolist()}")
        return pdgid_in_idx

    def _given_counts(self, cond: torch.Tensor, gt_mult: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(pt_mask, counts)`` as ``sample_counts`` does, for the given counts: no pass-through event.

        Raises:
            ValueError: ``gt_mult`` is not (B, S), a row per event and a column per species of ``ptypes``.
        """
        if gt_mult.shape != (len(cond), len(self.ptypes)):
            raise ValueError(f"gt_mult needs a row per event and a column per species of ptypes="
                             f"{self.ptypes.tolist()}: {(len(cond), len(self.ptypes))}, got {tuple(gt_mult.shape)}")
        return torch.zeros(len(cond), dtype=torch.bool, device=cond.device), gt_mult.to(cond.device).long()

    def _flow(self, cond_fm, mask, attn_mask, skip, rng):
        """Runs the flow on the events that need it; an event in ``skip`` (B,) keeps its tokens as built.

        The events skipped are the pass-through events, and with ``model_conf.exclude_absorbed`` the absorbed ones.

        Returns:
            ``(sols, mask, attn_mask)``, as ``LEGOLtng.forward`` returns them.
        """
        if skip.all():
            return cond_fm.clone(), mask, attn_mask
        if not skip.any():
            return self.model((cond_fm, mask, attn_mask), rng=rng)
        keep = ~skip
        sub, _, _ = self.model((cond_fm[keep], mask[keep], attn_mask[keep]), rng=rng_sub(rng, keep))
        sols = cond_fm.clone()
        sols[keep] = sub.to(sols.dtype)
        return sols, mask, attn_mask

    def pdgids_of(self, idx: torch.Tensor) -> torch.Tensor:
        """Returns the PDG ids (int64, exact: nuclei too) of the species indices ``idx``, 0 for none."""
        return F.pad(self.pdgids, (1, 0))[idx.long()]

    def _edep_excess(self, pid_in: torch.Tensor, e_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns each event's E_dep excess [MeV] by its incoming species, and whether its E_dep token is capped.

        Capped: the species has an excess, and its token is on the layer scale (not the margin scale).
        """
        excess, known = species_values(self.dp.edep_excess_mev or HAND_SET_EXCESS_MEV, pid_in, e_in)   # legacy
        return excess, known & ~self.dp.edep_margin_mask(pid_in)

    def _tokens(self, cond: torch.Tensor, n_slots: int) -> torch.Tensor:
        """Returns the tokens before the slots get their species, (B, n_prefix + 1 + n_slots, 8).

        The prefix: each conditioning scalar in channel 0 of its token, the E_dep token's channel 0 still 0, and the
        directions of every prefix token the (1, 1, 1) filler. Then the incoming token, its pdgid a vocabulary index,
        and a copy of it in every slot.
        """
        scalars, incoming = cond[:, :self.n_cond], cond[:, self.n_cond:].clone()
        incoming[:, -1] = self.model.convert_pdgids(incoming[:, -1])
        tok = F.pad(incoming.unsqueeze(1).expand(-1, 1 + n_slots, -1), (0, 0, self.n_prefix, 0))
        tok[:, [cond_slot(n) for n in self.cond_names], 0] = scalars
        _F(tok).non_p[..., 1:-1] = 1
        return tok

    def _edep_on_layer_scale(self, s: _F) -> torch.Tensor:
        """Returns the E_dep token on the ``norm_edep_layer`` scale: a ``model_conf.edep_margin`` token re-encoded."""
        if not self.dp.edep_margin:
            return s.edep
        e_in = self.dp.pen.from_scalar(s.in_mom, s.in_energy).norm(dim=-1)
        edep = self.dp.denorm_edep_margin(s.edep.double(), e_in.double(), self.pdgids_of(s.in_pid))
        return self.dp.norm_edep_layer(edep).to(s.edep.dtype)

    def _g4_cond(self, n, pos, mom, energy, density, size, pdgids, Z, A):
        """Builds ``cond``, (B * n, n_cond + 7), from Geant4-shaped arguments of 1 or B rows, each row ``n`` times."""
        device = next(self.model.parameters()).device
        pos, mom, energy, density, size, pdgids, Z, A = (
            None if t is None else t.to(device) for t in (pos, mom, energy, density, size, pdgids, Z, A))
        scalars = {"Density": density, "Z": Z, "A": A, "Size": size}
        B = self._g4_batch_size(pos, mom, energy, pdgids, size, scalars)

        col = lambda x: x.reshape(-1, 1).expand(B, 1).repeat_interleave(n, dim=0)
        mom = F.normalize(mom, dim=-1).reshape(-1, 3).expand(B, 3) * energy.reshape(-1, 1).expand(B, 1)
        cc  = torch.cat((mom, pos.reshape(-1, 3).expand(B, 3)), dim=-1).repeat_interleave(n, dim=0)
        return torch.cat((torch.cat([col(scalars[k]) for k in self.cond_names], dim=-1), cc, col(pdgids).to(cc.dtype)),
                         dim=-1)

    def _g4_batch_size(self, pos, mom, energy, pdgids, size, scalars) -> int:
        """Returns the batch size B the Geant4-shaped arguments broadcast to.

        Raises:
            ValueError: a conditioning scalar is missing, an argument has neither 1 nor B rows, or ``size`` has several
                rows while ``Size`` is not a conditioning scalar.
        """
        missing = [k for k in self.cond_names if scalars.get(k) is None]
        if missing:
            raise ValueError(f"gen_model_w_g4_args missing conditioning arrays for {missing}")
        if "Size" not in self.cond_names and size.shape[0] != 1:
            raise ValueError("Multiple sizes not yet supported (size is not a conditioning variable).")
        shapes = {"pos": pos.reshape(-1, 3).shape[0], "mom": mom.reshape(-1, 3).shape[0], "energy": energy.shape[0],
                  "pdgids": pdgids.shape[0], **{k: scalars[k].flatten().shape[0] for k in self.cond_names}}
        B = max(shapes.values())
        wrong = {k: v for k, v in shapes.items() if v not in (1, B)}
        if wrong:
            raise ValueError(f"Each argument must have either size 1 or batch size {B}; got {wrong}")
        return B


class GenerateIn(GenerateOut):
    """The reverse direction: the incoming particle from the outgoing set (legacy, needs ``train_inverse``)."""
    __call__ = generate_in
