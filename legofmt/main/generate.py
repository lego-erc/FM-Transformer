"""``GenerateOut``: the multiplicity and the flow checkpoint put together into a generator.

One call, in the order it runs:

```
cond      [*cond_scalars, mom * E_MeV (3), pos (3), pdgid] per incoming particle, as given;
          [*cond_scalars, e, dir (3), pos (3), pdgid] on the model's scales: the position ray-traced onto the cube,
          the energy split off the momentum (_model_cond); after the scalars it is the incoming token
counts    the multiplicity model samples the pass-through flag and the per-species counts (sample_counts)
slots     n_s = floor(c_s min(1, max_particles / C)),  C = sum_s c_s;  the slots the rounding frees take species
          drawn ~ c_s (fit_counts); the slots fill species by species (slot_species)
tokens    [*cond_scalars, E_dep, incoming, slot_1 .. slot_max_particles]: each slot a copy of the incoming token with
          its species index, 0 when empty (build_batch)
flow      LEGOLtng.forward flows the base to the outgoing particles and E_dep; pass-through events skip it
output    species indices back to pdgids; the E_dep token capped at norm(E_in + excess(species)) (_clamp_edep)
```

The momentum in ``cond`` must be energy-scaled, not unit-norm, or the event conditions on a ~1 MeV particle.
``gen_model_w_g4_args`` takes Geant4-shaped arguments instead. The energy balance and exit consistency
(``odeint_conf``) and ``GenerateIn``, the reverse direction, are :mod:`legofmt.legacy.generate`.
"""

from dataclasses import replace

import torch
import torch.nn.functional as F

from legofmt.cfm.project_model import is_compiled
from legofmt.compat import dynamic_shape_attention
from legofmt.data.write.norm import DataPrep
from legofmt.data.struct import _F, cond_slot, set_layout
from legofmt.geometry.energy_proj import EnergyProjections
from legofmt.geometry.raytracing_proj import CubeTrace
from legofmt.legacy.generate import balance_passthrough, close_balance, generate_in
from legofmt.main.modules import LEGOLtng
from legofmt.mod_comps.config import mult_energy_scale
from legofmt.multiplicity.model import MultModel
from legofmt.rng import REFILL, EventRNG

# the E_dep cap's margin for checkpoints from before model_conf.edep_excess_mev [MeV]
_EDEP_EXCESS_MEV = {11: 0.0, -11: 1.022, 22: 0.0, 2212: 0.0, 2112: 10.0, 211: 150.0, -211: 150.0}


def fit_counts(counts: torch.Tensor, max_particles: int, refill) -> torch.Tensor:
    """Fits each event's per-species counts into ``max_particles`` slots (module docstring, ``slots``).

    Args:
        counts:        (B, S), the sampled counts.
        max_particles: the slots per event.
        refill:        draws ``(B, max_particles)`` species from (B, S) weights; any species when an event has none.

    Returns:
        (B, S), the fitted counts.
    """
    total     = counts.sum(-1, keepdim=True)
    weights   = counts.float() + (total == 0)
    fitted    = (counts * (max_particles / total).clamp(max=1.0)).long()
    remaining = ((total > max_particles) * (max_particles - fitted.sum(-1, keepdim=True))).clamp(min=0)
    slots     = torch.arange(max_particles, device=counts.device)
    return fitted.scatter_add_(-1, refill(weights), (slots < remaining).long())


def slot_species(counts: torch.Tensor, max_particles: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns which slots hold a particle and each slot's species: the slots fill species by species.

    Args:
        counts:        (B, S), the fitted counts.
        max_particles: the slots per event.

    Returns:
        ``(occupied, species)``, both (B, max_particles); slot j's species is the number of cumulative counts <= j.
    """
    slots    = torch.arange(max_particles, device=counts.device)
    occupied = slots < counts.sum(-1, keepdim=True)
    bounds   = counts.cumsum(-1)[..., :-1].contiguous()
    return occupied, torch.searchsorted(bounds, slots.expand(len(counts), -1).contiguous(), right=True)


def _rng_kw(rng, rows=None) -> dict:
    """Returns ``rng=`` for a call only when keyed, so stubs and subclasses without the argument keep working."""
    return {} if rng is None else {"rng": rng if rows is None else rng.sub(rows)}


class GenerateOut(torch.nn.Module):
    """The generator: incoming particles in, outgoing particles and E_dep out.

    Args:
        flow_conf_path:       the flow checkpoint.
        mult_conf_path:       the multiplicity checkpoint.
        device:               where both models run.
        couple_in_out_pdgids: let the flow generate only the incoming species (``odeint_conf.filter_pdgid``).
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

        # the multiplicity model; its conditioning is a prefix of the flow's scalars, its energy scalar rescaled
        mult_conf     = torch.load(mult_conf_path, map_location=device, weights_only=False)
        self.gen_mult = MultModel(mult_conf).to(device)
        mm = mult_conf["config"]["mm_conf"]
        if mm.get("fwd_compile", False) and not is_compiled(self.gen_mult.model):
            # dynamic: the KV-cache decode grows the cache every step, so a static graph recompiled per
            # step until the recompile limit sent it to eager -- 115 vs 68 ms per 8192 events (plateSi pair)
            dynamic_shape_attention()
            self.gen_mult.model = torch.compile(self.gen_mult.model, dynamic=True)
        self.pdgid_in      = mm["ptypes_in"].to(device)
        self.ptypes        = mm["ptypes"].to(device)
        self.ptype_idx     = torch.searchsorted(self.ptypes, self.pdgids).clamp(max=len(self.ptypes) - 1)
        self.ptype_in_mask = self.ptypes[self.ptype_idx] == self.pdgids   # the flow's species the mult model counts
        self.n_mult_cond   = (len(mm["cond_scalars"]) if mm.get("cond_scalars")
                              else self.gen_mult.proj_in_.in_features - 7)
        self.mult_e_scale  = mult_energy_scale(mult_conf, rc)

        set_layout(self.cond_names)
        if couple_in_out_pdgids:
            rc.odeint_conf["filter_pdgid"] = self.pdgid_in

        # the scales: the incoming position's ray trace, the energies, the E_dep token's (for the E_in cap)
        self.proj_ray   = CubeTrace(mc.get("cuboid_dim"))
        self.cuboid_dim = self.proj_ray.cuboid_dim   # kept apart: no_raytrace models stub proj_ray
        self.pen        = EnergyProjections(cutoff_mev=rc.cutoff_mev, max_energy=rc.max_energy)
        self.dp         = DataPrep(flow_conf)

    def __call__(self, cond: torch.Tensor, prepped: bool = False, passthrough: bool = True, rng=None):
        """Generates the outgoing particles and E_dep of a batch of incoming particles.

        Args:
            cond:        ``[B, n_cond + 7] = [*cond_scalars, mom * E_MeV (3), pos (3), pdgid]``; the momentum must be
                energy-scaled.
            prepped:     skip the ray-trace and the energy split.
            passthrough: False never fires the pass-through token: the interacting branch, for a caller that drew the
                pass-through itself (``passthrough_prob``).
            rng:         makes every random draw a function of the event's key: independent of the batch, and shared by
                two models run on the same keys (``legofmt.rng.EventRNG``, one unique key per event).

        Returns:
            ``(sols, mask, attn_mask)`` in the ``(B, L, 8)`` layout.

        Raises:
            ValueError: ``rng`` does not hold one key per event.
        """
        if rng is not None and len(rng.keys) != len(cond):
            raise ValueError(f"rng needs one key per event: {len(rng.keys)} keys for {len(cond)} events")
        if rng is not None:  # keys on the events' device: rows are subset by device masks, draws meet device params
            rng = EventRNG(rng.keys.to(cond.device), rng.seed)

        cond = self._model_cond(cond, prepped)
        cond_fm, mask, attn_mask = self.gen_batch(cond, passthrough=passthrough, **_rng_kw(rng))
        incoming = cond[:, self.n_cond:]
        e_in     = self.pen.from_scalar(incoming[:, 1:4], incoming[:, 0:1]).norm(dim=-1)
        pid_in   = incoming[:, -1].long()

        pt_mask = mask.sum(-1) == 0
        pt_mask = balance_passthrough(self, pt_mask, attn_mask, e_in, pid_in)   # legacy: energy_balance
        sols, mask, attn_mask = self._flow(cond_fm, mask, attn_mask, pt_mask, rng)

        _F(sols).pdgids.copy_(F.pad(self.pdgids.to(sols.dtype), (1, 0))[_F(sols).pdgids.long()])
        _F(sols).edep.copy_(self._clamp_edep(_F(sols).edep, e_in, pid_in))
        close_balance(self, sols, e_in, pid_in)   # legacy: energy_balance, exit_consistency
        self.edep_layers = self._edep_layers(sols, pt_mask, e_in, pid_in, rng)
        return sols, mask, attn_mask

    def _flow(self, cond_fm, mask, attn_mask, pt_mask, rng):
        """Runs the flow on the events that interact; a pass-through event keeps its tokens as built.

        Returns:
            ``(sols, mask, attn_mask)``.
        """
        if pt_mask.all():
            return cond_fm.clone(), mask, attn_mask
        if not pt_mask.any():
            return self.model((cond_fm, mask, attn_mask), **_rng_kw(rng))
        keep = ~pt_mask
        sub, _, _ = self.model((cond_fm[keep], mask[keep], attn_mask[keep]), **_rng_kw(rng, keep))
        sols = cond_fm.clone()
        sols[keep] = sub.to(sols.dtype)
        return sols, mask, attn_mask

    def _clamp_edep(self, edep: torch.Tensor, e_in: torch.Tensor, pid_in: torch.Tensor) -> torch.Tensor:
        """Caps the E_dep token at ``norm_edep_layer(E_in + excess(species))``, out of place.

        Only the layer-scale token is capped (a margin token is capped by construction). The excess is the training
        data's largest (``model_conf.edep_excess_mev``), or for checkpoints from before it the hand-set
        ``_EDEP_EXCESS_MEV``; a species without one stays uncapped. Out of place, so the density design can
        differentiate through it.
        """
        margins = self.dp.edep_excess_mev or _EDEP_EXCESS_MEV
        own     = pid_in[..., None] == pid_in.new_tensor(list(margins))   # (..., n_species), one-hot or all false
        excess  = (own * e_in.new_tensor(list(margins.values()))).sum(-1)
        capped  = own.any(-1) & ~self.dp.edep_margin_mask(pid_in)
        return torch.where(capped, torch.minimum(edep, self.dp.norm_edep_layer(e_in + excess).to(edep)), edep)

    def _edep_layers(self, sols, pt_mask, e_in, pid_in, rng):
        """Returns the per-layer E_dep [MeV] the edep_ae decodes from the final E_dep token (None without an AE).

        Pass-through events deposit nothing.
        """
        ae = self.model.edep_ae
        if ae is None:
            return None
        layers, run = sols.new_zeros(sols.shape[0], ae.n_layers), ~pt_mask
        if run.any():
            edep = self.dp.denorm_edep_margin(_F(sols).edep[run].float(), e_in[run].float(), pid_in[run])
            layers[run] = ae.sample(self.model._edep_latents, edep, **_rng_kw(rng, run))
        return layers

    def _model_cond(self, cond: torch.Tensor, prepped: bool) -> torch.Tensor:
        """Returns ``cond`` on the model's scales (module docstring); ``prepped`` input already is."""
        if prepped:
            return cond.clone()
        scalars, mom, pos, pid = cond.split((self.n_cond, 3, 3, 1), dim=-1)
        pos = F.normalize(self.proj_ray(torch.cat((mom, pos), dim=-1))[..., 3:] / self.cuboid_dim.to(pos), dim=-1)
        dir_, e = self.pen.to_scalar(mom)
        return torch.cat((scalars, e, dir_, pos, pid), dim=-1)

    # ---- the multiplicity model -------------------------------------------------------------------------------------

    def gen_batch(self, cond: torch.Tensor, passthrough: bool = True, rng=None):
        """Samples the counts and builds the flow's padded ``(cond_fm, mask, attn_mask)`` from them."""
        return self.build_batch(cond, *self.sample_counts(cond, passthrough, **_rng_kw(rng)), **_rng_kw(rng))

    def sample_counts(self, cond: torch.Tensor, passthrough: bool = True,
                      rng=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples the multiplicity model for ``cond`` (model layout).

        Returns:
            ``(pt_mask, counts)``: the pass-through flags, and the raw per-slot counts as ``MultModel.forward``
            returns them (``MultModel.log_prob`` scores them).
        """
        mult_in, pdgid_in_idx = self._mult_in(cond)
        pt_mask = (self.gen_mult.sample_passthrough(mult_in, pdgid_in_idx, **_rng_kw(rng)) if passthrough
                   else torch.zeros(len(cond), dtype=torch.bool, device=cond.device))
        return pt_mask, self.gen_mult((mult_in, None, pdgid_in_idx), pt_mask=pt_mask, **_rng_kw(rng))

    def passthrough_prob(self, cond: torch.Tensor, prepped: bool = False) -> torch.Tensor:
        """Returns P(the incoming particle passes through unchanged).

        The probability the pass-through token's draw uses, 0 for species the head does not cover. ``cond`` as for
        ``__call__``.
        """
        return self.gen_mult.passthrough_prob(*self._mult_in(self._model_cond(cond, prepped)))

    def _mult_in(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the multiplicity model's input for ``cond`` and the incoming species' index into ``ptypes_in``.

        The input is the multiplicity model's scalars and the incoming particle without its pdgid; its energy scalar
        is rescaled to the multiplicity model's ``max_energy`` (``mult_e_scale``).

        Raises:
            ValueError: an incoming species the multiplicity model was not trained on.
        """
        pdgid_in = cond[:, -1].long()
        pdgid_in_idx = torch.searchsorted(self.pdgid_in, pdgid_in).clamp(max=len(self.pdgid_in) - 1)
        if (self.pdgid_in[pdgid_in_idx] != pdgid_in).any():
            raise ValueError(f"incoming pdgid(s) not in the mult model's ptypes_in={self.pdgid_in.tolist()}")
        mult_in = torch.cat((cond[:, :self.n_mult_cond], cond[:, self.n_cond:-1]), dim=-1)
        if self.mult_e_scale != 1.0:
            mult_in[:, self.n_mult_cond] = (mult_in[:, self.n_mult_cond] * self.mult_e_scale).clamp(0.0, 1.0)
        return mult_in, pdgid_in_idx

    # ---- the flow's tokens ------------------------------------------------------------------------------------------

    def build_batch(self, cond: torch.Tensor, pt_mask: torch.Tensor, counts: torch.Tensor, rng=None):
        """Builds the flow's padded tokens for the sampled counts (module docstring: slots, tokens).

        ``mask`` is 1 on the generated tokens (E_dep and the occupied slots), ``attn_mask`` True on the prefix, the
        incoming token and the occupied slots. A pass-through event generates nothing, deposits nothing, and every slot
        holds the full-energy atom.

        Args:
            cond:    (B, n_cond + 8), the model-layout conditioning.
            pt_mask: (B,), the pass-through events.
            counts:  the raw per-slot counts (``sample_counts``).
            rng:     keyed draws (``legofmt.rng.EventRNG``), or None.

        Returns:
            ``(cond_fm, mask, attn_mask)``.
        """
        max_particles = self.max_seq_l - (self.n_prefix + 1)
        slots  = torch.arange(max_particles, device=counts.device)
        refill = ((lambda w: torch.multinomial(w, max_particles, replacement=True)) if rng is None
                  else (lambda w: rng.categorical(w, REFILL, slots)))
        counts = fit_counts(counts[:, self.ptype_idx] * self.ptype_in_mask, max_particles, refill)
        occupied, species = slot_species(counts, max_particles)

        scalars, incoming = cond[:, :self.n_cond], cond[:, self.n_cond:].clone()
        incoming[:, -1] = self.model.convert_pdgids(incoming[:, -1])   # its pdgid as a vocabulary index
        cond_fm  = F.pad(incoming[:, None, :].expand(-1, 1 + max_particles, -1), (0, 0, self.n_prefix, 0))
        given    = [cond_slot(n) for n in self.cond_names]   # the cond scalars' prefix tokens
        cond_fm[:, given, 0] = scalars
        _F(cond_fm).non_p[..., 1:-1] = 1
        _F(cond_fm).out_pid.copy_(occupied * (species + 1))

        attn_mask = F.pad(occupied, (self.n_prefix + 1, 0), value=True)
        mask      = attn_mask.long()
        mask[:, given + [self.n_prefix]] = 0   # the cond scalars and the incoming particle are given

        _F(cond_fm).edep.masked_fill_(pt_mask, 0.0)
        _F(cond_fm).out_energy.masked_fill_(pt_mask[:, None], 0.0)
        mask.masked_fill_(pt_mask[:, None], 0)
        return cond_fm, mask, attn_mask

    # ---- Geant4-shaped arguments ------------------------------------------------------------------------------------

    def gen_model_w_g4_args(self, n, pos, mom, energy, density, size, pdgids, Z=None, A=None, passthrough=True,
                            rng=None):
        """Generates events from Geant4-shaped arguments.

        Broadcasts the per-gun arguments to ``B * n`` events. ``rng`` as in ``__call__``, with ``B * n`` keys
        (gun-major, as ``repeat_interleave`` orders the events).

        Returns:
            ``{per_event, per_particle, per_voxel}`` in *model* normalisation: ``per_event["E_dep"]`` is the normalised
                scalar, not MeV, always on the ``norm_edep_layer`` scale that callers decode with ``edep_log_min``
                (a ``model_conf.edep_margin`` token is re-encoded on it here).
        """
        sols, _, _ = self(self._g4_cond(n, pos, mom, energy, density, size, pdgids, Z, A), passthrough=passthrough,
                          **_rng_kw(rng))
        s = _F(sols)
        edep = s.edep
        if self.dp.edep_margin:
            e_in = self.dp.pen.from_scalar(s.in_mom, s.in_energy).norm(dim=-1)
            edep = self.dp.norm_edep_layer(self.dp.denorm_edep_margin(edep.double(), e_in.double(),
                                                                      s.in_pid)).to(edep.dtype)
        out = {
            "per_event": {"E_dep": edep, **{k: s.cond(k) for k in self.cond_names}},
            "per_particle": {"Incoming": s.in_p, "Outgoing": s.out_p},
            "per_voxel": {"E_dep": sols.new_empty(sols.shape[0], 0, 4)},
        }
        if self.edep_layers is not None:  # geom/slab_calo's own group, in model normalisation like E_dep
            out["per_slab"] = {"E_dep": self.dp.norm_edep_layer(self.edep_layers)}
        return out

    def passthrough_prob_w_g4_args(self, n, pos, mom, energy, density, size, pdgids, Z=None, A=None):
        """Returns ``passthrough_prob`` for the arguments of ``gen_model_w_g4_args``."""
        return self.passthrough_prob(self._g4_cond(n, pos, mom, energy, density, size, pdgids, Z, A))

    def _g4_cond(self, n, pos, mom, energy, density, size, pdgids, Z, A):
        """Builds ``cond`` from Geant4-shaped arguments, each of size 1 or B, every event repeated ``n`` times."""
        device = next(self.model.parameters()).device
        pos, mom, energy, density, size, pdgids, Z, A = (
            None if t is None else t.to(device) for t in (pos, mom, energy, density, size, pdgids, Z, A))
        scalars = {"Density": density, "Z": Z, "A": A, "Size": size}
        B = self._g4_batch_size(pos, mom, energy, pdgids, size, scalars)

        col = lambda x: x.reshape(-1, 1).expand(B, 1).repeat_interleave(n, 0)
        mom = F.normalize(mom, dim=-1).view(-1, 3).expand(B, 3) * energy.view(-1, 1).expand(B, 1)
        cc  = torch.cat((mom, pos.view(-1, 3).expand(B, 3)), dim=-1).repeat_interleave(n, dim=0)
        return torch.cat((torch.cat([col(scalars[k]) for k in self.cond_names], dim=-1), cc, col(pdgids).to(cc.dtype)),
                         dim=-1)

    def _g4_batch_size(self, pos, mom, energy, pdgids, size, scalars) -> int:
        """Returns the batch size B the Geant4-shaped arguments broadcast to.

        Raises:
            ValueError: a conditioning scalar is missing, an argument has neither size 1 nor B, or several sizes are
                given while Size is not a conditioning scalar.
        """
        missing = [k for k in self.cond_names if scalars.get(k) is None]
        if missing:
            raise ValueError(f"gen_model_w_g4_args missing conditioning arrays for {missing}")
        if "Size" not in self.cond_names and size.shape[0] != 1:
            raise ValueError("Multiple sizes not yet supported (size is not a conditioning variable).")
        shapes = {"pos": pos.view(-1, 3).shape[0], "mom": mom.view(-1, 3).shape[0], "energy": energy.shape[0],
                  "pdgids": pdgids.shape[0], **{k: scalars[k].reshape(-1).shape[0] for k in self.cond_names}}
        B = max(shapes.values())
        wrong = {k: v for k, v in shapes.items() if v not in (1, B)}
        if wrong:
            raise ValueError(f"Each argument must have either size 1 or batch size {B}; got {wrong}")
        return B


class GenerateIn(GenerateOut):
    """The reverse direction: the incoming particle from the outgoing set (legacy, needs ``train_inverse``)."""
    __call__ = generate_in
