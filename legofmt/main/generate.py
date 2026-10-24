"""``GenerateOut``: the multiplicity and flow checkpoints composed into a generator.

``gen_batch`` runs the multiplicity model to size the output slots and build the
padded ``(cond_fm, mask, attn_mask)`` tuple, then ``LEGOLtng.forward`` flows the
base distribution to particles. The conditioning momentum must be energy-scaled
rather than unit-norm, or the event conditions on a ~1 MeV particle. ``GenerateIn``
is the reverse direction and needs a ``train_inverse`` multiplicity checkpoint.
"""

import warnings
from dataclasses import replace

import torch
import torch.nn.functional as F

from legofmt.cfm.project_model import is_compiled
from legofmt.compat import dynamic_shape_attention
from legofmt.data.prep import DataPrep
from legofmt.data.struct import _F, DataStruct, set_layout
from legofmt.geometry.energy_proj import EnergyProjections
from legofmt.geometry.raytracing_proj import CubeTrace
from legofmt.main.modules import LEGOLtng
from legofmt.multiplicity.model import MultModel

_EDEP_EXCESS_MEV = {11: 0.0, -11: 1.022, 22: 0.0, 2212: 0.0, 2112: 10.0, 211: 150.0, -211: 150.0}


def _balance_energy(sols: torch.Tensor, e_in: torch.Tensor, pid_in: torch.Tensor, pen, dp) -> None:
    """Close the kinetic balance E_in = E_dep + sum E_out + 2 m_e (n_e+ out - n_e+ in) per event in place, keeping
    the generated E_dep. e+-/gamma in: every outgoing energy is scaled to that budget, and an absorbed one (nothing
    leaves) deposits the budget. Hadron in: only a survivor (the leading outgoing particle is the incoming species
    with > 90% of E_in) changes, to the budget minus the others; a nuclear interaction, whose Geant4 balance is
    open (fragments, binding), is left alone. Only valid where the data's own balance closes (the 0.1 MeV-cutoff
    plates, not a 10 MeV cutoff), hence opt-in."""
    s = _F(sols)
    e, pid = s.out_p[..., 0], s.out_p[..., -1]                        # (B, K) slots; per event (B, 1)
    e_in, pid_in = e_in.double()[:, None], pid_in[:, None]
    e_in_n = pen.to_scalar_e(e_in)
    live = (pid != 0) & e.isfinite()
    E = torch.where(live, pen.to_mev(e.double().nan_to_num(), e_in_n), 0.0)
    tot = E.sum(-1, keepdim=True)
    pair = 1.022 * ((live & (pid == -11)).sum(-1, keepdim=True) - (pid_in == -11).long())
    budget = (e_in - dp.denorm_edep_row(s.edep.double(), e_in[:, 0], pid_in[:, 0])[:, None] - pair).clamp_min(0.0)
    em = ((pid_in.abs() == 11) | (pid_in == 22)) & (e_in_n > 1e-6)  # at the cutoff the log ratio is 0/0
    lead_e, lead = E.max(-1, keepdim=True)
    surv = ~em & (e_in_n > 1e-6) & (lead_e > 0.9 * e_in) & (pid.gather(-1, lead) == pid_in)
    w = torch.where(em, E / tot, surv * F.one_hot(lead[:, 0], E.shape[-1]))  # who takes the residual budget - tot
    absorbed = (em & (tot == 0))[:, 0]                              # nothing leaves: the deposit is the budget
    s.edep[absorbed] = dp.norm_edep_row((e_in - pair)[:, 0], e_in[:, 0], pid_in[:, 0])[absorbed].to(s.edep.dtype)
    e.copy_(torch.where(w > 0, 1 - pen.to_scalar_e(E + w * (budget - tot)) / e_in_n, e.double()))


def _exit_outward(sols: torch.Tensor) -> None:
    """Mirror, in place, the momentum component of an outgoing particle that points into the face its exit position
    is on (the face is the dominant axis of the position direction, which is that of x / cuboid_dim). Position and
    direction are separate channels, so the flow can emit a particle leaving through +z while moving -z; Geant4 never
    does, and a chain hands such a particle to the next slab moving away from it."""
    out = _F(sols).out_p
    u, m = out[..., 4:7], out[..., 1:4]
    ax = u.abs().nan_to_num(-1.0).argmax(-1, keepdim=True)
    mn = m.gather(-1, ax)
    flip = (mn * u.gather(-1, ax).sign() < 0) & (out[..., -1:] != 0)
    m.scatter_(-1, ax, torch.where(flip, -mn, mn))


class GenerateOut(torch.nn.Module):

    flow_cls = LEGOLtng

    def __init__(
        self, flow_conf_path: str, mult_conf_path: str,
        device="cpu", couple_in_out_pdgids=False,
    ):
        super().__init__()
        flow_conf = torch.load(flow_conf_path, map_location=device, weights_only=False)
        self.model = self.flow_cls(flow_conf).to(device)
        self.model.rc = replace(self.model.rc, pdgid_is_idx=True)
        self.cond_names = tuple(self.model.rc.cond_scalars)
        self.n_prefix = self.model.rc.n_prefix
        self.n_cond = len(self.cond_names)  # = n_prefix - 1 (edep excluded)

        mult_conf = torch.load(mult_conf_path, map_location=device, weights_only=False)
        self.gen_mult = MultModel(mult_conf).to(device)

        if mult_conf["config"]["mm_conf"].get("fwd_compile", False) and not is_compiled(
            self.gen_mult.model
        ):
            # dynamic: the KV-cache decode grows the cache every step, so a static graph recompiled per
            # step until the recompile limit sent it to eager -- 115 vs 68 ms per 8192 events (plateSi pair)
            dynamic_shape_attention()
            self.gen_mult.model = torch.compile(self.gen_mult.model, dynamic=True)

        self.pdgid_in = mult_conf["config"]["mm_conf"]["ptypes_in"].to(device)
        self.ptypes = mult_conf["config"]["mm_conf"]["ptypes"].to(device)

        mm_cond = mult_conf["config"]["mm_conf"].get("cond_scalars")
        self.n_mult_cond = len(mm_cond) if mm_cond else self.gen_mult.proj_in_.in_features - 7
        if mm_cond and tuple(mm_cond) != self.cond_names[:self.n_mult_cond]:
            raise ValueError(
                f"mult cond_scalars {tuple(mm_cond)} is not a prefix of the flow's "
                f"{self.cond_names}; cond carries only the flow's scalars."
            )

        mm_cut = mult_conf["config"]["dl_conf"]["lds_args"].get("cutoff_mev")
        if mm_cut is not None and mm_cut != self.model.rc.cutoff_mev:
            warnings.warn(f"mult cutoff_mev {mm_cut} != flow {self.model.rc.cutoff_mev}: "
                          "the two energy scales disagree and mult_e_scale assumes they do not", stacklevel=2)
        mm_max = mult_conf["config"]["mm_conf"].get("max_energy")
        self.mult_e_scale = 1.0
        if mm_max and mm_max != self.model.rc.max_energy:
            cut = self.model.rc.cutoff_mev
            self.mult_e_scale = float(
                torch.tensor(self.model.rc.max_energy / cut).log()
                / torch.tensor(mm_max / cut).log()
            )

        set_layout(self.cond_names)

        if couple_in_out_pdgids:
            self.model.rc.odeint_conf["filter_pdgid"] = self.pdgid_in

        self.max_seq_l = flow_conf["config"]["model_conf"]["model_args"]["max_seq_l"]
        self.pdgids = flow_conf["config"]["model_conf"]["pdgids"].to(device)
        self.ptype_idx = torch.searchsorted(self.ptypes, self.pdgids).clamp(max=len(self.ptypes) - 1)
        self.ptype_in_mask = self.ptypes[self.ptype_idx] == self.pdgids
        self.proj_ray = CubeTrace(flow_conf["config"]["model_conf"].get("cuboid_dim"))
        self.cuboid_dim = self.proj_ray.cuboid_dim  # kept apart: no_raytrace models stub proj_ray
        self.pen = EnergyProjections(
            cutoff_mev=self.model.rc.cutoff_mev, max_energy=self.model.rc.max_energy,
        )
        self.dp = DataPrep(flow_conf)   # the E_dep row's normalisation (edep_log_min), for the E_in ceiling

    def __call__(self, cond: torch.Tensor, prepped: bool = False, passthrough: bool = True):
        """``cond`` is ``[B, n_cond + 7] = [*cond_scalars, mom * E_MeV (3), pos (3), pdgid]``;
        the momentum must be energy-scaled. ``prepped=True`` skips the ray-trace and
        the energy split. ``passthrough=False`` never fires the pass-through token: the
        interacting branch, for a caller that drew the pass-through itself (``passthrough_prob``).
        Returns ``(sols, mask, attn_mask)`` in the ``(B, L, 8)`` layout."""
        cond_model = self._model_cond(cond, prepped)
        cond_fm, mask, attn_mask = self.gen_batch(cond_model, passthrough=passthrough)
        nc = self.n_cond
        e_in = self.pen.from_scalar(cond_model[:, nc + 1:nc + 4], cond_model[:, nc:nc + 1]).norm(dim=-1)
        pid_in = cond_model[:, -1].long()
        pt_mask = mask.sum(-1) == 0
        ae = self.model.edep_ae
        if self.model.rc.odeint_conf.get("energy_balance", False) and ae is None:  # an edep_ae needs their latents
            # an e+-/gamma that leaves nothing deposits the balance budget, which overwrites the flow's E_dep
            em = (pid_in.abs() == 11) | (pid_in == 22)
            pt_mask = pt_mask | (em & (attn_mask[:, self.n_prefix + 1:].sum(-1) == 0)
                                 & (self.pen.to_scalar_e(e_in.double()) > 1e-6))
        if pt_mask.all():
            sols = cond_fm.clone()
        elif pt_mask.any():
            keep_events = ~pt_mask
            sub, _, _ = self.model(
                (cond_fm[keep_events], mask[keep_events], attn_mask[keep_events]))
            sols = cond_fm.clone()
            sols[keep_events] = sub.to(sols.dtype)
        else:
            sols, mask, attn_mask = self.model((cond_fm, mask, attn_mask))
        sols[..., -1] = torch.cat([sols.new_zeros(1), self.pdgids.to(sols.dtype)])[sols[..., -1].long()]
        edep = _F(sols).edep
        for pid, excess in (() if self.dp.edep_wall else _EDEP_EXCESS_MEV.items()):   # the wall row is capped already
            hit = pid_in == pid
            edep[..., hit] = torch.minimum(edep[..., hit], self.dp.norm_edep(e_in[hit] + excess).to(edep))
        if self.model.rc.odeint_conf.get("energy_balance", False):
            _balance_energy(sols, e_in, pid_in, self.pen, self.dp)
        if self.model.rc.odeint_conf.get("exit_consistency", False):
            _exit_outward(sols)
        # per-layer E_dep [MeV], decoded after the balance rewrote the E_dep row; pass-throughs deposit nothing
        self.edep_layers = None
        if ae is not None:
            self.edep_layers = sols.new_zeros(sols.shape[0], ae.n_layers)
            run = ~pt_mask
            if run.any():
                self.edep_layers[run] = ae.sample(self.model._edep_latents, self.dp.denorm_edep_row(
                    _F(sols).edep[run].float(), e_in[run].float(), pid_in[run]))
        return sols, mask, attn_mask

    def _model_cond(self, cond: torch.Tensor, prepped: bool) -> torch.Tensor:
        cond_model = cond.clone()
        if not prepped:
            nc = self.n_cond
            mom, pos = cond_model[:, nc:nc + 3], cond_model[:, nc + 3:nc + 6]
            pos = F.normalize(
                self.proj_ray(torch.cat((mom, pos), dim=-1))[..., 3:] / self.cuboid_dim.to(pos), dim=-1
            )
            dir_, e = self.pen.to_scalar(mom)
            cond_model = torch.cat(
                (cond_model[:, :nc], e, dir_, pos, cond_model[:, nc + 6:]), dim=-1
            )
        return cond_model

    def passthrough_prob(self, cond: torch.Tensor, prepped: bool = False) -> torch.Tensor:
        """P(the incoming particle passes through unchanged): the probability the pass-through
        token's draw uses, 0 for species the head does not cover. ``cond`` as for ``__call__``."""
        return self.gen_mult.passthrough_prob(*self._mult_in(self._model_cond(cond, prepped)))

    def gen_model_w_g4_args(self, n, pos, mom, energy, density, size, pdgids, Z=None, A=None, passthrough=True):
        """Geant4-shaped entry point: broadcast the per-gun arguments to ``B * n`` events
        and return ``{per_event, per_particle, per_voxel}`` in *model* normalisation
        (``per_event["E_dep"]`` is the normalised scalar, not MeV)."""
        sols, _, _ = self(self._g4_cond(n, pos, mom, energy, density, size, pdgids, Z, A), passthrough=passthrough)
        s = _F(sols)
        per_event = {"E_dep": s.edep}
        per_event.update({k: s.cond(k) for k in self.cond_names})
        out = {
            "per_event": per_event,
            "per_particle": {"Incoming": s.in_p, "Outgoing": s.out_p},
            "per_voxel": {"E_dep": sols.new_empty(sols.shape[0], 0, 4)},
        }
        if self.edep_layers is not None:  # geom/slab_calo's own group, in model normalisation like E_dep
            out["per_slab"] = {"E_dep": self.dp.norm_edep(self.edep_layers)}
        return out

    def passthrough_prob_w_g4_args(self, n, pos, mom, energy, density, size, pdgids, Z=None, A=None):
        """``passthrough_prob`` with the arguments of ``gen_model_w_g4_args``."""
        return self.passthrough_prob(self._g4_cond(n, pos, mom, energy, density, size, pdgids, Z, A))

    def _g4_cond(self, n, pos, mom, energy, density, size, pdgids, Z, A):
        device = next(self.model.parameters()).device
        pos, mom, energy, density, size, pdgids = (
            t.to(device) for t in (pos, mom, energy, density, size, pdgids)
        )
        Z = Z.to(device) if Z is not None else None
        A = A.to(device) if A is not None else None

        scalar_src = {"Density": density, "Z": Z, "A": A, "Size": size}
        missing = [k for k in self.cond_names if scalar_src.get(k) is None]
        if missing:
            raise ValueError(f"gen_model_w_g4_args missing conditioning arrays for {missing}")
        if "Size" not in self.cond_names and size.shape[0] != 1:
            raise ValueError("Multiple sizes not yet supported (size is not a conditioning variable).")

        shapes = {
            "pos": pos.view(-1, 3).shape[0],
            "mom": mom.view(-1, 3).shape[0],
            "energy": energy.shape[0],
            "pdgids": pdgids.shape[0],
            **{k: scalar_src[k].reshape(-1).shape[0] for k in self.cond_names},
        }
        B = max(shapes.values())
        err_size = {k: v for k, v in shapes.items() if v not in (1, B)}
        if err_size:
            raise ValueError(
                f"Each argument must have either size 1 or batch size {B}; got {err_size}"
            )

        mom = F.normalize(mom, dim=-1)

        def _scalar_col(x):
            return x.reshape(-1, 1).expand(B, 1).repeat_interleave(n, 0)

        e = energy.view(-1, 1).expand(B, 1)
        mom_b = mom.view(-1, 3).expand(B, 3)
        pos_b = pos.view(-1, 3).expand(B, 3)
        cc = torch.cat((mom_b * e, pos_b), dim=-1).repeat_interleave(n, dim=0)

        conds_b = torch.cat([_scalar_col(scalar_src[k]) for k in self.cond_names], dim=-1)
        pdgids_b = _scalar_col(pdgids).to(cc.dtype)
        return torch.cat((conds_b, cc, pdgids_b), dim=-1)

    def _mult_in(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pdgid_in = cond[:, -1].long()
        pdgid_in_idx = torch.searchsorted(self.pdgid_in, pdgid_in).clamp(max=len(self.pdgid_in) - 1)
        if (self.pdgid_in[pdgid_in_idx] != pdgid_in).any():
            raise ValueError(f"incoming pdgid(s) not in the mult model's ptypes_in={self.pdgid_in.tolist()}")
        mult_in = torch.cat(
            (cond[:, :self.n_mult_cond], cond[:, self.n_cond:self.n_cond + 7]), dim=-1
        )
        if self.mult_e_scale != 1.0:
            mult_in[:, self.n_mult_cond] = (
                mult_in[:, self.n_mult_cond] * self.mult_e_scale
            ).clamp(0.0, 1.0)
        return mult_in, pdgid_in_idx

    def gen_batch(self, cond: torch.Tensor, passthrough: bool = True):
        """Sample per-pdgid counts, truncate to ``max_seq_l``, and build the padded
        ``(cond_fm, mask, attn_mask)`` triple the flow consumes."""
        mult_in, pdgid_in_idx = self._mult_in(cond)
        pt_mask = (self.gen_mult.sample_passthrough(mult_in, pdgid_in_idx) if passthrough
                   else torch.zeros(len(cond), dtype=torch.bool, device=cond.device))
        mult = self.gen_mult((mult_in, None, pdgid_in_idx), pt_mask=pt_mask)
        mult = mult[:, self.ptype_idx] * self.ptype_in_mask

        max_particles = self.max_seq_l - (self.n_prefix + 1)
        total = mult.sum(-1, keepdim=True)
        scale = (max_particles / total).clamp(max=1.0)
        weights = mult.float() + (total == 0)  # the refill draws only species the model predicted
        mult = (mult * scale).long()
        scaled = total > max_particles
        remaining = (scaled * (max_particles - mult.sum(-1, keepdim=True))).clamp(min=0)
        dist = torch.multinomial(weights, max_particles, replacement=True)
        valid = (torch.arange(max_particles, device=mult.device) < remaining).long()
        mult.scatter_add_(-1, dist, valid)

        idx = torch.arange(max_particles, device=mult.device)
        occupied = idx < mult.sum(-1, keepdim=True)
        # species boundaries at the cumulative counts; the extra column absorbs
        # boundaries landing on max_particles so a saturated event's last slot is
        # not relabelled with a species the model predicted zero of
        bounds = mult.new_zeros(mult.shape[0], max_particles + 1)
        bounds.scatter_add_(-1, mult.cumsum(-1)[..., :-1], torch.ones_like(bounds))
        pdgid_pad = bounds.cumsum_(-1)[:, :max_particles]

        conds = cond[:, :self.n_cond]           # per-event conditioning scalars
        in_tok = cond[:, self.n_cond:].clone()  # [e, dir(3), pos(3), pdgid] (8 cols)
        in_tok[..., -1] = self.model.convert_pdgids(in_tok[..., -1])
        in_tok = in_tok[:, None, :]
        out_tok = in_tok.repeat(1, max_particles, 1)
        out_tok[..., -1] = occupied * (pdgid_pad + 1)
        cond_fm = torch.cat(
            (torch.zeros_like(in_tok).expand(-1, self.n_prefix, -1), in_tok, out_tok), dim=1
        )

        attn_mask = torch.cat(
            (torch.ones_like(occupied[:, :1]).repeat(1, self.n_prefix + 1), occupied), dim=1
        )
        mask = attn_mask.clone().long()
        cond_idx = [0, *range(2, self.n_prefix), self.n_prefix]  # cond scalars + incoming; edep stays 1
        mask[:, cond_idx] = 0
        cond_fm[:, 0, 0] = conds[:, 0]
        for j in range(1, self.n_cond):
            cond_fm[:, j + 1, 0] = conds[:, j]
        _F(cond_fm).non_p[..., 1:-1] = 1
        if pt_mask.any():
            edep = _F(cond_fm).edep
            edep[pt_mask] = 0.0
            out = _F(cond_fm).out_p
            out[pt_mask, :, 0] = 0.0
            mask = mask.clone()
            mask[pt_mask] = 0
        return cond_fm, mask, attn_mask


class GenerateIn(GenerateOut):

    @torch.no_grad()
    def __call__(self, batch):
        ds = batch if isinstance(batch, DataStruct) else DataStruct(*batch)
        f = ds.f.full.clone()
        m = torch.zeros_like(ds.m.full)
        m[:, self.n_prefix] = 1  # only the incoming slot is inferred
        ds = DataStruct(f, m, ds.am.full)
        conds = torch.cat(
            [ds.f.cond(n).view(-1, 1, 1) for n in self.cond_names], dim=-1
        ).expand(-1, ds.f.out_cc.shape[1], -1)
        out_tok = torch.cat((conds, ds.f.out_cc), dim=-1)
        out_pid = torch.searchsorted(
            self.ptypes, ds.f.out_p[..., -1].long()
        ).clamp(max=len(self.ptypes) - 1)
        pid_in_idx = self.gen_mult((out_tok, out_pid, ds.am.out_p.bool(), ds.f.edep))
        # The flow runs with pdgid_is_idx=True: swap the raw ids the mult
        # model consumed above for flow-vocabulary indices.
        f[..., -1] = self.model.convert_pdgids(f[..., -1]).to(f.dtype)
        f[:, self.n_prefix, -1] = self.model.convert_pdgids(self.pdgid_in[pid_in_idx].to(f.dtype))
        sols, _, _ = self.model(ds)
        sols[..., -1] = torch.cat(
            [sols.new_zeros(1), self.pdgids.to(sols.dtype)]
        )[sols[..., -1].long()]
        return _F(sols).in_p

