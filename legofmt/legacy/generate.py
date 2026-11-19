"""Generation paths the recipe no longer takes, kept so older studies still run unchanged:

    energy_balance     ``odeint_conf.energy_balance``: each event's kinetic balance closed after the flow
    exit_consistency   ``odeint_conf.exit_consistency``: an outgoing momentum that points into its exit face mirrored
    GenerateIn         the reverse direction: the incoming particle generated from the outgoing set (needs a
                       ``train_inverse`` multiplicity checkpoint)

``GenerateOut.__call__`` reaches the first two in one line each (``balance_passthrough``, ``close_balance``);
``GenerateIn`` is a two-line class in ``main/generate.py`` around ``generate_in``. Every function here takes the
generator as an argument, so this module does not import the production one.
"""

import torch
import torch.nn.functional as F

from legofmt.data.struct import _F, DataStruct


def balance_passthrough(gen, pt_mask: torch.Tensor, attn_mask: torch.Tensor, e_in: torch.Tensor,
                        pid_in: torch.Tensor) -> torch.Tensor:
    """Adds, with ``energy_balance``, the e+-/gamma events that leave nothing to the pass-through events.

    Their deposit is the balance budget, which overwrites the flow's E_dep; with an ``edep_ae`` nothing is added (the
    latents need the flow).
    """
    if not gen.model.rc.odeint_conf.get("energy_balance", False) or gen.model.edep_ae is not None:
        return pt_mask
    em = (pid_in.abs() == 11) | (pid_in == 22)
    leaves_nothing = attn_mask[:, gen.n_prefix + 1:].sum(-1) == 0
    return pt_mask | (em & leaves_nothing & (gen.pen.to_scalar_e(e_in.double()) > 1e-6))


def close_balance(gen, sols: torch.Tensor, e_in: torch.Tensor, pid_in: torch.Tensor) -> None:
    """Applies the ``odeint_conf`` opt-ins in place: ``energy_balance``, then ``exit_consistency``."""
    if gen.model.rc.odeint_conf.get("energy_balance", False):
        _balance_energy(sols, e_in, pid_in, gen.pen, gen.dp)
    if gen.model.rc.odeint_conf.get("exit_consistency", False):
        _exit_outward(sols)


def _balance_energy(sols: torch.Tensor, e_in: torch.Tensor, pid_in: torch.Tensor, pen, dp) -> None:
    """Closes each event's kinetic balance in place, keeping the generated E_dep.

    The balance is E_in = E_dep + sum E_out + 2 m_e (n_e+ out - n_e+ in). e+-/gamma in: every outgoing energy is scaled
    to that budget, and an absorbed one (nothing leaves) deposits the budget. Hadron in: only a survivor (the leading
    outgoing particle is the incoming species, with > 90% of E_in or with nothing but gammas / e+- beside it) changes,
    to the budget minus the others; a nuclear interaction, whose Geant4 balance is open (fragments, binding), is left
    alone. The second rule is what a thick block needs: a 1 GeV p / pi+ crossing the 108.8 mm W/Si stack keeps 72-81%,
    and Geant4 closes those events exactly (residual 0.00 MeV to q90). Only valid where the data's own balance closes
    (the 0.1 MeV-cutoff plates, not a 10 MeV cutoff), hence opt-in.
    """
    s = _F(sols)
    e, pid = s.out_p[..., 0], s.out_p[..., -1]                        # (B, K) slots; per event (B, 1)
    e_in, pid_in = e_in.double()[:, None], pid_in[:, None]
    e_in_n = pen.to_scalar_e(e_in)
    live = (pid != 0) & e.isfinite()
    E = torch.where(live, pen.to_mev(e.double().nan_to_num(), e_in_n), 0.0)
    tot = E.sum(-1, keepdim=True)
    pair = 1.022 * ((live & (pid == -11)).sum(-1, keepdim=True) - (pid_in == -11).long())
    budget = (e_in - dp.denorm_edep_margin(s.edep.double(), e_in[:, 0], pid_in[:, 0])[:, None] - pair).clamp_min(0.0)
    em = ((pid_in.abs() == 11) | (pid_in == 22)) & (e_in_n > 1e-6)  # at the cutoff the log ratio is 0/0
    lead_e, lead = E.max(-1, keepdim=True)
    is_lead = F.one_hot(lead[:, 0], E.shape[-1]).bool()
    em_rest = ~(live & ~is_lead & (pid.abs() != 11) & (pid != 22)).any(-1, keepdim=True)  # no other hadron / ion out
    surv = ~em & (e_in_n > 1e-6) & (pid.gather(-1, lead) == pid_in) & ((lead_e > 0.9 * e_in) | em_rest)
    w = torch.where(em, E / tot, surv * is_lead)                    # who takes the residual budget - tot
    absorbed = (em & (tot == 0))[:, 0]                              # nothing leaves: the deposit is the budget
    budget_edep = dp.norm_edep_margin((e_in - pair)[:, 0], e_in[:, 0], pid_in[:, 0]).to(s.edep)
    s.edep.copy_(torch.where(absorbed, budget_edep, s.edep))
    e.copy_(torch.where(w > 0, 1 - pen.to_scalar_e(E + w * (budget - tot)) / e_in_n, e.double()))


def _exit_outward(sols: torch.Tensor) -> None:
    """Mirrors, in place, an outgoing momentum component that points into the face the particle exits through.

    The face is the one its exit position is on: the dominant axis of the position direction, which is that of x /
    cuboid_dim. Position and direction are separate channels, so the flow can emit a particle leaving through +z while
    moving -z; Geant4 never does, and a chain hands such a particle to the next slab moving away from it.
    """
    out = _F(sols).out_p
    u, m = out[..., 4:7], out[..., 1:4]
    ax = u.abs().nan_to_num(-1.0).argmax(-1, keepdim=True)
    mn = m.gather(-1, ax)
    flip = (mn * u.gather(-1, ax).sign() < 0) & (out[..., -1:] != 0)
    m.scatter_(-1, ax, torch.where(flip, -mn, mn))


@torch.no_grad()
def generate_in(gen, batch):
    """Generates the incoming particle of each event from its outgoing set (``GenerateIn.__call__``).

    Args:
        gen:   the ``GenerateIn``.
        batch: a ``DataStruct`` or ``(f, m, am)``.

    Returns:
        The generated incoming tokens.
    """
    ds = batch if isinstance(batch, DataStruct) else DataStruct(*batch)
    f = ds.f.full.clone()
    m = torch.zeros_like(ds.m.full)
    m[:, gen.n_prefix] = 1  # only the incoming slot is inferred
    ds = DataStruct(f, m, ds.am.full)
    conds = torch.cat([ds.f.cond(n).view(-1, 1, 1) for n in gen.cond_names], dim=-1)
    out_tok = torch.cat((conds.expand(-1, ds.f.out_cc.shape[1], -1), ds.f.out_cc), dim=-1)
    out_pid = torch.searchsorted(gen.ptypes, ds.f.out_p[..., -1].long()).clamp(max=len(gen.ptypes) - 1)
    pid_in_idx = gen.gen_mult((out_tok, out_pid, ds.am.out_p.bool(), ds.f.edep))
    # The flow runs with pdgid_is_idx=True: swap the raw ids the mult
    # model consumed above for flow-vocabulary indices.
    f[..., -1] = gen.model.convert_pdgids(f[..., -1]).to(f.dtype)
    f[:, gen.n_prefix, -1] = gen.model.convert_pdgids(gen.pdgid_in[pid_in_idx].to(f.dtype))
    sols, _, _ = gen.model(ds)
    sols[..., -1] = F.pad(gen.pdgids.to(sols.dtype), (1, 0))[sols[..., -1].long()]
    return _F(sols).in_p
