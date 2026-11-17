"""E_dep scales superseded by the log scale and the margin token of :mod:`legofmt.data.prep`, kept so older
checkpoints and datasets still run.

    linear             ``u = E / max_energy``: a model without ``model_conf.edep_log_min``
    hand-set margins   the margin token's cap ``E_in + margin(species)`` for a checkpoint from before
                       ``model_conf.edep_excess_mev`` (the training data's own largest excess)
"""

from torch import Tensor

# What Geant4 can deposit above E_in [MeV], per incoming species: e+ annihilation; pions: absorption releases the rest
# mass, up to +178 MeV in the stack data; neutrons: binding, 3e-5 of events by up to 4.6 MeV. A species not listed gets
# the widest margin.
HAND_SET_MARGIN_MEV = {11: 0.0, -11: 1.022, 22: 0.0, 2212: 0.0, 2112: 10.0, 211: 200.0, -211: 200.0}


def norm_edep_linear(edep: Tensor, max_energy: float) -> Tensor:
    """Converts E_dep from MeV to the linear scale ``E / max_energy``."""
    return edep / max_energy


def denorm_edep_linear(u: Tensor, max_energy: float) -> Tensor:
    """Inverts ``norm_edep_linear``; every non-positive input (the ``overflow_delta`` sentinel) decodes to zero."""
    return u.clamp_min(0.0) * max_energy
