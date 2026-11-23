"""Energy scales superseded by those of :mod:`legofmt.data.write.norm`, kept so that older checkpoints and datasets
still run.

    linear             ``u = E / max_energy``: a model without ``model_conf.edep_log_min``
    hand-set margins   the margin token's cap ``E_in + margin(species)`` for a checkpoint without
                       ``model_conf.edep_excess_mev`` (the training data's own largest excess)
    hand-set excess    the layer-scale token's cap at ``E_in + excess(species)`` (``GenerateOut._clamp_edep``) for
                       such a checkpoint
    log outgoing       ``e = 1 - log(e_out / cutoff) / log(e_in / cutoff)``: the outgoing energy of a model without
                       ``model_conf.out_margin``, and what every dataset from before stores
"""

import torch
from torch import Tensor

# What Geant4 can deposit above E_in [MeV], per incoming species: e+ annihilation; pions: absorption releases the rest
# mass, up to +178 MeV in the stack data; neutrons: binding, 3e-5 of events by up to 4.6 MeV. A species not listed gets
# the widest margin.
HAND_SET_MARGIN_MEV = {11: 0.0, -11: 1.022, 22: 0.0, 2212: 0.0, 2112: 10.0, 211: 200.0, -211: 200.0}

# The same for the layer-scale token's cap. Its pions are set from the plates' largest excess (pi- 142.5 MeV), the
# margins above from the stack's (+178 MeV). A species not listed stays uncapped.
HAND_SET_EXCESS_MEV = {11: 0.0, -11: 1.022, 22: 0.0, 2212: 0.0, 2112: 10.0, 211: 150.0, -211: 150.0}


def norm_edep_linear(edep: Tensor, max_energy: float) -> Tensor:
    """Converts E_dep from MeV to the linear scale ``E / max_energy``."""
    return edep / max_energy


def denorm_edep_linear(u: Tensor, max_energy: float) -> Tensor:
    """Inverts ``norm_edep_linear``; every non-positive input (the ``overflow_delta`` sentinel) decodes to zero."""
    return u.clamp_min(0.0) * max_energy


def norm_out_energy_log(e_out: Tensor, e_in: Tensor, cutoff: float) -> Tensor:
    """Converts the outgoing energy from MeV to the log scale, held in [0, 1]: 0 from ``e_in`` on, 1 at the cutoff.

    ```
    e = 1 - log(e_out / cutoff) / log(e_in / cutoff)
    ```

    ``e_in`` [MeV] broadcasts against ``e_out``.
    """
    ratio = torch.log(e_out.clamp(min=1e-8) / cutoff) / torch.log(e_in.clamp(min=1e-8) / cutoff).clamp(min=1e-6)
    return 1 - ratio.clamp(0, 1)


def denorm_out_energy_log(e: Tensor, e_in: Tensor, cutoff: float) -> Tensor:
    """Inverts ``norm_out_energy_log``: ``e_out = cutoff (e_in / cutoff)**(1 - e)`` [MeV], ``e`` held in [0, 1]."""
    return cutoff * (e_in / cutoff) ** (1 - e.clamp(0.0, 1.0))
