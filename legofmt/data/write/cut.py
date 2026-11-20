"""The cut on raw particles: the step before ``DataPrep.prep``.

    {per_particle: {Incoming, Outgoing}, per_event}  ->  (particles, m, am, per_event)

A raw particle is ``[E_kin, p (3), x (3), pdgid]`` in MeV; the particles above ``cutoff_mev`` move to the front of
their event and NaN pads the rest.
"""

import torch
from torch import Tensor


class GetLEGOData:
    """Cuts a raw dataset at ``cutoff_mev`` and pads it.

    Args:
        cutoff_mev:    a particle below it is dropped.
        min_particles: an event left with fewer outgoing particles is dropped.
        device:        where the result goes.
        energy_kin:    cut on the kinetic energy; false (legacy): on |p|.
        dtype:         of the particles.
    """

    def __init__(self, cutoff_mev=10.0, min_particles=0, device="cpu", energy_kin=True, dtype=torch.float32):
        self.cutoff_mev = cutoff_mev
        self.min_particles = min_particles
        self.dev = device
        self.energy_kin = energy_kin
        self.dtype = dtype

    def __call__(self, data: dict, n_events: int | None = None) -> tuple[Tensor, Tensor, Tensor, dict]:
        """Drops the particles below ``cutoff_mev`` and the events left with fewer than ``min_particles`` outgoing.

        The cut applies to the incoming particle too.

        Args:
            data:     ``{per_particle: {Incoming, Outgoing}, per_event}``.
            n_events: keep a fixed random subset of this many events.

        Returns:
            ``(particles, m, am, per_event)``: (N, 1 + n_out, 8) with the incoming particle first, ``m`` 1 on the
            real outgoing particles, ``am`` True on every real particle.
        """
        raw       = data["per_particle"]
        particles = torch.cat((raw["Incoming"], raw["Outgoing"]), dim=-2).to(self.dtype)
        per_event = data["per_event"]
        energy    = particles[..., 0] if self.energy_kin else particles[..., 1:4].norm(dim=-1)   # false: legacy

        cut, n = _to_front(particles, keep=energy >= self.cutoff_mev)   # n: the particles kept, incoming included
        events = n > self.min_particles
        if n_events is not None:
            events = _fixed_subset(events, n_events)
        cut, per_event = cut[events], {k: v[events] for k, v in per_event.items()}

        am = ~cut.isnan().any(dim=-1)   # a real particle
        m  = am.long()
        m[:, 0] = 0                     # ... the flow generates: all but the incoming one
        return cut.to(self.dev), m.to(self.dev), am.to(self.dev), {k: v.to(self.dev) for k, v in per_event.items()}


# ---- plumbing --------------------------------------------------------------------------------------------------------

def _to_front(particles: Tensor, keep: Tensor) -> tuple[Tensor, Tensor]:
    """Returns each event's kept particles moved to its front, NaN after them, and how many there are."""
    n     = keep.sum(dim=-1)
    front = torch.arange(int(n.max()), device=n.device) < n[:, None]   # the first n slots of each event
    cut   = torch.full_like(particles[:, :int(n.max())], torch.nan)
    cut[front] = particles[keep]
    return cut, n


def _fixed_subset(events: Tensor, n_events: int) -> Tensor:
    """Returns the indices of ``n_events`` random events among those ``events`` marks, the same every run."""
    events = events.nonzero()[:, 0]
    return events[torch.randperm(len(events), generator=torch.Generator().manual_seed(0))[:n_events]]
