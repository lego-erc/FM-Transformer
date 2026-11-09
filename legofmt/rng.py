"""Opt-in per-event random numbers for generation (``rng=EventRNG(keys, seed)``).

Every draw is a pure function of (seed, event key, draw site, element): a counter-based SplitMix64 hash (the scheme of
DeepSeek's DeepInfer sampler) mapped to a uniform in (0, 1), then through the same transform the torch draw it
replaces used (inverse normal CDF, inverse CDF of a categorical, ``u < p``). The distributions are unchanged; what
changes is that an event's randomness no longer depends on its batch, its row or which rows a caller subsets, and
that two models generating the same keyed events share it -- so the difference between them carries no draw luck.

Without ``rng`` every call site keeps its torch draw (bit-identical to before). Keys must be unique within a sample:
two events with one key get the same randomness.
"""

import torch

# Draw sites. Fixed ids, so a new draw never shifts an existing one; GeomTrafos draws use ``stream`` and ``stream + 1``.
PASSTHROUGH, SYM_PT, SYM_COUNT, COUNT, REFILL, SYM_FLOW, BASE_E, EDEP, LATENT, AE_MIX, AE_NORM, AE_ZERO = range(1, 13)
BASE_MOM, BASE_POS, ISO_E, ISO_MOM, ISO_POS = 20, 22, 24, 26, 28


def _mix(z):
    """SplitMix64's finaliser on int64 (wrapping multiplies, logical shifts)."""
    srl = lambda x, n: (x >> n) & ((1 << (64 - n)) - 1)
    z = z + -0x61C8864680B583EB                     # 0x9E3779B97F4A7C15 as int64
    z = (z ^ srl(z, 30)) * -0x40A7B892E31B1A47      # 0xBF58476D1CE4E5B9
    z = (z ^ srl(z, 27)) * -0x6B2FB644ECCEEE15      # 0x94D049BB133111EB
    return z ^ srl(z, 31)


def species_rank_ids(pid: torch.Tensor) -> torch.Tensor:
    """(B, n) species per slot -> (B, n) ids (species, rank within the species), unique per row: a slot keeps its id
    when a particle of another species is added, whatever the slot order."""
    order = pid.argsort(dim=1, stable=True)
    s = pid.gather(1, order)
    idx = torch.arange(s.shape[1], device=s.device).expand_as(s)
    start = torch.cat((torch.ones_like(s[:, :1], dtype=torch.bool), s[:, 1:] != s[:, :-1]), dim=1)
    first = torch.where(start, idx, 0).cummax(dim=1).values
    return pid * 4096 + torch.empty_like(s).scatter_(1, order, idx - first)


class EventRNG:
    """Counter-based random numbers for ``B`` events with unique int64 ``keys``. ``elem`` names the element of a
    draw: an int (one draw per event, ``(B,)``), a 1-D tensor of ids shared by every event (``(B, n)``), or a
    tensor of per-event ids with leading dim ``B`` (its own shape)."""

    def __init__(self, keys, seed: int = 0):
        keys = torch.as_tensor(keys, dtype=torch.int64)
        if keys.unique().numel() != keys.numel():
            raise ValueError("EventRNG keys must be unique: two events with one key get the same randomness")
        self.keys, self.seed = keys, int(seed)

    def sub(self, rows) -> "EventRNG":
        """The same randomness for a subset / reordering of the events."""
        out = object.__new__(EventRNG)
        out.keys, out.seed = self.keys[rows], self.seed
        return out

    def uniform(self, stream: int, elem=0) -> torch.Tensor:
        """float64 in (0, 1)."""
        k = self.keys
        t = lambda v: torch.as_tensor(v, dtype=torch.int64, device=k.device)
        h = _mix(_mix(_mix(t(self.seed)) + k) + _mix(t(stream)))
        e = t(elem)
        h = h.view(-1, *([1] * e.dim())) if e.dim() < 2 else h.view(-1, *([1] * (e.dim() - 1)))
        z = _mix(h + e)
        return ((z >> 11) & ((1 << 53) - 1)).double().add_(0.5).mul_(2.0 ** -53)

    def normal(self, stream: int, elem=0) -> torch.Tensor:
        return torch.special.ndtri(self.uniform(stream, elem))

    def randint(self, high: int, stream: int, elem=0) -> torch.Tensor:
        return (self.uniform(stream, elem) * high).long()

    def categorical(self, probs: torch.Tensor, stream: int, elem=0) -> torch.Tensor:
        """Inverse-CDF draws from unnormalised ``probs (B, ..., K)``: one per row of ``probs`` (``elem`` shaped like
        its leading dims), or ``n`` per event for 2-D ``probs`` and 1-D ``elem``. Zero-weight columns are never hit."""
        cdf = probs.double().cumsum(-1)
        u = self.uniform(stream, elem).to(cdf.device)
        one = u.dim() == probs.dim() - 1
        target = (u * cdf[..., -1])[..., None] if one else u * cdf[..., -1:]
        idx = torch.searchsorted(cdf, target, right=True).clamp_(max=probs.shape[-1] - 1)
        return idx.squeeze(-1) if one else idx
