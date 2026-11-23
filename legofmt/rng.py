"""Opt-in per-event random numbers for generation (``rng=EventRNG(keys, seed)``).

Every draw is a pure function of (seed, event key, draw site, element): a counter-based SplitMix64 hash (the scheme of
DeepSeek's DeepInfer sampler) mapped to a uniform in (0, 1), then to the distribution of the torch draw it replaces
(inverse normal CDF, inverse CDF of a categorical, ``u < p``). The distributions are unchanged. An event's randomness
does not depend on its batch, its row or which rows a caller subsets, and two models generating the same keyed events
share it, so their outputs differ only because the models differ.

Without ``rng`` every call site draws from torch's generator. ``Noise`` and the functions ``uniform_like``,
``normal_like``, ``randint_like`` and ``categorical`` put both draws behind one call (``rng=None``: torch's).
``rng_sub`` subsets an ``rng`` that may be None; ``rng_on`` checks a caller's ``rng`` and moves its keys to the events'
device. The keys of one ``EventRNG`` must be unique: two events with one key get the same randomness.
"""

import torch

# Draw sites (``stream``). The ids are fixed, so a new site leaves the numbers of the existing ones unchanged; a
# direction draw uses ``stream`` and ``stream + 1``.
PASSTHROUGH, SYM_PT, SYM_COUNT, COUNT, REFILL, SYM_FLOW, BASE_E, EDEP, LATENT, AE_MIX, AE_NORM, AE_ZERO = range(1, 13)
BASE_MOM, BASE_POS, ISO_E, ISO_MOM, ISO_POS = 20, 22, 24, 26, 28


def _mix(z):
    """Applies one SplitMix64 step to ``z`` (int64): increment, then finaliser; multiplies wrap, shifts are logical."""
    srl = lambda x, n: (x >> n) & ((1 << (64 - n)) - 1)
    z = z + -0x61C8864680B583EB                     # 0x9E3779B97F4A7C15 as int64
    z = (z ^ srl(z, 30)) * -0x40A7B892E31B1A47      # 0xBF58476D1CE4E5B9
    z = (z ^ srl(z, 27)) * -0x6B2FB644ECCEEE15      # 0x94D049BB133111EB
    return z ^ srl(z, 31)


def species_rank_ids(pid: torch.Tensor) -> torch.Tensor:
    """Maps each slot's species to the id (species, rank within the species), unique within an event.

    The rank counts the slots of the same species in slot order, so a slot keeps its id when a particle of another
    species is added, wherever that particle is inserted.

    Args:
        pid: (B, n) int64, the species per slot.

    Returns:
        (B, n) int64 ids.
    """
    return pid * 4096 + _rank_among_equal(pid)


class EventRNG:
    """Counter-based random numbers for ``B`` events with unique int64 ``keys``.

    ``elem`` names the elements of a draw and sets its shape: an int gives one number per event (B,), a 1-D tensor of
    ids shared by every event gives (B, n), and a tensor of per-event ids (B, n, ...) gives its own shape.
    """

    def __init__(self, keys, seed: int = 0):
        keys = torch.as_tensor(keys, dtype=torch.int64)
        if keys.unique().numel() != keys.numel():
            raise ValueError("EventRNG keys must be unique: two events with one key get the same randomness")
        self.keys, self.seed = keys, int(seed)

    def sub(self, rows) -> "EventRNG":
        """Returns the ``EventRNG`` of the events ``rows`` (a subset or a reordering): each keeps its random numbers."""
        out = object.__new__(EventRNG)
        out.keys, out.seed = self.keys[rows], self.seed
        return out

    def uniform(self, stream: int, elem=0) -> torch.Tensor:
        """Samples uniforms in (0, 1), float64, at the draw site ``stream`` and the elements ``elem``.

        ```
        site = mix(mix(mix(seed) + key) + mix(stream))    per event
        z    = mix(site + elem)                           per element
        u    = (z's top 53 bits + 1/2) / 2**53
        ```
        """
        site = _mix(_mix(_mix(_int64(self.seed, self.keys)) + self.keys) + _mix(_int64(stream, self.keys)))
        e    = _int64(elem, self.keys)
        z    = _mix(_per_event(site, e) + e)
        return ((z >> 11) & ((1 << 53) - 1)).double().add_(0.5).mul_(2.0 ** -53)

    def normal(self, stream: int, elem=0) -> torch.Tensor:
        """Samples standard normals, float64; arguments as in ``uniform``."""
        return torch.special.ndtri(self.uniform(stream, elem))

    def randint(self, high: int, stream: int, elem=0) -> torch.Tensor:
        """Samples integers in ``[0, high)``; ``stream`` and ``elem`` as in ``uniform``."""
        return (self.uniform(stream, elem) * high).long()

    def categorical(self, probs: torch.Tensor, stream: int, elem=0) -> torch.Tensor:
        """Samples class indices from unnormalised ``probs`` (B, ..., K) by the inverse CDF.

        One index per distribution when ``elem`` gives the shape of ``probs`` without its last dim; ``n`` indices per
        event for ``probs`` (B, K) and a 1-D ``elem`` (n,). A class of weight 0 is never sampled.
        """
        cdf = probs.double().cumsum(-1)
        u   = self.uniform(stream, elem).to(cdf.device)
        one = u.dim() == probs.dim() - 1   # one index per distribution
        if one:
            u = u.unsqueeze(-1)
        idx = torch.searchsorted(cdf, u * cdf[..., -1:], right=True)   # the first k with u * total < cdf_k
        idx = idx.clamp_(max=probs.shape[-1] - 1)                       # u * total can round up to the total
        return idx.squeeze(-1) if one else idx


class Noise:
    """One call site's random numbers of shape ``shape``: torch's without ``rng``, the keyed ones with it.

    Torch's are float32 on ``device`` and consume its generator in call order. The keyed ones are float64, at the
    element ids ``elem``; ``stream`` names the draw site of each call.
    """

    def __init__(self, shape, device, rng: EventRNG | None = None, elem=0):
        self.shape, self.device, self.rng, self.elem = shape, device, rng, elem

    @classmethod
    def per_slot(cls, pid: torch.Tensor, rng: EventRNG | None = None) -> "Noise":
        """Constructs the noise for one draw per slot of ``pid`` (B, n), keyed on the slot's (species, rank)."""
        return cls(pid.shape, pid.device, rng, None if rng is None else species_rank_ids(pid.long()))

    def normal(self, stream: int) -> torch.Tensor:
        """Samples a standard normal per element."""
        if self.rng is None:
            return torch.randn(self.shape, device=self.device)
        return self.rng.normal(stream, self.elem)

    def uniform(self, stream: int) -> torch.Tensor:
        """Samples a uniform in (0, 1) per element."""
        if self.rng is None:
            return torch.rand(self.shape, device=self.device)
        return self.rng.uniform(stream, self.elem)


# Each function holds the two branches of one call site: torch's draw without ``rng``, the keyed draw at ``stream``
# with it.

def uniform_like(p: torch.Tensor, rng: EventRNG | None, stream: int, elem=0) -> torch.Tensor:
    """Samples uniforms in (0, 1) shaped like ``p``, in its dtype and on its device.

    ``p`` is (B,) for an int ``elem``, (B, n) for a 1-D tensor ``elem`` (the element ids of the keyed draw).
    """
    return torch.rand_like(p) if rng is None else rng.uniform(stream, elem).to(p)


def normal_like(p: torch.Tensor, rng: EventRNG | None, stream: int, elem=0) -> torch.Tensor:
    """Samples standard normals shaped like ``p``, in its dtype and on its device; ``elem`` as in ``uniform_like``."""
    return torch.randn_like(p) if rng is None else rng.normal(stream, elem).to(p)


def randint_like(high: int, x: torch.Tensor, rng: EventRNG | None, stream: int) -> torch.Tensor:
    """Samples an integer in ``[0, high)`` per event, shaped like ``x`` (B,) and on its device."""
    if rng is None:
        return torch.randint(high, x.shape, device=x.device)
    return rng.randint(high, stream).reshape(x.shape).to(x.device)


def categorical(probs: torch.Tensor, rng: EventRNG | None, stream: int, elem=0) -> torch.Tensor:
    """Samples class indices from the unnormalised ``probs`` (B, K).

    One per event for an int ``elem``; ``len(elem)`` per event, with replacement, for a 1-D tensor ``elem`` (the
    element ids of the keyed draw, as in ``EventRNG.categorical``).
    """
    if rng is not None:
        return rng.categorical(probs, stream, elem)
    if isinstance(elem, torch.Tensor):
        return torch.multinomial(probs, len(elem), replacement=True)
    return torch.multinomial(probs, 1).squeeze(-1)


def rng_sub(rng: EventRNG | None, rows) -> EventRNG | None:
    """Returns ``rng`` for the events ``rows``, a subset or a reordering (None stays None)."""
    return None if rng is None else rng.sub(rows)


def rng_on(rng: EventRNG | None, events: torch.Tensor) -> EventRNG | None:
    """Returns ``rng`` for the rows of ``events``, its keys on their device (None stays None).

    The keys go to the events' device because ``sub`` indexes them with masks on that device, and the draws are
    combined with tensors on it.

    Raises:
        ValueError: ``rng`` does not hold one key per row of ``events``.
    """
    if rng is None:
        return None
    if len(rng.keys) != len(events):
        raise ValueError(f"rng needs one key per event: {len(rng.keys)} keys for {len(events)} events")
    return EventRNG(rng.keys.to(events.device), rng.seed)


# ---- plumbing --------------------------------------------------------------------------------------------------------

def _rank_among_equal(v: torch.Tensor) -> torch.Tensor:
    """Ranks each entry of ``v`` (B, n) among the entries of its row with the same value, in slot order."""
    order = v.argsort(dim=1, stable=True)
    s = v.gather(1, order)
    idx = torch.arange(s.shape[1], device=s.device).expand_as(s)
    start = torch.cat((torch.ones_like(s[:, :1], dtype=torch.bool), s[:, 1:] != s[:, :-1]), dim=1)
    first = torch.where(start, idx, 0).cummax(dim=1).values
    return torch.empty_like(s).scatter_(1, order, idx - first)


def _int64(v, like: torch.Tensor) -> torch.Tensor:
    """Converts ``v`` to an int64 tensor on the device of ``like``."""
    return torch.as_tensor(v, dtype=torch.int64, device=like.device)


def _per_event(h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Shapes the per-event ``h`` (B,) to broadcast against the element ids ``e``.

    ``h + e`` is then ``(B, *e.shape)`` for a scalar ``e`` or ids shared by every event (n,), and of ``e``'s shape for
    per-event ids (B, n, ...).
    """
    return h.reshape(-1, *([1] * e.dim())) if e.dim() < 2 else h.reshape(-1, *([1] * (e.dim() - 1)))
