"""Pointwise operations with their singular points handled, so that the formulas using them read as formulas.

Masking a result with ``torch.where`` is not enough where an operation is singular: autograd still differentiates
the masked-out branch, and an infinite slope there turns the whole gradient into NaN. The guard has to sit on the
operation's input. ``masked_mean`` guards the empty mask; ``held`` holds a learned log-variance at or above its floor.
"""

import torch
from torch import Tensor


def safe_pow(base: Tensor, exponent) -> Tensor:
    """Computes ``base ** exponent`` autograd-safely: a ``base <= 0`` gets no gradient.

    At ``base = 0`` the slope ``exponent * base ** (exponent - 1)`` is infinite for ``exponent < 1``.
    """

    if base.requires_grad:
        return torch.where(base > 0, base, base.detach()) ** exponent

    return base ** exponent


def safe_div(num: Tensor, den: Tensor, ok: Tensor, otherwise: float) -> Tensor:
    """Computes ``num / den`` where ``ok``, the constant ``otherwise`` elsewhere.

    Never divides by a ``den`` outside ``ok``.
    """

    den = torch.where(ok, den, 1.0)
    return torch.where(ok, num / den, otherwise)


def safe_acos(x: Tensor) -> Tensor:
    """Computes acos autograd-safely: holds ``x`` inside [-1, 1] and avoids the infinite grad at +-1."""

    x = x.clamp(-1 + 1e-8, 1 - 1e-8)   # in fp32 the bound rounds to exactly +-1
    return torch.acos(torch.where(torch.abs(x) >= 1, x.detach(), x))


def detach_at_pole(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Detaches ``(x, y)`` where ``x**2 + y**2`` is subnormal (in fp32: within ~1e-19 of the pole ``x = y = 0``).

    The backward of atan2 and hypot overflows there. Without grad ``(x, y)`` are returned as they are: the mask would
    copy a strided input, and CPU's atan2 gives different bits on the copy than on the strided input.
    """

    if not (x.requires_grad or y.requires_grad):
        return x, y

    return _detach_where_pole(x, y)


def safe_atan2(y: Tensor, x: Tensor) -> Tensor:
    """Computes atan2 autograd-safely: avoids a NaN grad near and at the pole ``x = y = 0``.

    The pole is masked with and without grad, so both give the same bits: the mask copies a strided input, and CPU's
    atan2 gives different bits on the copy than on the strided input.
    """

    return torch.atan2(*_detach_where_pole(y, x))


def masked_mean(x: Tensor, w: Tensor, dim=None) -> Tensor:
    """Computes the ``w``-weighted mean ``sum(w x) / sum(w)`` of ``x`` over ``dim`` (every element by default).

    ``x`` and ``w`` are broadcast against each other, and ``w`` counts once per element: a (B, L, 1) mask counts every
    channel of a token. The weight sum is held at 1 or above, so an empty mask gives 0, not NaN.
    """
    x, w = torch.broadcast_tensors(x, w)
    return (x * w).sum(dim) / w.sum(dim).clamp(min=1)


def held(lv: Tensor, floor) -> Tensor:
    """Holds ``lv`` at ``floor`` or above, in place, and returns it.

    The clamp is in place and outside the graph, so ``lv`` keeps its gradient at the floor. A ``clamp`` in the graph
    has zero gradient below the floor, and a learned parameter would stay stuck there.
    """

    with torch.no_grad():
        lv.clamp_(min=floor)
    return lv


def _detach_where_pole(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Detaches ``(x, y)`` where ``x**2 + y**2`` is subnormal, with or without grad."""

    pole = x * x + y * y < torch.finfo(x.dtype).tiny
    return torch.where(pole, x.detach(), x), torch.where(pole, y.detach(), y)
