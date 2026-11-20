"""Pointwise operations with their singular points handled, so that the formulas using them read as formulas.

Masking a result with ``torch.where`` is not enough where an operation is singular: autograd still differentiates
the masked-out branch, and an infinite slope there turns the whole gradient into NaN. The guard has to sit on the
operation's input. ``masked_mean`` guards the empty mask; ``held`` holds a learned log-variance on its floor.
"""

import torch
from torch import Tensor


def safe_pow(base: Tensor, exponent) -> Tensor:
    """Computes ``base ** exponent`` autograd-safely: avoids a NaN grad for 0**(exp < 1)."""

    if base.requires_grad:
        return torch.where(base > 0, base, base.detach()) ** exponent

    return base ** exponent


def safe_div(num: Tensor, den: Tensor, ok: Tensor, otherwise: float) -> Tensor:
    """Computes ``num / den`` where ``ok``, the constant ``otherwise`` elsewhere.

    Never divides by a ``den`` that is not ok.
    """

    den = torch.where(ok, den, torch.ones_like(den))
    return torch.where(ok, num / den, torch.full_like(den, otherwise))


def safe_acos(x: Tensor) -> Tensor:
    """Computes acos autograd-safely: holds ``x`` inside [-1, 1] and avoids the infinite grad at +-1."""

    x = x.clamp(-1 + 1e-8, 1 - 1e-8)   # in fp32 the bound rounds to exactly +-1
    return torch.acos(torch.where(torch.abs(x) >= 1, x.detach(), x))


def detach_at_pole(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Detaches ``(x, y)`` where ``x^2 + y^2`` is subnormal (within ~1e-19 of the pole).

    The backward of atan2 and hypot overflows there. Without grad it returns ``(x, y)`` as they are: CPU's atan2 then
    keeps its strided-input path, so sampling stays bit-identical.
    """

    if not (x.requires_grad or y.requires_grad):
        return x, y

    return _detach_where_pole(x, y)


def safe_atan2(y: Tensor, x: Tensor) -> Tensor:
    """Computes atan2 autograd-safely: avoids a NaN grad near and at the pole ``x = y = 0``.

    It masks with and without grad: the mask's copy of a strided input decides which CPU atan2 kernel runs, and so
    the bits.
    """

    return torch.atan2(*_detach_where_pole(y, x))


def masked_mean(x: Tensor, w: Tensor, dim=None) -> Tensor:
    """Computes the ``w``-weighted mean ``sum(w x) / sum(w)`` of ``x`` over ``dim`` (every element by default).

    The weight sum is held at 1 or above, so an empty mask gives 0, not NaN.
    """

    return (x * w).sum(dim) / w.sum(dim).clamp(min=1)


def held(lv: Tensor, floor) -> Tensor:
    """Holds ``lv`` at ``floor`` or above, in place, and returns it.

    In place and without grad, so the gradient stays live on the floor: below it a clamp() in the graph has zero
    gradient, and a learned parameter would latch there.
    """

    with torch.no_grad():
        lv.clamp_(min=floor)
    return lv


def _detach_where_pole(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Detaches ``(x, y)`` where ``x^2 + y^2`` is subnormal, with or without grad."""

    pole = x * x + y * y < torch.finfo(x.dtype).tiny
    return torch.where(pole, x.detach(), x), torch.where(pole, y.detach(), y)
