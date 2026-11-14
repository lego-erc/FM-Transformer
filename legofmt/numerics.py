"""Pointwise operations with their singular points handled, so that the formulas using them read as formulas.

Masking a result with ``torch.where`` is not enough where an operation is singular: autograd still differentiates
the masked-out branch, and an infinite slope there turns the whole gradient into NaN. The guard has to sit on the
operation's input.
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
    return torch.acos(torch.where(x.abs() >= 1, x.detach(), x))


def detach_at_pole(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Detaches ``(x, y)`` where ``x^2 + y^2`` is subnormal (within ~1e-19 of the pole).

    The backward of atan2 and hypot overflows there.
    """

    pole = x * x + y * y < torch.finfo(x.dtype).tiny
    return torch.where(pole, x.detach(), x), torch.where(pole, y.detach(), y)


def safe_atan2(y: Tensor, x: Tensor) -> Tensor:
    """Computes atan2 autograd-safely: avoids a NaN grad near and at the pole ``x = y = 0``."""

    return torch.atan2(*detach_at_pole(y, x))
