"""Minibatch OT coupling of the base draw to the data (``model_conf.ot_coupling``; training only)."""

import torch
from torch import Tensor

from flow_matching.utils.manifolds import Sphere

from legofmt.data.struct import DataStruct, _F

try:
    from torch_linear_assignment import batch_linear_assignment
except ImportError:
    batch_linear_assignment = None
try:  # skips TLA's NaN / infeasibility .item() checks: our costs are finite and square
    from torch_linear_assignment._triton import batch_linear_assignment as _lap_cuda_unchecked
except ImportError:
    _lap_cuda_unchecked = None

_OT_COUPLING_REQUIRES_LAP = (
    "ot_coupling=True requires `torch-linear-assignment` (a pixi dependency). "
    "Install it or set model_conf.ot_coupling=False."
)


def ot_couple(base: Tensor, ds_t: DataStruct, data: Tensor, manifold) -> Tensor:
    """Per-event Hungarian assignment of base slots to data slots, blocked across
    pdgids and valid/pad pairs; base samples are pdgid-independent so permuting is free."""
    if batch_linear_assignment is None:
        raise RuntimeError(_OT_COUPLING_REQUIRES_LAP)
    base = base.where(ds_t.am.full.unsqueeze(-1), data)
    pid  = ds_t.f.out_pid
    blocked = (
        ds_t.am.out_p.unsqueeze(-1).logical_xor(ds_t.am.out_p.unsqueeze(-2))
        | (pid.unsqueeze(-1) != pid.unsqueeze(-2))
    )
    out = _F(base).out_p
    tgt = ds_t.f.out_cc.unsqueeze(-2).split(manifold.ambient_dims, dim=-1)
    ref = out.unsqueeze(-3).split(manifold.ambient_dims, dim=-1)
    cost = sum(
        (((a * b).sum(-1).float().clamp(-1 + 1e-6, 1 - 1e-6).acos()
          if isinstance(mf, Sphere) else (a - b).norm(dim=-1).float()) ** 2)
        for mf, a, b in zip(manifold.manifolds, tgt, ref)
    ) + blocked * 1e6
    assign = (_lap_cuda_unchecked(cost, validation="off") if cost.is_cuda and _lap_cuda_unchecked is not None
              else batch_linear_assignment(cost)).long()
    out[:] = torch.take_along_dim(out, assign.unsqueeze(-1), dim=1)
    return base
