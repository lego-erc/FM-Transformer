"""Minibatch OT coupling of the base sample to the data (``model_conf.ot_coupling``; training only).

Each event's base slots are permuted onto its data slots by the assignment that minimises the summed cost

```
cost[i, j] = sum_k d_k(data_i, base_j)^2     k: the token's manifold factors
d_k        = the arc on a sphere, the Euclidean distance otherwise
```

A pair is blocked (cost 1e6) when its slots differ in pdgid, or one of them is padding. Within a species an event's
base slots are i.i.d., so permuting them leaves the base distribution as it is.
"""

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


_BLOCKED = 1e6   # the cost of a blocked pair


def distance(manifold, a: Tensor, b: Tensor) -> Tensor:
    """Computes ``d_k(a, b)`` of one manifold factor: the arc on a sphere, the Euclidean distance otherwise."""
    if isinstance(manifold, Sphere):
        return torch.acos((a * b).sum(-1).float().clamp(-1 + 1e-6, 1 - 1e-6))
    return (a - b).norm(dim=-1).float()


def ot_couple(base: Tensor, ds_t: DataStruct, manifold) -> Tensor:
    """Permutes each event's base slots onto its data slots by optimal assignment.

    Args:
        base:     (B, N, 7), the base sample.
        ds_t:     the batch; its tokens replace the sample on padding.
        manifold: the outgoing token's product manifold.

    Returns:
        ``base`` (B, N, 7) with each event's outgoing slots permuted.

    Raises:
        RuntimeError: ``torch-linear-assignment`` is not installed.
    """
    require_lap()
    base   = torch.where(ds_t.am.full.unsqueeze(-1), base, ds_t.f.model_in)
    out    = _F(base).out_p                                            # a view: writing it permutes base
    cost   = (
        _pair_cost(ds_t.f.out_cc, out, manifold)
        + _BLOCKED * _blocked(ds_t.am.out_p, ds_t.f.out_pid)
    )
    out[:] = torch.take_along_dim(out, _assign(cost).unsqueeze(-1), dim=1)
    return base


def require_lap() -> None:
    """Raises unless ``torch-linear-assignment`` is installed.

    Raises:
        RuntimeError: it is not.
    """
    if batch_linear_assignment is None:
        raise RuntimeError("ot_coupling=True requires `torch-linear-assignment` (a pixi dependency). "
                           "Install it or set model_conf.ot_coupling=False.")


# ---- plumbing --------------------------------------------------------------------------------------------------------

def _pair_cost(data: Tensor, base: Tensor, manifold) -> Tensor:
    """Computes ``cost[b, i, j] = sum_k d_k(data_i, base_j)^2`` (B, n, n) over the manifold's factors."""
    data_i = data.unsqueeze(2).split(manifold.ambient_dims, dim=-1)   # per factor (B, n, 1, .)
    base_j = base.unsqueeze(1).split(manifold.ambient_dims, dim=-1)   # per factor (B, 1, n, .)
    return sum(distance(mf, a, b) ** 2 for mf, a, b in zip(manifold.manifolds, data_i, base_j))


def _blocked(valid: Tensor, pid: Tensor) -> Tensor:
    """Returns the pairs (B, n, n) whose slots differ in pdgid, or of which one is padding."""
    return (valid.unsqueeze(2) != valid.unsqueeze(1)) | (pid.unsqueeze(2) != pid.unsqueeze(1))


def _assign(cost: Tensor) -> Tensor:
    """Solves the assignment of each event's cost matrix (unchecked on CUDA: our costs are finite and square)."""
    if cost.is_cuda and _lap_cuda_unchecked is not None:
        return _lap_cuda_unchecked(cost, validation="off").long()
    return batch_linear_assignment(cost).long()
