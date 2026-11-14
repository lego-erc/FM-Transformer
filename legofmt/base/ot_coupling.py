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

_OT_COUPLING_REQUIRES_LAP = (
    "ot_coupling=True requires `torch-linear-assignment` (a pixi dependency). "
    "Install it or set model_conf.ot_coupling=False."
)


_BLOCKED = 1e6   # the cost of a blocked pair


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
    if batch_linear_assignment is None:
        raise RuntimeError(_OT_COUPLING_REQUIRES_LAP)
    base    = base.where(ds_t.am.full.unsqueeze(-1), ds_t.f.model_in)
    out     = _F(base).out_p                                                    # a view: writing it permutes base
    valid   = ds_t.am.out_p
    pid     = ds_t.f.out_pid
    blocked = (valid[:, :, None] != valid[:, None, :]) | (pid[:, :, None] != pid[:, None, :])

    data_i  = ds_t.f.out_cc[:, :, None].split(manifold.ambient_dims, dim=-1)    # per factor (B, n, 1, .)
    base_j  = out[:, None].split(manifold.ambient_dims, dim=-1)                 # per factor (B, 1, n, .)
    cost    = sum(
        (torch.acos((a * b).sum(-1).float().clamp(-1 + 1e-6, 1 - 1e-6)) if isinstance(mf, Sphere)
         else (a - b).norm(dim=-1).float()) ** 2
        for mf, a, b in zip(manifold.manifolds, data_i, base_j)
    ) + _BLOCKED * blocked

    assign  = (_lap_cuda_unchecked(cost, validation="off") if cost.is_cuda and _lap_cuda_unchecked is not None
               else batch_linear_assignment(cost)).long()
    out[:]  = torch.take_along_dim(out, assign.unsqueeze(-1), dim=1)
    return base
