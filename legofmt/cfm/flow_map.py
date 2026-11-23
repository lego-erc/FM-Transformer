"""The flow-map term ``L_fm``: self-distillation of the field into few-step sampling.

```
step      d = 2**-k,  k ~ U{0, ..., sections - 1};   t0 ~ the grid of d in [0, 1)
teacher   two half steps from s0 = s_{t0}, without gradient:
            s_mid = exp_{s0}(d/2 v(s0, t0; d/2))
            s_end = exp_{s_mid}(d/2 v(s_mid, t0 + d/2; d/2))
          for the smallest d the teacher is called with d = 0, the instantaneous velocity, which the velocity loss
          trains: no step of half the smallest d is ever trained
target    log_{s0}(s_end) / d          the mean velocity over the step; through the logmap, as the manifold has
                                       sphere factors
L_fm      the mean of |v(s0, t0; d) - target|**2 over the channels of the generated tokens
```

``s = (x, z)`` is the state of ``cfm/state.py``: the tokens and the ``edep_ae`` latent. The latent takes the same two
half steps (Euclidean, from ``z_{t0} = z_0 + t0 (z_1 - z_0)`` on its straight path), and its error joins the mean as
K more entries. Only the generated tokens move. The step sizes ``d = 2**-k`` are the rungs ``k`` of a ladder.
``one_step_euler_loss`` takes the ``LEGOLtng`` as its first argument.
"""

import torch
from torch import Tensor

from legofmt.cfm.state import LatentManifold, State, moved, no_latent, state_field


def one_step_euler_loss(lego, base: Tensor, ds_t, pdgid_idx: Tensor, model=None, z=None) -> Tensor:
    """Computes ``L_fm`` in fp32.

    The loss is ~5e-5 and bf16 autocast inflates it ~390x, so it is computed outside autocast.

    Args:
        lego:      the ``LEGOLtng``.
        base:      (B, L, 7), the batch's ``x_0``.
        ds_t:      the batch, in the frame and units of the velocity loss.
        pdgid_idx: (B, L, 1), the tokens' species indices.
        model:     the field to call; None: ``lego.model``.
        z:         ``(z_0, z_1)`` of the ``edep_ae`` latent, each (B, K), or None.

    Returns:
        The loss (a scalar).
    """
    with torch.autocast(base.device.type, enabled=False):
        z = None if z is None else tuple(a.float() for a in z)
        model = lego.model if model is None else model
        return _one_step_euler_loss_fp32(lego, base.float(), ds_t, pdgid_idx, model, z)


def _one_step_euler_loss_fp32(lego, base: Tensor, ds_t, pdgid_idx: Tensor, model, z=None) -> Tensor:
    """Computes ``L_fm`` as the module docstring has it, on fp32 inputs; arguments as in ``one_step_euler_loss``."""
    field, man, gen = _ladder_inputs(lego, ds_t, pdgid_idx, model)
    sections = lego.rc.one_step_euler_sections
    with torch.no_grad():
        k     = _uniform_rung(sections, base)                                     # k ~ U{0, ..., sections - 1}
        d     = 2.0 ** -k
        t0    = (torch.rand_like(d) * (1.0 / d).round()).floor() * d               # on the grid of d
        s0    = _on_path(lego, base, ds_t, z, t0)                                  # s_{t0}
        half  = d / 2
        d_t   = torch.where(half >= 2.0 ** -(sections - 1), half, 0.0)   # the teacher's d; for the smallest d: 0
        s_mid = moved(gen, s0,    man.expmap(s0,    _per_event(half) * field(s0, t0, d=d_t)))
        s_end = moved(gen, s_mid, man.expmap(s_mid, _per_event(half) * field(s_mid, t0 + half, d=d_t)))
        tgt   = man.logmap(s0, s_end) / _per_event(d)
    return _mean_squared_error(field(s0, t0, d=d), tgt, gen & ds_t.am.full.unsqueeze(-1), latent=z is not None)


# ---- plumbing ------------------------------------------------------------------------------------------------------

def _ladder_inputs(lego, ds_t, pdgid_idx: Tensor, model):
    """Returns ``(field, man, gen)``: the field and the manifold's maps on a ``State``, the generated tokens (B, L, 1).

    The field is ``model`` with the batch's keyword inputs (``state_field``).
    """
    mask = ds_t.m.full
    ckw  = dict(mask=mask, attn_mask=ds_t.am.full, types=lego.types_embd[:, :mask.shape[1]], pdgids=pdgid_idx)
    return state_field(model, ckw), LatentManifold(lego.model.manifold), (mask == 1).unsqueeze(-1)


def _uniform_rung(sections: int, base: Tensor) -> Tensor:
    """Samples one rung ``k`` per event, uniform on {0, ..., sections - 1}: (B, 1) in the dtype of ``base``."""
    return (torch.randint(1, sections + 1, (base.shape[0], 1), device=base.device) - 1).to(base.dtype)


def _on_path(lego, base: Tensor, ds_t, z, t0: Tensor) -> State:
    """Returns the state at ``t0``: the tokens on the training path, the latent on its straight path.

    The tokens' path runs from ``base`` to the data. The latent's is ``z_0 + t0 (z_1 - z_0)`` for ``z = (z_0, z_1)``;
    without a latent (``z`` None) it is the empty one (B, 0).
    """
    x_t = lego.ps.sample(base, ds_t.f.model_in.float(), t0.squeeze(-1)).x_t
    z_t = no_latent(t0) if z is None else z[0] + t0 * (z[1] - z[0])
    return State(x_t, z_t)


def _per_event(c: Tensor) -> State:
    """Returns one scalar per event ``c`` (B, 1) as a ``State``: (B, 1, 1) for the tokens, (B, 1) for the latent."""
    return State(c.unsqueeze(-1), c)


def _mean_squared_error(pred: State, tgt: State, cells: Tensor, latent: bool) -> Tensor:
    """Returns the mean of ``|pred - tgt|**2`` over the channels of the tokens ``cells`` (B, L, 1) and the latent.

    The latent's entries count only with ``latent``.
    """
    err, n = ((pred.x - tgt.x) ** 2 * cells).sum(), cells.sum() * pred.x.shape[-1]
    if latent:
        err, n = err + ((pred.z - tgt.z) ** 2).sum(), n + pred.z.numel()
    return err / n.clamp(min=1)
