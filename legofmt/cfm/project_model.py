"""The field on the manifold: ``ProjectModel`` wraps ``CFMTrafo_x`` (``cfm_trafo_x.py``) between two projections.

```
x_p = proj_M(x)                  the state, onto the manifold
v   = proj_T(x_p, field(x_p))    the field's output, onto the tangent space at x_p
```

Both act on every attended token; padding passes through both as it is. The field sees ``x_p`` detached unless
``no_detach`` is set. In training the path's ``x_t`` carries no graph anyway; a solve, which may be differentiated
(density design), and ``log_likelihood`` set the flag (``solvers._field_sees_graph``).

``model_conf.energy_border`` gives the energy ``e`` of the generated outgoing tokens a border:

```
e_b = clamp(e, 0, 1)             what the field sees in the place of e
v_e = (e_b - e) / (1 - t)        the energy's velocity, wherever e is outside [0, 1]
```

The energy's training path runs from a base in [0, 1] to a target in [0, 1], so no path leaves that range. An energy
outside it was put there by a step error, a solver's or the flow-map teacher's, and there the field is trained for
none of its outputs, the directions' velocities included. So the field is evaluated on the border, and the energy
moves on the straight line that reaches the border at t = 1. Its velocity is the same all along that line, which
makes it the mean velocity of a flow-map step of any ``d`` as well.

``LEGOLtng`` assigns the wrapper as ``self.model``, which is where the state_dict keys ``model.vf.*`` come from.
"""

import torch
from torch import Tensor, nn

from legofmt.data.struct import _F, _M
from legofmt.geometry.product_manifold import ProductManifold
from legofmt.legacy.cfm_trafo_x import pin_per_token

# ``energy_border``: ``1 - t`` is held at this or more. At t = 1 no time is left to reach the border, and rk4's last
# stage and an adaptive solver evaluate the field there.
MIN_TIME_LEFT = 1e-2


def on_border(x: Tensor, mask: Tensor) -> Tensor:
    """Returns ``x`` (B, L, in_dim) with the energy of the generated outgoing tokens held in [0, 1].

    ``mask`` is (B, L), 1 on the generated tokens. Every other token and channel is returned as it is.
    """
    e = _F(x).out_energy
    x = x.clone()   # e stays a view of the caller's tensor
    _F(x).out_energy.copy_(torch.where(_M(mask).out_p == 1, e.clamp(0.0, 1.0), e))
    return x


def to_border(v: Tensor, x: Tensor, x_b: Tensor, t: Tensor) -> Tensor:
    """Returns ``v`` with the energy's velocity ``(e_b - e) / (1 - t)`` wherever ``on_border`` moved the energy.

    Args:
        v:   (B, L, in_dim), the field's velocity.
        x:   (B, L, in_dim), the state.
        x_b: (B, L, in_dim), the state as the field saw it (``on_border``).
        t:   the time, as ``ProjectModel.forward`` takes it.
    """
    gap  = _F(x_b).energy - _F(x).energy                                       # e_b - e; 0 inside the border
    left = (1 - torch.atleast_2d(t)).clamp(min=MIN_TIME_LEFT).unsqueeze(-1)    # the time left, per event
    v_e  = torch.where(gap != 0, gap / left, _F(v).energy)
    return torch.cat((v_e, v[..., 1:]), dim=-1)


class ProjectModel(nn.Module):
    """The field between the projection onto the manifold and the projection onto its tangent space."""

    def __init__(self, vf: nn.Module, manifold: ProductManifold, energy_border: bool = False, **kwargs) -> None:
        super().__init__()
        self.vf            = vf
        self.manifold      = manifold
        self.energy_border = energy_border   # model_conf.energy_border (module docstring)
        self.no_detach     = kwargs.get("no_detach", False)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        mask: Tensor,
        attn_mask: Tensor,
        types: Tensor,
        pdgids: Tensor | None = None,
        d: Tensor | None = None,
        z: Tensor | None = None,
    ) -> Tensor:
        """Evaluates the field at ``proj_M(x)`` and projects its velocity onto the tangent space there.

        Args:
            x:         (B, L, in_dim), the state.
            t:         the time: a scalar, or one per event (B, 1). Legacy: one per token (B, L).
            mask:      (B, L), 1 on generated tokens.
            attn_mask: (B, L) bool, False on padding.
            types:     (1, L), each position's type index.
            pdgids:    (B, L, 1), each token's species index.
            d:         the step size, shaped like ``t``; None: 0, the instantaneous velocity. Read only by a field
                with ``step_cond``.
            z:         (B, lat_dim), the event's latent (``model_args.lat_dim``); Euclidean, not projected.

        Returns:
            ``v`` (B, L, in_dim); with ``z`` the pair ``(v, v_z)``, ``v_z`` (B, lat_dim) the latent's velocity.
        """
        x_p, x_seen = self._prep_x(x, attn_mask)                   # proj_M(x), and what the field sees
        if self.energy_border: x_seen = on_border(x_seen, mask)    # with the outgoing energy held in [0, 1],
        v, v_z      = self._call_field(x_seen, t, d, z, mask, attn_mask, types, pdgids)
        v           = torch.where(attn_mask.unsqueeze(-1), self.manifold.proju(x_p, v), v)   # proj_T; padding as is
        if self.energy_border: v = to_border(v, x_p, x_seen, t)    # and an energy outside moving straight to the border
        return v if z is None else (v, v_z)

    # ---- plumbing -----------------------------------------------------------------------------------------------

    def _prep_x(self, x: Tensor, attn_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Projects ``x`` onto the manifold.

        Returns:
            ``(x_proj, x_att)``: the projection, and what the field sees. That is the projection on every attended
            token, so a conditioning token's filler values off the manifold never reach the field, and ``x`` on
            padding; detached unless ``no_detach``.
        """
        x_proj = self.manifold.projx(x)
        x_att  = torch.where(attn_mask.unsqueeze(-1), x_proj, x)
        if not self.no_detach:
            x_att.detach_()
        return x_proj, x_att

    def _call_field(self, x: Tensor, t: Tensor, d: Tensor | None, z: Tensor | None, mask: Tensor, attn_mask: Tensor,
                    types: Tensor, pdgids: Tensor | None) -> tuple[Tensor, Tensor | None]:
        """Calls the field with one ``(t, d)`` per event; returns ``(v, v_z)``, ``v_z`` None without a latent.

        ``t`` and ``d`` are made 2-D. ``d`` reaches only a ``step_cond`` field, where None becomes 0, the
        instantaneous velocity.
        """
        t = torch.atleast_2d(t)
        if getattr(self.vf, "step_cond", False):
            d = torch.zeros_like(t) if d is None else torch.atleast_2d(d)
        else:
            d = None  # the field ignores it
        # legacy: per-token times
        if t.shape[-1] != 1 or (d is not None and d.shape[-1] != 1): t, d = pin_per_token(t, d, mask, attn_mask)
        if z is None:
            return self.vf(x, mask, attn_mask, types, pdgids, t=t, d=d), None
        return self.vf(x, mask, attn_mask, types, pdgids, t=t, d=d, z=z)


# ---- plumbing ------------------------------------------------------------------------------------------------------

def uncompiled(module: nn.Module) -> nn.Module:
    """Returns the module underneath a ``torch.compile`` wrapper, or the module itself."""
    return getattr(module, "_orig_mod", module)


def is_compiled(module: nn.Module) -> bool:
    """Returns whether ``module`` is a ``torch.compile`` wrapper."""
    return hasattr(module, "_orig_mod")
