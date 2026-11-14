"""The vector field on the manifold: ``ProjectModel`` wraps the field (:class:`~legofmt.cfm.cfm_trafo_x.CFMTrafo_x`)
between two projections.

```
x_p = proj_M(x)                  the state, onto the manifold
v   = proj_T(x_p, field(x_p))    the field's output, onto the tangent space at x_p
```

on every attended token; padding passes through both as it is. The field sees ``x_p`` detached unless
``no_detach``: the path's ``x_t`` carries no graph in training, a solve that is differentiated (density design) and
``log_likelihood`` set the flag.

``LEGOLtng`` assigns the wrapper as ``self.model``, which is where the state_dict keys ``model.vf.*`` come from.
"""

import torch
from torch import Tensor, nn

from legofmt.geometry.product_manifold import ProductManifold
from legofmt.legacy.cfm_trafo_x import pin_per_token


class ProjectModel(nn.Module):
    """The field between the projection onto the manifold and the projection onto its tangent space."""

    def __init__(self, vf: nn.Module, manifold: ProductManifold, **kwargs) -> None:
        super().__init__()
        self.vf        = vf
        self.manifold  = manifold
        self.no_detach = kwargs.get("no_detach", False)

    def _prep_x(self, x: Tensor, attn_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Projects ``x`` onto the manifold.

        The network sees the projected value on every ``attn_mask`` slot (off-manifold conditioning filler never reaches
        it), detached unless ``no_detach``. Returns ``(x_proj, x_att)``: the projection, and what the network sees.
        """
        x_proj = self.manifold.projx(x)
        x_att  = torch.where(attn_mask.unsqueeze(-1), x_proj, x)
        if not self.no_detach:
            x_att.detach_()
        return x_proj, x_att

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
        """Evaluates the field at ``x`` projected onto the manifold; the velocity comes back in its tangent space.

        Args:
            x:         (B, L, in_dim), the state.
            t:         the time: a scalar, or one per event (B, 1).
            mask:      (B, L), 1 on generated tokens.
            attn_mask: (B, L) bool, False on padding.
            types:     (1, L), each position's type.
            pdgids:    (B, L, 1), each token's species index.
            d:         the step size, shaped like ``t``; None: 0, the instantaneous velocity. Read only by a field
                with ``step_cond``.
            z:         (B, K), the field's latent (``model_args.lat_dim``); Euclidean, not projected.

        Returns:
            ``v`` (B, L, in_dim); with ``z`` ``(v, v_z)``.
        """
        x_proj, x_att = self._prep_x(x, attn_mask)
        t = torch.atleast_2d(t)
        if getattr(self.vf, "step_cond", False):
            d = torch.zeros_like(t) if d is None else torch.atleast_2d(d)
        else:
            d = None  # the field ignores it
        if t.shape[-1] != 1 or (d is not None and d.shape[-1] != 1): t, d = pin_per_token(t, d, mask, attn_mask)
        out = self.vf(x_att, mask, attn_mask, types, pdgids, t=t, d=d, **({} if z is None else {"z": z}))
        v, v_z = (out, None) if z is None else out
        v = torch.where(attn_mask.unsqueeze(-1), self.manifold.proju(x_proj, v), v)
        return v if z is None else (v, v_z)


def uncompiled(module: nn.Module) -> nn.Module:
    """Returns the module underneath a ``torch.compile`` wrapper, or the module itself."""
    return getattr(module, "_orig_mod", module)


def is_compiled(module: nn.Module) -> bool:
    """Returns whether ``module`` is a ``torch.compile`` wrapper."""
    return hasattr(module, "_orig_mod")
