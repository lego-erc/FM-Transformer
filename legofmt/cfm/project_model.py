"""Manifold-projection wrapper around the flow's vector field (:class:`~legofmt.cfm.cfm_trafo_x.CFMTrafo_x`).

``LEGOLtng`` assigns it as ``self.model``, which is where the state_dict keys ``model.vf.*`` come from.
"""

import torch
from torch import Tensor, nn

from legofmt.geometry.product_manifold import ProductManifold


class ProjectModel(nn.Module):
    """Projection Wrapper for Riemannian FM Model."""

    def __init__(self, vf: nn.Module, manifold: ProductManifold, **kwargs) -> None:
        super().__init__()
        self.vf        = vf
        self.manifold  = manifold
        self.no_detach = kwargs.get("no_detach", False)

    def _prep_x(self, x: Tensor, attn_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Project ``x`` onto the manifold. The network sees the projected value on
        every ``attn_mask`` slot (off-manifold conditioning filler never reaches it),
        detached unless ``no_detach`` -- which ``log_likelihood`` sets."""
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
        """With the field's latent token (``model_args.lat_dim``) pass the latent ``z`` (B, K) and get
        ``(v, v_z)``; ``z`` is Euclidean and not projected."""
        x_proj, x_att = self._prep_x(x, attn_mask)
        t = torch.atleast_2d(t)
        if getattr(self.vf, "step_cond", False):
            d = torch.zeros_like(t) if d is None else torch.atleast_2d(d)
        else:
            d = None  # the field ignores it
        if t.shape[-1] != 1 or (d is not None and d.shape[-1] != 1):  # per token: pin the conditions here
            t = torch.where(mask == 1, t.expand_as(attn_mask), 1.0)  # conditions get t=1
            if d is not None:
                d = torch.where(mask == 1, d.expand_as(attn_mask), 0.0)  # and are not transported
        out = self.vf(x_att, mask, attn_mask, types, pdgids, t=t, d=d, **({} if z is None else {"z": z}))
        v, v_z = (out, None) if z is None else out
        v = torch.where(attn_mask.unsqueeze(-1), self.manifold.proju(x_proj, v), v)
        return v if z is None else (v, v_z)


def uncompiled(module: nn.Module) -> nn.Module:
    """The module underneath a ``torch.compile`` wrapper, or the module itself."""
    return getattr(module, "_orig_mod", module)


def is_compiled(module: nn.Module) -> bool:
    return hasattr(module, "_orig_mod")
