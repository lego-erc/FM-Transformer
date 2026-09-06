"""Manifold-projection wrapper around the flow's vector field.

Split out of ``main/modules.py``: a plain ``nn.Module`` with no Lightning
contact, so it belongs beside the :class:`~legofmt.cfm.cfm_trafo_x.CFMTrafo_x`
it wraps. ``LEGOLtng`` assigns it as ``self.model``, which is where its
state_dict keys (``model.vf.*``) come from -- the class's module path does not
appear in a checkpoint.
"""

import torch
from torch import Tensor, nn

from legofmt.data.struct import _F

from legofmt.geometry.geom_trafos import GeomTrafos
from legofmt.geometry.path_sample_mult import ProductManifold


class ProjectModel(nn.Module):
    """Projection Wrapper for Riemannian FM Model."""

    def __init__(self, vf: nn.Module, manifold: ProductManifold, **kwargs) -> None:
        super().__init__()
        self.vf          = vf
        self.manifold    = manifold
        self.geom_trafos = GeomTrafos()
        self.cond_cube   = kwargs.get("cond_cube", False)
        self.no_detach   = kwargs.get("no_detach", False)

    def _prep_x(self, x: Tensor, attn_mask: Tensor) -> tuple[Tensor, Tensor]:
        x_proj = self.manifold.projx(x)
        x_att  = torch.where(attn_mask.unsqueeze(-1), x_proj, x)
        if not self.no_detach:
            x_att.detach_()
        if self.cond_cube:
            x_att = x_att.clone()
            in_p  = _F(x_att).in_p
            in_p.copy_(self.geom_trafos.to_cube(in_p))
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
    ) -> Tensor:
        x_proj, x_att = self._prep_x(x, attn_mask)
        t = torch.atleast_2d(t).expand_as(attn_mask)
        t = torch.where(mask == 1, t, 1.0)  # conditions get t=1
        if getattr(self.vf, "step_cond", False):
            d = torch.zeros_like(t) if d is None else torch.atleast_2d(d).expand_as(attn_mask)
            d = torch.where(mask == 1, d, 0.0)  # conditions are not transported
        v = self.vf(x_att, mask, attn_mask, types, pdgids, t=t, d=d)
        v_proj = self.manifold.proju(x_proj, v)
        return torch.where(attn_mask.unsqueeze(-1), v_proj, v)
