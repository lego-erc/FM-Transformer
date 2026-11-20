"""``CubeTrace``: project positions onto the cube surface along the momentum ray.

Traces along the momentum ray to the exit face of the cube through the point
itself (half-size ``|x|_inf``), so ``t = 0`` when the momentum already points
out of that face. With ``cuboid_dim`` (``d``) the cube is the cuboid of those
edge ratios through the point (half-sizes ``|x/d|_inf * d``). Applied at prep
time (``proj_ray``) and again in ``GenerateOut``. It is not recorded in the
checkpoint, so ``no_raytrace`` models must stub ``gen.proj_ray``.

```
x' = x + p t,   t = min over the axes of (h sign(p) - x) / p,   h = |x / d|_inf d
```
"""

import torch

from legofmt.geometry.geom_trafos import on_position


def exit_time(p, x, d):
    """Returns the ray parameter t at which x + p t first leaves, through x, the cuboid of edge ratios ``d``.

    ```
    t = min over the axes of (h sign(p) - x) / p,   h = |x / d|_inf d   the half-sizes of the cuboid through x
    ```
    """
    sign = torch.where(p > 0, 1.0, -1.0).to(p.dtype)
    h    = (torch.abs(x) / d).max(dim=-1, keepdim=True).values * d
    p    = sign * torch.abs(p).clamp(min=1e-8)   # |p| held at 1e-8 or above, its sign kept
    t    = (h * sign - x) / p
    return t.min(-1, keepdim=True).values


class CubeTrace:
    """Callable: ``(mom, pos)`` 6-vectors -> the same with the position advanced to the cuboid's surface.

    Args:
        cuboid_dim: the cuboid's edge lengths; only their ratios are used. None: a cube.
    """

    def __init__(self, cuboid_dim=None):
        # edge lengths of a cuboid target; only their ratios are used, so a cube is exactly [1, 1, 1]
        d = torch.tensor(cuboid_dim or [1.0, 1.0, 1.0])
        self.cuboid_dim = d / d.max()

    def __call__(self, *args, **kwargs):
        return self.project_particles_cc(*args, **kwargs)

    def get_time(self, p, x):
        """Returns the ray parameter t at which x + p*t first exits the cuboid of shape ``cuboid_dim`` through x."""
        return exit_time(p, x, self.cuboid_dim.to(x))

    def project_particles_cc(self, cc):
        """Advances the position x to x + p t, t = ``get_time(p, x)``; momentum passes through unchanged."""
        return on_position(cc, lambda p, x: x + p * self.get_time(p, x))
