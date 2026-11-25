"""``CubeTrace``: moves a position along its momentum ray onto the cuboid's surface.

```
x' = x + p t,   t = min over the axes of (h sign(p) - x) / p,   h = |x / d|_inf d
```

The cuboid is the one with the edge ratios ``d`` (``cuboid_dim``; a cube without it) whose surface holds the point
``x`` itself: its half-edges are ``h``. The ray is traced to the face it leaves that cuboid through, so ``t = 0``
when the momentum already points out of the face ``x`` lies on.

``DataPrep`` applies it to the incoming position when it writes a dataset (``proj_ray``). ``GenerateOut`` applies it
to the incoming position of every call that is not ``prepped``, whatever ``proj_ray`` says: for a model trained on
data that was not traced, ``gen.proj_ray`` has to be stubbed.

``CubeTrace.exit_across_edge``: ``GenerateOut`` moves each generated exit point that lies just across an edge from the
face its momentum leaves through onto that face.
"""

import torch

from legofmt.geometry.geom_trafos import on_position

EDGE_BAND = 0.95   # exit_across_edge: an exit point is near an edge when its second |component| exceeds this share
                   # of its largest


def exit_time(p, x, d):
    """Returns the ray parameter ``t`` (..., 1) at which ``x + p t`` leaves the cuboid whose surface holds ``x``.

    ```
    t = min over the axes of (h sign(p) - x) / p,   h = |x / d|_inf d   that cuboid's half-edges; d: its edge ratios
    ```
    """
    sign = torch.where(p > 0, 1.0, -1.0).to(p.dtype)
    h    = (torch.abs(x) / d).amax(-1, keepdim=True) * d
    p    = sign * torch.abs(p).clamp(min=1e-8)   # |p| held at 1e-8 or above, its sign kept
    t    = (h * sign - x) / p
    return t.amin(-1, keepdim=True)


class CubeTrace:
    """A callable: ``(mom, pos)`` 6-vectors -> the same with the position advanced to the cuboid's surface.

    Args:
        cuboid_dim: the cuboid's edge lengths; only their ratios are used. None: a cube.
    """

    def __init__(self, cuboid_dim=None):
        d = torch.tensor(cuboid_dim or [1.0, 1.0, 1.0])
        self.cuboid_dim = d / d.max()   # the edge ratios: the longest edge is 1

    def __call__(self, *args, **kwargs):
        return self.project_particles_cc(*args, **kwargs)

    def get_time(self, p, x):
        """Returns ``exit_time(p, x, cuboid_dim)``: the ray parameter ``t`` at which ``x + p t`` leaves the cuboid."""
        return exit_time(p, x, self.cuboid_dim.to(x))

    def project_particles_cc(self, cc):
        """Advances the position ``x`` of ``cc`` (..., 6) to ``x + p t``, ``t = get_time(p, x)``; ``p`` is kept."""
        return on_position(cc, lambda p, x: x + p * self.get_time(p, x))

    def exit_across_edge(self, p, u):
        """Returns the exit points ``u`` with each one that lies just across an edge mirrored over it, out of place.

        ```
        a, b    the axes of u's largest and second-largest |component|: the face u lies on, and the neighbouring
                face across the nearest edge
        moved   p_a sign(u_a) < 0              p points into the face u lies on
                |u_b| > EDGE_BAND |u_a|        u lies near the edge to the neighbouring face
                p_b sign(u_b) > 0              p leaves through the neighbouring face
        u'      |u_a| and |u_b| swapped, signs kept: the same distance to the edge, |u'| = |u|
        ```

        Exit point and momentum are separate channels of the flow, so it can put a particle on one face moving out
        through the next; Geant4 never does.

        ``p`` and ``u`` are (..., 3): momentum directions, and exit points as the direction of x / cuboid_dim (on
        which every cuboid is the unit cube).
        """
        mag, ax = u.abs().sort(-1, descending=True)
        a, b    = ax[..., :1], ax[..., 1:2]
        sa, sb  = u.gather(-1, a).sign(), u.gather(-1, b).sign()
        moved   = (p.gather(-1, a) * sa < 0) & (mag[..., 1:2] > EDGE_BAND * mag[..., :1]) & (p.gather(-1, b) * sb > 0)
        mirror  = u.scatter(-1, a, sa * mag[..., 1:2]).scatter(-1, b, sb * mag[..., :1])
        return torch.where(moved, mirror, u)
