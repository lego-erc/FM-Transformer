"""``CubeTrace``: project positions onto the cube surface along the momentum ray.

Traces along the momentum ray to the exit face of the cube through the point
itself (half-size ``|x|_inf``), so ``t = 0`` when the momentum already points
out of that face. With ``cuboid_dim`` (``d``) the cube is the cuboid of those
edge ratios through the point (half-sizes ``|x/d|_inf * d``). Applied at prep
time (``proj_ray``) and again in ``GenerateOut``. It is not recorded in the
checkpoint, so ``no_raytrace`` models must stub ``gen.proj_ray``.
"""

import torch


class CubeTrace:

    def __init__(self, cuboid_dim=None):
        # edge lengths of a cuboid target; only their ratios are used, so a cube is exactly [1, 1, 1]
        d = torch.tensor(cuboid_dim or [1.0, 1.0, 1.0])
        self.cuboid_dim = d / d.max()

    def __call__(self, *args, **kwargs):
        return self.project_particles_cc(*args, **kwargs)

    def get_time(self, p, x):
        """Ray parameter t at which x + p*t first exits the cuboid of shape ``cuboid_dim`` through x."""
        p_sign = torch.where(p > 0, 1.0, -1.0).to(p.dtype)
        d = self.cuboid_dim.to(x)
        x_abs_max = (x.abs() / d).max(dim=-1, keepdim=True).values * d
        p_ = p_sign * p.abs().clamp(min=1e-8)
        return ((x_abs_max * p_sign - x) / p_).min(-1, keepdim=True).values

    def project_particles_cc(self, cc):
        """Advance the position by p*get_time(p, x); momentum passes through unchanged."""
        p, x = cc.split(3, -1)
        t = self.get_time(p, x)
        return torch.cat((p, x + p * t), -1)