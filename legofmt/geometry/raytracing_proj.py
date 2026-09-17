"""``CubeTrace``: project positions onto the cube surface along the momentum ray.

Traces along the momentum ray to the exit face of the cube through the point
itself (half-size ``|x|_inf``), so ``t = 0`` when the momentum already points
out of that face. Applied at prep time (``proj_ray``) and again in
``GenerateOut``. It is not recorded in the checkpoint, so ``no_raytrace``
models must stub ``gen.proj_ray``.
"""

import torch


class CubeTrace:

    def __call__(self, *args, **kwargs):
        return self.project_particles_cc(*args, **kwargs)

    def get_time(self, p, x):
        """Ray parameter t at which x + p*t first exits the cube of half-size |x|_inf."""
        p_sign = torch.where(p > 0, 1.0, -1.0).to(p.dtype)
        x_abs_max = x.abs().max(dim=-1, keepdim=True).values
        p_ = p_sign * p.abs().clamp(min=1e-8)
        return ((x_abs_max * p_sign - x) / p_).min(-1, keepdim=True).values

    def project_particles_cc(self, cc):
        """Advance the position by p*get_time(p, x); momentum passes through unchanged."""
        p, x = cc.split(3, -1)
        t = self.get_time(p, x)
        return torch.cat((p, x + p * t), -1)