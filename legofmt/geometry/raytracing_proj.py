import torch


class CubeTrace:
    """Utility functions for raytracing and projection onto the cube surface."""

    def __call__(self, *args, **kwargs):
        return self.project_particles_cc(*args, **kwargs)

    def get_time(self, p, x):
        p_sign = torch.where(p > 0, 1.0, -1.0).to(p.dtype)
        x_abs_max = x.abs().max(dim=-1, keepdim=True).values
        p_ = p_sign * p.abs().clamp(min=1e-8)
        return ((x_abs_max * p_sign - x) / p_).min(-1, keepdim=True).values

    def project_particles_cc(self, cc):
        p, x = cc.split(3, -1)
        t = self.get_time(p, x)
        return torch.cat((p, x + p * t), -1)