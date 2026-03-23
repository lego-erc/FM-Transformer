import torch

from legofmt.geometry.vmf_sampling import VMF


class CubeTrace:
    """Utility functions for raytracing and projection onto the cube surface."""

    def __init__(self):
        self.vmf_utils = VMF()

    def __call__(self, *args, **kwargs):
        return self.project_particles_cc(*args, **kwargs)

    def get_time(self, p, x):
        p_sign = p.sgn()
        p_sign -= p_sign.eq(0).int()
        x_abs_max = x.abs().max(dim=-1, keepdim=True).values
        p_ = p_sign * p.abs().clamp(min=1e-8)
        t_surface = (x_abs_max * p_sign - x) / p_
        return t_surface.min(-1, keepdim=True).values

    def project_particles_cc(self, cc):
        p, x = cc.split(3, -1)
        t = self.get_time(p, x)
        return torch.cat((p, x + p * t), -1)

    def iso_half_sphere(self, p_x: torch.Tensor) -> torch.Tensor:
        """Project samples to outward half of Sphere.

        Projects a point x and a momentum p (both in R^3) to the outward half of the unit sphere
        centered at the origin. The sign of the maximum absolute coordinate of x is used to determine
        the sign of the corresponding coordinate of p.

        Args:
            p_x: Tensor of shape (..., 6) containing points x and momenta p in R^3.

        Returns:
            Tensor of shape (..., 6) containing the projected points and momenta.
        """
        p, x = p_x.split(3, -1)
        coord_max = x.abs().argmax(dim=-1)
        coord_max_sgn = torch.sign(x[..., coord_max])
        p[..., coord_max] = coord_max_sgn * p[..., coord_max].abs()
        return torch.cat((p, x), dim=-1)
