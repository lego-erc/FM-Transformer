import torch


class GeomTrafos:
    """Utility geometry and coordinate transformations."""

    def _batched(self, coords, c_dims, f_name):
        out = getattr(self, f_name)(coords.unfold(-1, c_dims, c_dims))
        return out.flatten(-2, -1)

    def to_cc(self, sph):
        if sph.shape[-1] != 2:
            return self._batched(sph, 2, "to_cc")
        theta, phi = sph.movedim(-1, 0)
        x = phi.cos() * theta.sin()
        y = phi.sin() * theta.sin()
        z = theta.cos()
        return torch.stack((x, y, z), dim=-1)

    def to_sph(self, cc):
        if cc.shape[-1] != 3:
            return self._batched(cc, 3, "to_sph")
        x, y, z = cc.movedim(-1, 0)
        theta = torch.acos(z.clamp(1e-8 - 1, 1 - 1e-8))
        phi = torch.atan2(y, x)
        return torch.stack((theta, phi), dim=-1)

    def to_cube(self, p_and_x, d=1.0):
        p, x_ = p_and_x.split(3, -1)
        x = x_ / x_.abs().max(-1, keepdim=True).values * d
        return torch.cat((p, x), dim=-1)

    def rotate(self, sph, alpha, beta):
        """Rotate about x-axis by alpha, then about z-axis by beta, in spherical coords."""
        st, ct = sph[..., 0:1].sin(), sph[..., 0:1].cos()
        sp, cp = sph[..., 1:2].sin(), sph[..., 1:2].cos()
        sa, ca = alpha.sin(), alpha.cos()
        theta = torch.acos((st * sp * sa + ct * ca).clamp(-1 + 1e-8, 1 - 1e-8))
        phi = torch.atan2(st * sp * ca - ct * sa, st * cp) + beta
        return torch.cat((theta, phi), dim=-1)

    def sample(self, n: tuple, loc_cc, kappa: torch.Tensor, bs_frac: float = 0.0, tanh_theta: bool = False):
        loc = self.to_sph(loc_cc).expand((*n, -1))
        if bs_frac > 0.0:
            loc = loc.clone()
        loc_theta, loc_phi = loc.split(1, -1)
        if bs_frac > 0.0:
            loc_theta[: round(bs_frac * n[0])] = loc_theta[0] + torch.pi
        if not tanh_theta:
            samples_theta = ((
                2 / kappa * torch.randn(n, device=loc_cc.device) + torch.pi
            ) % (2 * torch.pi) - torch.pi).abs()
        else:
            samples_theta = torch.pi * ((torch.randn(n, device=loc_cc.device) / kappa).tanh()).abs()
        samples_phi = 2 * torch.pi * torch.rand_like(samples_theta)
        sph = torch.stack((samples_theta, samples_phi), dim=-1)
        return self.to_cc(self.rotate(sph, loc_theta, loc_phi + torch.pi / 2))

    def sample_iso(self, n: tuple, mpct, device=None, **kwargs):
        samples_phi = 2 * torch.pi * torch.rand((*n, mpct), device=device)
        samples_cos_theta = 2 * torch.rand((*n, mpct), device=device) - 1
        samples_theta = torch.acos(samples_cos_theta)
        angles = torch.stack((samples_theta, samples_phi), dim=-1)
        return self.to_cc(angles).flatten(-2)
