"""Cartesian <-> spherical conversion, direction sampling, and the box map.

``to_cuboid`` is an L-infinity normalisation (sphere -> unit cube surface) scaled per axis to the box of half-widths
``d``; it rescales only the position half of a ``(mom, pos)`` 6-vector, and momentum passes through.
"""

import torch
import torch.nn.functional as F


def zoom_polar(v: torch.Tensor, u: torch.Tensor, theta_s: float, inverse: bool = False,
               mix: float = 0.0) -> torch.Tensor:
    """Remap the polar angle of unit vectors ``v`` about unit axes ``u`` to
    theta' = mix * theta + (1 - mix) * pi * log1p(theta / theta_s) / log1p(pi / theta_s), keeping the azimuth
    (``inverse`` undoes it, by Newton for mix > 0): a bijection of the sphere that magnifies deflections below
    ~theta_s; ``mix`` keeps the large angles from being squeezed. ``v`` on +-u is left as is."""
    cos = (v * u).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    perp = v - cos * u
    sin = perp.norm(dim=-1, keepdim=True)
    theta = torch.atan2(sin, cos)
    k = torch.log1p(torch.tensor(torch.pi / theta_s, dtype=torch.float64)) / torch.pi
    if not inverse:
        new = mix * theta + (1 - mix) * torch.log1p(theta / theta_s) / k
    else:
        new = theta_s * torch.expm1(theta * k)
        for _ in range(40 if mix else 0):   # increasing and concave: from the first step on, monotone from below
            f = mix * new + (1 - mix) * torch.log1p(new / theta_s) / k - theta
            new = (new - f / (mix + (1 - mix) / (k * (theta_s + new)))).clamp(0.0, torch.pi)
    out = torch.cos(new) * u + torch.sin(new) * perp / sin.clamp_min(1e-30)
    return torch.where(sin > 0, out, v)


class GeomTrafos:

    def _batched(self, coords, c_dims, f_name):
        out = getattr(self, f_name)(coords.unfold(-1, c_dims, c_dims))
        return out.flatten(-2, -1)

    def to_cc(self, sph):
        """Spherical (theta, phi) -> cartesian unit vector, batched over trailing pairs."""
        if sph.shape[-1] != 2:
            return self._batched(sph, 2, "to_cc")
        theta, phi = sph.movedim(-1, 0)
        return torch.stack((phi.cos() * theta.sin(), phi.sin() * theta.sin(), theta.cos()), dim=-1)

    def to_sph(self, cc):
        """Cartesian unit vector -> spherical (theta, phi), batched over trailing triples."""
        if cc.shape[-1] != 3:
            return self._batched(cc, 3, "to_sph")
        cc = F.normalize(cc, dim=-1)
        x, y, z = cc.movedim(-1, 0)
        if cc.requires_grad:   # on (or subnormally close to) the z axis, where a gun along z sits, hypot / atan2 have
            # no usable gradient: detach the inputs there (values unchanged; sampling keeps CPU's strided-input path)
            pole = x * x + y * y < torch.finfo(x.dtype).tiny
            x, y = torch.where(pole, x.detach(), x), torch.where(pole, y.detach(), y)
        theta = torch.atan2(torch.hypot(x, y), z)
        phi = torch.atan2(y, x)
        return torch.stack((theta, phi), dim=-1)

    def to_cuboid(self, p_and_x, d=1.0):
        """d: a scalar (a cube) or per-axis half-widths; for the box's half-edges it inverts DataPrep's stored
        direction of x / cuboid_dim."""
        p, x_ = p_and_x.split(3, -1)
        x = x_ / x_.abs().max(-1, keepdim=True).values.clamp_min(1e-8) * d
        return torch.cat((p, x), dim=-1)

    def rotate(self, sph, alpha, beta):
        """Rotate about x-axis by alpha, then about z-axis by beta, in spherical coords."""
        st, ct = sph[..., 0:1].sin(), sph[..., 0:1].cos()
        sp, cp = sph[..., 1:2].sin(), sph[..., 1:2].cos()
        sa, ca = alpha.sin(), alpha.cos()
        c = (st * sp * sa + ct * ca).clamp(-1 + 1e-8, 1 - 1e-8)   # in fp32 the bound rounds to exactly +-1
        theta = torch.acos(torch.where(c.abs() >= 1, c.detach(), c))   # acos has an infinite slope at +-1
        y, x = st * sp * ca - ct * sa, st * cp
        # within ~1e-19 of the pole x^2 + y^2 is subnormal and atan2's backward overflows: no gradient there
        pole = x * x + y * y < torch.finfo(x.dtype).tiny
        phi = torch.atan2(torch.where(pole, y.detach(), y), torch.where(pole, x.detach(), x)) + beta
        return torch.cat((theta, phi), dim=-1)

    def sample(
        self, n: tuple, loc_cc, kappa: torch.Tensor,
        bs_frac: float = 0.0, tanh_theta: bool = False, ang_pow=None,
    ):
        """Directions concentrated (kappa) around loc_cc, wrapped-normal or
        tanh-normal theta; bs_frac sends the first round(bs_frac*B) events --
        all their particles -- to the antipode. ``ang_pow`` raises the
        tanh-normal theta to a power, so kappa sets the angular scale and
        ang_pow the shape: one moment each, instead of one for both."""
        loc = self.to_sph(loc_cc).expand((*n, -1))
        if bs_frac > 0.0:
            loc = loc.clone()
            loc[:round(bs_frac * n[0]), ..., 0] += torch.pi
        loc_theta, loc_phi = loc.split(1, -1)
        if tanh_theta:
            t = (torch.randn(n, device=loc_cc.device) / kappa).tanh().abs()
            if ang_pow is not None:   # a CUDA normal draw is exactly 0 ~once in 2**24, where t^p has an infinite slope
                pos = t > 0
                t = torch.where(pos, torch.where(pos, t, torch.ones_like(t)).pow(ang_pow), torch.zeros_like(t))
            samples_theta = torch.pi * t
        else:
            z = 2 / kappa * torch.randn(n, device=loc_cc.device)
            samples_theta = ((z + torch.pi) % (2 * torch.pi) - torch.pi).abs()   # wrapped normal
        samples_phi = 2 * torch.pi * torch.rand_like(samples_theta)
        sph = torch.stack((samples_theta, samples_phi), dim=-1)
        return self.to_cc(self.rotate(sph, loc_theta, loc_phi + torch.pi / 2))

    def sample_iso(self, n: tuple, mpct, device=None):
        """mpct isotropic unit directions per event."""
        phi = 2 * torch.pi * torch.rand((*n, mpct), device=device)
        theta = torch.acos(2 * torch.rand((*n, mpct), device=device) - 1)
        return self.to_cc(torch.stack((theta, phi), dim=-1)).flatten(-2)
