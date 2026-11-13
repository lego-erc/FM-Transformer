"""Cartesian <-> spherical conversion, rotation, the polar zoom, the box map and the chord through the box.

``to_cuboid`` is an L-infinity normalisation (sphere -> unit cube surface) scaled per axis to the box of half-widths
``d``; it rescales only the position half of a ``(mom, pos)`` 6-vector, and momentum passes through.
"""

import torch
import torch.nn.functional as F

from legofmt.numerics import detach_at_pole, safe_acos, safe_atan2, safe_div


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


def cuboid_chord(u: torch.Tensor, pos: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Length ``(..., 1)`` of the line along ``u`` through a cuboid, in longest-edge lengths. ``d`` are the half-edges
    with the longest one 1, ``pos`` the stored direction of a point on the surface, dir(x / d).

    ```
    line       x(s) = p + s u
    per axis   it leaves through the face at  sign(u) d  after  s = (sign(u) d - p) / u    (ahead)
               and, run backwards, through   -sign(u) d  after  s = (sign(u) d + p) / u    (behind)
    chord      (nearest face ahead + nearest face behind) / 2
    ```

    Counting both sides gives the same chord whether the entry or the exit point is stored."""
    d      = d.to(u)
    p      = pos / torch.abs(pos).amax(-1, keepdim=True).clamp_min(1e-8) * d   # the surface point
    # an axis the line is parallel to gets 4.0, beyond any face, so it never ends the chord
    moving = torch.abs(u) > 1e-6
    ahead  = safe_div(torch.sign(u) * d - p, u, moving, otherwise=4.0)
    behind = safe_div(p + torch.sign(u) * d, u, moving, otherwise=4.0)
    return ((ahead.amin(-1) + behind.amin(-1)).clamp(0.0, 3.5) / 2).unsqueeze(-1)   # 3.5: above a cube's diagonal


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
        if cc.requires_grad:   # a gun along z sits on the pole; sampling (no grad) keeps CPU's strided-input path
            x, y = detach_at_pole(x, y)
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
        theta = safe_acos(st * sp * sa + ct * ca)
        y, x = st * sp * ca - ct * sa, st * cp
        phi = safe_atan2(y, x) + beta
        return torch.cat((theta, phi), dim=-1)

    def direction_about(self, axis, theta, phi):
        """Unit vectors at polar angle ``theta`` and azimuth ``phi`` about the unit vectors ``axis``, one per element
        of ``theta``; ``axis`` broadcasts against them."""
        axis_theta, axis_phi = self.to_sph(axis).expand((*theta.shape, -1)).split(1, -1)
        about_z = torch.stack((theta, phi), dim=-1)
        return self.to_cc(self.rotate(about_z, axis_theta, axis_phi + torch.pi / 2))
