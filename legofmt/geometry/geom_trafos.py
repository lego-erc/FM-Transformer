"""Cartesian <-> spherical conversion, rotation, the polar zoom, the map onto the cuboid and the chord through it.

```
cartesian   x = sin(theta) cos(phi),   y = sin(theta) sin(phi),   z = cos(theta)
spherical   theta = atan2(hypot(x, y), z),   phi = atan2(y, x)

rotated     about x by alpha   z' = sin(theta) sin(phi) sin(alpha) + cos(theta) cos(alpha),   theta' = acos(z')
                               y' = sin(theta) sin(phi) cos(alpha) - cos(theta) sin(alpha),   x' = x
            then about z by beta                                                             phi' = atan2(y', x') + beta

on_cuboid   x / |x|_inf * d                        onto the surface of the cuboid with half-edges d

zoom        theta' = mix theta + (1 - mix) pi log1p(theta / theta_s) / log1p(pi / theta_s)
```

The formulas take one tensor per component. ``GeomTrafos`` applies them to components stacked on the last dimension,
``(..., 2)`` / ``(..., 3)``, and ``to_cuboid`` only to the position half of a ``(mom, pos)`` 6-vector (the momentum
passes through).
"""

import torch
import torch.nn.functional as F

from legofmt.numerics import detach_at_pole, safe_acos, safe_atan2, safe_div


def cartesian(theta, phi):
    """Converts spherical (theta, phi) to the unit vector (x, y, z)."""
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return x, y, z


def spherical(x, y, z):
    """Converts the unit vector (x, y, z) to spherical (theta, phi)."""
    x, y  = detach_at_pole(x, y)   # with grad only: the backward overflows at the pole
    theta = torch.atan2(torch.hypot(x, y), z)
    phi   = torch.atan2(y, x)
    return theta, phi


def rotated(theta, phi, alpha, beta):
    """Rotates the direction (theta, phi) about the x-axis by alpha, then about the z-axis by beta."""
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    sin_p, cos_p = torch.sin(phi), torch.cos(phi)
    sin_a, cos_a = torch.sin(alpha), torch.cos(alpha)
    z     = sin_t * sin_p * sin_a + cos_t * cos_a   # about x by alpha
    theta = safe_acos(z)
    y     = sin_t * sin_p * cos_a - cos_t * sin_a
    x     = sin_t * cos_p
    phi   = safe_atan2(y, x) + beta                 # then about z by beta
    return theta, phi


def on_cuboid(x, d):
    """Scales the directions ``x`` onto the surface of the cuboid with half-edges ``d``: ``x / |x|_inf * d``."""
    norm = torch.abs(x).amax(-1, keepdim=True).clamp(min=1e-8)   # |x|_inf, held at 1e-8 or above
    return x / norm * d


def zoom_angle(theta, theta_s, mix):
    """Zooms the polar angle ``theta``: magnifies the angles below ~``theta_s``, keeps 0 and pi.

    ```
    theta' = mix theta + (1 - mix) log1p(theta / theta_s) / k,   k = log1p(pi / theta_s) / pi
    ```
    """
    k = torch.log1p(_float64(torch.pi / theta_s)) / torch.pi
    return (
        mix * theta
        + (1 - mix) * torch.log1p(theta / theta_s) / k
    )


def unzoom_angle(theta_z, theta_s, mix):
    """Inverts ``zoom_angle``: in closed form for mix = 0, otherwise by 40 Newton steps that start from the closed form.

    ```
    theta = theta_s expm1(k theta_z)                                                    mix = 0
    theta = theta - (zoom_angle(theta) - theta_z) / (mix + (1 - mix) / (k (theta_s + theta)))    one Newton step
    ```
    """
    k     = torch.log1p(_float64(torch.pi / theta_s)) / torch.pi
    theta = theta_s * torch.expm1(theta_z * k)
    # zoom_angle is increasing and concave, so after the first step the iterates rise monotonically to the root
    for _ in range(40 if mix else 0):
        f     = zoom_angle(theta, theta_s, mix) - theta_z
        theta = (theta - f / (mix + (1 - mix) / (k * (theta_s + theta)))).clamp(0.0, torch.pi)
    return theta


def polar_about(v, u):
    """Splits the unit vectors ``v`` about the unit axes ``u``.

    Returns ``(theta, perp, sin)``: the polar angle (..., 1), the part of ``v`` perpendicular to ``u`` (..., 3), and
    that part's length ``sin(theta)`` (..., 1).
    """
    cos   = (v * u).sum(-1, keepdim=True).clamp(-1.0, 1.0)
    perp  = v - cos * u
    sin   = perp.norm(dim=-1, keepdim=True)
    theta = torch.atan2(sin, cos)
    return theta, perp, sin


def from_polar_about(theta, u, perp, sin):
    """Returns the unit vectors at the polar angle ``theta`` about ``u``, in the plane of ``u`` and ``perp``."""
    return (
        torch.cos(theta) * u
        + torch.sin(theta) * perp / sin.clamp(min=1e-30)   # sin = 0: v on +-u, which the caller keeps
    )


def zoom_polar(v: torch.Tensor, u: torch.Tensor, theta_s: float, inverse: bool = False,
               mix: float = 0.0) -> torch.Tensor:
    """Zooms the polar angle of the unit vectors ``v`` about the unit axes ``u`` and keeps the azimuth.

    ``zoom_angle`` on the polar angle; ``inverse``: ``unzoom_angle``. A bijection of the sphere that magnifies the
    deflections below ~``theta_s``; ``mix > 0`` keeps the large angles from being squeezed. A ``v`` on ``+-u`` is
    returned as it is.
    """
    theta, perp, sin = polar_about(v, u)
    theta = unzoom_angle(theta, theta_s, mix) if inverse else zoom_angle(theta, theta_s, mix)
    out   = from_polar_about(theta, u, perp, sin)
    return torch.where(sin > 0, out, v)


def cuboid_chord(u: torch.Tensor, pos: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Computes the length of the line along ``u`` through a cuboid, in units of its longest edge.

    ``u`` (..., 3) is the line's unit direction. ``d`` (3,) are the half-edges, the longest one 1. ``pos`` (..., 3) is
    the direction the tokens store for a point ``x`` on the surface, ``normalize(x / d)``. Returns ``(..., 1)``.

    ```
    line       x(s) = p + s u
    per axis   it leaves through the face at  sign(u) d  after  s = (sign(u) d - p) / u    (ahead)
               and, run backwards, through   -sign(u) d  after  s = (sign(u) d + p) / u    (behind)
    chord      (nearest face ahead + nearest face behind) / 2
    ```

    Counting both sides gives the same chord whether the entry or the exit point is stored.
    """
    d      = d.to(u)
    p      = on_cuboid(pos, d)                                        # the surface point
    # an axis the line is parallel to gets 4.0, beyond any face, so it never ends the chord
    moving = torch.abs(u) > 1e-6
    ahead  = safe_div(torch.sign(u) * d - p, u, moving, otherwise=4.0)
    behind = safe_div(p + torch.sign(u) * d, u, moving, otherwise=4.0)
    chord  = (ahead.amin(-1) + behind.amin(-1)).clamp(0.0, 3.5) / 2   # 3.5: above a cube's diagonal
    return chord.unsqueeze(-1)


class GeomTrafos:
    """The coordinate maps between cartesian unit vectors and spherical (theta, phi), and onto the cuboid."""

    def to_cc(self, sph):
        """Converts spherical (theta, phi) to cartesian unit vectors: ``cartesian`` per trailing pair.

        ``(..., 2k) -> (..., 3k)``.
        """
        return _per_group(cartesian, sph, 2)

    def to_sph(self, cc):
        """Converts cartesian vectors to spherical (theta, phi): ``spherical`` per trailing triple, normalised.

        ``(..., 3k) -> (..., 2k)``.
        """
        return _per_group(spherical, cc, 3, unit=True)

    def to_cuboid(self, p_and_x, d=1.0):
        """Maps the position direction of ``p_and_x`` (..., 6) onto the surface of the cuboid with half-edges ``d``.

        ``on_cuboid`` on the position half; the momentum half passes through. ``d`` is a scalar (a cube) or the
        per-axis half-edges. With the cuboid's own half-edges it inverts the direction ``DataPrep`` stores,
        ``normalize(x / cuboid_dim)``.
        """
        return on_position(p_and_x, lambda p, x: on_cuboid(x, d))

    def rotate(self, sph, alpha, beta):
        """Rotates spherical (theta, phi) about the x-axis by alpha, then about the z-axis by beta: ``rotated``."""
        theta, phi = rotated(sph[..., 0:1], sph[..., 1:2], alpha, beta)
        return torch.cat((theta, phi), dim=-1)

    def direction_about(self, axis, theta, phi):
        """Returns the unit vectors at polar angle ``theta`` and azimuth ``phi`` about the unit vectors ``axis``.

        The direction (theta, phi) about z, rotated onto ``axis``: about x by the axis' theta, then about z by its
        phi + pi/2. One per element of ``theta``; ``axis`` broadcasts against them.
        """
        about_z, axis_theta, axis_phi = _against_axis(theta, phi, self.to_sph(axis))
        return self.to_cc(self.rotate(about_z, axis_theta, axis_phi + torch.pi / 2))


# ---- plumbing --------------------------------------------------------------------------------------------------------

def on_position(p_and_x, new_position):
    """Replaces the position half ``x`` of ``(mom, pos)`` 6-vectors by ``new_position(p, x)``; ``p`` passes through."""
    p, x = p_and_x.split(3, dim=-1)
    return torch.cat((p, new_position(p, x)), dim=-1)


def _per_group(formula, packed, n, unit=False):
    """Applies ``formula`` to each trailing group of ``n`` components of ``packed``, one tensor per component.

    The formula's ``m`` results per group are stacked the same way: ``(..., k n) -> (..., k m)``. ``unit`` normalises
    each group first. The components are handed over as views of ``packed``, not as copies: CPU's atan2 picks its
    kernel by the input's strides, so a contiguous copy can change the last bits of the result.
    """
    one    = packed.shape[-1] == n
    groups = packed if one else packed.unflatten(-1, (-1, n))
    if unit:
        groups = F.normalize(groups, dim=-1)
    out = torch.stack(formula(*groups.unbind(-1)), dim=-1)
    return out if one else out.flatten(-2, -1)


def _against_axis(theta, phi, axis_sph):
    """Stacks ``(theta, phi)`` to (..., 2) and expands the axis' spherical angles to theta's shape, each (..., 1)."""
    axis_theta, axis_phi = axis_sph.expand((*theta.shape, -1)).split(1, dim=-1)
    return torch.stack((theta, phi), dim=-1), axis_theta, axis_phi


def _float64(v):
    """Converts the number ``v`` to a float64 tensor."""
    return torch.tensor(v, dtype=torch.float64)
