"""Sampler options superseded by the recipe's draw (:mod:`legofmt.base.gen_base`), kept so older configs and
checkpoints still run unchanged:

    scale_dist   ``trunc_norm`` (the default of a config without the key), ``uniform``, ``logit_norm``
    base_dist    ``iso`` (isotropic momentum and position directions), ``iso_pos`` (isotropic positions only)
    tanh_theta   off: a wrapped-normal polar angle; on without a shape exponent (configs without ``ang_shape``)
    bs_frac      > 0: a share of the events drawn about the antipode
    iso          the fully isotropic start of inverse-mask events (``mask_conf.p_forward`` < 1)

``GenerateBase`` derives from ``LegacySampler``: its ``__call__`` hands every draw that is not the recipe's to the one
here, and ``rd_scale`` / ``poles`` / ``iso_dirs`` / ``iso`` stay reachable on it (``lego_eval`` tests for ``iso_dirs``).
The methods read the parameters ``GenerateBase`` holds (``kappa``, ``ang_pow``, ``sm_scale``, ``sm_pow``).
"""

import torch
import torch.nn.functional as F

from legofmt import rng as R
from legofmt.base.base_dists import Angles, Energy
from legofmt.geometry.geom_trafos import GeomTrafos

_GEOM = GeomTrafos()


def sample_dirs(
    n: tuple, loc_cc, kappa: torch.Tensor,
    bs_frac: float = 0.0, tanh_theta: bool = False, ang_pow=None,
    rng=None, elem=None, stream: int = 0,
):
    """Directions concentrated (kappa) around loc_cc, wrapped-normal or
    tanh-normal theta; bs_frac sends the first round(bs_frac*B) events --
    all their particles -- to the antipode. ``ang_pow`` raises the
    tanh-normal theta to a power, so kappa sets the angular scale and
    ang_pow the shape: one moment each, instead of one for both. ``rng``
    (legofmt.rng.EventRNG) draws theta / phi from ``stream`` / ``stream + 1``
    at the per-event element ids ``elem`` (shape ``n``)."""
    normal = lambda: torch.randn(n, device=loc_cc.device) if rng is None else rng.normal(stream, elem).to(loc_cc)
    loc = _GEOM.to_sph(loc_cc).expand((*n, -1))
    if bs_frac > 0.0:
        loc = loc.clone()
        loc[:round(bs_frac * n[0]), ..., 0] += torch.pi
    loc_theta, loc_phi = loc.split(1, -1)
    if tanh_theta and ang_pow is not None:
        samples_theta = torch.pi * Angles(kappa, ang_pow).sample(normal())
    elif tanh_theta:   # no shape exponent (configs without ang_shape)
        samples_theta = torch.pi * (normal() / kappa).tanh().abs()
    else:
        z = 2 / kappa * normal()
        samples_theta = ((z + torch.pi) % (2 * torch.pi) - torch.pi).abs()   # wrapped normal
    samples_phi = 2 * torch.pi * (torch.rand_like(samples_theta) if rng is None
                                  else rng.uniform(stream + 1, elem).to(samples_theta))
    sph = torch.stack((samples_theta, samples_phi), dim=-1)
    return _GEOM.to_cc(_GEOM.rotate(sph, loc_theta, loc_phi + torch.pi / 2))


def sample_iso(n: tuple, mpct, device=None, rng=None, elem=None, stream: int = 0):
    """mpct isotropic unit directions per event; ``rng`` as in ``sample_dirs``."""
    ids = None if rng is None else elem[..., None] * mpct + torch.arange(mpct, device=elem.device)
    rand = lambda s: (torch.rand((*n, mpct), device=device) if rng is None
                      else rng.uniform(stream + s, ids).float().to(device))
    phi = 2 * torch.pi * rand(0)
    theta = torch.acos(2 * rand(1) - 1)
    return _GEOM.to_cc(torch.stack((theta, phi), dim=-1)).flatten(-2)


class LegacySampler:
    """The sampler's options and every draw but the recipe's."""

    def __init__(self, base_conf: dict):
        self.base_dist = base_conf.get("base_dist", "poles")
        self.tanh_theta = base_conf.get("tanh_theta", False)
        self.bs_frac = base_conf.get("bs_frac", 0.0)
        self.scale_dist = base_conf.get("scale_dist", "trunc_norm")

        if self.base_dist not in ("poles", "iso", "iso_pos"):
            raise ValueError(f"base_dist must be 'poles', 'iso', or 'iso_pos', got {self.base_dist!r}")

    def is_recipe(self) -> bool:
        """Returns whether the options are the recipe's, the draw ``GenerateBase.__call__`` spells out."""
        return (self.scale_dist == "sm_norm" and self.base_dist == "poles" and bool(self.tanh_theta)
                and self.bs_frac == 0 and self.ang_pow is not None)

    def __call__(self, ds_t, rng=None):
        """Samples ``x_0`` for a batch with any option but the recipe's.

        Joins the data's prefix and incoming token with a draw per outgoing slot (``poles`` / ``iso_dirs``), then
        writes ``z ~ N(0, 1)`` on the E_dep cell. Arguments and result as in ``GenerateBase.__call__``.
        """
        f, shape = ds_t.f, ds_t.f.out_pid.shape
        elem = None if rng is None else R.species_rank_ids(f.out_pid.long())
        if self.base_dist == "iso":
            tokens = self.iso_dirs(shape, incoming_rt=f.in_cc, rng=rng, elem=elem)
        else:
            tokens = self.poles(shape, incoming_rt=f.in_cc, iso_pos=(self.base_dist == "iso_pos"), rng=rng, elem=elem)
        x0 = torch.cat((f.non_cc, tokens), dim=1)
        x0[:, 1, 0] = R.Noise(x0.shape[:1], x0.device, rng).normal(R.EDEP).to(x0)
        return x0

    def rd_scale(self, shape, e_in, rng=None, elem=None):
        """Energy prior for the outgoing slots per scale_dist; returns (B, n_out, 1)."""
        u = None if rng is None else rng.uniform(R.BASE_E, elem)[..., None]   # float64 until the end
        normal = lambda: torch.randn((*shape, 1), device=e_in.device) if u is None else torch.special.ndtri(u)
        if self.scale_dist == "trunc_norm":
            if u is None:
                u = torch.nn.init.trunc_normal_(e_in.new_empty((*shape, 1)), std=1.0, a=-1.0, b=0.0) + 1.0
            else:  # N(0, 1) truncated to [-1, 0] by its inverse CDF
                lo = torch.special.ndtr(torch.tensor(-1.0, dtype=u.dtype))
                u = torch.special.ndtri(lo + u * (0.5 - lo)) + 1.0
        elif self.scale_dist == "uniform":
            u = torch.rand((*shape, 1), device=e_in.device) if u is None else u
        elif self.scale_dist == "sm_norm":
            u = Energy(self.sm_scale, self.sm_pow).sample(normal()[..., 0])[..., None]   # parameters: per slot
        elif self.scale_dist == "logit_norm":
            u = torch.sigmoid(normal())  # smooth, symmetric on (0,1)
        else:
            raise ValueError("Unknown scale_dist")
        return e_in.view(-1, 1, 1) * u.to(e_in)

    def poles(self, shape, incoming_rt, iso_pos=False, rng=None, elem=None, **kwargs):
        p_cc = F.normalize(incoming_rt[..., 1:4], dim=-1)
        loc_cc = incoming_rt[..., -3:]
        e_sc = self.rd_scale(shape, torch.ones_like(incoming_rt[..., 0:1]), rng=rng, elem=elem)
        if iso_pos:
            x = sample_iso(shape, 1, device=incoming_rt.device, rng=rng, elem=elem, stream=R.BASE_POS)
        else:
            x = sample_dirs(
                shape, loc_cc, self.kappa, self.bs_frac, self.tanh_theta, self.ang_pow,
                rng=rng, elem=elem, stream=R.BASE_POS)
        p_ = sample_dirs(shape, p_cc, self.kappa, 0.0, self.tanh_theta, self.ang_pow,
                         rng=rng, elem=elem, stream=R.BASE_MOM)
        base = torch.cat((e_sc, p_, x), dim=-1)
        return torch.cat((incoming_rt, base), dim=1)

    def iso_dirs(self, shape, incoming_rt, rng=None, elem=None, **kwargs):
        # naive base: isotropic momentum/position directions; energy scale from
        # rd_scale (per scale_dist), E_dep token from __call__.
        e_sc = self.rd_scale(shape, torch.ones_like(incoming_rt[..., 0:1]), rng=rng, elem=elem)
        p_ = sample_iso(shape, 1, device=incoming_rt.device, rng=rng, elem=elem, stream=R.BASE_MOM)
        x  = sample_iso(shape, 1, device=incoming_rt.device, rng=rng, elem=elem, stream=R.BASE_POS)
        base = torch.cat((e_sc, p_, x), dim=-1)
        return torch.cat((incoming_rt, base), dim=1)

    @torch.no_grad()
    def iso(self, shape, device, rng=None):
        """Fully isotropic x_0 for inverse-mask events, on all tokens; unrelated
        to base_dist='iso', which routes to iso_dirs."""
        elem = None if rng is None else torch.arange(shape[1], device=device).expand(shape)
        e = (torch.rand((*shape, 1), device=device) if rng is None
             else rng.uniform(R.ISO_E, elem)[..., None].float().to(device))
        p = sample_iso(shape, 1, device=device, rng=rng, elem=elem, stream=R.ISO_MOM)
        x = sample_iso(shape, 1, device=device, rng=rng, elem=elem, stream=R.ISO_POS)
        return torch.cat((e, p, x), dim=-1)
