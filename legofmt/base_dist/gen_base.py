"""Samplers for the flow's base distribution -- the ``x_0`` it integrates from.

``base_dist`` picks the direction prior (``poles`` concentrates around the
incoming direction; ``iso`` / ``iso_pos`` do not), ``scale_dist`` the energy
prior, and ``insert_add`` fills the edep row. The sampler must match the
manifold the flow integrates on.
"""

import torch
import torch.nn.functional as F

from legofmt import rng as R
from legofmt.geometry.geom_trafos import GeomTrafos


class GenerateBase:
    """Samples the flow's base distribution: directions from base_dist
    (poles/iso/iso_pos), the energy prior from scale_dist, the E_dep row from
    insert_add."""

    def __init__(self, config: dict):
        self.geom_trafos = GeomTrafos()
        base_conf = config.get("base_conf") or {}
        self.base_dist = base_conf.get("base_dist", "poles")
        self.tanh_theta = base_conf.get("tanh_theta", False)
        self.kappa = base_conf.get("kappa", torch.tensor(10.0))
        self.e_dep_max = base_conf.get("e_dep_max", 1.)
        self.bs_frac = base_conf.get("bs_frac", 0.0)
        self.scale_dist = base_conf.get("scale_dist", "trunc_norm")
        self.sm_scale = base_conf.get("sm_scale", 0.5)  # sm_norm tanh temperature; larger -> flatter energy base
        self.edep_mu = 0.0   # per-event edep base params; overwritten by LEGOLtng's base_head
        self.edep_sig = 1.0
        self.ang_pow = None  # per-event angular shape; set by base_head when ang_shape
        self.sm_pow = 1.0    # sm_norm shape q: u = 1 - tanh(sm |z|^q); set per slot by base_head_lik

        if self.base_dist not in ("poles", "iso", "iso_pos"):
            raise ValueError(f"base_dist must be 'poles', 'iso', or 'iso_pos', got {self.base_dist!r}")

    def __call__(self, shape, incoming_rt, rng=None, elem=None):
        """``rng`` (legofmt.rng.EventRNG) keys the draws per event, ``elem`` (B, n_out) per slot."""
        if self.base_dist == "iso":
            return self.iso_dirs(shape, incoming_rt=incoming_rt, rng=rng, elem=elem)
        return self.poles(shape, incoming_rt=incoming_rt, iso_pos=(self.base_dist == "iso_pos"), rng=rng, elem=elem)

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
            u = 1 - torch.tanh(normal().abs().pow(self.sm_pow) * self.sm_scale)
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
            x = self.geom_trafos.sample_iso(shape, 1, device=incoming_rt.device, rng=rng, elem=elem, stream=R.BASE_POS)
        else:
            x = self.geom_trafos.sample(
                shape, loc_cc, self.kappa, self.bs_frac, self.tanh_theta, self.ang_pow,
                rng=rng, elem=elem, stream=R.BASE_POS)
        p_ = self.geom_trafos.sample(shape, p_cc, self.kappa, 0.0, self.tanh_theta, self.ang_pow,
                                     rng=rng, elem=elem, stream=R.BASE_MOM)
        base = torch.cat((e_sc, p_, x), dim=-1)
        return torch.cat((incoming_rt, base), dim=1)

    def iso_dirs(self, shape, incoming_rt, rng=None, elem=None, **kwargs):
        # naive base: isotropic momentum/position directions; energy scale from
        # rd_scale (per scale_dist), E_dep row from insert_add.
        e_sc = self.rd_scale(shape, torch.ones_like(incoming_rt[..., 0:1]), rng=rng, elem=elem)
        p_ = self.geom_trafos.sample_iso(shape, 1, device=incoming_rt.device, rng=rng, elem=elem, stream=R.BASE_MOM)
        x  = self.geom_trafos.sample_iso(shape, 1, device=incoming_rt.device, rng=rng, elem=elem, stream=R.BASE_POS)
        base = torch.cat((e_sc, p_, x), dim=-1)
        return torch.cat((incoming_rt, base), dim=1)

    @torch.no_grad()
    def iso(self, shape, device, rng=None):
        """Fully isotropic x_0 for inverse-mask events, on all rows; unrelated
        to base_dist='iso', which routes to iso_dirs."""
        elem = None if rng is None else torch.arange(shape[1], device=device).expand(shape)
        e = (torch.rand((*shape, 1), device=device) if rng is None
             else rng.uniform(R.ISO_E, elem)[..., None].float().to(device))
        p = self.geom_trafos.sample_iso(shape, 1, device=device, rng=rng, elem=elem, stream=R.ISO_MOM)
        x = self.geom_trafos.sample_iso(shape, 1, device=device, rng=rng, elem=elem, stream=R.ISO_POS)
        return torch.cat((e, p, x), dim=-1)

    def insert_add(self, base, rng=None):
        """Fills the E_dep row's energy cell from the (learned) sigmoid-normal
        edep_mu/edep_sig, in place."""
        mu, sig = self.edep_mu, self.edep_sig
        if torch.is_tensor(mu):
            mu, sig = mu.squeeze(-1), sig.squeeze(-1)
        z = torch.randn_like(base[:, 1, 0]) if rng is None else rng.normal(R.EDEP).to(base)
        base[:, 1, 0] = self.e_dep_max * torch.sigmoid(mu + sig * z)
        return base
