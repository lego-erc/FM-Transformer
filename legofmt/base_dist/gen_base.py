"""Samplers for the flow's base distribution -- the ``x_0`` it integrates from.

``base_dist`` picks the direction prior (``poles`` concentrates around the
incoming direction; ``iso`` / ``iso_pos`` do not), ``scale_dist`` the energy
prior, and ``insert_add`` fills the edep row. The sampler must match the
manifold the flow integrates on.
"""

import torch
import torch.nn.functional as F

from legofmt.geometry.geom_trafos import GeomTrafos


class GenerateBase:

    def __init__(self, config: dict):
        self.geom_trafos = GeomTrafos()
        self.cutoff_mev = config["dl_conf"]["lds_args"].get("cutoff_mev", 10.0)
        base_conf = config.get("base_conf")
        self.base_dist = base_conf.get("base_dist", "poles")
        self.tanh_theta = base_conf.get("tanh_theta", False)
        self.kappa = base_conf.get("kappa", torch.tensor(10.0))
        self.e_dep_max = base_conf.get("e_dep_max", 1.)
        self.bs_frac = base_conf.get("bs_frac", 0.0)
        self.scale_dist = base_conf.get("scale_dist", "trunc_norm")
        self.sm_scale = base_conf.get("sm_scale", 0.5)  # sm_norm tanh temperature; larger -> flatter energy base
        self.edep_mu = 0.0   # per-event edep base params; overwritten by LEGOLtng's base_head
        self.edep_sig = 1.0

        if self.base_dist not in ("poles", "iso", "iso_pos"):
            raise ValueError(f"base_dist must be 'poles', 'iso', or 'iso_pos', got {self.base_dist!r}")

    def __call__(self, shape, incoming_rt):
        if self.base_dist == "iso":
            return self.iso_dirs(shape, incoming_rt=incoming_rt)
        return self.poles(shape, incoming_rt=incoming_rt, iso_pos=(self.base_dist == "iso_pos"))

    @torch.no_grad()
    def rd_scale(self, shape, e_in):
        if self.scale_dist == "trunc_norm":
            u = torch.nn.init.trunc_normal_(
                e_in.new_empty((*shape, 1)), std=1.0, a=-1.0, b=0.0,
            ) + 1.0
        elif self.scale_dist == "uniform":
            u = torch.rand((*shape, 1), device=e_in.device)
        elif self.scale_dist == "sm_norm":
            u = 1 - torch.tanh(torch.randn((*shape, 1), device=e_in.device).abs() * self.sm_scale)
        elif self.scale_dist == "logit_norm":
            u = torch.sigmoid(torch.randn((*shape, 1), device=e_in.device))  # smooth, symmetric on (0,1)
        else:
            raise ValueError("Unknown scale_dist")
        return e_in.view(-1, 1, 1) * u

    @torch.no_grad()
    def poles(self, shape, incoming_rt, iso_pos=False, **kwargs):
        e_in = incoming_rt[..., 0:1]
        p_cc = F.normalize(incoming_rt[..., 1:4], dim=-1)
        loc_cc = incoming_rt[..., -3:]
        e_sc = self.rd_scale(shape, torch.ones_like(e_in))
        if iso_pos:
            x = self.geom_trafos.sample_iso(shape, 1, device=incoming_rt.device)
        else:
            x = self.geom_trafos.sample(shape, loc_cc, self.kappa, self.bs_frac, self.tanh_theta)
        p_ = self.geom_trafos.sample(shape, p_cc, self.kappa, 0.0, self.tanh_theta)
        base = torch.cat((e_sc, p_, x), dim=-1)
        return torch.cat((incoming_rt, base), dim=1)

    @torch.no_grad()
    def iso_dirs(self, shape, incoming_rt, **kwargs):
        # naive base: isotropic momentum/position directions; energy scale unchanged
        # (rd_scale, i.e. sm_norm) and e_dep base inherited from insert_add.
        e_in = incoming_rt[..., 0:1]
        e_sc = self.rd_scale(shape, torch.ones_like(e_in))
        p_ = self.geom_trafos.sample_iso(shape, 1, device=incoming_rt.device)
        x  = self.geom_trafos.sample_iso(shape, 1, device=incoming_rt.device)
        base = torch.cat((e_sc, p_, x), dim=-1)
        return torch.cat((incoming_rt, base), dim=1)

    @torch.no_grad()
    def iso(self, shape, device):
        e = torch.rand((*shape, 1), device=device)
        p = self.geom_trafos.sample_iso(shape, 1, device=device)
        x = self.geom_trafos.sample_iso(shape, 1, device=device)
        return torch.cat((e, p, x), dim=-1)

    @torch.no_grad()
    def insert_add(self, base):
        mu, sig = self.edep_mu, self.edep_sig
        if torch.is_tensor(mu):
            mu, sig = mu.squeeze(-1), sig.squeeze(-1)
        base[:, 1, 0] = self.e_dep_max * torch.sigmoid(mu + sig * torch.randn_like(base[:, 1, 0]))
        return base
