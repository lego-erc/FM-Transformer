import torch
import torch.nn.functional as F

from legofmt.geometry.geom_trafos import GeomTrafos


class GenerateBase:
    def __init__(self, config: dict):
        self.geom_trafos = GeomTrafos()
        self.cutoff_mev = config["dl_conf"]["lds_args"].get("cutoff_mev", 10.0)
        base_conf = config.get("base_conf")
        base_dist = base_conf.get("base_dist", "poles")
        self.tanh_theta = base_conf.get("tanh_theta", False)
        self.kappa = base_conf.get("kappa", torch.tensor(10.0))
        self.e_dep_max = base_conf.get("e_dep_max", 1.)
        self.bs_frac = base_conf.get("bs_frac", 0.0)
        self.scale_dist = base_conf.get("scale_dist", "trunc_norm")

        if base_dist == "poles":
            self.func = self.poles

        else:
            raise ValueError("base_dist's other than poles are currently deprecated")

    def __call__(self, shape, incoming_rt):
        return self.func(shape, incoming_rt=incoming_rt)

    @torch.no_grad()
    def rd_scale(self, shape, e_in):
        if self.scale_dist == "trunc_norm":
            u = torch.nn.init.trunc_normal_(
                e_in.new_empty((*shape, 1)), std=1.0, a=-1.0, b=0.0,
            ) + 1.0
        elif self.scale_dist == "uniform":
            u = torch.rand((*shape, 1), device=e_in.device)
        elif self.scale_dist == "sm_norm":
            u = 1 - torch.tanh(torch.randn((*shape, 1), device=e_in.device).abs() / 2)
        else:
            raise ValueError("Unknown scale_dist")
        return e_in.view(-1, 1, 1) * u

    @torch.no_grad()
    def poles(self, shape, incoming_rt, **kwargs):
        e_in = incoming_rt[..., 0:1]
        p_cc = F.normalize(incoming_rt[..., 1:4], dim=-1)
        loc_cc = incoming_rt[..., -3:]
        e_sc = self.rd_scale(shape, e_in)
        x = self.geom_trafos.sample(shape, loc_cc, self.kappa, self.bs_frac, self.tanh_theta)
        p_ = self.geom_trafos.sample(shape, p_cc, self.kappa, 0.0, self.tanh_theta)
        base = torch.cat((e_sc, p_, x), dim=-1)
        base = torch.cat((incoming_rt, base), dim=1)
        return base

    @torch.no_grad()
    def insert_add(self, base):
        base[:, 1, 0] = self.e_dep_max * torch.sigmoid(torch.randn_like(base[:, 1, 0]))
        return base
