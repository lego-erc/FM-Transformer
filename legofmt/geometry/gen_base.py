import torch
import torch.nn.functional as F

from legofmt.geometry.vmf_sampling import VMF


class GenerateBase:
    def __init__(self, config: dict):
        self.vmf_utils = VMF()
        self.cutoff_mev = config["dl_conf"]["lds_args"].get("cutoff_mev", 10.0)
        base_conf = config.get("base_conf")
        base_dist = base_conf.get("base_dist", "iso")
        self.tanh_theta = base_conf.get("tanh_theta", False)
        self.kappa = base_conf.get("kappa", torch.tensor(10.0))
        self.e_dep_max = base_conf.get("e_dep_max", 1.)
        self.bs_frac = base_conf.get("bs_frac", 0.0)
        self.scale_dist = base_conf.get("scale_dist", "trunc_norm")

        if base_dist == "iso":
            self.func = self.iso

        elif base_dist == "iso_half":
            self.func = self.iso_half

        elif base_dist == "iso_3dmom":
            self.scale_dist = "trunc_norm"
            self.func = self.iso_3dmom

        elif base_dist == "poles":
            self.exp_dist = torch.distributions.Exponential(8)
            self.func = self.poles

    def __call__(self, shape, incoming_rt):
        return self.func(shape, incoming_rt=incoming_rt)

    @torch.no_grad()
    def iso(self, shape, incoming_rt=None, **kwargs):
        rd_scale = self.rd_scale(shape, incoming_rt.device)
        base = self.vmf_utils.sample_iso(shape, 2, **kwargs)
        p, x = base.split(3, dim=-1)
        base = torch.cat((rd_scale * p, x), dim=-1)
        base = torch.cat((incoming_rt, base), dim=1)
        return base

    @torch.no_grad()
    def rd_scale(self, shape, p_norm):
        base_range = torch.log(p_norm / self.cutoff_mev).view(-1, 1, 1)
        if self.scale_dist == "trunc_norm":
            return base_range * (
                torch.nn.init.trunc_normal_(
                    torch.empty((*shape, 1), device=p_norm.device),
                    std=1.0 / base_range,
                    a=-1.0,
                    b=0.0,
                )
                + 1.0
            )
        if self.scale_dist == "uniform":
            return base_range * torch.rand((*shape, 1), device=p_norm.device)
        if self.scale_dist == "sm_norm":
            return base_range * (
                1 - torch.tanh(torch.randn((*shape, 1), device=p_norm.device).abs() / 2)
            ) + 1

    @torch.no_grad()
    def iso_3dmom(self, shape, **kwargs):
        iso_base = self.iso(shape, **kwargs)
        rd_scale = self.rd_scale(shape, iso_base.device)
        return iso_base * torch.cat((rd_scale, torch.ones_like(rd_scale)), dim=-1)

    @torch.no_grad()
    def poles(self, shape, incoming_rt, **kwargs):
        p_norm = incoming_rt[..., :3].norm(dim=-1, keepdim=True)
        p_cc = F.normalize(incoming_rt[..., :3], dim=-1)
        loc_cc = incoming_rt[..., -3:]
        rd_scale = self.rd_scale(shape, p_norm)
        x = self.vmf_utils.sample(shape, loc_cc, self.kappa, self.bs_frac, self.tanh_theta)
        p_ = self.vmf_utils.sample(shape, p_cc, self.kappa, 0.0, self.tanh_theta)
        p_sc = rd_scale * p_
        base = torch.cat((p_sc, x), dim=-1)
        base = torch.cat((incoming_rt, base), dim=1)
        return base

    @torch.no_grad()
    def insert_add(self, base):
        base[:, 1, 0] = self.e_dep_max * torch.sigmoid(torch.randn_like(base[:, 1, 0]))
        return base
