import torch

from legofmt.geometry.vmf_sampling import VMF


class GenerateBase:
    def __init__(self, config: dict):
        self.vmf_utils = VMF()
        base_dist = config.get("base_dist", "iso")
        self.kappa = config.get("kappa", torch.tensor(10.0))
        self.base_range = config.get("base_range", 1.0)
        self.e_dep_max = config.get("e_dep_max", 0.4)
        self.bs_frac = config.get("bs_frac", 0.0)
        self.scale_dist = config.get("scale_dist", "trunc_norm")

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
    def rd_scale(self, shape, device):
        if self.scale_dist == "trunc_norm_legacy":
            return (
                -((torch.randn((*shape, 1), device=device).abs() * 2) % self.base_range)
                + self.base_range
            )
        if self.scale_dist == "trunc_norm":
            return self.base_range * (
                torch.nn.init.trunc_normal_(
                    torch.empty((*shape, 1), device=device), std=1.0 / self.base_range, a=-1.0, b=0.0
                )
                + 1.0
            )
        if self.scale_dist == "uniform":
            return self.base_range * torch.rand((*shape, 1), device=device)
        if self.scale_dist == "sm_norm":
            return self.base_range * (torch.sigmoid(torch.randn((*shape, 1), device=device)))
        if self.scale_dist == "exp":
            safety = 1e-1
            return (
                self.base_range
                * self.exp_dist.sample((*shape, 1)).to(device)
                % (1.0 - safety)
            ) + safety

    @torch.no_grad()
    def iso_3dmom(self, shape, **kwargs):
        iso_base = self.iso(shape, **kwargs)
        rd_scale = self.rd_scale(shape, iso_base.device)
        return iso_base * torch.cat((rd_scale, torch.ones_like(rd_scale)), dim=-1)

    @torch.no_grad()
    def poles(self, shape, incoming_rt=None, loc_cc=None, **kwargs):
        if incoming_rt is not None:
            loc_cc = incoming_rt[..., -3:]
        rd_scale = self.rd_scale(shape, loc_cc.device)
        x = self.vmf_utils.sample(shape, loc_cc, self.kappa, self.bs_frac)
        p_ = self.vmf_utils.sample(shape, x, self.kappa, 0.0)
        p_to_x_sign = torch.einsum("...ij, ...ij -> ...i", x, p_).sgn()
        p = torch.einsum("...i, ...ij -> ...ij", p_to_x_sign, p_)
        p_sc = rd_scale * p
        base = torch.cat((p_sc, x), dim=-1)
        if incoming_rt is not None:
            base = torch.cat((incoming_rt, base), dim=1)
        return base

    @torch.no_grad()
    def extend_add(self, base):
        rd = self.e_dep_max * torch.sigmoid(torch.randn_like(base[:, :1, :1]))
        ext = torch.zeros_like(base[:, :1])
        ext[..., 0] = rd.squeeze(-1)
        return torch.cat((ext, base), dim=1)