import pytorch_lightning as ltng
import torch
import json
from flow_matching.solver import RiemannianODESolver
from flow_matching.utils import ModelWrapper
from flow_matching.utils.manifolds import Manifold
from legofmt.data.dataloaders import LEGODataset
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch_lap_cuda_lib import solve_lap as slap

from legofmt.cfm.cfm_trafo_x import CFMTrafo_x
from legofmt.geometry.energy_proj import EnergyProjections
from legofmt.geometry.gen_base import GenerateBase
from legofmt.geometry.path_sample_mult import ProductPathSampler
from legofmt.geometry.raytracing_proj import CubeTrace


class ProjectModel(ModelWrapper, nn.Module):
    """Projection Wrapper for Riemannian FM Model."""

    def __init__(
        self,
        vf: nn.Module,
        manifold: Manifold,
        **kwargs: dict,
    ) -> None:
        nn.Module.__init__(self)
        self.vf = vf
        self.manifold = manifold
        self.kwargs = kwargs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        attn_mask: torch.Tensor,
        types: torch.Tensor,
        pdgids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        proj_mask = attn_mask * (types > (types.max() - 2))
        x_2d = x.flatten(0, -2)
        pm_flat = proj_mask.flatten()
        x_projx = self.manifold.projx(x_2d[pm_flat])
        x_2d[pm_flat] = x_projx
        x = x_2d.view_as(x).detach()
        t = torch.atleast_2d(t).expand_as(attn_mask)
        t_mask = mask.squeeze(-1) == 1
        t = t_mask * t + ~t_mask
        v = self.vf(t, x, mask, attn_mask, types, pdgids)
        v_2d = v.flatten(0, -2)
        v_proj = self.manifold.proju(x_projx, v_2d[pm_flat])
        v_2d[pm_flat] = v_proj
        return v_2d.view_as(v)


class LEGOLtng(ltng.LightningModule, nn.Module):
    def __init__(self, config: dict) -> None:
        """As example input.

            config = {
            "dl_conf": {
                "path": DATA_PATH,
                "cutoff_mev": 20,
                "min_particles": 1,
                "max_e": False,
                "is_filtered": True
            },
            "base_conf": {
                "base_range": 1.,
                "kappa": torch.tensor(40.),
                "bs_frac": 0.,
                "base_dist": "poles",
            },
            "model_conf": {
                    "h_dim": 2**7,
                    "ntokens": 9,
                    "in_dim": 6,
                    "nlayers": 4,
                    "nhead": 8,
                    "dropout": 0.1,
                    "ff_mult": 4,
                    "manifold": man_euc_sph,
                    "proj_ray": True,
                    "proj_en": "log",
            },
            "opt_conf": {
                "opt": schedulefree.AdamWScheduleFree,
                "lr": 1e-3
            },
        }
        """
        super().__init__()
        state_dict = config.get("state_dict")
        config = config.get("config", config)
        dpath = config.get("dl_conf").get("path")
        config["dl_conf"]["data_path"] = dpath + "/data.pt"
        with open(dpath + "/meta.json") as f:
            meta_dict = json.load(f)
        ntokens = meta_dict["ntokens"]
        ptensor = torch.tensor([0] + meta_dict["particles"])
        self.register_buffer("pdgids_template", ptensor)
        config["model_conf"]["npdgids"] = self.pdgids_template.shape[0]
        config["model_conf"]["ntokens"] = ntokens
        model_conf = config.get("model_conf").copy()
        self.t_dist = model_conf.pop("t_dist", "uniform")
        self.manifold = model_conf.pop("manifold")
        self.proj_ray = model_conf.pop("proj_ray", True)
        self.proj_en = model_conf.pop("proj_en", False)
        self.ot_coupling = model_conf.pop("ot_coupling", False)
        self.proj_en_out = model_conf.pop("proj_en_out", False)
        self.pen = EnergyProjections(self.proj_en)
        self.model = ProjectModel(CFMTrafo_x(**model_conf), self.manifold)
        if state_dict is not None:
            self.model.vf.load_state_dict(state_dict, strict=False)

        self.gen_base = GenerateBase(config.get("base_conf").copy())
        self.ppa = CubeTrace()

        opt_conf = config.get("opt_conf").copy()
        opt = opt_conf.pop("opt")
        self.opt = opt(self.model.parameters(), **opt_conf)

        self.dl_conf = config.get("dl_conf").copy()
        self.types_embd = nn.Parameter(
            torch.arange(model_conf.get("ntokens"), dtype=torch.int64)
            .clamp_max(2 if self.dl_conf.get("include_add", True) else 1)
            .view(1, -1),
            requires_grad=False,
        )

        self.odeint_conf = config.get("odeint_conf", {})

    def on_fit_start(self) -> None:
        self.loss_fn = nn.MSELoss()
        self.ps = ProductPathSampler(self.manifold)
        self.model.train()
        self.opt.train()

    @torch.no_grad()
    def convert_pdgids(self, pdgids: Tensor) -> Tensor:
        data_particles = pdgids.nan_to_num(int(self.pdgids_template[0]))
        bool_comp = data_particles.unsqueeze(-1) == self.pdgids_template.view(1, 1, -1)
        idx_template = torch.arange(bool_comp.shape[-1], device=self.device).view(
            1, 1, -1
        )
        return (bool_comp * idx_template).sum(dim=-1)
    
    @torch.no_grad()
    def extend_data_add(self, batch: tuple, pdgid_idx: Tensor, base: Tensor) -> tuple:
        cc, mask, attn_mask, data_add = batch
        in_dim = self.model.vf.in_dim
        if isinstance(data_add, torch.Tensor):
            data_add = data_add.view(mask.shape[0], -1)
            da_nft = torch.tensor([data_add.shape[-1]], device=self.device)
            da_nt = torch.floor_divide(da_nft, in_dim) + 1
            da_tk = torch.ones_like(cc[:, :1]).repeat(1, 1, da_nt.int())
            da_tk[:, 0, :da_nft] = data_add.view(-1, da_nft)
            da_tk = da_tk.view(-1, da_nt, in_dim)
            da_tk[:, 0, 0] /= cc[:, 0, :3].norm(dim=-1)
            zr = torch.ones_like(da_tk[..., :1], dtype=torch.long)
            cc = torch.cat((da_tk, cc), dim=1)
            if pdgid_idx is not None:
                pdgid_idx = torch.cat(
                    (
                        torch.zeros_like(pdgid_idx[:, :da_nt], device=self.device),
                        pdgid_idx,
                    ),
                    dim=1,
                )
        else:
            zr = torch.ones_like(mask[:, :data_add], dtype=torch.long)
        mask = torch.cat((zr, mask), dim=1)
        attn_mask = torch.cat((zr.bool().squeeze(-1), attn_mask), dim=1)
        base = self.gen_base.extend_add(base)
        return cc, mask, attn_mask, base

    @torch.no_grad()
    def _prep(self, batch: tuple, _batch_idx: int | Tensor) -> Tensor:
        cc, mask, attn_mask, data_add = batch
        in_dim = self.model.vf.in_dim
        pdgid_idx = self.convert_pdgids(cc[..., -1])
        cc = cc[..., -in_dim - 1 : -1]
        cc = cc.nan_to_num(1)
        cc = self.manifold.projx(cc)
        if self.proj_en is not False:
            cc = self.pen(cc)
        if self.proj_ray:
            cc[:, 0] = self.ppa(cc[:, 0])
        base = self.gen_base(cc * torch.ones_like(mask[0]), device=self.device)
        if self.ot_coupling:
            cost = attn_mask[:, 1:].unsqueeze(-1) * torch.cdist(cc[:, 1:], base[:, 1:])
            assign = slap(cost, cost.device)
            base[:, 1:] = base[:, 1:].gather(
                1, assign.unsqueeze(-1).expand_as(base[:, 1:])
            )
        batch = (cc, mask, attn_mask, data_add)
        cc, mask, attn_mask, base = self.extend_data_add(batch, pdgid_idx, base)

        return base, cc, mask, attn_mask, pdgid_idx

    def _step(self, batch: tuple, _batch_idx: int | Tensor) -> Tensor:
        with torch.no_grad():
            base, target, mask, attn_mask, pdgid_idx = self._prep(batch, _batch_idx)
            if self.t_dist == "sm_norm":
                t = torch.sigmoid(torch.randn_like(base[:, 0, 0]))
            elif self.t_dist == "uniform":
                t = torch.rand_like(base[:, 0, 0])
            ps_ = self.ps.sample(base, target, t)
        v_out = self.model(
            ps_.x_t,
            ps_.t,
            mask=mask,
            attn_mask=attn_mask,
            types=self.types_embd,
            pdgids=pdgid_idx,
        )
        v_target = ps_.dx_t * attn_mask.unsqueeze(-1)
        return self.loss_fn(v_out, v_target)

    def training_step(self, batch: tuple, _batch_idx: int | Tensor) -> Tensor:
        loss = self._step(batch, _batch_idx)
        make_used_ = sum(
            p.sum() * 0.0 for p in self.model.vf.vf.attn_layers.parameters()
        )
        if loss.isnan():
            for name, p in self.model.named_parameters():
                if p.isnan().any():
                    break
            raise ValueError(f"NaN loss encountered during training. \n {name}")
        return loss + make_used_

    def validation_step(self, batch: tuple, _batch_idx: int | Tensor) -> Tensor:
        loss = self._step(batch, _batch_idx)
        with torch.no_grad():
            self.log(
                "Validation Loss",
                loss.clone().item(),
                on_step=True,
                on_epoch=True,
                # batch_size=self._step_bs,
                logger=True,
                sync_dist=True,
            )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.opt

    def train_dataloader(self) -> DataLoader:
        bs = self.dl_conf.pop("bs", 2**12)
        num_workers = self.dl_conf.pop("num_workers", 32)
        dataset_train = LEGODataset(**self.dl_conf)
        return DataLoader(
            dataset_train, batch_size=bs, shuffle=True, num_workers=num_workers
        )

    @torch.no_grad()
    def forward(self, batch: tuple, _batch_idx: int | Tensor | None = None) -> Tensor:
        split_size = self.odeint_conf.get("split_size", 2**16 - 1)
        return_base = self.odeint_conf.get("return_base", False)
        step_size = self.odeint_conf.get("step_size", 0.04)
        return_timesteps = self.odeint_conf.get("return_timesteps", False)
        filter_pdgid = self.odeint_conf.get("filter_pdgid", None)
        if return_timesteps:
            time_grid = torch.arange(
                0, 1 + step_size, step=step_size, device=self.device
            ).clamp_max(1)
        else:
            time_grid = torch.tensor([0.0, 1.0], device=self.device)
        method = self.odeint_conf.get("method", "midpoint")
        base, _, mask, attn_mask, pdgids_idx = self._prep(batch, _batch_idx)
        if return_base:
            return base.masked_fill(~attn_mask.unsqueeze(-1), torch.nan)
        init_state_tp = base.split(split_size, 0)
        mask_tp = mask.split(split_size, 0)
        attn_mask_tp = attn_mask.split(split_size, 0)
        if pdgids_idx is not None:
            pdgids_idx_tp = pdgids_idx.split(split_size, 0)

        for idx in range(len(init_state_tp)):
            solver = RiemannianODESolver(
                velocity_model=self.model, manifold=self.manifold
            )
            sols_ = solver.sample(
                x_init=init_state_tp[idx],
                step_size=step_size,
                method=method,
                projx=False,
                proju=False,
                return_intermediates=return_timesteps,
                time_grid=time_grid,
                mask=mask_tp[idx],
                attn_mask=attn_mask_tp[idx],
                types=self.types_embd,
                pdgids=pdgids_idx_tp[idx] if pdgids_idx is not None else None,
            )
            del solver
            sols_ = sols_.masked_fill(~attn_mask_tp[idx].unsqueeze(-1), torch.nan)

            if filter_pdgid is not None:
                filter_pdgid_idx = self.convert_pdgids(filter_pdgid)
                pdgid_mask = torch.isin(pdgids_idx_tp[idx], filter_pdgid_idx)
                sols_ = sols_.masked_fill(~pdgid_mask.unsqueeze(-1), torch.nan)

            try:
                sols = torch.cat((sols, sols_), dim=-3)
            except NameError:
                sols = sols_
            del sols_
            torch.cuda.empty_cache()

        return sols.contiguous()
