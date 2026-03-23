import pytorch_lightning as ltng
import torch
import json
from flow_matching.solver import RiemannianODESolver
from flow_matching.utils import ModelWrapper
from flow_matching.utils.manifolds import Euclidean, Sphere
from legofmt.data.dataloaders import LEGODataset
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch_lap_cuda_lib import solve_lap as slap

from legofmt.cfm.cfm_trafo_x import CFMTrafo_x
from legofmt.geometry.energy_proj import EnergyProjections
from legofmt.geometry.gen_base import GenerateBase
from legofmt.geometry.path_sample_mult import ProductPathSampler, ProductManifold
from legofmt.geometry.raytracing_proj import CubeTrace


class ProjectModel(ModelWrapper, nn.Module):
    """Projection Wrapper for Riemannian FM Model."""

    def __init__(
        self,
        vf: nn.Module,
        manifold: ProductManifold,
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
        dpath = config.get("dl_conf").get("lds_args").get("path")
        if dpath[-3:] != ".pt":
            config["dl_conf"]["data_path"] = dpath + "/data_prepped.pt"
            with open(dpath + "/meta.json") as f:
                meta_dict = json.load(f)
                self.ntokens = meta_dict["ntokens"]
                ptensor = torch.tensor(meta_dict["particles"], dtype=torch.int64)
                self.register_buffer("pdgids_template", ptensor.contiguous())
                config["model_conf"]["model_args"]["npdgids"] = self.pdgids_template.shape[0] + 1
            if "ntokens" not in config["model_conf"]["model_args"]:
                config["model_conf"]["model_args"]["ntokens"] = self.ntokens
        else:
            self.pdgids_template = None
        model_conf = config.get("model_conf")
        self.t_dist = model_conf.get("t_dist", "uniform")
        self.manifold = eval(model_conf.get("manifold"))
        self.ot_coupling = model_conf.get("ot_coupling", False)
        self.proj_en_out = model_conf.get("proj_en_out", False)
        self.loss_sc_fac = model_conf.get("loss_sc", 0.0)
        self.model = ProjectModel(CFMTrafo_x(**model_conf.get("model_args")), self.manifold)
        if state_dict is not None:
            self.model.vf.load_state_dict(state_dict, strict=False)

        self.gen_base = GenerateBase(config.get("base_conf").copy())
        self.ppa = CubeTrace()

        opt_conf = config.get("opt_conf").copy()
        opt = opt_conf.pop("opt")
        self.opt = opt(self.model.parameters(), **opt_conf)

        self.dl_conf = config.get("dl_conf")
        self.register_buffer("types_embd",
            torch.arange(self.ntokens, dtype=torch.int64).clamp_max(3).view(1, -1)
        )

        self.odeint_conf = config.get("odeint_conf", {})

    @torch.no_grad()
    def on_fit_start(self) -> None:
        self.loss_fn = nn.MSELoss()
        self.ps = ProductPathSampler(self.manifold)
        self.model.train()
        self.opt.train()

    @torch.no_grad()
    def convert_pdgids(self, pdgids: Tensor) -> Tensor:
        cond = torch.isnan(pdgids) | (pdgids == 0) | (pdgids >= 1e8)
        pdgid_idx = torch.searchsorted(self.pdgids_template, pdgids.contiguous()) + 1
        return pdgid_idx.masked_fill_(cond, 0)

    @torch.no_grad()
    def gen_base_wrapper(self, batch: tuple) -> Tensor:
        target, mask_, attn_mask_ = batch
        cc, mask, attn_mask = target[:, 2:], mask_[:, 2:], attn_mask_[:, 2:]
        base = self.gen_base(mask[:, 1:].shape[:-1], cc[:, :1])
        if self.ot_coupling and self.model.training:
            base = attn_mask.unsqueeze(-1) * base + ~attn_mask.unsqueeze(-1) * cc
            cost = attn_mask[:, 1:].unsqueeze(-1) * torch.cdist(cc[:, 1:], base[:, 1:])
            assign = slap(cost, cost.device)
            base[:, 1:] = base[:, 1:].gather(
                1, assign.unsqueeze(-1).expand_as(base[:, 1:])
            )
        base = self.gen_base.extend_add(base) # E_dep
        base = torch.cat((target[:, :1], base), dim=1) # Density
        return base

    def _step(self, batch: tuple, _batch_idx: int | Tensor) -> Tensor:
        with torch.no_grad():
            target, mask, attn_mask = batch
            energy, cc, pdgids = target.split([1, 6, 1], dim=-1)
            base = self.gen_base_wrapper((cc, mask, attn_mask))
            pdgid_idx = self.convert_pdgids(pdgids)
            if self.t_dist == "sm_norm":
                t = torch.sigmoid(torch.randn_like(base[:, 0, 0]))
            elif self.t_dist == "uniform":
                t = torch.rand_like(base[:, 0, 0])
            ps_ = self.ps.sample(base, cc, t)
        v_out = self.model(
            ps_.x_t,
            ps_.t,
            mask=mask,
            attn_mask=attn_mask,
            types=self.types_embd,
            pdgids=pdgid_idx,
        )
        if self.loss_sc_fac > 0:
            pred_sc = (1 - ps_.t).unsqueeze(-1) * v_out + ps_.x_t
            pred_sc_ft = (pred_sc * attn_mask.unsqueeze(-1))[..., :3]
            target_sc = (target * attn_mask.unsqueeze(-1))[..., :3]
            loss_sc = self.loss_fn(pred_sc_ft, target_sc)
        else:
            loss_sc = 0.0
        v_target = ps_.dx_t * attn_mask.unsqueeze(-1)
        loss = self.loss_fn(v_out, v_target) + self.loss_sc_fac * loss_sc
        return loss

    def training_step(self, batch: tuple, _batch_idx: int | Tensor) -> Tensor:
        loss = self._step(batch, _batch_idx)
        if loss.isnan():
            for name, p in self.model.named_parameters():
                if p.isnan().any():
                    break
            raise ValueError(f"NaN loss encountered during training. \n {name}")
        return loss

    @torch.no_grad()
    def validation_step(self, batch: tuple, _batch_idx: int | Tensor) -> Tensor:
        loss = self._step(batch, _batch_idx)
        with torch.no_grad():
            self.log(
                "Validation Loss",
                loss.item(),
                on_step=True,
                on_epoch=True,
                # batch_size=self._step_bs,
                logger=True,
                sync_dist=True,
            )
        return loss

    @torch.no_grad()
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.opt

    @torch.no_grad()
    def train_dataloader(self) -> DataLoader:
        num_workers = self.dl_conf.get("num_workers", 32)
        dataset_train = LEGODataset(**self.dl_conf.get("lds_args"))
        return DataLoader(
            dataset_train,
            batch_size=self.dl_conf.get("bs", 2**12),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    @torch.no_grad()
    def forward(self, batch: tuple, _batch_idx: int | Tensor | None = None) -> Tensor:
        self.model.eval()
        self.opt.eval()

        split_size = self.odeint_conf.get("split_size", 2**16 - 1)
        return_base = self.odeint_conf.get("return_base", False)
        step_size = self.odeint_conf.get("step_size", 0.04)
        return_timesteps = self.odeint_conf.get("return_timesteps", False)
        filter_pdgid = self.odeint_conf.get("filter_pdgid", None)
        method = self.odeint_conf.get("method", "midpoint")

        if return_timesteps:
            time_grid = torch.arange(
                0, 1 + step_size, step=step_size, device=self.device
            ).clamp_max(1)
        else:
            time_grid = torch.tensor([0.0, 1.0], device=self.device)

        target, mask, attn_mask = batch

        if target.shape[-2] < self.ntokens:
            pad_len = self.ntokens - target.shape[-2]
            target = torch.cat((target, target[:, -1:].expand(-1, pad_len, -1)), dim=-2)
        if mask.shape[1] < self.ntokens:
            pad_len = self.ntokens - mask.shape[1]
            mask = torch.cat((mask, torch.zeros_like(mask[:,  -1:]).expand(-1, pad_len, -1)), dim=1)
        if attn_mask.shape[1] < self.ntokens:
            pad_len = self.ntokens - attn_mask.shape[1]
            attn_mask = torch.cat((attn_mask, torch.zeros_like(attn_mask[:, -1:]).expand(-1, pad_len)), dim=1)

        energy, cc, pdgids = target.split([1, 6, 1], dim=-1)
        pdgids_idx = self.convert_pdgids(pdgids)

        base = self.gen_base_wrapper((cc, mask, attn_mask))
        if return_base:
            return base.masked_fill(~attn_mask.unsqueeze(-1), torch.nan)

        init_state_tp = base.split(split_size, 0)
        mask_tp = mask.split(split_size, 0)
        attn_mask_tp = attn_mask.split(split_size, 0)
        pdgids_idx_tp = pdgids_idx.split(split_size, 0)

        sols_list = []
        solver = RiemannianODESolver(
            velocity_model=self.model, manifold=self.manifold
        )
        for idx in range(len(init_state_tp)):
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
                pdgids=pdgids_idx_tp[idx],
            )
            sols_ = sols_.masked_fill(~attn_mask_tp[idx].unsqueeze(-1), torch.nan)

            if filter_pdgid is not None:
                filter_pdgid_idx = self.convert_pdgids(filter_pdgid)
                pdgid_mask = torch.isin(pdgids_idx_tp[idx], filter_pdgid_idx)
                sols_ = sols_.masked_fill(~pdgid_mask, torch.nan)

            sols_list.append(sols_)

        return torch.cat(sols_list, dim=-3).contiguous()
