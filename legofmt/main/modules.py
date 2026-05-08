import lightning as ltng
import torch
import json
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from flow_matching.utils.manifolds import Euclidean, Sphere
from legofmt.data.dataloaders import LEGODataset
from torch import Tensor, nn
from torch.utils.data import DataLoader

try:
    from torch_lap_cuda_lib import solve_lap as slap
except ImportError:
    slap = None

from legofmt.geometry.vmf_sampling import VMF
from legofmt.cfm.cfm_trafo_x import CFMTrafo_x
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
        self.vmf = VMF()
        self.cond_cube = kwargs.get("cond_cube", False)
        self.no_detach = kwargs.get("no_detach", False)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        attn_mask: torch.Tensor,
        types: torch.Tensor,
        pdgids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pm = attn_mask.unsqueeze(-1)
        x_proj_dense = self.manifold.projx(x)
        x = torch.where(pm, x_proj_dense, x)
        if not self.no_detach:
            x = x.detach()
        t = torch.atleast_2d(t).expand_as(attn_mask)
        t = torch.where(mask.squeeze(-1) == 1, t, 1.)
        if self.cond_cube:
            x_cube = x.clone()
            x_cube[:, 2:3] = self.vmf.to_cube(x_cube[:, 2:3])
            x_surr = x_cube
        else:
            x_surr = x
        v = self.vf(t, x_surr, mask, attn_mask, types, pdgids)
        v_proj_dense = self.manifold.proju(x_proj_dense, v)
        return torch.where(pm, v_proj_dense, v)


class LEGOLtng(ltng.LightningModule):
    def __init__(self, config: dict) -> None:
        """As example input.

            config = {
            "dl_conf": {
                "data": DATA_PATH/data_prepped,
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
                    "max_seq_l": 9,
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
        model_conf = config.get("model_conf")
        dpath = config.get("dl_conf").get("lds_args").get("data")
        if dpath[-3:] != ".pt" and state_dict is None:
            config["dl_conf"]["data_path"] = dpath + "/data_prepped.pt"
            with open(dpath + "/meta.json") as f:
                meta_dict = json.load(f)
                self.max_seq_l = meta_dict["ntokens"]
                ptensor = (
                    torch.tensor(meta_dict["particles"], dtype=torch.int64)
                    .sort()
                    .values
                )
                self.register_buffer("pdgids_template", ptensor.contiguous())
                model_conf["model_args"]["npdgids"] = self.pdgids_template.shape[0] + 1
            if "max_seq_l" not in model_conf["model_args"]:
                model_conf["model_args"]["max_seq_l"] = self.max_seq_l
            if "pdgids" not in model_conf["model_args"]:
                model_conf["pdgids"] = ptensor
        elif state_dict is not None:
            if model_conf["model_args"].get("ntokens", False):
                model_conf["model_args"]["max_seq_l"] = model_conf["model_args"].pop(
                    "ntokens"
                )
            self.max_seq_l = model_conf["model_args"]["max_seq_l"]
            self.register_buffer("pdgids_template", model_conf["pdgids"])
        self.t_dist = model_conf.get("t_dist", "uniform")
        self.t_dist_scale = model_conf.get("t_dist_scale", 1.4)
        _MANIFOLD_NS = {
            "ProductManifold": ProductManifold,
            "Euclidean": Euclidean,
            "Sphere": Sphere,
        }
        self.manifold = eval(
            model_conf.get("manifold"), {"__builtins__": {}}, _MANIFOLD_NS
        )
        self.ot_coupling = model_conf.get("ot_coupling", False)
        self.proj_en_out = model_conf.get("proj_en_out", False)
        self.pdgid_is_idx = model_conf.get("pdgid_is_idx", False)
        self.loss_sc_fac = model_conf.get("loss_sc", 0.0)
        self.min_snr_gamma = model_conf.get("min_snr_gamma", 0.0)
        cond_cube = model_conf.get("cond_cube", False)
        if state_dict is None:
            model_conf["model_args"].setdefault("ntypes", 4)
        self.model = ProjectModel(
            CFMTrafo_x(**model_conf.get("model_args")),
            self.manifold,
            cond_cube=cond_cube,
        )

        self.gen_base = GenerateBase(config.copy())
        self.ppa = CubeTrace()

        opt_conf = config.get("opt_conf").copy()
        opt = opt_conf.pop("opt")
        self._sched_conf = opt_conf.pop("scheduler", None)
        self.opt = opt(self.model.parameters(), **opt_conf)
        self._opt_is_sf = hasattr(self.opt, "train") and callable(getattr(self.opt, "train", None))

        self.dl_conf = config.get("dl_conf")
        self.register_buffer(
            "types_embd",
            torch.arange(self.max_seq_l, dtype=torch.int64).clamp_max(3).view(1, -1),
        )

        self.odeint_conf = config.get("odeint_conf", {})

        if state_dict is not None:
            self.model.vf.load_state_dict(state_dict, strict=False)

    def _opt_train(self) -> None:
        """No-op unless the optimizer has a schedule-free .train() method."""
        if self._opt_is_sf:
            self.opt.train()

    def _opt_eval(self) -> None:
        """No-op unless the optimizer has a schedule-free .eval() method."""
        if self._opt_is_sf:
            self.opt.eval()

    @torch.no_grad()
    def on_fit_start(self) -> None:
        self.loss_fn = nn.MSELoss()
        self.ps = ProductPathSampler(self.manifold)
        self.model.train()
        self._opt_train()

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
            am = attn_mask[:, 1:]
            base = base.where(attn_mask.unsqueeze(-1), cc)
            inf_cond = am.unsqueeze(-1).logical_xor(am.unsqueeze(-2))
            cost = torch.cdist(cc[:, 1:], base[:, 1:])
            cost = cost + inf_cond * 1e6
            assign = slap(cost, cost.device).long()
            base[:, 1:] = torch.take_along_dim(base[:, 1:], assign.unsqueeze(-1), dim=1)
        base = torch.cat((target[:, :2], base), dim=1)
        base = self.gen_base.insert_add(base)  # E_dep
        return base

    def _step(self, batch: tuple, _batch_idx: int | Tensor) -> Tensor:
        with torch.no_grad():
            target, mask, attn_mask = batch
            energy, cc, pdgids = target.split([1, 6, 1], dim=-1)
            base = self.gen_base_wrapper((cc, mask, attn_mask))
            pdgid_idx = self.convert_pdgids(pdgids)
            if self.t_dist == "sm_norm":
                t = torch.sigmoid(self.t_dist_scale * torch.randn_like(base[:, 0, 0]))
            elif self.t_dist == "sd3":
                u = torch.rand_like(base[:, 0, 0])
                t = 1 - u + self.t_dist_scale / 3 * ((torch.pi / 2 * u).sin()**2 - u)
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
            target_sc = (cc * attn_mask.unsqueeze(-1))[..., :3]
            loss_sc = self.loss_fn(pred_sc_ft, target_sc)
        else:
            loss_sc = 0.0
        sq = (v_out - ps_.dx_t)**2
        if self.min_snr_gamma > 0:
            t_s = ps_.t.clamp(1e-3, 1 - 1e-3)
            snr = ((1 - t_s) / t_s)**2
            sq = sq * (snr.clamp_max(self.min_snr_gamma) / snr).view(-1, 1, 1)
        losses = torch.stack([s.mean() for s in (sq[:, 3:] * (mask[:, 3:] == 1)).chunk(2, -1)])
        loss_v = sq[:, 1].mean() + (losses * losses.detach().mean() / losses.detach()).sum()
        loss = loss_v + self.loss_sc_fac * loss_sc
        return loss

    def training_step(self, batch: tuple, _batch_idx: int | Tensor) -> Tensor:
        loss = self._step(batch, _batch_idx)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: tuple, _batch_idx: int | Tensor) -> Tensor:
        loss = self._step(batch, _batch_idx)
        with torch.no_grad():
            self.log(
                "Validation Loss",
                loss,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
        return loss

    @torch.no_grad()
    def configure_optimizers(self):
        if self._sched_conf is None:
            return self.opt
        cfg = dict(self._sched_conf)
        sched_cls = cfg.pop("cls")
        interval = cfg.pop("interval", "step")
        scheduler = sched_cls(self.opt, **cfg)
        return {
            "optimizer": self.opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": interval},
        }

    @torch.no_grad()
    def train_dataloader(self) -> DataLoader:
        num_workers = self.dl_conf.get("num_workers", 4)
        dataset_train = LEGODataset(**self.dl_conf.get("lds_args"))
        return DataLoader(
            dataset_train,
            batch_size=self.dl_conf.get("bs", 2**12),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            multiprocessing_context="fork" if num_workers > 0 else None,
        )

    def chunked(self, fn, *tensors, split_size=None, dim=0, cat_dim=None):

        if split_size is None or split_size >= tensors[0].shape[dim]:
            return fn(*tensors)
        if cat_dim is None:
            cat_dim = dim
        out = []
        for chunk in zip(*(t.split(split_size, dim) for t in tensors)):
            out.append(fn(*chunk))
        return torch.cat(out, dim=cat_dim)

    @torch.no_grad()
    def solve(
        self,
        batch: tuple,
        x_init: Tensor | None = None,
        reverse: bool = False,
        compute_ll: bool = False,
        log_p0=None,
        split_size: int | None = None,
        step_size: float = 0.04,
        method: str = "midpoint",
        time_grid: Tensor | None = None,
        return_intermediates: bool = False,
    ) -> Tensor:
        if self.model.training:
            self.model.eval()
            self._opt_eval()

        target, mask, attn_mask = batch
        _, cc, pdgids = target.split([1, 6, 1], dim=-1)
        pdgids_idx = pdgids.int() if self.pdgid_is_idx else self.convert_pdgids(pdgids)
        cc = cc.where(attn_mask.unsqueeze(-1), cc[:, 2:3, :])
        if x_init is not None and x_init.shape == cc.shape:
            x_init = x_init.where(attn_mask.unsqueeze(-1), x_init[:, 2:3, :])

        solver = ODESolver(velocity_model=self.model)
        common = dict(step_size=step_size, method=method, types=self.types_embd)

        if compute_ll:
            self.model.no_detach = True
            try:
                _, log_ll = solver.compute_likelihood(
                    x_1=cc,
                    log_p0=log_p0,
                    mask=mask,
                    attn_mask=attn_mask,
                    pdgids=pdgids_idx,
                    **common,
                )
            finally:
                self.model.no_detach = False
            return log_ll

        if x_init is None:
            x_init = self.gen_base_wrapper((cc, mask, attn_mask))
        if time_grid is None:
            time_grid = x_init.new_tensor([1.0, 0.0] if reverse else [0.0, 1.0])

        def _sample(x_init, mask, attn_mask, pdgids_idx):
            return solver.sample(
                x_init=x_init,
                time_grid=time_grid,
                mask=mask,
                attn_mask=attn_mask,
                pdgids=pdgids_idx,
                return_intermediates=return_intermediates,
                **common,
            )

        return self.chunked(
            _sample,
            x_init,
            mask,
            attn_mask,
            pdgids_idx,
            split_size=split_size,
            cat_dim=-3,
        )

    @torch.no_grad()
    def forward(self, batch: tuple, _batch_idx: int | Tensor | None = None) -> tuple:
        if self.model.training:
            self.model.eval()
            self._opt_eval()

        cfg = self.odeint_conf
        if (cfg.get("fwd_compile", False) 
        and not (hasattr(self.model, "_orig_mod")
        or hasattr(self.model.vf, "_orig_mod"))):
            self.model = torch.compile(self.model, dynamic=False)

        target, mask, attn_mask = batch
        _, cc, pdgids = target.split([1, 6, 1], dim=-1)
        base = self.gen_base_wrapper((cc, mask, attn_mask))
        densities = base[:, :1, :1].expand_as(base[..., :1])

        if cfg.get("return_base", False):
            sols = base.masked_fill(~attn_mask.unsqueeze(-1), torch.nan)
        else:
            step_size = cfg.get("step_size", 0.04)
            time_grid = (
                torch.arange(
                    0, 1 + step_size, step=step_size, device=self.device
                ).clamp_max(1)
                if cfg.get("return_timesteps", False)
                else None
            )
            sols = self.solve(
                batch,
                x_init=base,
                split_size=cfg.get("split_size"),
                step_size=step_size,
                method=cfg.get("method", "midpoint"),
                time_grid=time_grid,
                return_intermediates=cfg.get("return_timesteps", False),
            )
            sols.masked_fill_(~attn_mask.unsqueeze(-1), torch.nan)
            filter_pdgid = cfg.get("filter_pdgid")
            if filter_pdgid is not None:
                pdgids_idx = (
                    pdgids.int() if self.pdgid_is_idx else self.convert_pdgids(pdgids)
                )
                keep = torch.isin(pdgids_idx, self.convert_pdgids(filter_pdgid)) | (
                    pdgids_idx == 0
                )
                sols.masked_fill_(~keep, torch.nan)
                pdgids = pdgids.masked_fill(~keep, 0)

        if sols.dim() == 4:
            T = sols.shape[0]
            densities = densities.unsqueeze(0).expand(T, -1, -1, -1)
            pdgids = pdgids.unsqueeze(0).expand(T, -1, -1, -1)
        return torch.cat((densities, sols, pdgids), dim=-1), mask, attn_mask
