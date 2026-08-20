import torch
from torch import Tensor
from torch.utils.data import Dataset

from .struct import DataStruct


class GetLEGOData:
    def __init__(
        self,
        cutoff_mev=10.0,
        min_particles=0,
        device="cpu",
        energy_kin=True,
        **kwargs,
    ):
        self.dev = device
        self.dtype = kwargs.pop("dtype", torch.float32)
        self.min_particles = min_particles
        self.cutoff_mev = cutoff_mev
        self.energy_kin = energy_kin

    def __call__(self, *args, **kwargs):
        return self.dataset_cutoff(*args, **kwargs)

    def dataset_compact(self, data):
        data_pp = data.get("per_particle")
        data_add = data.get("per_event")
        data_pp = torch.cat((data_pp["Incoming"], data_pp["Outgoing"]), dim=-2)
        return data_pp.to(self.dtype), data_add

    def dataset_cutoff(
        self,
        data: dict,
        n_events: (int | None) = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict]:
        dataset, data_add = self.dataset_compact(data)
        e_valid = dataset[..., 0] if self.energy_kin else dataset[..., 1:4].norm(dim=-1)
        mask_valid = e_valid >= self.cutoff_mev
        max_valid = mask_valid.sum(dim=-1).max()
        mask_valid_sorted = mask_valid.sort(dim=-1, descending=True).values[:, :max_valid]
        dataset_valid = torch.full_like(dataset[:, :max_valid], torch.nan)
        dataset_valid[mask_valid_sorted] = dataset[mask_valid]
        keep = ~dataset_valid[:, : self.min_particles + 1, 0].isnan().any(dim=-1)
        data_pp = dataset_valid[keep]
        data_add = {k: v[keep] for k, v in data_add.items()}
        if n_events is not None:
            rd_idx = torch.randperm(data_pp.shape[0], device="cpu")[:n_events]
            data_pp = data_pp[rd_idx]
            data_add = {k: v[rd_idx] for k, v in data_add.items()}
        particle_nan = ~data_pp.isnan().any(dim=-1)
        attn_mask = particle_nan.to(torch.int64)
        mask = attn_mask.clone()
        mask[:, 0] = 0
        data_add = {k: v.to(self.dev) for k, v in data_add.items()}
        return data_pp.to(self.dev), mask.to(self.dev), attn_mask.to(self.dev).bool(), data_add


class LEGODataset(Dataset):
    def __init__(self, data: (str | dict | tuple), *, prep=None, **kwargs) -> None:
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        if isinstance(data, str):
            path = data if data.endswith(".pt") else data + "/data_prepped.pt"
            data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            if prep is None:
                raise ValueError(
                    "LEGODataset(dict, ...) requires `prep` (e.g. DataPrep(config)); "
                    "GetLEGOData yields pre-format_add layout that DataStruct misaligns."
                )
            data = prep(GetLEGOData(**kwargs)(data))
        self.data = DataStruct(*data)

        frac = kwargs.get("frac")
        if frac:
            n = int(len(self.data) * frac)
            idxs = torch.randperm(len(self.data), device=self.device)[:n]
            self.data = self.data[idxs]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int | Tensor) -> DataStruct:
        return self.data[idx]
