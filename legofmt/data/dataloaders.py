import torch
from torch import Tensor
from torch.utils.data import Dataset


class GetLEGOData:
    def __init__(
        self,
        cutoff_mev=0.0,
        min_particles=0,
        device="cpu",
        is_filtered=False,
        **kwargs,
    ):
        self.dev = device
        self.dtype = kwargs.pop("dtype", torch.float32)
        self.min_particles = min_particles
        self.cutoff_mev = cutoff_mev

        if is_filtered:
            self.func = self.get_filtered
        elif cutoff_mev is not None:
            self.func = self.dataset_cutoff

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def dataset_compact(self, path):
        data = torch.load(path, map_location="cpu", weights_only=False)

        data_pp = data.get("per_particle")
        data_add = data.get("per_event")
        if isinstance(data_add, torch.Tensor):
            data_add = data_add.to(self.dtype)
        elif isinstance(data_add, dict):
            data_add = data_add["E_dep"].to(self.dtype)

        if isinstance(data_pp, dict):
            data_pp = torch.cat((data_pp["Incoming"], data_pp["Outgoing"]), dim=-2)

        if data_pp.shape[-1] == 16:
            data_pp_filtered = torch.cat(
                (data_pp[:, 0:1, :8], data_pp[..., 8:]), dim=1
            ).to(self.dtype)
        elif data_pp.shape[-1] == 8:
            data_pp_filtered = data_pp.to(self.dtype)
        return data_pp_filtered, data_add

    def dataset_cutoff(
        self,
        path: str,
        n_events: (int | None) = None,
    ) -> tuple[Tensor, Tensor, Tensor, torch.distributions.Categorical]:
        """Load a dataset from the given path and preprocesses it to give particles above an energy threshold.

        Parameters
        ----------
        path : str
            The path to the dataset.

        Returns
        -------
        dataset : Tensor
            The preprocessed dataset with shape (bs, particles, features).
        mask : Tensor
            Inference type mask for the particles, 1 is RV to be flown, 0 is condition.
        attn_mask : Tensor
            Attention mask for the transformer.

        """
        dataset, data_add = self.dataset_compact(path)
        mask_valid = dataset[..., 0] >= self.cutoff_mev
        max_valid = mask_valid.sum(dim=-1).max()
        mask_valid_sorted = mask_valid.sort(dim=-1, descending=True).values[
            :, :max_valid
        ]
        dataset_valid = torch.empty_like(dataset)[:, :max_valid].fill_(torch.nan)
        dataset_valid[mask_valid_sorted] = dataset[mask_valid]
        idx_rel_events = ~dataset_valid[:, : self.min_particles + 1, 0].isnan().any(
            dim=-1
        )
        data_pp = dataset_valid[idx_rel_events]
        if n_events is not None:
            rd_idx = torch.randperm(data_pp.shape[0], device="cpu")[
                :n_events
            ]
            data_pp = data_pp[rd_idx]
        particle_nan = ~data_pp.isnan().any(dim=-1)
        attn_mask = particle_nan.to(torch.int64)
        mask = attn_mask.clone().unsqueeze(2)
        mask[:, 0] = 0
        return {
            "per_particle": (
                data_pp.to(self.dev),
                mask.to(self.dev),
                attn_mask.to(self.dev).bool()
            ),
            "per_event": data_add[idx_rel_events].to(self.dev),
        }

    def get_filtered(
        self,
        path: str,
        **kwargs,
    ) -> dict:
        dataset = torch.load(path, map_location="cpu")
        data_pp = dataset.get("per_particle")
        data_add = dataset.get("per_event").to(self.dtype)
        return {
            "per_particle": data_pp,
            "per_event": data_add,
        }


class LEGODataset(Dataset):
    def __init__(self, path: str, **kwargs) -> None:
        super().__init__()
        path = kwargs.pop("data_path", path)
        n_events = kwargs.pop("n_events", None)
        self.include_add = kwargs.pop("include_add", False)
        get_lego_data = GetLEGOData(**kwargs)
        self.data = get_lego_data(path, n_events=n_events)
        self.data_pp, self.mask, self.attn_mask = self.data.get("per_particle")
        self.data_coord = self.data_pp[..., 1:-1]
        self.data_add = self.data.get("per_event")
        self.length = self.data_pp.shape[0]
        self.max_particles = self.data_pp.shape[1]
        self.device = kwargs.get("device", "cpu")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int | Tensor) -> tuple[Tensor]:
        if self.include_add:
            return (
                self.data_pp[idx].to(self.device),
                self.mask[idx].to(self.device),
                self.attn_mask[idx].to(self.device),
                self.data_add[idx].to(self.device),
            )
        return (
            self.data_pp[idx].to(self.device),
            self.mask[idx].to(self.device),
            self.attn_mask[idx].to(self.device),
        )
