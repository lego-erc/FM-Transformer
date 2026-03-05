import torch
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule

from x_transformers import ContinuousTransformerWrapper, Decoder

import torch.nn.functional as F
import schedulefree

from legofmt.data.dataloaders import LEGODataset


class MultLoader(torch.utils.data.Dataset):
    def __init__(self, path, max_particles):
        dataset = LEGODataset(path=path, cutoff_mev=10, min_particles=0)
        density = dataset.target[:, 0, 1]
        energy, cc, pdgid = dataset.target[:, 2:].split((1, 6, 1), dim=-1)
        pdgid_in = pdgid[:, 0].squeeze()

        self.em_count = (pdgid[:, 1:] == 11).sum(dim=1).flatten().clamp_max(max_particles - 1)
        self.gamma_count = (pdgid[:, 1:] == 22).sum(dim=1).flatten().clamp_max(max_particles - 1)
        self.counts = torch.stack((self.em_count, self.gamma_count), dim=1)
        self.pdgid_in_idx = (pdgid_in == pdgid_in.unique().view(-1, 1)).nonzero()[:, 0]
        self.incoming_cc = cc[:, 0]

    def __len__(self):
        return self.incoming_cc.shape[0]

    def __getitem__(self, idx):
        return (self.incoming_cc[idx], self.counts[idx], self.pdgid_in_idx[idx])


class MultModel(LightningModule, torch.nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        state_dict = config.get("state_dict", None)
        if "config" in config.keys():
            config = config.get("config")
        self.mm_conf = config.get("mm_conf", {}).copy()
        self.max_particles = self.mm_conf.pop("max_particles", 10)
        dropout = self.mm_conf.pop("dropout", 0.1)
        h_dim = self.mm_conf.pop("h_dim", 512)
        n_layers = self.mm_conf.pop("n_layers", 6)
        n_heads = self.mm_conf.pop("n_heads", 8)
        self.max_seq_len = self.mm_conf.pop("max_seq_len", 3)
        in_dim = self.mm_conf.pop("in_dim", 6)
        self.n_ptypes = self.mm_conf.pop("n_ptypes", 2)
        self.focal_gamma = self.mm_conf.pop("ce_focal_gamma", 2)
        lr = self.mm_conf.pop("lr", 1e-3)
        self.bs = self.mm_conf.pop("bs", 2**12)
        self.num_workers = self.mm_conf.pop("num_workers", 16)
        self.path = self.mm_conf.pop("path")
        use_abs_pos_emb = self.mm_conf.pop("use_abs_pos_emb", True)
        self.model = ContinuousTransformerWrapper(
            max_seq_len=self.max_seq_len,
            emb_dropout=dropout,
            use_abs_pos_emb=use_abs_pos_emb,
            attn_layers=Decoder(
                dim=h_dim,
                depth=n_layers,
                heads=n_heads,
                layer_dropout=dropout,
                attn_dropout=dropout,
                ff_dropout=dropout,
                dim_condition=h_dim,
                **self.mm_conf,
            )
        )

        self.proj_in_ = torch.nn.Sequential(
            torch.nn.Linear(in_dim, h_dim),
            torch.nn.Mish(),
            torch.nn.Linear(h_dim, h_dim),
        )

        self.embd_in_ = torch.nn.ModuleList([
            torch.nn.Embedding(self.max_particles, h_dim)
            for _ in range(self.max_seq_len - 1)
        ])

        self.embd_pp_ = torch.nn.Embedding(self.n_ptypes, h_dim)

        self.proj_out_ = torch.nn.ModuleList([
            torch.nn.Linear(h_dim, self.max_particles)
            for _ in range(self.max_seq_len - 1)
        ])

        if state_dict is not None:
            self.load_state_dict(state_dict, strict=True)
        self.opt = schedulefree.AdamWScheduleFree(self.parameters(), lr=lr, weight_decay=0.01)

    def proj_in(self, x):
        return F.mish(self.proj_in_(x))

    def on_fit_start(self):
        self.opt.train()
        self.train()

    def on_fit_end(self):
        self.opt.eval()
        self.eval()

    def training_step(self, batch, batch_idx):
        in_cc, counts, pdgid_in_idx = batch
        in_embd = self.proj_in(in_cc)
        pdgid_embd = in_embd + self.embd_pp_(pdgid_in_idx)
        gt_embds = torch.stack(
            [self.embd_in_[i](counts[:, i]) for i in range(self.max_seq_len - 2)], dim=1
        )
        in_seq = torch.cat((pdgid_embd.unsqueeze(1), gt_embds), dim=1)
        out = self.model(in_seq, mask=None, condition=pdgid_embd)
        logits = torch.stack(
            [self.proj_out_[i](out[:, i]) for i in range(self.max_seq_len - 1)], dim=1
        )

        loss_model = F.cross_entropy(logits.reshape(-1, self.max_particles), counts.reshape(-1), reduction="mean")

        make_used_ = sum(p.sum() * 0.0 for p in self.model.attn_layers.parameters())
        make_used_ += sum(p.sum() * 0.0 for p in self.embd_in_.parameters())
        make_used_ += sum(p.sum() * 0.0 for p in self.proj_out_.parameters())
        make_used_ += sum(p.sum() * 0.0 for p in self.embd_pp_.parameters())
        return loss_model + make_used_
    
    def configure_optimizers(self):
        return self.opt

    def train_dataloader(self):
        path = self.path + "/data_prepped.pt"
        dataset_train = MultLoader(path, self.max_particles)
        return DataLoader(
            dataset_train,
            batch_size=self.bs,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    @torch.no_grad()
    def forward(self, batch: (tuple | torch.Tensor)):
        self.opt.eval()
        self.model.eval()
        if isinstance(batch, tuple):
            in_cc, _, pdgid_in_idx = batch
        elif isinstance(batch, torch.Tensor):
            in_cc = batch.clone()
            if in_cc.shape[-1] == 8:
                in_cc = in_cc[..., 1:-1]
            in_cc = in_cc.view(-1, 6)
        in_embd = self.proj_in(in_cc)
        pdgid_embd = (in_embd + self.embd_pp_(pdgid_in_idx)).unsqueeze(1)
        counts = torch.empty(pdgid_embd.shape[0], self.max_seq_len - 1, dtype=torch.long, device=in_cc.device)

        for i in range(self.max_seq_len - 1):
            out = self.model(pdgid_embd, mask=None, condition=pdgid_embd[:, 0])[:, -1]
            logits = self.proj_out_[i](out)

            sampled = torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
            embd = self.embd_in_[i](sampled).unsqueeze(1)
            pdgid_embd = torch.cat((pdgid_embd, embd), dim=1)
            counts[:, i] = sampled

        return counts
