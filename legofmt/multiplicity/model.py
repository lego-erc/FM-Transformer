import torch
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule

from x_transformers import ContinuousTransformerWrapper, Decoder
from x_transformers import ContinuousAutoregressiveWrapper

import schedulefree

from legofmt.data.dataloaders import LEGODataset

class MultLoader(torch.utils.data.Dataset):
    def __init__(self, path):
        dataset = LEGODataset(path=path, cutoff_mev=10, min_particles=1)
        energy, cc, pdgid = dataset.target[:, 2:].split((1, 6, 1), dim=-1)

        cc_em_in = cc[(pdgid[:, 0] == 11).flatten()]
        pdgid_em_in = pdgid[(pdgid[:, 0] == 11).flatten(), 1:]

        self.em_count = (pdgid_em_in 
        == 11).sum(dim=1).flatten().clamp_max(9)
        self.gamma_count = (pdgid_em_in == 22).sum(dim=1).flatten().clamp_max(9)

        self.em_oh = torch.nn.functional.one_hot(self.em_count, num_classes=10)
        self.gamma_oh = torch.nn.functional.one_hot(self.gamma_count, num_classes=10)

        self.incoming_cc = cc_em_in[:, :1]
        
    def __len__(self):
        return self.incoming_cc.shape[0]
    
    def __getitem__(self, idx):
        return (self.incoming_cc[idx], self.em_oh[idx], self.gamma_oh[idx])

class CrossEntropyLossWrapper(torch.nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = self.loss_fn(pred.flatten(0, 1), target.flatten(0, 1).argmax(-1))
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class MultModel(LightningModule, torch.nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        state_dict = config.get("state_dict", None)
        if "config" in config.keys():
            config = config.get("config")
        self.mm_conf = config.get("mm_conf", {}).copy()
        max_particles = self.mm_conf.pop("max_particles", 10)
        dropout = self.mm_conf.pop("dropout", 0.1)
        h_dim = self.mm_conf.pop("h_dim", 512)
        n_layers = self.mm_conf.pop("n_layers", 6)
        n_heads = self.mm_conf.pop("n_heads", 8)
        max_seq_len = self.mm_conf.pop("max_seq_len", 3)
        in_dim = self.mm_conf.pop("in_dim", 6)
        ce_focal_gamma = self.mm_conf.pop("ce_focal_gamma", 2)
        lr = self.mm_conf.pop("lr", 1e-3)
        self.bs = self.mm_conf.pop("bs", 2**12)
        self.num_workers = self.mm_conf.pop("num_workers", 16)
        self.path = self.mm_conf.pop("path")
        model = ContinuousTransformerWrapper(
            dim_in = max_particles,
            dim_out = max_particles,
            max_seq_len = max_seq_len,
            emb_dropout=dropout,
            use_abs_pos_emb = False,
            attn_layers = Decoder(
                dim = h_dim,
                depth = n_layers,
                heads = n_heads,
                layer_dropout = dropout,
                attn_dropout = dropout,
                ff_dropout = dropout,
                **self.mm_conf,
            )
        )

        self.register_buffer("proj_in", torch.empty(max_particles, in_dim))
        torch.nn.init.xavier_normal_(self.proj_in, gain=1.0)

        self.model = ContinuousAutoregressiveWrapper(model, loss_fn=CrossEntropyLossWrapper(gamma=ce_focal_gamma))

        if state_dict is not None:
            self.load_state_dict(state_dict, strict=True)
        self.opt = schedulefree.AdamWScheduleFree(self.parameters(), lr=lr)

    def on_fit_start(self):
        self.opt.train()
        self.model.train()

    def training_step(self, batch, batch_idx):
        in_cc, em_count, gamma_count = batch
        in_embd = torch.einsum("ilj,kj->ik", in_cc, self.proj_in)
        batch = torch.stack((in_embd, em_count, gamma_count), dim=1)
        make_used_ = sum(
            p.sum() * 0.0 for p in self.model.net.attn_layers.parameters()
        )
        loss = self.model(batch) + make_used_
        return loss

    @torch.no_grad()
    def forward(self, batch):
        self.opt.eval()
        self.model.eval()
        in_cc = batch[0].unsqueeze(1)
        in_embd = torch.einsum("ilj,kj->ilk", in_cc, self.proj_in)
        
        out = in_embd
        for _ in range(2):
            logits = self.model.net(out)
            last_logits = logits[:, -1:]
            sampled = torch.distributions.Categorical(logits=last_logits).sample()
            one_hot = torch.nn.functional.one_hot(sampled, num_classes=self.proj_in.shape[0]).float()
            out = torch.cat((out, one_hot), dim=1)
            
        n_p = out[:, 1:].argmax(dim=-1)
        return n_p
    
    def configure_optimizers(self):
        return self.opt
    
    def train_dataloader(self):
        path = self.path + "/data_prepped.pt"
        dataset_train = MultLoader(path)
        return DataLoader(
            dataset_train, batch_size=self.bs, shuffle=True, num_workers=self.num_workers
        )