"""``MultModel``: the autoregressive per-pdgid count predictor.

An x-transformers Decoder over one slot per outgoing PDG id, conditioned on the
incoming particle and the per-event scalars, trained with cross-entropy.
``mm_conf.train_inverse`` adds ``InvModel``, which predicts the incoming PID
from the outgoing set and backs ``GenerateIn``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

from lightning import LightningModule

from x_transformers import ContinuousTransformerWrapper, Decoder, Encoder

from legofmt.compat import fp32_attention, needs_fp32_attention

from legofmt.data.dataloaders import LEGODataset, make_loader
from legofmt.data.prep import DataPrep
from legofmt.data.struct import cond_scalars
from legofmt.geometry.geom_trafos import GeomTrafos
from legofmt.geometry.symmetry_projections import CubeSymmetry
from legofmt.mod_comps.config import resolve_mult_config
from legofmt.mod_comps.optimizers import (
    build_optimizer, opt_eval, opt_is_schedulefree, opt_train, schedulefree_adamw,
)


_NEUTRAL_PDGIDS = (22, 2112, 130, 310, 12, -12, 14, -14, 3122)


def _passthrough_allowed(rc) -> torch.Tensor:
    allowed = rc.mm_conf.get("passthrough_pdgids", _NEUTRAL_PDGIDS)
    allowed = {int(p) for p in allowed}
    return torch.tensor([int(p) in allowed for p in rc.ptypes_in], dtype=torch.bool)


class _ResidualHead(nn.Module):
    """depth x pre-norm [Linear -> Mish -> Linear] residual blocks, then a
    linear read-out; the shape of the trunk's own slot-0 computation."""

    def __init__(self, h: int, d: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.RMSNorm(h), nn.Linear(h, d), nn.Mish(), nn.Linear(d, h))
            for _ in range(depth)
        ])
        self.out = nn.Sequential(nn.RMSNorm(h), nn.Linear(h, 1))

    def forward(self, x):
        for b in self.blocks:
            x = x + b(x)
        return self.out(x)


class MultLoader(torch.utils.data.Dataset):
    """Flattens a prepped dataset into (in_tok, counts, pdgid_in_idx) plus the
    inverse-model tensors when train_inverse; assumes mm_conf.cond_scalars is
    the layout the file was written with.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        self.device = device
        mm_conf = config.get("mm_conf")
        lds_conf = config.get("dl_conf").get("lds_args").copy()
        max_particles = mm_conf.get("max_out_particles")
        ptypes = mm_conf.get("ptypes", torch.tensor([11, 22]))
        ptypes_in = mm_conf.get("ptypes_in", torch.tensor([11, 22]))
        self.train_inverse = mm_conf.get("train_inverse", False)

        ds = LEGODataset(**lds_conf, prep=DataPrep({
            "max_energy": mm_conf["max_energy"],
            "cutoff_mev": lds_conf.get("cutoff_mev"),
            "cond_scalars": mm_conf.get("cond_scalars", cond_scalars()),
            "edep_log_min": mm_conf.get("edep_log_min"),
        })).data
        meta_cond = config.get("additional", {}).get("data_meta", {}).get("cond_scalars")
        if meta_cond and tuple(meta_cond) != tuple(mm_conf["cond_scalars"]):
            raise ValueError(
                f"mm_conf.cond_scalars={tuple(mm_conf['cond_scalars'])} != dataset layout {tuple(meta_cond)}"
            )
        ds_f = ds.f
        # in_dim must be len(cond_scalars) + 7
        conds = torch.cat([ds_f.cond(n).unsqueeze(-1) for n in cond_scalars()], dim=-1)

        self.pdgid_in_idx = torch.searchsorted(
            ptypes_in, ds_f.in_p[..., 0, -1].contiguous()
        ).clamp(0, ptypes_in.shape[0] - 1)
        self.in_tok = torch.cat((conds, ds_f.in_cc.squeeze(-2)), dim=-1)
        self.counts = (ds_f.out_p[..., -1:] == ptypes.view(1, 1, -1)).sum(1).clamp_max(max_particles - 1)
        self.passthrough = None
        if mm_conf.get("passthrough_head", False):
            n_out = ds.am.out_p.sum(-1)
            self.passthrough = ((ds_f.edep.reshape(-1) <= 0) & (n_out == 1)).float()

        if self.train_inverse:
            out_cc = ds_f.out_cc.nan_to_num()
            self.out_tok = torch.cat(
                (conds.unsqueeze(1).expand(-1, out_cc.shape[1], -1), out_cc), dim=-1
            ).contiguous()
            self.out_pid_idx = torch.searchsorted(ptypes, ds_f.out_p[..., -1].long()).clamp(max=ptypes.shape[0] - 1)
            self.out_mask = ds.am.out_p.bool().clone()
            self.edep = ds_f.edep.clone()

    def __len__(self):
        return self.in_tok.shape[0]

    def __getitem__(self, idx):
        base = (
            self.in_tok[idx].to(self.device),
            self.counts[idx].to(self.device),
            self.pdgid_in_idx[idx].to(self.device),
        )
        pt = () if self.passthrough is None else (self.passthrough[idx].to(self.device),)
        if not self.train_inverse:
            return base + pt
        return base + (
            self.out_tok[idx].to(self.device),
            self.out_pid_idx[idx].to(self.device),
            self.out_mask[idx].to(self.device),
            self.edep[idx].to(self.device),
        ) + pt


class InvModel(nn.Module):
    """Inverse-direction predictor: outgoing shower (set) -> incoming PID."""

    def __init__(self, rc):
        super().__init__()
        self.rc = rc
        self.geom_trafos = GeomTrafos()

        self.model = ContinuousTransformerWrapper(
            max_seq_len=rc.mm_conf.get("max_out_particles", 0) + 1,
            emb_dropout=rc.dropout,
            use_abs_pos_emb=rc.use_abs_pos_emb,
            post_emb_norm=rc.post_emb_norm,
            attn_layers=Encoder(
                dim=rc.inv_h_dim,
                depth=rc.inv_n_layers,
                heads=rc.inv_n_heads,
                attn_dropout=rc.dropout,
                ff_dropout=rc.dropout,
                dim_condition=rc.inv_h_dim,
                **rc.inv_model_args,
            ),
        )
        if needs_fp32_attention(rc.inv_model_args):
            fp32_attention(self.model)

        self.proj_in_ = nn.Linear(rc.in_dim, rc.inv_h_dim)
        self.embd_out_ = nn.Embedding(rc.ptypes.shape[0], rc.inv_h_dim)
        self.proj_cond_ = nn.Linear(1, rc.inv_h_dim)
        self.embd_query_ = nn.Parameter(torch.randn(rc.inv_h_dim) * 0.02)
        self.proj_pid_ = nn.Linear(rc.inv_h_dim, rc.n_ptypes_in)

    def proj_in(self, x):
        x = x.clone()
        x[..., -6:] = self.geom_trafos.to_cube(x[..., -6:], d=self.rc.pos_scale)
        return self.proj_in_(x)

    def forward(self, out_tok, out_pid_idx, out_mask, edep):
        """Incoming-PID logits from the outgoing set's tokens, pids, mask and E_dep."""
        tok = self.proj_in(out_tok) + self.embd_out_(out_pid_idx)
        out = self.model(
            tok, mask=out_mask, condition=self.proj_cond_(edep.unsqueeze(-1)),
            prepend_embeds=self.embd_query_.expand(tok.shape[0], 1, -1),
            prepend_mask=out_mask.new_ones(tok.shape[0], 1),
        )
        return self.proj_pid_(out[:, 0])


class MultModel(LightningModule):
    """Autoregressive per-pdgid count decoder: one slot per ptypes entry with
    fused per-slot count heads (proj_out_w/b), conditioned on the incoming token.
    """

    def __init__(self, full_config: dict):
        super().__init__()
        rc = resolve_mult_config(full_config)
        self.rc = rc

        self.register_buffer("ptypes", rc.ptypes)
        self.register_buffer("ptypes_in", rc.ptypes_in)
        self.geom_trafos = GeomTrafos()
        self.sym = CubeSymmetry() if rc.canon_sym else None

        self.model = ContinuousTransformerWrapper(
            max_seq_len=rc.max_seq_len,
            emb_dropout=rc.dropout,
            use_abs_pos_emb=rc.use_abs_pos_emb,
            post_emb_norm=rc.post_emb_norm,
            attn_layers=Decoder(
                dim=rc.h_dim,
                depth=rc.n_layers,
                heads=rc.n_heads,
                attn_dropout=rc.dropout,
                ff_dropout=rc.dropout,
                dim_condition=rc.h_dim,
                **rc.model_args,
            ),
        )
        if needs_fp32_attention(rc.model_args):
            fp32_attention(self.model)

        self.proj_in_ = torch.nn.Linear(rc.in_dim, rc.h_dim)

        self.embd_pp_ = torch.nn.Embedding(rc.n_ptypes_in, rc.h_dim)
        self.embd_in_ = torch.nn.Embedding((rc.max_seq_len - 1) * rc.max_particles, rc.h_dim)
        self.register_buffer(
            "_in_offsets",
            torch.arange(rc.max_seq_len - 1, dtype=torch.long) * rc.max_particles,
            persistent=False,
        )
        self.pt_head = None
        if rc.mm_conf.get("passthrough_head", False):
            pt_dim = rc.mm_conf.get("passthrough_dim", rc.h_dim)
            depth = rc.mm_conf.get("passthrough_depth", 0)
            self.pt_head = _ResidualHead(rc.h_dim, pt_dim, depth) if depth else nn.Sequential(
                nn.Linear(rc.h_dim, pt_dim), nn.Mish(), nn.Linear(pt_dim, 1),
            )
        self.pt_on_trunk = rc.mm_conf.get("passthrough_on", "in_embd") == "trunk"
        self.register_buffer("_pt_allowed", _passthrough_allowed(rc), persistent=False)
        self.proj_out_w = torch.nn.Parameter(torch.empty(rc.max_seq_len, rc.h_dim, rc.max_particles))
        self.proj_out_b = torch.nn.Parameter(torch.empty(rc.max_seq_len, rc.max_particles))
        _bound = 1.0 / (rc.h_dim ** 0.5)
        for _i in range(rc.max_seq_len):
            torch.nn.init.kaiming_uniform_(self.proj_out_w[_i].transpose(0, 1), a=5 ** 0.5)
            torch.nn.init.uniform_(self.proj_out_b[_i], -_bound, _bound)

        # Optional inverse-PID co-training head (separate params, no sharing).
        # Built at construction time and, when present, trained on every step,
        # so DDP sees no unused parameters in either mode (plain ``ddp`` works).
        self.inv = InvModel(rc) if rc.train_inverse else None

        if rc.state_dict is not None:
            self.load_state_dict(rc.state_dict, strict=False)

        if rc.opt_conf is None:
            self.opt = schedulefree_adamw(
                self.parameters(),
                lr=rc.mm_conf.get("lr", 1e-3),
                betas=(0.95, 0.999),
                weight_decay=rc.mm_conf.get("weight_decay", 0.0),
                warmup_steps=rc.mm_conf.get("warmup_steps", 0),
            )
            self._sched = None
        else:
            self.opt, self._sched = build_optimizer(self.parameters(), rc.opt_conf)
        self._opt_is_sf = opt_is_schedulefree(self.opt)

    def proj_in(self, x):
        x = x.clone()
        if self.sym is not None:
            face = self.sym.face_of(x[..., -3:])
            dirs = self.sym.canonicalize(torch.stack((x[..., -6:-3], x[..., -3:]), dim=-2), face)
            x[..., -6:-3], x[..., -3:] = dirs[..., 0, :], dirs[..., 1, :]
        x[..., -6:] = self.geom_trafos.to_cube(x[..., -6:], d=self.rc.pos_scale)
        return self.proj_in_(x)

    def _opt_train(self):
        opt_train(self.opt)

    def _opt_eval(self):
        opt_eval(self.opt)

    def on_fit_start(self):
        self._opt_train()

    def on_fit_end(self):
        self._opt_eval()

    def training_step(self, batch, batch_idx):
        in_tok, counts, pdgid_in_idx = batch[:3]

        in_embd = self.proj_in(in_tok) + self.embd_pp_(pdgid_in_idx)
        in_seq = torch.cat(
            (in_embd.unsqueeze(1), self.embd_in_(counts[:, : self.rc.max_seq_len - 1] + self._in_offsets)),
            dim=1,
        )
        out_f = self.model(in_seq, mask=None, condition=in_embd)
        logits_f = torch.einsum("bsh,shp->bsp", out_f, self.proj_out_w) + self.proj_out_b
        ce = F.cross_entropy(
            logits_f.reshape(-1, self.rc.max_particles), counts.reshape(-1), reduction="none",
        ).view(counts.shape).mean(-1)
        total, pt_logs = None, {}
        if self.pt_head is None:
            loss_count = ce.mean()
            total = loss_count
        else:
            allowed = self._pt_allowed[pdgid_in_idx]
            w = 1.0 - batch[-1].float() * allowed
            loss_count = (ce * w).sum() / w.sum().clamp(min=1.0)
            total = loss_count
            if allowed.any():
                pt_in = out_f[:, 0] if self.pt_on_trunk else in_embd
                loss_pt = F.binary_cross_entropy_with_logits(
                    self.pt_head(pt_in).squeeze(-1)[allowed], batch[-1].float()[allowed],
                )
                total = loss_count + loss_pt
                pt_logs = {"loss/passthrough": loss_pt.detach()}

        if self.inv is None:
            self.log_dict(
                {"train_loss": total, "loss/counts": loss_count.detach(), **pt_logs},
                prog_bar=True, sync_dist=False,  # per-step all-reduce only for logging stalled DDP ranks
            )
            return total

        out_tok, out_pid_idx, out_mask, edep = batch[3:7]
        loss_inv = F.cross_entropy(self.inv(out_tok, out_pid_idx, out_mask, edep), pdgid_in_idx)
        loss = total + loss_inv
        self.log_dict(
            {"train_loss": loss, "loss/counts": loss_count.detach(),
             "loss/pid_in": loss_inv.detach(), **pt_logs},
            prog_bar=True, sync_dist=False,  # per-step all-reduce only for logging stalled DDP ranks
        )
        return loss

    @torch.no_grad()
    def sample_passthrough(self, in_tok: torch.Tensor, pdgid_in_idx: torch.Tensor) -> torch.Tensor:
        """Bernoulli draw of "this particle did not interact", masked to neutrals."""
        if self.pt_head is None:
            return torch.zeros(in_tok.shape[0], dtype=torch.bool, device=in_tok.device)
        in_embd = self.proj_in(in_tok) + self.embd_pp_(pdgid_in_idx)
        pt_in = in_embd
        if self.pt_on_trunk:
            # slot 0 is causal, so this equals training's out_f[:, 0]
            pt_in = self.model(in_embd.unsqueeze(1), mask=None, condition=in_embd)[:, 0]
        p = self.pt_head(pt_in).squeeze(-1).sigmoid()
        return (torch.rand_like(p) < p) & self._pt_allowed[pdgid_in_idx]

    def configure_optimizers(self):
        if self._sched is None:
            return self.opt
        return {"optimizer": self.opt, "lr_scheduler": self._sched}

    def setup(self, stage: str | None = None) -> None:
        if getattr(self, "_val_ds", None) is not None:
            return
        full = MultLoader(self.rc.config)
        n_val = max(1, int(len(full) * self.rc.val_conf.get("val_frac", 0.01)))
        gen = torch.Generator().manual_seed(self.rc.val_conf.get("seed", 0))
        self._train_ds, self._val_ds = random_split(
            full, [len(full) - n_val, n_val], generator=gen,
        )

    def _make_loader(self, dataset, *, shuffle: bool):
        return make_loader(
            dataset,
            bs=self.rc.mm_conf.get("bs", 2**12),
            shuffle=shuffle,
            num_workers=self.rc.dl_conf.get("num_workers", 4),
            batched_sampler=True,
        )

    def train_dataloader(self):
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self._val_ds, shuffle=False)

    def on_validation_epoch_start(self) -> None:
        # rows: [0] generated counts, [1] true counts, [2, 0] event count
        self._val_acc = torch.zeros(3, self.rc.max_seq_len, device=self.device)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """Free-running per-species counts, not the loss.

        The loss saturates at the data's irreducible entropy while the sampled
        counts are still several percent off, so only a rollout metric tracks
        the thing that is actually wrong. ``force_eager`` keeps the 19 decode
        shapes out of the dynamo cache the compiled training step lives in.
        """
        in_tok, counts, pdgid_in_idx = batch[:3]
        with torch.compiler.set_stance("force_eager"):
            skip = self.sample_passthrough(in_tok, pdgid_in_idx)
            gen = self((in_tok, None, pdgid_in_idx), skip=skip)
        self._val_acc[0] += gen.sum(0)
        self._val_acc[1] += counts.sum(0)
        self._val_acc[2, 0] += in_tok.shape[0]

    def on_validation_epoch_end(self) -> None:
        acc = self._val_acc
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(acc, op=torch.distributed.ReduceOp.SUM)
        n = acc[2, 0].clamp(min=1)
        gen, true = acc[0] / n, acc[1] / n
        # keep comes from the reduced sums, so every rank logs the same key set;
        # per-rank keys would leave the ranks issuing different collectives
        keep = true > self.rc.val_conf.get("count_floor", 0.005)
        ratio = gen[keep] / true[keep]
        self.log_dict({
            "val/n_out_ratio": gen.sum() / true.sum().clamp(min=1e-9),
            "val/count_mae": (ratio - 1).abs().mean(),
        }, sync_dist=False)
        for i in keep.nonzero(as_tuple=True)[0].tolist():
            self.log(f"val/count_ratio/{int(self.ptypes[i])}", gen[i] / true[i], sync_dist=False)

    @torch.no_grad()
    def forward(self, batch: (tuple | torch.Tensor), skip: torch.Tensor | None = None):
        """Inference only: a 3-D batch[0] runs InvModel (outgoing set -> incoming
        PID index), a 2-D one decodes counts autoregressively with a KV cache;
        puts the module in eval mode."""
        self._opt_eval()
        self.eval()
        if batch[0].dim() == 3:
            if self.inv is None:
                raise RuntimeError(
                    "MultModel was built with train_inverse=False; inverse-PID "
                    "inference is unavailable. Retrain with mm_conf.train_inverse=true."
                )
            return self.inv(*batch[:4]).argmax(-1)

        in_tok, _, pdgid_in_idx = batch
        counts = in_tok.new_zeros(in_tok.shape[0], self.rc.max_seq_len, dtype=torch.long)
        if skip is not None and skip.any():
            pid = self.ptypes_in[pdgid_in_idx[skip]]
            counts[skip, torch.searchsorted(self.ptypes, pid).clamp(max=self.rc.max_seq_len - 1)] = 1
            keep = ~skip
            if not keep.any():
                return counts
            sub = self.forward((in_tok[keep], None, pdgid_in_idx[keep]))
            counts[keep] = sub
            return counts

        in_embd = self.proj_in(in_tok) + self.embd_pp_(pdgid_in_idx)
        x = in_embd.unsqueeze(1)

        cache = None
        for i in range(self.rc.max_seq_len):
            out, cache = self.model(
                x,
                mask=None,
                condition=in_embd,
                return_intermediates=True,
                cache=cache,
                input_not_include_cache=(i > 0),
            )
            logits = out[:, -1] @ self.proj_out_w[i] + self.proj_out_b[i]
            sampled = torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
            counts[:, i] = sampled
            if i < self.rc.max_seq_len - 1:
                x = self.embd_in_(sampled + i * self.rc.max_particles).unsqueeze(1)
        return counts
