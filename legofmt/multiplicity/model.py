"""``MultModel``: the count model, an autoregressive predictor of how many particles of each species leave.

```
sequence       [incoming, n_1, ..., n_{S-1}]: one slot per outgoing species s, in ptypes order
incoming       h_in = proj_in([scalars, e, mom, pos on the cuboid]) + embd_pp[incoming species]
               the directions in the canonical cube frame (canon_sym; sym_aug: a random cube symmetry on top)
count token    embd_in[s, n_s]: the count of species s, read by the slots after it
trunk          a causal Decoder over the sequence, h_in also its adaptive-norm condition;  out_s at slot s
counts         P(n_s | n_<s, incoming) = softmax(out_s W_s + b_s)             one linear head per slot
pass-through   P(no interaction | incoming) = sigmoid(pt_head(out_0))         for the species the head covers

loss           ce_b    = mean_s CE(out_s W_s + b_s, n_s)
               L_count = sum_b w_b ce_b / sum_b w_b     w_b = 0 for a covered pass-through event, else 1
               L_pt    = the BCE of the head, over the covered species
               L       = L_count + L_pt                 (L = mean_b ce_b without the head)
sampling       the pass-through flag first; then slot by slot, each count fed back as the next token (KV cache)
```

``MultLoader`` is its dataset: per event the incoming token, the counts per species, the pass-through flag. Superseded
(:mod:`legofmt.legacy.mult_model`): the inverse-species model ``InvModel``, the residual and ``in_embd`` pass-through
heads; a config without ``opt_conf`` and the schedule-free switch are :mod:`legofmt.legacy.optimizers`.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import random_split

from lightning import LightningModule

from x_transformers import ContinuousTransformerWrapper, Decoder

from legofmt.compat import fp32_attention, needs_fp32_attention

from legofmt.data.load.packed import load_dataset
from legofmt.data.load.loader import loader_seed, make_loader
from legofmt.data.write.norm import DataPrep
from legofmt.data.struct import cond_scalars
from legofmt.geometry.geom_trafos import GeomTrafos
from legofmt.geometry.symmetry_projections import CubeSymmetry
from legofmt.legacy.mult_model import InvModel, ResidualHead, inverse_pid, inverse_tensors
from legofmt.legacy.optimizers import ScheduleFreeSwitch, default_schedulefree, opt_is_schedulefree
from legofmt.mod_comps.config import resolve_mult_config
from legofmt.mod_comps.optimizers import build_optimizer
from legofmt.rng import COUNT, PASSTHROUGH, SYM_COUNT, SYM_PT

# the species the pass-through head covers unless mm_conf.passthrough_pdgids says otherwise: the neutrals
_NEUTRAL_PDGIDS = (22, 2112, 130, 310, 12, -12, 14, -14, 3122)


def slot_ce(logits: Tensor, counts: Tensor) -> Tensor:
    """Returns each event's cross-entropy of its counts, averaged over the slots: (B, S, P), (B, S) -> (B,)."""
    return F.cross_entropy(logits.flatten(0, 1), counts.flatten(), reduction="none").view(counts.shape).mean(-1)


class MultLoader(torch.utils.data.Dataset):
    """The count model's dataset: per event ``(in_tok, counts, pdgid_in_idx[, pt_target])``.

    ```
    in_tok         [*cond_scalars, e, mom (3), pos (3)] of the incoming particle
    counts         the particles out per species of ptypes, held below max_out_particles
    pdgid_in_idx   the incoming species' index into ptypes_in
    pt_target      1 for a pass-through event: no deposit and one particle out (passthrough_head)
    ```

    Args:
        config: the count model's config; ``mm_conf.cond_scalars`` is the layout the file was written with.
    """

    def __init__(self, config: dict):
        mm_conf, lds_conf = config["mm_conf"], config["dl_conf"]["lds_args"]
        ptypes    = mm_conf.get("ptypes", torch.tensor([11, 22]))
        ptypes_in = mm_conf.get("ptypes_in", torch.tensor([11, 22]))
        lds = load_dataset(**lds_conf, prep=DataPrep({
            "max_energy": mm_conf["max_energy"],
            "cutoff_mev": lds_conf.get("cutoff_mev"),
            "cond_scalars": mm_conf.get("cond_scalars", cond_scalars()),
            "edep_log_min": mm_conf.get("edep_log_min"),
        }))
        conds, incoming = torch.cat([lds.cond(n).unsqueeze(-1) for n in cond_scalars()], dim=-1), lds.incoming

        self.in_tok       = torch.cat((conds, incoming[:, :7]), dim=-1)
        self.counts       = lds.species_counts(ptypes).clamp_max(mm_conf.get("max_out_particles") - 1)
        self.pdgid_in_idx = torch.searchsorted(ptypes_in, incoming[:, -1].contiguous()).clamp(0, len(ptypes_in) - 1)
        self.pt_target    = None
        if mm_conf.get("passthrough_head", False):
            self.pt_target = ((lds.edep.reshape(-1) <= 0) & (lds.n_out == 1)).float()
        self.train_inverse = mm_conf.get("train_inverse", False)
        if self.train_inverse:   # legacy: InvModel's inputs
            self.out_tok, self.out_pid_idx, self.out_mask, self.edep = inverse_tensors(lds, conds, ptypes)

    def __len__(self):
        return self.in_tok.shape[0]

    def __getitem__(self, idx):
        ts = (self.in_tok, self.counts, self.pdgid_in_idx)
        if self.train_inverse:
            ts += (self.out_tok, self.out_pid_idx, self.out_mask, self.edep)
        if self.pt_target is not None:
            ts += (self.pt_target,)
        return tuple(t[idx] for t in ts)


class MultModel(ScheduleFreeSwitch, LightningModule):
    """The count model (module docstring): network, loss, sampling, and its Lightning hooks.

    Args:
        full_config: ``{"config": ..., "state_dict": ...}`` as a checkpoint stores it, or the config alone.
    """

    def __init__(self, full_config: dict):
        super().__init__()
        self.rc = rc = resolve_mult_config(full_config)
        mm = rc.mm_conf

        self.register_buffer("ptypes", rc.ptypes)
        self.register_buffer("ptypes_in", rc.ptypes_in)
        self.geom_trafos = GeomTrafos()
        self.register_buffer("pos_scale", rc.pos_scale, persistent=False)
        self.sym = CubeSymmetry() if rc.canon_sym else None

        # the trunk
        self.model = ContinuousTransformerWrapper(
            max_seq_len=rc.max_seq_len, emb_dropout=rc.dropout, use_abs_pos_emb=rc.use_abs_pos_emb,
            post_emb_norm=rc.post_emb_norm,
            attn_layers=Decoder(dim=rc.h_dim, depth=rc.n_layers, heads=rc.n_heads, attn_dropout=rc.dropout,
                                ff_dropout=rc.dropout, dim_condition=rc.h_dim, **rc.model_args),
        )
        if needs_fp32_attention(rc.model_args):
            fp32_attention(self.model)

        # the tokens: the incoming particle, and one embedding table for every (slot, count)
        self.proj_in_ = nn.Linear(rc.in_dim, rc.h_dim)
        self.embd_pp_ = nn.Embedding(rc.n_ptypes_in, rc.h_dim)
        self.embd_in_ = nn.Embedding((rc.max_seq_len - 1) * rc.max_particles, rc.h_dim)
        self.register_buffer("_in_offsets", torch.arange(rc.max_seq_len - 1, dtype=torch.long) * rc.max_particles,
                             persistent=False)

        # the pass-through head, and the incoming species it covers
        self.pt_head = None
        if mm.get("passthrough_head", False):
            pt_dim, depth = mm.get("passthrough_dim", rc.h_dim), mm.get("passthrough_depth", 0)
            self.pt_head = (ResidualHead(rc.h_dim, pt_dim, depth) if depth   # legacy
                            else nn.Sequential(nn.Linear(rc.h_dim, pt_dim), nn.Mish(), nn.Linear(pt_dim, 1)))
        self.pt_on_trunk = mm.get("passthrough_on", "in_embd") == "trunk"   # in_embd: legacy
        covered = {int(p) for p in mm.get("passthrough_pdgids", _NEUTRAL_PDGIDS)}
        covered = torch.tensor([int(p) in covered for p in rc.ptypes_in], dtype=torch.bool)
        self.register_buffer("_pt_pdgid_allowed", covered, persistent=False)

        # the count heads: one Linear(h_dim, max_particles) per slot, fused into one table, each with nn.Linear's init
        self.proj_out_w = nn.Parameter(torch.empty(rc.max_seq_len, rc.h_dim, rc.max_particles))
        self.proj_out_b = nn.Parameter(torch.empty(rc.max_seq_len, rc.max_particles))
        bound = 1.0 / rc.h_dim ** 0.5
        for i in range(rc.max_seq_len):
            nn.init.kaiming_uniform_(self.proj_out_w[i].transpose(0, 1), a=5 ** 0.5)
            nn.init.uniform_(self.proj_out_b[i], -bound, bound)

        self.inv = InvModel(rc) if rc.train_inverse else None   # legacy: its own parameters, trained on every step

        if rc.state_dict:
            missing = self.load_state_dict(rc.state_dict, strict=False).missing_keys
            if missing:
                warnings.warn(f"MultModel checkpoint lacks {missing}: those parameters keep a random init",
                              stacklevel=2)

        if rc.opt_conf is None: self.opt, self._sched = default_schedulefree(self.parameters(), mm), None   # legacy
        else: self.opt, self._sched = build_optimizer(self.parameters(), rc.opt_conf)
        self._opt_is_sf = opt_is_schedulefree(self.opt)   # read by lego_eval

    # ---- the network ------------------------------------------------------------------------------------------------

    def proj_in(self, x: Tensor, rng=None, stream: int = SYM_COUNT) -> Tensor:
        """Embeds the incoming token: its directions in the canonical cube frame, its position on the cuboid.

        Out of place throughout: ``log_prob`` differentiates through it.

        Args:
            x:      (B, in_dim), ``[*cond_scalars, e, mom (3), pos (3)]``.
            rng:    keyed draws (``legofmt.rng.EventRNG``) for the ``sym_aug`` symmetry, or None.
            stream: the keyed draw's site.

        Returns:
            (B, h_dim).
        """
        scalars, mom, pos = x.split((x.shape[-1] - 6, 3, 3), dim=-1)
        if self.sym is not None:
            face = self.sym.face_of(pos)
            g    = ((torch.randint(8, face.shape, device=face.device) if rng is None
                     else rng.randint(8, stream).view(face.shape).to(face.device)) if self.rc.sym_aug else None)
            mom, pos = self.sym.canonicalize(torch.stack((mom, pos), dim=-2), face, g).unbind(-2)
        on_cuboid = self.geom_trafos.to_cuboid(torch.cat((mom, pos), dim=-1), d=self.pos_scale)
        return self.proj_in_(torch.cat((scalars, on_cuboid), dim=-1))

    def _teacher_logits(self, in_tok: Tensor, counts: Tensor, pdgid_in_idx: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Runs the trunk over the true counts (teacher forcing).

        Returns:
            ``(logits, out, h_in)``: the count logits (B, S, P), the trunk's outputs (B, S, h_dim), the incoming
            embedding (B, h_dim).
        """
        h_in   = self.proj_in(in_tok) + self.embd_pp_(pdgid_in_idx)
        tokens = self.embd_in_(counts[:, :self.rc.max_seq_len - 1] + self._in_offsets)
        out    = self.model(torch.cat((h_in.unsqueeze(1), tokens), dim=1), mask=None, condition=h_in)
        return torch.einsum("bsh,shp->bsp", out, self.proj_out_w) + self.proj_out_b, out, h_in

    def _pt_logit(self, out_0: Tensor, h_in: Tensor) -> Tensor:
        """Returns the pass-through head's logit (B,), from the trunk's first output (``h_in``: the legacy head)."""
        return self.pt_head(out_0 if self.pt_on_trunk else h_in).squeeze(-1)

    # ---- the loss ----------------------------------------------------------------------------------------------------

    def training_step(self, batch: tuple, batch_idx) -> Tensor:
        """Returns the loss of a ``MultLoader`` batch (module docstring)."""
        in_tok, counts, pdgid_in_idx = batch[:3]
        logits, out, h_in = self._teacher_logits(in_tok, counts, pdgid_in_idx)
        ce = slot_ce(logits, counts)
        pt_logs = {}
        if self.pt_head is None:
            loss_count = total = ce.mean()
        else:
            passed, covered = batch[-1].float(), self._pt_pdgid_allowed[pdgid_in_idx]
            w          = 1.0 - passed * covered   # a covered pass-through event has no counts to learn
            loss_count = (ce * w).sum() / w.sum().clamp(min=1.0)
            bce        = F.binary_cross_entropy_with_logits(self._pt_logit(out[:, 0], h_in), passed,
                                                            reduction="none")
            loss_pt    = (bce * covered).sum() / covered.sum().clamp(min=1)
            total      = loss_count + loss_pt
            pt_logs    = {"loss/passthrough": loss_pt.detach()}

        logs = {"loss/counts": loss_count.detach()}
        if self.inv is not None:   # legacy
            loss_inv = F.cross_entropy(self.inv(*batch[3:7]), pdgid_in_idx)
            total = total + loss_inv
            logs["loss/pid_in"] = loss_inv.detach()
        # sync_dist=False: a per-step all-reduce only for logging stalled DDP ranks
        self.log_dict({"train_loss": total, **logs, **pt_logs}, prog_bar=True, sync_dist=False)
        return total

    def log_prob(self, in_tok: Tensor, counts: Tensor, pdgid_in_idx: Tensor, pt_mask: Tensor | None = None) -> Tensor:
        """Returns log P of a ``forward`` sample, ``(B,)``.

        The count slots' log-probabilities (teacher-forced) unless the pass-through token fired, plus the head's
        Bernoulli term for the species it covers (``pt_mask`` as ``sample_passthrough`` drew it). Differentiable in
        ``in_tok``: the score-function term of density design's gradient.

        Raises:
            NotImplementedError: under ``sym_aug``, where ``proj_in`` samples a new cube symmetry per call and would
                score the counts under another one than they were sampled with.
        """
        if self.rc.sym_aug:
            raise NotImplementedError("log_prob under sym_aug: proj_in draws a new cube transform per call, so the "
                                      "counts would be scored under another one than they were sampled with")
        self._opt_eval()
        self.eval()
        logits, out, h_in = self._teacher_logits(in_tok, counts, pdgid_in_idx)
        lp = logits.log_softmax(-1).gather(-1, counts.unsqueeze(-1)).squeeze(-1).sum(-1)
        if self.pt_head is None:
            return lp
        pt    = torch.zeros_like(lp, dtype=torch.bool) if pt_mask is None else pt_mask
        logit = self._pt_logit(out[:, 0], h_in)
        bern  = torch.where(pt, F.logsigmoid(logit), F.logsigmoid(-logit)) * self._pt_pdgid_allowed[pdgid_in_idx]
        return torch.where(pt, bern, lp + bern)

    # ---- sampling ----------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def passthrough_prob(self, in_tok: Tensor, pdgid_in_idx: Tensor, rng=None) -> Tensor:
        """Returns P("this particle did not interact"), 0 for the species the head does not cover."""
        if self.pt_head is None:
            return torch.zeros(in_tok.shape[0], device=in_tok.device)
        h_in = self.proj_in(in_tok, rng, SYM_PT) + self.embd_pp_(pdgid_in_idx)
        # slot 0 is causal, so the one-token pass equals training's out[:, 0]
        out_0 = self.model(h_in.unsqueeze(1), mask=None, condition=h_in)[:, 0] if self.pt_on_trunk else h_in
        return self.pt_head(out_0).squeeze(-1).sigmoid() * self._pt_pdgid_allowed[pdgid_in_idx]

    def sample_passthrough(self, in_tok: Tensor, pdgid_in_idx: Tensor, rng=None) -> Tensor:
        """Samples the Bernoulli "this particle did not interact".

        A keyed sample (``rng``, a ``legofmt.rng.EventRNG``) evaluates the head in eval mode: dropout would be
        randomness no key covers.
        """
        if rng is not None:
            self._opt_eval()
            self.eval()
        p = self.passthrough_prob(in_tok, pdgid_in_idx, rng)
        return (torch.rand_like(p) if rng is None else rng.uniform(PASSTHROUGH).to(p)) < p

    @torch.no_grad()
    def forward(self, batch: tuple, pt_mask: Tensor | None = None, rng=None) -> Tensor:
        """Samples the counts per species; puts the module in eval mode.

        Args:
            batch:   ``(in_tok, _, pdgid_in_idx)``.
            pt_mask: (B,), the pass-through events (``sample_passthrough``): each gets its incoming particle, once.
            rng:     keyed draws (``legofmt.rng.EventRNG``), or None.

        Returns:
            (B, S) counts.
        """
        self._opt_eval()
        self.eval()
        if batch[0].dim() == 3: return inverse_pid(self, batch)   # legacy: InvModel on an outgoing set
        in_tok, _, pdgid_in_idx = batch
        if pt_mask is None or not pt_mask.any():
            return self._decode(in_tok, pdgid_in_idx, rng)

        counts = in_tok.new_zeros(in_tok.shape[0], self.rc.max_seq_len, dtype=torch.long)
        slot   = torch.searchsorted(self.ptypes, self.ptypes_in[pdgid_in_idx[pt_mask]])
        counts[pt_mask, slot.clamp(max=self.rc.max_seq_len - 1)] = 1
        keep = ~pt_mask
        if keep.any():
            counts[keep] = self._decode(in_tok[keep], pdgid_in_idx[keep], None if rng is None else rng.sub(keep))
        return counts

    def _decode(self, in_tok: Tensor, pdgid_in_idx: Tensor, rng=None) -> Tensor:
        """Samples the counts slot by slot, each fed back as the next token; the trunk runs with a KV cache."""
        h_in   = self.proj_in(in_tok, rng) + self.embd_pp_(pdgid_in_idx)
        counts = in_tok.new_zeros(in_tok.shape[0], self.rc.max_seq_len, dtype=torch.long)
        x, cache = h_in.unsqueeze(1), None
        for s in range(self.rc.max_seq_len):
            out, cache = self.model(x, mask=None, condition=h_in, return_intermediates=True, cache=cache,
                                    input_not_include_cache=(s > 0))
            probs = (out[:, -1] @ self.proj_out_w[s] + self.proj_out_b[s]).softmax(-1)
            counts[:, s] = (torch.multinomial(probs, 1).squeeze(-1) if rng is None
                            else rng.categorical(probs, COUNT, s))
            if s < self.rc.max_seq_len - 1:
                x = self.embd_in_(counts[:, s] + self._in_offsets[s]).unsqueeze(1)
        return counts

    # ---- Lightning: validation on sampled counts -------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        # rows: [0] generated counts, [1] true counts, [2, 0] event count
        self._val_acc = torch.zeros(3, self.rc.max_seq_len, device=self.device)

    @torch.no_grad()
    def validation_step(self, batch: tuple, batch_idx) -> None:
        """Accumulates free-running per-species counts beside the true ones; the loss is not validated.

        The loss saturates at the data's irreducible entropy while the sampled counts are still several percent off, so
        only a rollout metric tracks the thing that is actually wrong. ``force_eager`` keeps the 19 decode shapes out of
        the dynamo cache the compiled training step lives in. It runs in fp32, like GenerateOut: under bf16-mixed the
        count logits (|logit| up to ~100) moved per-species counts by up to 3.7%, the size of the errors being tracked.
        """
        in_tok, counts, pdgid_in_idx = batch[:3]
        with torch.compiler.set_stance("force_eager"), torch.autocast(in_tok.device.type, enabled=False):
            pt_mask = self.sample_passthrough(in_tok, pdgid_in_idx)
            gen = self((in_tok, None, pdgid_in_idx), pt_mask=pt_mask)
        self._val_acc[0] += gen.sum(0)
        self._val_acc[1] += counts.sum(0)
        self._val_acc[2, 0] += in_tok.shape[0]

    def on_validation_epoch_end(self) -> None:
        """Logs the generated / true mean count: in total, per species, and the ratios' mean absolute deviation."""
        acc = self._val_acc
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(acc, op=torch.distributed.ReduceOp.SUM)
        n = acc[2, 0].clamp(min=1)
        gen, true = acc[0] / n, acc[1] / n
        # keep_species comes from the reduced sums, so every rank logs the same key set;
        # per-rank keys would leave the ranks issuing different collectives
        keep_species = true > self.rc.val_conf.get("count_floor", 0.005)
        ratio = gen[keep_species] / true[keep_species]
        self.log_dict({
            "val/n_out_ratio": gen.sum() / true.sum().clamp(min=1e-9),
            "val/count_mae": (ratio - 1).abs().mean(),
        }, sync_dist=False)
        for i in keep_species.nonzero(as_tuple=True)[0].tolist():
            self.log(f"val/count_ratio/{int(self.ptypes[i])}", gen[i] / true[i], sync_dist=False)

    # ---- Lightning: data and optimizer ------------------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Loads the dataset once and splits off the validation events."""
        if getattr(self, "_val_ds", None) is not None:
            return
        full  = MultLoader(self.rc.config)
        n_val = max(1, int(len(full) * self.rc.val_conf.get("val_frac", 0.01)))
        gen   = torch.Generator().manual_seed(self.rc.val_conf.get("seed", 0))
        self._train_ds, self._val_ds = random_split(full, [len(full) - n_val, n_val], generator=gen)

    def train_dataloader(self):
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self._val_ds, shuffle=False)

    def _make_loader(self, dataset, *, shuffle: bool):
        """Returns ``make_loader`` with the config's batch size, workers and seed."""
        return make_loader(dataset, bs=self.rc.mm_conf.get("bs", 2**12), shuffle=shuffle,
                           num_workers=self.rc.dl_conf.get("num_workers", 4), seed=loader_seed(self.rc.dl_conf))

    def configure_optimizers(self):
        """Returns the optimizer, its schedule rebuilt on the trainer's exact step count."""
        if self._sched is None:
            return self.opt
        # the trainer's exact count, not scripts/train.py's dataset_size estimate
        self.rc.opt_conf["scheduler"]["total_steps"] = int(self.trainer.estimated_stepping_batches)
        self.opt, self._sched = build_optimizer(self.parameters(), self.rc.opt_conf)
        return {"optimizer": self.opt, "lr_scheduler": self._sched}

    def on_fit_start(self):
        self._opt_train()   # legacy: schedule-free

    def on_fit_end(self):
        self._opt_eval()    # legacy: schedule-free
