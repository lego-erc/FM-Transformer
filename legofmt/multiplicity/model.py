"""``MultModel``: the multiplicity model, an autoregressive model of the number of outgoing particles per species.

```
sequence       S slots, one per outgoing species s = 0 .. S-1 in ptypes order; the tokens are
               [incoming, n_0, ..., n_{S-2}], and slot s reads the incoming token and the counts before n_s
incoming       h_in = proj_in([scalars, e, mom, pos on the cuboid]) + embd_pp[incoming species]
               the directions in the canonical cube frame (canon_sym; sym_aug: a random symmetry of that face on top)
count token    embd_in[s, n_s]: the count of species s, read by the slots after it
trunk          a causal Decoder over the sequence, h_in also its adaptive-norm condition;  out_s at slot s
counts         P(n_s | n_<s, incoming) = softmax(out_s W_s + b_s)             one linear head per slot
pass-through   P(no interaction | incoming) = sigmoid(pt_head(out_0))         for the species the head covers

loss           ce_b    = mean_s CE(out_s W_s + b_s, n_s)
               L_count = sum_b w_b ce_b / sum_b w_b     w_b = 0 for a covered pass-through event, else 1
               L_pt    = -mean log Bernoulli(pass-through | sigmoid(pt_head(out_0))), over the covered events
               L       = L_count + L_pt                 (L = mean_b ce_b without the head)
log P          sum_s log softmax(out_s W_s + b_s)[n_s] + log Bernoulli(pass-through | sigmoid(pt_head(out_0)))
               a pass-through event: the Bernoulli term alone; a species the head does not cover: the sum alone
sampling       the pass-through flag first; then slot by slot, each count fed back as the next token (KV cache)
```

An event is covered when the pass-through head covers its incoming species. The formulas are module functions
(``count_log_prob``, ``bernoulli_log_prob``). ``MultLoader`` is the model's dataset: per event the incoming token, the
counts per species, the pass-through flag. Superseded (:mod:`legofmt.legacy.mult_model`): the inverse-species model
``InvModel``, the residual and ``in_embd`` pass-through heads; a config without ``opt_conf`` and the schedule-free
switch are :mod:`legofmt.legacy.optimizers`.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lightning import LightningModule

from x_transformers import ContinuousTransformerWrapper, Decoder

from legofmt.compat import fp32_attention, needs_fp32_attention

from legofmt.data.load.packed import load_dataset
from legofmt.data.load.loader import loader_seed, make_loader, split_val
from legofmt.data.write.norm import DataPrep
from legofmt.data.struct import cond_scalars, vocab_index
from legofmt.geometry.geom_trafos import GeomTrafos
from legofmt.geometry.symmetry_projections import CubeSymmetry
from legofmt.legacy.mult_model import InvModel, ResidualHead, inverse_pid, inverse_tensors
from legofmt.legacy.optimizers import ScheduleFreeSwitch, default_schedulefree, opt_is_schedulefree
from legofmt.config import resolve_mult_config
from legofmt.optimizers import build_optimizer
from legofmt.numerics import masked_mean
from legofmt.rng import COUNT, PASSTHROUGH, SYM_COUNT, SYM_PT, categorical, randint_like, rng_sub, uniform_like

# the species the pass-through head covers unless mm_conf.passthrough_pdgids says otherwise: the neutrals
_NEUTRAL_PDGIDS = (22, 2112, 130, 310, 12, -12, 14, -14, 3122)


def count_log_prob(logits: Tensor, counts: Tensor) -> Tensor:
    """Returns each slot's ``log softmax(logits_s)[n_s]``: (B, S, P), (B, S) -> (B, S)."""
    return torch.log_softmax(logits, dim=-1).gather(-1, counts.unsqueeze(-1)).squeeze(-1)


def bernoulli_log_prob(logit: Tensor, x: Tensor) -> Tensor:
    """Returns ``log P(x)`` of the Bernoulli with ``P(True) = sigmoid(logit)``.

    ``log sigmoid(logit)`` where ``x`` is True, ``log sigmoid(-logit)`` where it is False.
    """
    return torch.where(x, F.logsigmoid(logit), F.logsigmoid(-logit))


class MultLoader(torch.utils.data.Dataset):
    """The multiplicity model's dataset: per event ``(in_tok, counts, pdgid_in_idx[, pt_target])``.

    ```
    in_tok         [*cond_scalars, e, mom (3), pos (3)] of the incoming particle
    counts         the outgoing particles per species of ptypes, at most max_out_particles - 1 each
    pdgid_in_idx   the incoming species' index into ptypes_in
    pt_target      1 for a pass-through event: no deposit and one outgoing particle (passthrough_head)
    ```

    With the legacy ``train_inverse``, ``InvModel``'s four tensors come before ``pt_target``.

    Args:
        config: the multiplicity model's config, with the defaults ``resolve_mult_config`` fills in. The token layout
            must be set to ``mm_conf.cond_scalars``, the layout the dataset was written with (``resolve_mult_config``
            does that).
    """

    def __init__(self, config: dict):
        mm_conf, lds_conf = config["mm_conf"], config["dl_conf"]["lds_args"]
        ptypes    = mm_conf.get("ptypes", torch.tensor([11, 22]))
        ptypes_in = mm_conf.get("ptypes_in", torch.tensor([11, 22]))
        lds = load_dataset(**lds_conf, prep=DataPrep(_prep_config(mm_conf, lds_conf)))
        conds, incoming = torch.stack([lds.cond(n) for n in cond_scalars()], dim=-1), lds.incoming

        self.in_tok       = torch.cat((conds, incoming[:, :7]), dim=-1)
        self.counts       = lds.species_counts(ptypes).clamp(max=mm_conf.get("max_out_particles") - 1)
        self.pdgid_in_idx = vocab_index(ptypes_in, incoming[:, -1].contiguous())[0]
        self.pt_target    = None
        if mm_conf.get("passthrough_head", False):
            self.pt_target = ((lds.edep.flatten() <= 0) & (lds.n_out == 1)).float()
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
        in_tok, counts, *rest = (t[idx] for t in ts)
        return (in_tok, counts.long(), *rest)   # stored as int16; a batch needs int64 class targets


class MultModel(ScheduleFreeSwitch, LightningModule):
    """The multiplicity model (module docstring): its network, loss, sampling and Lightning hooks.

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

        # the tokens: the incoming particle, and one table with an embedding for every (slot, count)
        self.proj_in_ = nn.Linear(rc.in_dim, rc.h_dim)
        self.embd_pp_ = nn.Embedding(rc.n_ptypes_in, rc.h_dim)
        self.embd_in_ = nn.Embedding((rc.max_seq_len - 1) * rc.max_particles, rc.h_dim)
        self.register_buffer("_in_offsets", torch.arange(rc.max_seq_len - 1, dtype=torch.long) * rc.max_particles,
                             persistent=False)

        # the pass-through head, and the incoming species it covers
        self.pt_head     = _pt_head(rc, mm)
        self.pt_on_trunk = mm.get("passthrough_on", "in_embd") == "trunk"   # in_embd: legacy
        self.register_buffer("_pt_pdgid_allowed", _pt_covered(mm, rc.ptypes_in), persistent=False)

        # the count heads: one Linear(h_dim, max_particles) per slot, stacked in one tensor, each with nn.Linear's init
        self.proj_out_w = nn.Parameter(torch.empty(rc.max_seq_len, rc.h_dim, rc.max_particles))
        self.proj_out_b = nn.Parameter(torch.empty(rc.max_seq_len, rc.max_particles))
        self._init_count_heads()

        self.inv = InvModel(rc) if rc.train_inverse else None   # legacy: its own parameters, trained on every step

        if rc.state_dict:
            self._load_checkpoint(rc.state_dict)
        if rc.opt_conf is None: self.opt, self._sched = default_schedulefree(self.parameters(), mm), None   # legacy
        else: self.opt, self._sched = build_optimizer(self.parameters(), rc.opt_conf)
        self._opt_is_sf = opt_is_schedulefree(self.opt)   # read by lego_eval

    # ---- the network ------------------------------------------------------------------------------------------------

    def proj_in(self, x: Tensor, rng=None, stream: int = SYM_COUNT) -> Tensor:
        """Embeds the incoming token: its directions in the canonical cube frame, its position on the cuboid.

        Without the species, which ``embd_pp_`` adds. Out of place throughout: ``log_prob`` differentiates through it.

        Args:
            x:      (B, in_dim), ``[*cond_scalars, e, mom (3), pos (3)]``.
            rng:    keys the ``sym_aug`` symmetry on the event (``legofmt.rng.EventRNG``), or None.
            stream: the draw site's id (``legofmt.rng``), which a keyed symmetry uses.

        Returns:
            (B, h_dim), the embedding.
        """
        scalars, mom, pos = x.split((x.shape[-1] - 6, 3, 3), dim=-1)
        if self.sym is not None:   # the canonical cube frame; sym_aug: a random symmetry of that face on top
            face = self.sym.face_of(pos)
            g    = randint_like(8, face, rng, stream) if self.rc.sym_aug else None
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
        """Returns the pass-through head's logit, (B,), from the trunk's output at slot 0.

        The legacy ``in_embd`` head reads the incoming embedding ``h_in`` instead.
        """
        return self.pt_head(out_0 if self.pt_on_trunk else h_in).squeeze(-1)

    # ---- the loss ----------------------------------------------------------------------------------------------------

    def training_step(self, batch: tuple, batch_idx) -> Tensor:
        """Returns the loss of a ``MultLoader`` batch (module docstring)."""
        in_tok, counts, pdgid_in_idx = batch[:3]
        logits, out, h_in = self._teacher_logits(in_tok, counts, pdgid_in_idx)
        ce = -count_log_prob(logits, counts).mean(-1)   # each event's cross-entropy, averaged over the slots
        loss_pt = loss_inv = None
        if self.pt_head is None:
            loss_count = total = ce.mean()
        else:
            passed, covered = batch[-1].float(), self._pt_pdgid_allowed[pdgid_in_idx]
            loss_count = masked_mean(ce, 1.0 - passed * covered)   # a covered pass-through event has no count loss
            loss_pt    = masked_mean(-bernoulli_log_prob(self._pt_logit(out[:, 0], h_in), passed.bool()), covered)
            total      = loss_count + loss_pt
        if self.inv is not None:   # legacy
            loss_inv = F.cross_entropy(self.inv(*batch[3:7]), pdgid_in_idx)
            total    = total + loss_inv
        self._log_losses(total, counts=loss_count, passthrough=loss_pt, pid_in=loss_inv)
        return total

    def log_prob(self, in_tok: Tensor, counts: Tensor, pdgid_in_idx: Tensor, pt_mask: Tensor | None = None) -> Tensor:
        """Returns log P of a ``forward`` sample, (B,).

        The sum of the slots' log-probabilities of ``counts``, plus the pass-through head's Bernoulli term where it
        covers the incoming species; a pass-through event has the Bernoulli term alone. Differentiable in ``in_tok``:
        the score-function term of the density design's gradient.

        Args:
            in_tok:       (B, in_dim), the incoming tokens.
            counts:       (B, S), the counts, as ``forward`` returned them.
            pdgid_in_idx: (B,), the incoming species' index into ``ptypes_in``.
            pt_mask:      (B,), the pass-through flags, as ``sample_passthrough`` returned them; None: none passed.

        Raises:
            NotImplementedError: under ``sym_aug``, where ``proj_in`` samples a new cube symmetry per call and would
                score the counts under another one than they were sampled with.
        """
        self._check_scorable()
        self._eval_mode()
        logits, out, h_in = self._teacher_logits(in_tok, counts, pdgid_in_idx)
        lp = count_log_prob(logits, counts).sum(-1)
        if self.pt_head is None:
            return lp
        pt   = torch.zeros_like(lp, dtype=torch.bool) if pt_mask is None else pt_mask
        bern = bernoulli_log_prob(self._pt_logit(out[:, 0], h_in), pt) * self._pt_pdgid_allowed[pdgid_in_idx]
        return torch.where(pt, bern, lp + bern)   # a pass-through event has no counts to score

    # ---- sampling ----------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def passthrough_prob(self, in_tok: Tensor, pdgid_in_idx: Tensor, rng=None) -> Tensor:
        """Returns each event's probability, (B,), that the incoming particle passes through unchanged.

        0 for a species the pass-through head does not cover, and without a head. ``rng`` keys the ``sym_aug``
        symmetry of ``proj_in``.
        """
        if self.pt_head is None:
            return torch.zeros(in_tok.shape[0], device=in_tok.device)
        h_in = self.proj_in(in_tok, rng, SYM_PT) + self.embd_pp_(pdgid_in_idx)
        return torch.sigmoid(self._pt_logit(self._slot0(h_in), h_in)) * self._pt_pdgid_allowed[pdgid_in_idx]

    def sample_passthrough(self, in_tok: Tensor, pdgid_in_idx: Tensor, rng=None) -> Tensor:
        """Samples each event's pass-through flag, (B,), with probability ``passthrough_prob``.

        With ``rng`` (a ``legofmt.rng.EventRNG``) the module is put in eval mode first: dropout is randomness that no
        key covers.
        """
        if rng is not None:
            self._eval_mode()
        p = self.passthrough_prob(in_tok, pdgid_in_idx, rng)
        return uniform_like(p, rng, PASSTHROUGH) < p

    @torch.no_grad()
    def forward(self, batch: tuple, pt_mask: Tensor | None = None, rng=None) -> Tensor:
        """Samples the counts per species; puts the module in eval mode.

        Args:
            batch:   ``(in_tok, _, pdgid_in_idx)``; the second entry is not read.
            pt_mask: (B,), the pass-through events (``sample_passthrough``): each gets one particle of its incoming
                species and nothing else.
            rng:     keys every count on the event (``legofmt.rng.EventRNG``), or None.

        Returns:
            (B, S), the counts per species of ``ptypes``.
        """
        self._eval_mode()
        if batch[0].dim() == 3: return inverse_pid(self, batch)   # legacy: InvModel on an outgoing set
        in_tok, _, pdgid_in_idx = batch
        if pt_mask is None or not pt_mask.any():
            return self._decode(in_tok, pdgid_in_idx, rng)

        counts = self._incoming_once(in_tok, pdgid_in_idx, pt_mask)
        keep = ~pt_mask
        if keep.any():
            counts[keep] = self._decode(in_tok[keep], pdgid_in_idx[keep], rng_sub(rng, keep))
        return counts

    def _decode(self, in_tok: Tensor, pdgid_in_idx: Tensor, rng=None) -> Tensor:
        """Samples the counts slot by slot, each fed back as the next token; the trunk runs with a KV cache."""
        h_in   = self.proj_in(in_tok, rng) + self.embd_pp_(pdgid_in_idx)
        counts = in_tok.new_zeros(in_tok.shape[0], self.rc.max_seq_len, dtype=torch.long)
        x, cache = h_in.unsqueeze(1), None   # the incoming token first
        for s in range(self.rc.max_seq_len):
            out, cache   = self._trunk_step(x, h_in, cache, s)
            probs        = torch.softmax(out[:, -1] @ self.proj_out_w[s] + self.proj_out_b[s], dim=-1)
            counts[:, s] = categorical(probs, rng, COUNT, s)
            if s < self.rc.max_seq_len - 1:   # the count token of species s, read by the slots after it
                x = self.embd_in_(counts[:, s] + self._in_offsets[s]).unsqueeze(1)
        return counts

    # ---- Lightning: validation on sampled counts -------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        # rows: [0] the generated counts, [1] the true counts, both summed per species; [2, 0] the number of events
        self._val_acc = torch.zeros(3, self.rc.max_seq_len, device=self.device)

    @torch.no_grad()
    def validation_step(self, batch: tuple, batch_idx) -> None:
        """Accumulates the sampled counts per species beside the true ones; the loss is not validated.

        The loss saturates at the data's entropy while the sampled counts are still several percent off, so only a
        metric on sampled counts shows the error. ``force_eager`` keeps the decode's shapes (one per slot) out of the
        dynamo cache that holds the compiled training step. The decode runs in fp32, as in ``GenerateOut``: under bf16
        autocast the count logits (|logit| up to ~100) move the counts per species by up to 3.7%, the size of the
        errors being tracked.
        """
        in_tok, counts, pdgid_in_idx = batch[:3]
        with torch.compiler.set_stance("force_eager"), torch.autocast(in_tok.device.type, enabled=False):
            pt_mask = self.sample_passthrough(in_tok, pdgid_in_idx)
            gen = self((in_tok, None, pdgid_in_idx), pt_mask=pt_mask)
        self._val_acc[0] += gen.sum(0)
        self._val_acc[1] += counts.sum(0)
        self._val_acc[2, 0] += in_tok.shape[0]

    def on_validation_epoch_end(self) -> None:
        """Logs the generated / true mean count: in total, per species, and the species' mean of ``|ratio - 1|``.

        The last two cover the species whose true mean count is above ``val_conf.count_floor``.
        """
        acc = _summed_over_ranks(self._val_acc)
        n = acc[2, 0].clamp(min=1)
        gen, true = acc[0] / n, acc[1] / n
        # keep_species comes from the reduced sums, so every rank logs the same key set;
        # per-rank keys would leave the ranks issuing different collectives
        keep_species = true > self.rc.val_conf.get("count_floor", 0.005)
        ratio = gen[keep_species] / true[keep_species]
        self.log_dict({
            "val/n_out_ratio": gen.sum() / true.sum().clamp(min=1e-9),
            "val/count_mae": torch.abs(ratio - 1).mean(),
        }, sync_dist=False)
        for i in keep_species.nonzero(as_tuple=True)[0].tolist():
            self.log(f"val/count_ratio/{int(self.ptypes[i])}", gen[i] / true[i], sync_dist=False)

    # ---- Lightning: data and optimizer ------------------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Loads the dataset once and splits off the validation events.

        Where resolving the config has read the dataset already (``rc.data``), that one is used.
        """
        if getattr(self, "_val_ds", None) is not None:
            return
        data = MultLoader(self.rc.config) if self.rc.data is None else self.rc.data
        self._train_ds, self._val_ds = split_val(data, self.rc.val_conf)

    def train_dataloader(self):
        return self._make_loader(self._train_ds, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self._val_ds, shuffle=False)

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

    # ---- plumbing ---------------------------------------------------------------------------------------------------

    def _init_count_heads(self) -> None:
        """Initialises each slot's count head as ``nn.Linear(h_dim, max_particles)`` would, slot by slot."""
        bound = 1.0 / self.rc.h_dim ** 0.5
        for i in range(self.rc.max_seq_len):
            nn.init.kaiming_uniform_(self.proj_out_w[i].transpose(0, 1), a=5 ** 0.5)
            nn.init.uniform_(self.proj_out_b[i], -bound, bound)

    def _load_checkpoint(self, state_dict: dict) -> None:
        """Loads ``state_dict``, warning about the parameters it lacks (they keep their random init)."""
        missing = self.load_state_dict(state_dict, strict=False).missing_keys
        if missing:
            warnings.warn(f"MultModel checkpoint lacks {missing}: those parameters keep a random init",
                          stacklevel=3)

    def _eval_mode(self) -> None:
        """Puts the module in eval mode, the schedule-free optimizer's weights too (legacy: schedule-free)."""
        self._opt_eval()
        self.eval()

    def _check_scorable(self) -> None:
        """Raises NotImplementedError under ``sym_aug``, where ``log_prob`` cannot score a sample (see its Raises)."""
        if self.rc.sym_aug:
            raise NotImplementedError("log_prob under sym_aug: proj_in draws a new cube transform per call, so the "
                                      "counts would be scored under another one than they were sampled with")

    def _slot0(self, h_in: Tensor) -> Tensor:
        """Returns the trunk's output at slot 0 for the incoming token alone (``h_in`` itself for the legacy head).

        The trunk is causal, so slot 0's output on this one-token sequence equals training's ``out[:, 0]``.
        """
        return self.model(h_in.unsqueeze(1), mask=None, condition=h_in)[:, 0] if self.pt_on_trunk else h_in

    def _trunk_step(self, x: Tensor, h_in: Tensor, cache, s: int):
        """Runs the trunk on the new token ``x`` of slot ``s`` with the KV cache of the slots before it.

        Returns:
            ``(out, cache)``: the outputs, and the cache grown by ``x``.
        """
        return self.model(x, mask=None, condition=h_in, return_intermediates=True, cache=cache,
                          input_not_include_cache=(s > 0))

    def _incoming_once(self, in_tok: Tensor, pdgid_in_idx: Tensor, pt_mask: Tensor) -> Tensor:
        """Returns counts (B, S) that hold one particle of the incoming species on each pass-through event, else 0."""
        counts  = in_tok.new_zeros(in_tok.shape[0], self.rc.max_seq_len, dtype=torch.long)
        slot, _ = vocab_index(self.ptypes, self.ptypes_in[pdgid_in_idx[pt_mask]])
        counts[pt_mask, slot] = 1
        return counts

    def _log_losses(self, total: Tensor, **parts) -> None:
        """Logs the loss and its parts; a None part is skipped.

        Unsynced: an all-reduce per step, only for logging, stalls DDP ranks.
        """
        logs = {f"loss/{k}": v.detach() for k, v in parts.items() if v is not None}
        self.log_dict({"train_loss": total, **logs}, prog_bar=True, sync_dist=False)

    def _make_loader(self, dataset, *, shuffle: bool):
        """Constructs the DataLoader (``make_loader``) with the config's batch size, workers and seed."""
        return make_loader(dataset, bs=self.rc.mm_conf.get("bs", 2**12), shuffle=shuffle,
                           num_workers=self.rc.dl_conf.get("num_workers", 4), seed=loader_seed(self.rc.dl_conf))


# ---- plumbing --------------------------------------------------------------------------------------------------------

def _prep_config(mm_conf: dict, lds_conf: dict) -> dict:
    """Returns the ``DataPrep`` config the multiplicity model's dataset is read with."""
    return {
        "max_energy": mm_conf["max_energy"],
        "cutoff_mev": lds_conf.get("cutoff_mev"),
        "cond_scalars": mm_conf.get("cond_scalars", cond_scalars()),
        "edep_log_min": mm_conf.get("edep_log_min"),
    }


def _pt_head(rc, mm: dict) -> nn.Module | None:
    """Builds the pass-through head (``mm_conf.passthrough_head``), or None."""
    if not mm.get("passthrough_head", False):
        return None
    pt_dim, depth = mm.get("passthrough_dim", rc.h_dim), mm.get("passthrough_depth", 0)
    if depth: return ResidualHead(rc.h_dim, pt_dim, depth)   # legacy
    return nn.Sequential(nn.Linear(rc.h_dim, pt_dim), nn.Mish(), nn.Linear(pt_dim, 1))


def _pt_covered(mm: dict, ptypes_in: Tensor) -> Tensor:
    """Returns, (n_ptypes_in,), which incoming species the pass-through head covers.

    ``mm_conf.passthrough_pdgids``; by default the neutrals, ``_NEUTRAL_PDGIDS``.
    """
    return torch.isin(ptypes_in, ptypes_in.new_tensor(list(mm.get("passthrough_pdgids", _NEUTRAL_PDGIDS))))


def _summed_over_ranks(acc: Tensor) -> Tensor:
    """Sums ``acc`` over the DDP ranks, in place, and returns it; without DDP ``acc`` is returned unchanged."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(acc, op=torch.distributed.ReduceOp.SUM)
    return acc
