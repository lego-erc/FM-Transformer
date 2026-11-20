"""The flow's training loss, and the training step that computes it (a ``LEGOLtng`` mixin).

```
time       t = 1 - u + s/3 (sin(pi/2 u)**2 - u)            u ~ U(0, 1),  s = t_dist_scale  (SD3 mode sampling)
path       x_t, dx_t on the geodesic from x_0 ~ base to x_1 = the data;  z_t = z_0 + t (z_1 - z_0) for the latent
cells      L_c = the mean of |v - dx_t|**2 over the generated tokens' channels of cell c
             energy    channel 0, without the E_dep token when edep_cell
             out_dir   channels 1:4
             out_pos   channels 4:7
             edep      the E_dep token's channel 0 (edep_cell)
             lat       |v_z - (z_1 - z_0)|**2 (edep_ae)
total      sum_c exp(-lv_c) L_c + lv_c                     learned_loss_weights (Kendall), lv_c >= its floor
           sum_c L_c                                       otherwise
flow map   + fac (lv_f + every exp(-lv_f) L_fm)            on the steps it fires (every one_step_euler_every-th)
           + fac lv_f                                      on the others: in expectation the Kendall term
```

The formulas are module functions (``sd3_time``, ``loss_cells``, ``kendall_total``); ``TrainStep`` puts the batch
into the frame the flow learns in, samples the path, and reduces. Inverse events (``mask_conf.p_forward < 1``), the
``sm_norm`` time and reflow are :mod:`legofmt.legacy.train_step`. ``reduce_loss`` returns ``(total, logs)`` so that
it runs without a Trainer; ``LEGOLtngDirect`` (``legacy/reflow.py``) reuses it.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from legofmt.data.struct import DataStruct, _F

from legofmt.distill.distill import one_step_euler_loss
from legofmt.log_metrics.val_metrics import UNSYNCED_PREFIXES

# each loss cell's channels of a token
_CELLS = {"energy": slice(0, 1), "out_dir": slice(1, 4), "out_pos": slice(4, 7)}


def sd3_time(u, s):
    """Returns SD3's mode-sampled time for ``u ~ U(0, 1)`` and scale ``s``; ``s = 0`` is uniform."""
    return 1 - u + s / 3 * (torch.sin(torch.pi / 2 * u) ** 2 - u)


def loss_cells(sq: Tensor, gen: Tensor, edep_cell: bool, sq_lat: Tensor | None = None) -> dict[str, Tensor]:
    """Returns each loss cell's mean squared error, in the order of the loss weights.

    Args:
        sq:        (B, L, 7), the squared error per token and channel.
        gen:       (B, L, 1), the generated tokens.
        edep_cell: the E_dep token's energy channel is a cell of its own.
        sq_lat:    the squared error of the edep_ae latent's velocity, or None.

    Returns:
        ``{energy, out_dir, out_pos[, edep][, lat]}``.
    """
    n  = gen.sum().clamp(min=1)
    sq = sq * gen
    cells = {c: sq[..., ch].sum() / (n * (ch.stop - ch.start)) for c, ch in _CELLS.items()}
    if edep_cell:
        n_ed, sum_ed = gen[:, 1].sum().clamp(min=1), _F(sq).edep.sum()
        cells["energy"] = (sq[..., 0:1].sum() - sum_ed) / (n - n_ed).clamp(min=1)
        cells["edep"] = sum_ed / n_ed
    if sq_lat is not None:
        cells["lat"] = sq_lat.mean()
    return cells


def kendall_total(cells: dict[str, Tensor], lv: Tensor) -> Tensor:
    """Returns ``sum_c exp(-lv_c) L_c + lv_c``: each cell weighted by its learned inverse variance, plus the barrier."""
    w = torch.exp(-lv)
    return sum(w[i] * L for i, L in enumerate(cells.values())) + lv.sum()


class TrainStep:
    """Training-step mixin for :class:`~legofmt.main.modules.LEGOLtng`."""

    def _step(self, ds_t: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        """Computes one batch's loss, as the module docstring has it.

        Args:
            ds_t:       the batch.
            _batch_idx: unused (Lightning's signature).

        Returns:
            The loss (a scalar).
        """
        layers = ds_t.edep_layers
        ds_t, model = self._length_bucket(ds_t)
        with torch.no_grad():
            ds_t = self._flow_frame(ds_t)
            base = self.gen_base_wrapper(ds_t, all_forward=not self.rc.inverse_events)
            z0, z1 = self._latent_ends(layers, base)
            pdgid_idx = self.convert_pdgids(ds_t.f.pdgids)
            t = self._sample_t(ds_t)
            if self.reflow_teacher is not None: ds_t = self._reflow_coupling(ds_t, base)   # legacy: reflow
            ds_t = DataStruct(self._edep_whiten(ds_t.f.full, ds_t), ds_t.m.full, ds_t.am.full)   # base's E_dep unit
            path = self.ps.sample(base, ds_t.f.model_in, t)
            z_t  = None if z0 is None else z0 + path.t * (z1 - z0)

        v, v_z = self._velocity(model, path, ds_t, pdgid_idx, z_t)
        sq_lat = None if z_t is None else (v_z - (z1 - z0)) ** 2
        loss, logs = self.reduce_loss((v - path.dx_t) ** 2, ds_t, sq_lat)
        if logs:
            self.log_dict(logs, on_step=True, on_epoch=False, logger=True, sync_dist=False)
        return self._flow_map_loss(loss, base, ds_t, pdgid_idx, model, z=None if z0 is None else (z0, z1))

    def training_step(self, batch: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        return self._step(batch, _batch_idx)

    @torch.no_grad()
    def validation_step(self, batch: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        bs = len(batch)
        loss = self._step(batch, _batch_idx)
        self.log("Validation Loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        for name, val in self.val_metrics(self, batch).items():
            # population-dependent keys are logged unsynced (deadlock: see UNSYNCED_PREFIXES)
            self.log(name, val, on_epoch=True, sync_dist=not name.startswith(UNSYNCED_PREFIXES), batch_size=bs)
        return loss

    # ---- the loss -----------------------------------------------------------------------------------------------

    def reduce_loss(self, sq: Tensor, ds_t: DataStruct, sq_lat: Tensor | None = None) -> tuple[Tensor, dict]:
        """Reduces the squared velocity errors to the loss: ``loss_cells``, summed or weighted by ``kendall_total``.

        Args:
            sq:     (B, L, 7), the squared error per token and channel.
            ds_t:   the batch; ``m`` and ``am`` pick the generated tokens.
            sq_lat: the squared error of the latent's velocity, or None.

        Returns:
            ``(total, logs)``: the loss, and in training the unweighted cells (and the weights' variances) to log.
        """
        cells = loss_cells(sq, ((ds_t.m.full == 1) & (ds_t.am.full == 1)).unsqueeze(-1), self.rc.edep_cell, sq_lat)
        if self.rc.learned_loss_weights:
            with torch.no_grad():   # in place: below the floor clamp() has zero gradient, and a cell would latch there
                self.lv.clamp_(min=self.lv_floor)
            total = kendall_total(cells, self.lv)
        else:
            total = sum(cells.values())
        if not self.training:
            return total, {}

        logs = {f"loss/{c}": L.detach() for c, L in cells.items()}   # unweighted: comparable across weightings
        logs["loss/raw"] = (cells["energy"] + cells["out_dir"] + cells["out_pos"]).detach()
        if self.rc.learned_loss_weights:
            var = self.lv.detach().exp()
            logs.update({f"loss_weight/var_{c.removeprefix('out_')}": var[i] for i, c in enumerate(cells)})
        return total, logs

    def _flow_map_loss(self, loss: Tensor, base: Tensor, ds_t: DataStruct, pdgid_idx: Tensor, model=None,
                       z=None) -> Tensor:
        """Adds the flow-map term (module docstring) to ``loss``; validation evaluates ``L_fm`` every step.

        Args:
            loss:      the velocity loss.
            base:      the batch's x_0.
            ds_t:      the batch, in the frame and units of the velocity loss.
            pdgid_idx: the tokens' species indices.
            model:     the field (the piecewise one in split mode).
            z:         ``(z_0, z_1)`` of the edep_ae latent, or None.

        Returns:
            ``loss`` with the term added.
        """
        fac, every = self.rc.one_step_euler_fac, self.rc.one_step_euler_every
        if fac <= 0:
            return loss
        lv_f = None
        if hasattr(self, "lv_flow"):
            # L_fm (~5e-5) runs three decades below the velocity cells, so its weight has a floor of its own;
            # in place, so the gradient stays live on the floor
            with torch.no_grad():
                self.lv_flow.clamp_(min=self.rc.max_loss_weight_flow)
            lv_f = self.lv_flow.squeeze()
            loss = loss + fac * lv_f
        if self.training and self.global_step % every != 0:
            return loss

        l_fm = one_step_euler_loss(self, base, ds_t, pdgid_idx, model, z=z)
        w = fac * (every if self.training else 1)
        if lv_f is not None:
            w = w * torch.exp(-lv_f)
        if self.training:
            logs = {"loss/one_step_euler": l_fm.detach()}
            if lv_f is not None:
                logs["loss_weight/var_flow"] = lv_f.detach().exp()
            self.log_dict(logs, on_step=True, on_epoch=False, logger=True, sync_dist=False)
        return loss + w * l_fm

    # ---- the batch, the path, the field ---------------------------------------------------------------------------

    @torch.no_grad()
    def _flow_frame(self, ds_t: DataStruct) -> DataStruct:
        """Returns the batch as the flow learns it: the zero atoms moved, the canonical frame, the zoomed directions.

        ```
        overflow    an outgoing energy 0 (the full-energy atom) and an E_dep 0 (no deposit) -> -overflow_delta
        canon_sym   every event rotated onto the canonical cube face; sym_aug: a random cube symmetry on top
        dir_zoom    the outgoing polar angles zoomed about the incoming direction
        ```
        """
        m = self._sample_mask(ds_t) if self.rc.inverse_events else ds_t.m.full   # legacy: inverse events
        ds_t = DataStruct(ds_t.f.full, m, ds_t.am.full)
        if self.rc.overflow_delta > 0:
            ds_t = self._shift_overflow_targets(ds_t)
        if self.sym is not None:
            face = self.sym.face_of(ds_t.f.in_pos)
            g    = torch.randint(8, face.shape, device=face.device) if self.rc.sym_aug else None
            ds_t = DataStruct(self._canon_dirs(ds_t.f.full, face, ds_t.m.fwd, g=g), m, ds_t.am.full)
        if self.rc.dir_zoom:
            ds_t = DataStruct(self._zoom_dirs(ds_t.f.full, ds_t.am.full, ds_t.m.fwd, pdgids=ds_t.f.pdgids), m,
                              ds_t.am.full)
        return ds_t

    def _shift_overflow_targets(self, ds_t: DataStruct) -> DataStruct:
        """Returns a copy of the batch with its zero atoms at ``-overflow_delta``, outside [0, 1]."""
        f = ds_t.f.full.clone()
        out_e, edep = _F(f).out_energy, _F(f).edep
        out_e.masked_fill_((out_e == 0) & ds_t.am.out_p.bool(), -self.rc.overflow_delta)
        edep.masked_fill_(edep == 0, -self.rc.overflow_delta)
        return DataStruct(f, ds_t.m.full, ds_t.am.full)

    def _sample_t(self, ds_t: DataStruct) -> Tensor:
        """Samples each event's time ``t`` (``sd3_time``)."""
        if self.rc.t_dist != "sd3" or self.rc.t_dist_shift != 1.0: return self._sample_t_options(ds_t)   # legacy
        return sd3_time(torch.rand_like(ds_t.f.d), self.rc.t_dist_scale)

    def _latent_ends(self, layers: Tensor | None, base: Tensor) -> tuple[Tensor | None, Tensor | None]:
        """Returns the edep_ae latent's ``(z_0, z_1)``: noise, and the encoding of the true per-layer deposits.

        Raises:
            ValueError: the dataset has no per-layer E_dep.
        """
        if self.edep_ae is None:
            return None, None
        if layers is None:
            raise ValueError("model_conf.edep_ae needs a dataset with per-layer E_dep (E_dep_layers).")
        z1 = self.edep_ae.encode(self._edep_dp.denorm_edep_layer(layers).to(base.dtype))
        return torch.randn_like(z1), z1

    def _velocity(self, model, path, ds_t: DataStruct, pdgid_idx: Tensor,
                  z_t: Tensor | None) -> tuple[Tensor, Tensor | None]:
        """Returns the field's velocity at ``(x_t, t)``, and the latent's (None without one).

        ``d = 0`` goes in as a tensor, not None: the flow-map calls pass one, and None is a dynamo guard of its own.
        """
        m, am = ds_t.m.full, ds_t.am.full
        step = {"d": torch.zeros_like(path.t)} if getattr(self.model.vf, "step_cond", False) else {}
        lat  = {} if z_t is None else {"z": z_t}
        v = model(path.x_t, path.t, mask=m, attn_mask=am, types=self.types_embd[:, :am.shape[1]], pdgids=pdgid_idx,
                  **step, **lat)
        return v if lat else (v, None)

    # ---- length buckets -------------------------------------------------------------------------------------------

    def _length_bucket(self, ds_t: DataStruct):
        """Returns the batch cut to its length bucket, and the field to run (once per length piece in split mode)."""
        if self.rc.len_buckets is None:
            return ds_t, self.model
        ds_t, pieces = self._len_pieces(ds_t)
        return ds_t, self._piecewise(pieces) if len(pieces) > 1 else self.model

    def _len_pieces(self, ds_t: DataStruct) -> tuple[DataStruct, list[tuple[slice, int]]]:
        """Cuts the batch to its length bucket; in split mode also into ``len_split`` pieces by length.

        Returns:
            ``(ds_t, pieces)``: the batch (in split mode sorted by length), and per piece its rows and its bucket
            length, the bucket of its longest event.
        """
        lengths = ds_t.am.full.sum(1)
        cpu = self.__dict__.pop("_cpu_lengths", None)   # on_before_batch_transfer's copy: no host sync
        if cpu is None or len(cpu) != len(lengths):
            cpu = lengths.cpu()
        n = len(cpu)
        k = self.rc.len_split if self.rc.len_mode == "split" and n % self.rc.len_split == 0 else 1
        size = n // k
        if k > 1:  # sorted by length, so each piece's longest event is its last
            ds_t = ds_t[lengths.sort(stable=True).indices]
            longest = cpu.sort().values[size - 1::size]
        else:
            longest = cpu.max().view(1)
        Ls = [next(b for b in self.rc.len_buckets if b >= x) for x in longest.tolist()]
        return ds_t.trim(Ls[-1]), [(slice(j * size, (j + 1) * size), L) for j, L in enumerate(Ls)]

    def _piecewise(self, pieces: list[tuple[slice, int]]):
        """Returns the field run once per length piece, each cut to its own length and padded back."""
        def run(x, t, mask, attn_mask, types, pdgids, d=None):
            return torch.cat([F.pad(self.model(
                x[i, :L].contiguous(), t[i, :L].contiguous(), mask[i, :L].contiguous(),
                attn_mask[i, :L].contiguous(), types[:, :L], pdgids[i, :L].contiguous(),
                None if d is None else d[i, :L].contiguous(),
            ), (0, 0, 0, x.shape[1] - L)) for i, L in pieces])
        return run
