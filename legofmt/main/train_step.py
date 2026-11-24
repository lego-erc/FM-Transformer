"""The flow's training loss, and the training step that computes it (a ``LEGOLtng`` mixin).

```
time       t = 1 - u + s/3 (sin(pi/2 u)**2 - u)            u ~ U(0, 1),  s = t_dist_scale  (SD3 mode sampling)
           t -> t^(1/k)                                     k = t_dist_shift: density ~ k t^(k-1), k < 1 toward t = 0
path       x_t, dx_t on the geodesic from x_0 ~ base to x_1 = the data;  z_t = z_0 + t (z_1 - z_0) for the latent
cells      L_c = the mean of |v - dx_t|**2 over the generated tokens' channels of cell c
             energy    channel 0, without the E_dep token when edep_cell
             out_dir   channels 1:4
             out_pos   channels 4:7
             edep      the E_dep token's channel 0 (edep_cell)
             lat       |v_z - (z_1 - z_0)|**2 (edep_ae)
total      sum_c (exp(-lv_c) L_c + lv_c)                   learned_loss_weights (Kendall), lv_c >= its floor
           sum_c L_c                                       otherwise
flow map   + fac (lv_f + every exp(-lv_f) L_fm)            where global_step % every == 0: L_fm is evaluated
           + fac lv_f                                      on the other steps
                                                           fac = one_step_euler_fac, every = one_step_euler_every
joint      one_step_euler_frac = q > 0 (every = 1): a random share q of the events takes L_fm instead of the cells, in
           the same forward (Frans et al.); the teacher runs on those events alone (``_joint_step``)
```

Averaged over the steps, the two flow-map lines are ``fac (lv_f + exp(-lv_f) L_fm)``, the Kendall form of a cell;
without learned loss weights ``lv_f`` is 0.

The formulas are module functions (``sd3_time``, ``time_shift``, ``latent_path``, ``loss_cells``, ``kendall_total``,
``flow_map_term``); ``TrainStep`` puts the batch into the frame the flow learns in, samples the path, and reduces the
squared errors to the loss. Inverse events (``mask_conf.p_forward < 1``), the ``sm_norm`` time distribution and
reflow are :mod:`legofmt.legacy.train_step`. ``reduce_loss`` returns ``(total, logs)`` so that it
runs without a Trainer; ``LEGOLtngDirect`` (``legacy/reflow.py``) reuses it.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from legofmt.data.load.loader import smallest_bucket
from legofmt.data.struct import DataStruct, _F
from legofmt.numerics import held, masked_mean

from legofmt.cfm.flow_map import ladder_error, ladder_targets, one_step_euler_loss, top_rung
from legofmt.cfm.state import State, no_latent, state_field
from legofmt.main.val_metrics import UNSYNCED_PREFIXES

def sd3_time(u, s):
    """Returns SD3's mode-sampled time for ``u ~ U(0, 1)`` and scale ``s``; ``s = 0`` is uniform."""
    return 1 - u + s / 3 * (torch.sin(torch.pi / 2 * u) ** 2 - u)


def time_shift(t, k):
    """Returns ``t^(1/k)``: a uniform ``t`` gets the density ``k t^(k-1)``; k > 1 moves it toward 1, k < 1 toward 0."""
    return t if k == 1.0 else t.clamp(min=0) ** (1 / k)


def latent_path(z0: Tensor, z1: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """Returns the edep_ae latent's point ``z_t = z_0 + t (z_1 - z_0)`` and its velocity ``z_1 - z_0``."""
    return z0 + t * (z1 - z0), z1 - z0


def loss_cells(sq: Tensor, gen: Tensor, edep: Tensor | None = None, sq_lat: Tensor | None = None) -> Tensor:
    """Returns each loss cell's mean squared error (C,), in the order of the loss weights ``lv``.

    ```
    energy    the energy channel of the generated tokens          (with ``edep``: without the E_dep tokens)
    out_dir   the direction channels of the generated tokens
    out_pos   the position channels of the generated tokens
    edep      the energy channel of the generated E_dep tokens    (with ``edep``)
    lat       the edep_ae latent's velocity                       (with ``sq_lat``)
    ```

    Args:
        sq:     (B, L, 7), the squared error per token and channel.
        gen:    (B, L, 1), bool: the generated tokens.
        edep:   (B, L, 1), bool: the generated E_dep tokens when E_dep is a loss cell of its own (``edep_cell``),
            else None.
        sq_lat: (B, K), the squared error of the edep_ae latent's velocity, or None.
    """
    energy, out_dir, out_pos = sq.split((1, 3, 3), dim=-1)
    particles = gen if edep is None else gen & ~edep
    cells = [masked_mean(energy, particles), masked_mean(out_dir, gen), masked_mean(out_pos, gen)]
    if edep is not None:
        cells.append(masked_mean(energy, edep))
    if sq_lat is not None:
        cells.append(sq_lat.mean())
    return torch.stack(cells)


def kendall_total(cells: Tensor, lv: Tensor) -> Tensor:
    """Returns ``sum_c (exp(-lv_c) L_c + lv_c)``: each cell times its learned inverse variance, plus its log-variance.

    The ``+ lv_c`` keeps a variance from growing without bound.
    """
    return (torch.exp(-lv) * cells).sum() + lv.sum()


def flow_map_term(l_fm: Tensor, fac: float, every: int, lv_f: Tensor | None = None) -> Tensor:
    """Returns ``fac every exp(-lv_f) L_fm``, the flow-map term of a step that evaluates ``L_fm``.

    Without ``lv_f`` (no learned loss weights): ``fac every L_fm``.
    """
    w = fac * every
    if lv_f is not None:
        w = w * torch.exp(-lv_f)
    return w * l_fm


class TrainStep:
    """Training-step mixin for :class:`~legofmt.main.modules.LEGOLtng`."""

    def _step(self, ds_t: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        """Computes one batch's loss (module docstring).

        Args:
            ds_t:       the batch, as the DataLoader delivers it.
            _batch_idx: unused (Lightning's signature).

        Returns:
            The loss (a scalar).
        """
        layers = ds_t.edep_layers   # read first: the batches rebuilt below do not carry the per-layer E_dep
        ds_t, model = self._length_bucket(ds_t)
        with torch.no_grad():
            ds_t = self._flow_frame(ds_t)
            base = self.gen_base_wrapper(ds_t, all_forward=not self.rc.inverse_events)
            z    = self._latent_ends(layers, base)
            pdgid_idx = self.species_idx(ds_t.f.pdgids)
            t = self._sample_t(ds_t)
            if self.reflow_teacher is not None: ds_t = self._reflow_coupling(ds_t, base)   # legacy: reflow
            # the data's E_dep token in units of its prior, which gen_base_wrapper set above (no second head call)
            f    = self._edep_whiten(ds_t.f.full, ds_t, drawn=True)
            ds_t = DataStruct(f, ds_t.m.full, ds_t.am.full)
            path = self.ps.sample(base, ds_t.f.model_in, t)
            z_t, dz = (None, None) if z is None else latent_path(*z, path.t)

        if self.training and self.rc.one_step_euler_frac > 0:
            return self._joint_step(model, path, base, ds_t, pdgid_idx, z, z_t, dz)
        v, v_z = self._velocity(model, path, ds_t, pdgid_idx, z_t)
        sq_lat = None if z is None else (v_z - dz) ** 2
        loss, logs = self.reduce_loss((v - path.dx_t) ** 2, ds_t, sq_lat)
        self._log_step(logs)
        return self._flow_map_loss(loss, base, ds_t, pdgid_idx, model, z=z)

    def training_step(self, batch: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        return self._step(batch, _batch_idx)

    @torch.no_grad()
    def validation_step(self, batch: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        loss = self._step(batch, _batch_idx)
        self._log_validation(loss, len(batch))
        self._log_val_metrics(self.val_metrics(self, batch), len(batch))
        return loss

    # ---- the loss -----------------------------------------------------------------------------------------------

    def reduce_loss(self, sq: Tensor, ds_t: DataStruct, sq_lat: Tensor | None = None) -> tuple[Tensor, dict]:
        """Reduces the squared velocity errors to the loss: ``loss_cells``, summed or weighted by ``kendall_total``.

        Args:
            sq:     (B, L, 7), the squared error per token and channel.
            ds_t:   the batch; ``m`` and ``am`` pick the generated tokens.
            sq_lat: (B, K), the squared error of the latent's velocity, or None.

        Returns:
            ``(total, logs)``: the loss, and the entries to log (``_loss_logs``; empty outside training).
        """
        gen   = _generated(ds_t)
        cells = loss_cells(sq, gen, _edep_token(gen) if self.rc.edep_cell else None, sq_lat)
        if self.rc.learned_loss_weights:
            total = kendall_total(cells, held(self.lv, self.lv_floor))
        else:
            total = cells.sum()
        return total, (self._loss_logs(cells) if self.training else {})

    def _joint_step(self, model, path, base: Tensor, ds_t: DataStruct, pdgid_idx: Tensor, z, z_t, dz) -> Tensor:
        """Computes a training batch's loss with the flow map in the velocity loss's forward (``one_step_euler_frac``).

        ```
        rows      a random share q of the events: they take L_fm, the others the velocity cells
        teacher   their rungs' targets, on those events alone (ladder_targets; self.model, as split mode's pieces
                  are cut from the whole batch)
        forward   once over every event, in fp32: the velocity rows at (x_t, t, d = 0), the flow-map rows at (s0, t0, d)
        ```
        Arguments as ``_step`` has them: the field, the path sample, x_0, the batch, the species indices, the latent's
        ``(z_0, z_1)``, its ``z_t`` and velocity ``dz`` (None without a latent).
        """
        q    = max(1, round(self.rc.one_step_euler_frac * len(ds_t)))
        perm = torch.randperm(len(ds_t), device=base.device)
        rows, vel = perm[:q], perm[q:]
        s0, t0, d, tgt = ladder_targets(self, base[rows], ds_t[rows], pdgid_idx[rows], self.model,
                                        z=None if z is None else (z[0][rows], z[1][rows]))
        z_in  = no_latent(path.x_t) if z_t is None else z_t
        s_in  = State(path.x_t.index_put((rows,), s0.x), z_in.index_put((rows,), s0.z))
        am    = ds_t.am.full
        field = state_field(model, dict(mask=ds_t.m.full, attn_mask=am, types=self.types_embd[:, :am.shape[1]],
                                        pdgids=pdgid_idx))
        with torch.autocast(base.device.type, enabled=False):
            v = field(s_in, path.t.index_put((rows,), t0), d=torch.zeros_like(path.t).index_put((rows,), d))

        velocity_rows = DataStruct(ds_t.f.full, ds_t.m.full.index_fill(0, rows, 0), am)
        sq_lat = None if z is None else ((v.z - dz) ** 2)[vel]
        loss, logs = self.reduce_loss((v.x - path.dx_t) ** 2, velocity_rows, sq_lat)
        self._log_step(logs)
        l_fm = ladder_error(State(v.x[rows], v.z[rows]), tgt, ds_t[rows], latent=z is not None)
        return self._flow_map_loss(loss, base, ds_t, pdgid_idx, model, z=z, l_fm=l_fm)

    def _flow_map_loss(self, loss: Tensor, base: Tensor, ds_t: DataStruct, pdgid_idx: Tensor, model=None,
                       z=None, l_fm: Tensor | None = None) -> Tensor:
        """Adds the flow-map term (module docstring) to ``loss``; validation evaluates ``L_fm`` on every step.

        Args:
            loss:      the velocity loss.
            base:      (B, L, 7), the batch's x_0.
            ds_t:      the batch, in the frame and units of the velocity loss.
            pdgid_idx: (B, L, 1), the tokens' species indices.
            model:     the field to call (in split mode the piecewise one); None: ``self.model``.
            z:         ``(z_0, z_1)`` of the edep_ae latent, or None.
            l_fm:      ``L_fm`` when the step has computed it (``_joint_step``); None: computed here.

        Returns:
            ``loss`` with the term added; ``loss`` itself when ``one_step_euler_fac`` is 0.
        """
        fac, every = self.rc.one_step_euler_fac, self.rc.one_step_euler_every
        if fac <= 0:
            return loss
        lv_f = self._flow_map_lv()
        if lv_f is not None:
            loss = loss + fac * lv_f                                  # the log-variance term, on every step
        if self.training and self.global_step % every != 0:
            return loss

        if l_fm is None:
            l_fm = one_step_euler_loss(self, base, ds_t, pdgid_idx, model, z=z)
        self._log_step({"loss/one_step_euler": l_fm,
                        "loss_weight/var_flow": None if lv_f is None else torch.exp(lv_f.detach()),
                        "flow_map/top_rung": None if self.rc.one_step_euler_unlock <= 0
                                             else l_fm.new_tensor(float(top_rung(self)))})
        return loss + flow_map_term(l_fm, fac, every if self.training else 1, lv_f)

    # ---- the batch, the path, the field ---------------------------------------------------------------------------

    @torch.no_grad()
    def _flow_frame(self, ds_t: DataStruct) -> DataStruct:
        """Returns the batch as the flow learns it: the canonical frame, the zoomed directions.

        ```
        canon_sym   every event rotated onto the canonical cube face; sym_aug: a random symmetry of that face on top
        dir_zoom    the outgoing polar angles zoomed about the incoming direction
        ```
        """
        m = self._sample_mask(ds_t) if self.rc.inverse_events else ds_t.m.full   # legacy: inverse events
        ds_t = DataStruct(ds_t.f.full, m, ds_t.am.full)
        if self.rc.overflow_delta > 0: ds_t = self._shift_overflow_targets(ds_t)   # legacy: overflow_delta
        ds_t, _ = self._enter_frame(ds_t, zoom=True, aug=self.rc.sym_aug)
        return ds_t

    def _sample_t(self, ds_t: DataStruct) -> Tensor:
        """Samples each event's time ``t``, (B,), with ``sd3_time``, shifted by ``time_shift``."""
        if self.rc.t_dist != "sd3": return time_shift(self._sample_t_options(ds_t), self.rc.t_dist_shift)   # legacy
        return time_shift(sd3_time(torch.rand_like(ds_t.f.d), self.rc.t_dist_scale), self.rc.t_dist_shift)

    def _latent_ends(self, layers: Tensor | None, base: Tensor) -> tuple[Tensor, Tensor] | None:
        """Returns the edep_ae latent's ``(z_0, z_1)``, (B, K) each: noise, and the encoding of the true E_dep layers.

        None without an edep_ae.

        Args:
            layers: (B, n_layers), the batch's E_dep layers on the layer scale, or None.
            base:   the batch's x_0; ``z_1`` takes its dtype.

        Raises:
            ValueError: the dataset has no per-layer E_dep.
        """
        if self.edep_ae is None:
            return None
        if layers is None:
            raise ValueError("model_conf.edep_ae needs a dataset with per-layer E_dep (E_dep_layers).")
        z1 = self.edep_ae.encode(self._edep_dp.denorm_edep_layer(layers).to(base.dtype))
        return torch.randn_like(z1), z1

    def _velocity(self, model, path, ds_t: DataStruct, pdgid_idx: Tensor,
                  z_t: Tensor | None) -> tuple[Tensor, Tensor | None]:
        """Returns the field's velocity at ``(x_t, t)``, and the latent's velocity (None without ``z_t``)."""
        am = ds_t.am.full
        v  = model(path.x_t, path.t, mask=ds_t.m.full, attn_mask=am, types=self.types_embd[:, :am.shape[1]],
                   pdgids=pdgid_idx, **self._field_extras(path, z_t))
        return v if z_t is not None else (v, None)

    # ---- plumbing -------------------------------------------------------------------------------------------------

    def _flow_map_lv(self) -> Tensor | None:
        """Returns the flow-map term's log-variance ``lv_f``, held at its floor or above; None when it is not learned.

        ``L_fm`` (~5e-5) is three decades smaller than the velocity cells, so its floor is its own
        (``max_loss_weight_flow``).
        """
        if not hasattr(self, "lv_flow"):
            return None
        return held(self.lv_flow, self.rc.max_loss_weight_flow).squeeze(0)

    def _field_extras(self, path, z_t: Tensor | None) -> dict:
        """Returns the field's optional inputs: the step ``d = 0`` (``step_cond``), the latent's ``z_t`` (edep_ae).

        ``d = 0`` is passed as a tensor, not as None: the flow-map calls pass a tensor, and a None would compile a
        second graph.
        """
        step = {"d": torch.zeros_like(path.t)} if getattr(self.model.vf, "step_cond", False) else {}
        lat  = {} if z_t is None else {"z": z_t}
        return {**step, **lat}

    def _loss_logs(self, cells: Tensor) -> dict[str, Tensor]:
        """Returns the cells' log entries, by log name.

        ```
        loss/<cell>              each cell unweighted, so that runs with different loss weights compare
        loss/raw                 the sum of the first three cells (energy, out_dir, out_pos)
        loss_weight/var_<cell>   each cell's learned variance exp(lv_c) (learned_loss_weights); <cell> without "out_"
        ```
        """
        names = _cell_names(self.rc.edep_cell, len(cells))
        cells = cells.detach()
        logs  = {f"loss/{c}": L for c, L in zip(names, cells)}
        logs["loss/raw"] = cells[:3].sum()
        if self.rc.learned_loss_weights:
            var = torch.exp(self.lv.detach())
            logs.update({f"loss_weight/var_{c.removeprefix('out_')}": v for c, v in zip(names, var)})
        return logs

    def _log_step(self, logs: dict) -> None:
        """Logs a training step's scalars, detached and unsynced; nothing outside training. A None value is skipped."""
        logs = {k: v.detach() for k, v in logs.items() if v is not None}
        if self.training and logs:
            self.log_dict(logs, on_step=True, on_epoch=False, logger=True, sync_dist=False)

    def _log_validation(self, loss: Tensor, bs: int) -> None:
        """Logs the validation loss, per step and per epoch, synced."""
        self.log("Validation Loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)

    def _log_val_metrics(self, metrics: dict, bs: int) -> None:
        """Logs the validation metrics per epoch, synced across the DDP ranks except the ``UNSYNCED_PREFIXES`` keys."""
        for name, val in metrics.items():
            self.log(name, val, on_epoch=True, sync_dist=not name.startswith(UNSYNCED_PREFIXES), batch_size=bs)

    # ---- length buckets -------------------------------------------------------------------------------------------

    def _length_bucket(self, ds_t: DataStruct):
        """Returns the batch cut to its length bucket, and the field to call on it.

        In split mode with several length pieces the field is ``_piecewise``: ``self.model`` run once per piece.
        """
        if self.rc.len_buckets is None:
            return ds_t, self.model
        ds_t, pieces = self._len_pieces(ds_t)
        return ds_t, self._piecewise(pieces) if len(pieces) > 1 else self.model

    def _len_pieces(self, ds_t: DataStruct) -> tuple[DataStruct, list[tuple[slice, int]]]:
        """Cuts the batch to its length bucket; in split mode also into ``len_split`` pieces by length.

        A batch that ``len_split`` does not divide stays one piece.

        Returns:
            ``(ds_t, pieces)``: the batch (in split mode sorted by length), and per piece its rows (a slice) and its
            length: the bucket of its longest event.
        """
        lengths = ds_t.am.full.sum(1)
        cpu = self.__dict__.pop("_cpu_lengths", None)   # on_before_batch_transfer's CPU copy: no GPU sync here
        if cpu is None or len(cpu) != len(lengths):
            cpu = lengths.cpu()
        n = len(cpu)
        k = self.rc.len_split if self.rc.len_mode == "split" and n % self.rc.len_split == 0 else 1
        size = n // k
        if k > 1:  # sorted by length, so each piece's longest event is its last
            ds_t = ds_t[lengths.sort(stable=True).indices]
            longest = cpu.sort().values[size - 1::size]
        else:
            longest = cpu.max().unsqueeze(0)
        Ls = smallest_bucket(longest, self.rc.len_buckets).tolist()
        return ds_t.trim(Ls[-1]), [(slice(j * size, (j + 1) * size), L) for j, L in enumerate(Ls)]

    def _piecewise(self, pieces: list[tuple[slice, int]]):
        """Returns a function that runs the field once per length piece.

        Each piece is cut to its own length; its velocity is padded back to the batch's length.
        """
        def run(x, t, mask, attn_mask, types, pdgids, d=None):
            return torch.cat([F.pad(self.model(
                x[i, :L].contiguous(), t[i, :L].contiguous(), mask[i, :L].contiguous(),
                attn_mask[i, :L].contiguous(), types[:, :L], pdgids[i, :L].contiguous(),
                None if d is None else d[i, :L].contiguous(),
            ), (0, 0, 0, x.shape[1] - L)) for i, L in pieces])
        return run


def _generated(ds_t: DataStruct) -> Tensor:
    """Returns the generated tokens, (B, L, 1): the real tokens (``am``) on which the loss mask ``m`` is 1."""
    return ((ds_t.m.full == 1) & (ds_t.am.full == 1)).unsqueeze(-1)


def _edep_token(gen: Tensor) -> Tensor:
    """Returns the generated E_dep tokens, (B, L, 1): ``gen`` on token 1, false on every other."""
    edep = torch.zeros_like(gen)
    _F(edep).edep.copy_(_F(gen).edep)
    return edep


def _cell_names(edep_cell: bool, n: int) -> tuple[str, ...]:
    """Returns the names of ``loss_cells``' ``n`` entries."""
    names = ("energy", "out_dir", "out_pos", "edep", "lat") if edep_cell else ("energy", "out_dir", "out_pos", "lat")
    return names[:n]
