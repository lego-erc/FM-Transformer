"""The flow's training step, a ``LEGOLtng`` mixin: mask sampling, target shifting, loss reduction, flow-map term.

``reduce_loss`` returns ``(total, logs)`` and leaves the ``log_dict`` call to its caller, so the reduction runs without
a Trainer. ``LEGOLtngDirect`` (``distill/reflow.py``) overrides ``_step`` and reuses ``_sample_mask`` /
``reduce_loss``; the tests drive ``_step`` directly. Everything read through ``self`` beyond these methods (``rc``,
``model``, ``ps``, ``sym``, ``lv``, ``lv_flow``, ``gen_base_wrapper``, ...) is owned by ``LEGOLtng``.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from legofmt.data.struct import DataStruct, _F

from legofmt.distill.distill import one_step_euler_loss
from legofmt.log_metrics.val_metrics import UNSYNCED_PREFIXES


class TrainStep:
    """Training-step mixin for :class:`~legofmt.main.modules.LEGOLtng`."""

    def _shift_overflow_targets(self, ds_t: DataStruct) -> DataStruct:
        """On a clone, write ``-overflow_delta`` into the zero-valued outgoing-energy
        and E_dep cells, a target outside ``[0, 1]``. An outgoing 0 is the full-energy
        atom (E_out >= E_in; the column stores ``1 - log ratio``), an E_dep 0 no deposit."""
        f = ds_t.f.full.clone()
        delta = self.rc.overflow_delta
        op = _F(f).out_p
        op[..., 0].masked_fill_((op[..., 0] == 0) & ds_t.am.out_p.bool(), -delta)
        edep = _F(f).edep
        edep.masked_fill_(edep == 0, -delta)
        return DataStruct(f, ds_t.m.full, ds_t.am.full)

    def reduce_loss(self, sq: Tensor, ds_t: DataStruct, sq_lat: Tensor | None = None) -> tuple[Tensor, dict]:
        """The weighted training loss, and the scalars the caller should log. ``sq_lat`` is the squared error of
        the ``model_conf.edep_ae`` latent's velocity, its own cell."""
        g        = ((ds_t.m.full == 1) & (ds_t.am.full == 1)).unsqueeze(-1)
        loss_lat = None if sq_lat is None else sq_lat.mean()
        denom    = g.sum().clamp(min=1)
        out      = sq * g
        loss_ed  = None
        if self.rc.edep_cell:  # the E_dep row is a cell of its own
            n_ed    = g[:, 1].sum().clamp(min=1)
            loss_ed = out[:, 1, 0].sum() / n_ed
            loss_e  = (out[..., 0:1].sum() - out[:, 1, 0].sum()) / (denom - n_ed).clamp(min=1)
        else:
            loss_e  = out[..., 0:1].sum() / denom
        # one cell per channel, in lv's order: energy, dir, pos[, edep][, lat]
        cells = {"energy": loss_e, "out_dir": out[..., 1:4].sum() / (denom * 3),
                 "out_pos": out[..., 4:7].sum() / (denom * 3)}
        cells.update({k: v for k, v in (("edep", loss_ed), ("lat", loss_lat)) if v is not None})
        if self.rc.learned_loss_weights:
            # clamp in place: below the floor clamp() has zero gradient, so a cell
            # that overshoots would latch there and stop responding to its loss
            with torch.no_grad():
                self.lv.clamp_(min=self.lv_floor)
            w = torch.exp(-self.lv)
            total = sum(w[i] * v for i, v in enumerate(cells.values())) + self.lv.sum()  # + the log(sigma^2) barrier
        else:
            total = sum(cells.values())
        if not self.training:
            return total, {}
        # unweighted, so these stay comparable across loss-weight settings
        logs = {f"loss/{k}": v.detach() for k, v in cells.items()}
        logs["loss/raw"] = (loss_e + cells["out_dir"] + cells["out_pos"]).detach()
        if self.rc.learned_loss_weights:
            var = self.lv.detach().exp()
            logs.update({f"loss_weight/var_{k.removeprefix('out_')}": var[i] for i, k in enumerate(cells)})
        return total, logs

    @torch.no_grad()
    def _sample_mask(self, ds_t: DataStruct) -> Tensor:
        """Per-event coin flip (``mask_conf.p_forward``) between the data mask and its
        inverse (generate the incoming from the outgoing set); scalar rows stay conditioning."""
        if not self.rc.mask_conf:
            return ds_t.m.full
        fwd = ds_t.m.full
        inv = (fwd == 0) & (ds_t.am.full == 1)
        inv[:, :self.rc.n_prefix] = 0  # all conditioning scalars stay conditioning
        pick = torch.rand(fwd.shape[0], device=fwd.device) < self.rc.mask_conf.get("p_forward", 0.5)
        return torch.where(pick.unsqueeze(-1), fwd, inv.long())

    def _sample_t(self, ds_t: DataStruct) -> Tensor:
        """Training-time ``t`` per event: logit-normal (``sm_norm``) or SD3 mode sampling
        (``sd3``; scale 0 is uniform); ``t_dist_shift`` applies ``t ** (1/shift)``."""
        if self.rc.t_dist == "sm_norm":
            t = torch.sigmoid(self.rc.t_dist_scale * torch.randn_like(ds_t.f.d))
        elif self.rc.t_dist == "sd3":
            u = torch.rand_like(ds_t.f.d)
            t = 1 - u + self.rc.t_dist_scale / 3 * ((torch.pi / 2 * u).sin() ** 2 - u)
        else:
            raise ValueError(f"unknown t_dist: {self.rc.t_dist!r}")
        if self.rc.t_dist_shift != 1.0:
            t = t.clamp(min=0) ** (1.0 / self.rc.t_dist_shift)
        return t

    @torch.no_grad()
    def _teacher_transport(self, ds_t: DataStruct, base: Tensor, **solve_kwargs) -> Tensor:
        """The reflow teacher's transport of ``base`` on the generated slots, the data
        elsewhere. The E_dep row is generated but its direction columns are the
        (1,1,1) filler; transporting them gave that row an off-manifold target."""
        tgt = self.reflow_teacher.solve(ds_t, x_init=base, **solve_kwargs)
        tgt = torch.where((ds_t.m.full == 1).unsqueeze(-1), tgt, ds_t.f.model_in)
        tgt[:, :self.rc.n_prefix, 1:] = ds_t.f.non_cc[..., 1:]
        return tgt

    def _len_pieces(self, ds_t: DataStruct) -> tuple[DataStruct, list[tuple[slice, int]]]:
        lengths = ds_t.am.full.sum(1)
        cpu = self.__dict__.pop("_cpu_lengths", None)
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
        def run(x, t, mask, attn_mask, types, pdgids, d=None):
            return torch.cat([F.pad(self.model(
                x[i, :L].contiguous(), t[i, :L].contiguous(), mask[i, :L].contiguous(),
                attn_mask[i, :L].contiguous(), types[:, :L], pdgids[i, :L].contiguous(),
                None if d is None else d[i, :L].contiguous(),
            ), (0, 0, 0, x.shape[1] - L)) for i, L in pieces])
        return run

    def _flow_map_loss(
        self, loss: Tensor, base: Tensor, ds_t: DataStruct, pdgid_idx: Tensor, model=None, z=None,
    ) -> Tensor:
        """Add ``fac * (lv_flow + every * exp(-lv_flow) * one_step_euler)`` on firing
        steps and the barrier alone otherwise, so the per-step expectation equals
        the ungated Kendall term; validation evaluates it every step with ``every=1``."""
        fac = self.rc.one_step_euler_fac
        if fac <= 0:
            return loss
        every = self.rc.one_step_euler_every
        lvf = None
        if hasattr(self, "lv_flow"):
            # A Kendall cell settles at lv = log(L); one_step_euler (~5e-5) runs three
            # decades below the velocity losses, so it needs a lower floor of its own.
            with torch.no_grad():  # in place, so the gradient stays live on the floor
                self.lv_flow.clamp_(min=self.rc.max_loss_weight_flow)
            lvf = self.lv_flow.squeeze()
            loss = loss + fac * lvf
        if self.training and self.global_step % every != 0:
            return loss

        sc = one_step_euler_loss(self, base, ds_t, pdgid_idx, model, z=z)
        w = fac * (every if self.training else 1)
        if lvf is not None:
            w = w * torch.exp(-lvf)
        if self.training:
            fm_logs = {"loss/one_step_euler": sc.detach()}
            if lvf is not None:
                fm_logs["loss_weight/var_flow"] = lvf.detach().exp()
            self.log_dict(fm_logs, on_step=True, on_epoch=False, logger=True, sync_dist=False)
        return loss + w * sc

    def _step(self, ds_t: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        model, layers = self.model, ds_t.edep_layers
        if self.rc.len_buckets is not None:
            ds_t, pieces = self._len_pieces(ds_t)
            if len(pieces) > 1:
                model = self._piecewise(pieces)
        with torch.no_grad():
            m, am = self._sample_mask(ds_t), ds_t.am.full  # fixed from here on; only the features change
            ds_t = DataStruct(ds_t.f.full, m, am)
            if self.rc.overflow_delta > 0:
                ds_t = self._shift_overflow_targets(ds_t)
            fwd = m[:, self.rc.n_prefix] == 0  # forward events: the incoming slot conditions
            if self.sym is not None:
                face = self.sym.face_of(ds_t.f.in_cc[..., 0, 4:7])
                g    = torch.randint(8, face.shape, device=face.device) if self.rc.sym_aug else None
                ds_t = DataStruct(self._canon_dirs(ds_t.f.full, face, fwd, g=g), m, am)
            if self.rc.dir_zoom:
                ds_t = DataStruct(self._zoom_dirs(ds_t.f.full, am, fwd, pdgids=ds_t.f.pdgids), m, am)
            base = self.gen_base_wrapper(
                ds_t, all_forward=not self.rc.mask_conf or self.rc.mask_conf.get("p_forward", 0.5) >= 1)
            z0 = z1 = None
            if self.edep_ae is not None:  # the latent token's target (the AE's encoding of the true layers) and base
                if layers is None:
                    raise ValueError("model_conf.edep_ae needs a dataset with per-layer E_dep (E_dep_layers).")
                z1 = self.edep_ae.encode(self._edep_dp.denorm_edep(layers).to(base.dtype))
                z0 = torch.randn_like(z1)
            pdgid_idx = self.convert_pdgids(ds_t.f.pdgids)
            t = self._sample_t(ds_t)
            if self.reflow_teacher is not None and self.training and self.global_step % self.rc.reflow_every == 0:
                # reflow: couple base to the teacher's transport of it
                tgt  = self._teacher_transport(ds_t, base, **self.rc.reflow_kwargs)
                ds_t = DataStruct(torch.cat((tgt, ds_t.f.pdgids), dim=-1), m, am)
            if self.rc.edep_whiten:  # from here to the loss (flow map included) the E_dep row is whitened
                base = self._edep_whiten(base, ds_t)
                ds_t = DataStruct(self._edep_whiten(ds_t.f.full, ds_t), m, am)
            ps_ = self.ps.sample(base, ds_t.f.model_in, t)
            lat = {} if z0 is None else {"z": z0 + ps_.t * (z1 - z0)}
        # an explicit zero d, not None: the flow-map calls pass a tensor, and a None is a
        # separate dynamo guard (tests/test_compile_guards.py)
        step_extras = {"d": torch.zeros_like(ps_.t)} if getattr(self.model.vf, "step_cond", False) else {}
        v_out = model(ps_.x_t, ps_.t, mask=m, attn_mask=am, types=self.types_embd[:, :am.shape[1]],
                      pdgids=pdgid_idx, **step_extras, **lat)
        sq_lat = None
        if lat:
            v_out, v_z = v_out
            sq_lat = (v_z - (z1 - z0)) ** 2
        loss, logs = self.reduce_loss((v_out - ps_.dx_t) ** 2, ds_t, sq_lat)
        if logs:
            self.log_dict(logs, on_step=True, on_epoch=False, logger=True, sync_dist=False)
        return self._flow_map_loss(loss, base, ds_t, pdgid_idx, model, z=None if z0 is None else (z0, z1))

    def training_step(self, batch: tuple, _batch_idx: int | Tensor) -> Tensor:
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

