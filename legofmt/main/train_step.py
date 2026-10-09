"""The flow's training step: mask sampling, target shifting, loss reduction.

Split out of ``main/modules.py``. A mixin like ``Solvers`` / ``BaseDist``,
though here by choice rather than necessity: nothing outside this repo touches
these members. In-repo consumers are ``LEGOLtngDirect`` (``distill/reflow.py``,
which overrides ``_step`` and calls ``_sample_mask`` / ``reduce_loss``) and the
tests, which drive ``_step`` directly.

``reduce_loss`` returns ``(total, logs)`` and leaves the ``log_dict`` call to
its caller, so the reduction can be exercised without a Trainer attached.

Members resolved through ``self`` and owned by ``LEGOLtng``: ``rc``, ``model``,
``ps``, ``sym``, ``lv``, ``lv_flow``, ``gen_base``, ``val_metrics``,
``reflow_teacher``, ``convert_pdgids``, ``_canon_dirs``, ``gen_base_wrapper``.
"""

import torch
from torch import Tensor

from legofmt.data.struct import DataStruct, _F

from legofmt.distill.distill import one_step_euler_loss
from legofmt.log_metrics.val_metrics import UNSYNCED_PREFIXES


class TrainStep:
    """Training-step mixin for :class:`~legofmt.main.modules.LEGOLtng`."""

    def _shift_overflow_targets(self, ds_t: DataStruct) -> DataStruct:
        """On a clone, write ``-overflow_delta`` into the zero-valued outgoing-energy
        and E_dep cells: 'absorbed / no deposit' becomes a target outside ``[0, 1]``."""
        f = ds_t.f.full.clone()
        delta = self.rc.overflow_delta
        op = _F(f).out_p
        op[..., 0].masked_fill_((op[..., 0] == 0) & ds_t.am.out_p.bool(), -delta)
        edep = _F(f).edep
        edep.masked_fill_(edep == 0, -delta)
        return DataStruct(f, ds_t.m.full, ds_t.am.full)

    def reduce_loss(self, sq: Tensor, ds_t: DataStruct,
                    w: Tensor | None = None) -> tuple[Tensor, dict]:
        """The weighted training loss, and the scalars the caller should log.

        ``w`` is a per-event weight that scales both numerator and denominator.
        Pass-through events come in at 0: the gate owns them, so the flow is not
        asked to model a mode it never generates.
        """
        g        = ((ds_t.m.full == 1) & (ds_t.am.full == 1)).unsqueeze(-1)
        if w is not None:
            g    = g * w.view(-1, 1, 1)
        denom    = g.sum().clamp(min=1)
        out      = sq * g
        loss_e   = out[..., 0:1].sum() / denom
        loss_ed  = None
        if self.rc.edep_cell:
            n_ed    = g[:, 1].sum().clamp(min=1)
            loss_ed = out[:, 1, 0].sum() / n_ed
            loss_e  = (out[..., 0:1].sum() - out[:, 1, 0].sum()) / (denom - n_ed).clamp(min=1)
        loss_dir = out[..., 1:4].sum() / (denom * 3)
        loss_x   = out[..., 4:7].sum() / (denom * 3)
        logs     = {}
        if not self.rc.learned_loss_weights:
            total = loss_e + loss_dir + loss_x + (loss_ed if loss_ed is not None else 0.0)
        else:
            # clamp in place: below the floor clamp() has zero gradient, so a cell
            # that overshoots would latch there and stop responding to its loss
            lo = torch.full_like(self.lv, self.rc.max_loss_weight)
            if self.rc.edep_cell:
                lo[3] = self.rc.max_loss_weight_edep
            with torch.no_grad():
                self.lv.clamp_(min=lo)
            lv  = self.lv.clamp(min=lo)  # (3,) or (4,), one per channel
            w   = torch.exp(-lv)
            total = (
                w[0] * loss_e + w[1] * loss_dir + w[2] * loss_x
                + (w[3] * loss_ed if loss_ed is not None else 0.0)
                + lv.sum()  # the +log(sigma^2) barrier; zero at init
            )
            v = lv.detach().exp()
            logs = {"loss_weight/var_energy": v[0], "loss_weight/var_dir": v[1],
                    "loss_weight/var_pos": v[2]}
            if loss_ed is not None:
                logs["loss_weight/var_edep"] = v[3]
        out_logs = {}
        if self.training:
            out_logs = {
                # unweighted, so these stay comparable across loss-weight settings
                "loss/energy": loss_e.detach(),
                **({"loss/edep": loss_ed.detach()} if loss_ed is not None else {}),
                "loss/out_dir": loss_dir.detach(),
                "loss/out_pos": loss_x.detach(),
                "loss/raw": (loss_e + loss_dir + loss_x).detach(),
                **logs,
            }
        return total, out_logs

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

    def _flow_map_loss(
        self, loss: Tensor, base: Tensor, ds_t: DataStruct, pdgid_idx: Tensor,
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
            lvf = self.lv_flow.clamp(min=self.rc.max_loss_weight_flow).squeeze()
            loss = loss + fac * lvf
        if self.training and self.global_step % every != 0:
            return loss

        sc = one_step_euler_loss(self, base, ds_t, pdgid_idx)
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
        n0 = self.rc.n_prefix + 1
        with torch.no_grad():
            m      = self._sample_mask(ds_t)
            gen_sp = (m[:, self.rc.n_prefix] == 0).unsqueeze(-1)
            m      = m.clone()
            m[:, n0:] |= gen_sp
            ds_t = DataStruct(ds_t.f.full, m, ds_t.am.full)
            if self.rc.overflow_delta > 0:
                ds_t = self._shift_overflow_targets(ds_t)
            if self.sym is not None:
                fwd  = ds_t.m.full[:, self.rc.n_prefix] == 0
                face = self.sym.face_of(ds_t.f.in_cc[..., 0, 4:7])
                g    = torch.randint(8, face.shape, device=face.device) if self.rc.sym_aug else None
                ds_t = DataStruct(self._canon_dirs(ds_t.f.full, face, fwd, g=g), ds_t.m.full, ds_t.am.full)
            pt_tgt  = ((ds_t.f.edep <= 0) & (ds_t.am.out_p.sum(-1) == 1)).float()
            pt_idx  = self._in_species(ds_t)
            pt_ok   = self.pt_allowed(ds_t, pt_idx) & gen_sp.squeeze(-1)
            w       = 1.0 - pt_tgt * pt_ok
            base = self.gen_base_wrapper(ds_t)
            s1   = self.convert_pdgids(ds_t.f.pdgids).squeeze(-1)  # NaN pad -> class 0
            t    = self._sample_t(ds_t)
            rows = (torch.arange(m.shape[1], device=m.device) >= n0) & gen_sp
            s_t  = torch.where(
                rows & (torch.rand_like(s1, dtype=t.dtype) >= t.unsqueeze(-1)),
                torch.randint_like(s1, self.rc.model_args["npdgids"]), s1,
            )
            pad  = (~ds_t.am.full & rows).unsqueeze(-1)
            ds_t = DataStruct(
                torch.cat((torch.where(pad, base, ds_t.f.model_in), ds_t.f.pdgids), dim=-1),
                m, ds_t.am.full | rows,
            )
            if (
                self.reflow_teacher is not None and self.training
                and self.global_step % self.rc.reflow_every == 0
            ):  # reflow: couple base to the teacher's transport of it
                tgt = self.reflow_teacher.solve(ds_t, x_init=base, **self.rc.reflow_kwargs)
                tgt = torch.where((ds_t.m.full == 1).unsqueeze(-1), tgt, ds_t.f.model_in)
                # the E_dep row is generated but its direction columns are the (1,1,1)
                # filler; transporting them gave that row an off-manifold target
                tgt[:, :self.rc.n_prefix, 1:] = ds_t.f.non_cc[..., 1:]
                ds_t = DataStruct(
                    torch.cat((tgt, ds_t.f.pdgids), dim=-1), ds_t.m.full, ds_t.am.full,
                )
            ps_ = self.ps.sample(base, ds_t.f.model_in, t)
        step_extras = ({"d": torch.zeros_like(ps_.t)}
                       if getattr(self.model.vf, "step_cond", False) else {})
        v_out, sp_logits = self.model(
            ps_.x_t, ps_.t,
            mask=ds_t.m.full, attn_mask=ds_t.am.full,
            types=self.types_embd, pdgids=s_t, return_species=True, **step_extras,
        )
        sq = (v_out - ps_.dx_t) ** 2
        loss, logs = self.reduce_loss(sq, ds_t, w)
        if pt_ok.any():
            loss_pt = torch.nn.functional.binary_cross_entropy_with_logits(
                self.pt_logits(ds_t, pt_idx)[pt_ok], pt_tgt[pt_ok])
            loss = loss + self.rc.passthrough_fac * loss_pt
            if self.training:
                logs = {**logs, "loss/passthrough": loss_pt.detach()}
        rows = rows & (w > 0).unsqueeze(-1)
        if rows.any():
            lg, tg  = sp_logits[rows], s1[rows]
            loss_sp = torch.nn.functional.cross_entropy(lg, tg)
            loss    = loss + self.rc.species_fac * loss_sp
            if self.training:
                logs = {**logs, "loss/species": loss_sp.detach(),
                        "loss/species_acc": (lg.argmax(-1) == tg).float().mean()}
        if logs:
            self.log_dict(logs, on_step=True, on_epoch=False, logger=True, sync_dist=False)

        loss = self._flow_map_loss(loss, base, ds_t, s_t)

        return loss

    def training_step(self, batch: tuple, _batch_idx: int | Tensor) -> Tensor:
        return self._step(batch, _batch_idx)

    @torch.no_grad()
    def validation_step(self, batch: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        bs = len(batch)
        loss = self._step(batch, _batch_idx)
        self.log("Validation Loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        for name, val in self.val_metrics(self, batch).items():
            # population-dependent keys carry no collective: a rank whose shard holds no
            # events of a species emits no key for it, and syncing those would leave the
            # ranks issuing different numbers of collectives and deadlock NCCL
            sync = not name.startswith(UNSYNCED_PREFIXES)
            self.log(name, val, on_epoch=True, sync_dist=sync, batch_size=bs)
        return loss

