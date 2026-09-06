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
``reflow_teacher``, ``convert_pdgids``, ``_canon_dirs``, ``gen_base_wrapper``,
``_base_dist_loss``.
"""

import torch
from torch import Tensor

from legofmt.data.struct import DataStruct, _F

from legofmt.distill.distill import one_step_euler_loss


class TrainStep:
    """Training-step mixin for :class:`~legofmt.main.modules.LEGOLtng`."""

    def _shift_overflow_targets(self, ds_t: DataStruct) -> DataStruct:
        f = ds_t.f.full.clone()
        if self.rc.overflow_delta > 0:
            op = _F(f).out_p
            op[..., 0].masked_fill_((op[..., 0] == 0) & ds_t.am.out_p.bool(),
                                    -self.rc.overflow_delta)
        if self.rc.edep_overflow_delta > 0:
            edep = _F(f).edep
            edep.masked_fill_(edep == 0, -self.rc.edep_overflow_delta)
        return DataStruct(f, ds_t.m.full, ds_t.am.full)

    def reduce_loss(
        self, sq: Tensor, ds_t: DataStruct, t: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """The weighted training loss, and the scalars the caller should log."""
        g        = ((ds_t.m.full == 1) & (ds_t.am.full == 1)).unsqueeze(-1)
        denom    = g.sum().clamp(min=1)
        out      = sq * g
        loss_e   = out[..., 0:1].sum() / denom
        loss_dir = out[..., 1:4].sum() / (denom * 3)
        loss_x   = out[..., 4:7].sum() / (denom * 3)
        logs     = {}
        if not self.rc.uncert_weighting:
            total = loss_e + loss_dir + loss_x
        else:
            if t is None:
                raise ValueError(
                    "uncert_weighting=True needs the per-event t, which the "
                    "time-independent (LEGOLtngDirect) path does not have."
                )
            nb  = self.rc.uncert_bins
            idx = (t.detach().clamp(0, 1) * nb).long().clamp(max=nb - 1)
            lv  = self.lv.view(nb, 3)[idx].clamp(min=self.rc.uncert_min)  # (B, 3)
            w   = torch.exp(-lv)
            total = (
                (out[..., 0:1] * w[:, 0].view(-1, 1, 1)).sum() / denom
                + (out[..., 1:4] * w[:, 1].view(-1, 1, 1)).sum() / (denom * 3)
                + (out[..., 4:7] * w[:, 2].view(-1, 1, 1)).sum() / (denom * 3)
                + lv.mean(0).sum()  # the +log(sigma^2) barrier; zero at init
            )
            v = lv.detach().exp().mean(0)
            logs = {"uncert/var_energy": v[0], "uncert/var_dir": v[1], "uncert/var_pos": v[2]}
        out_logs = {}
        if self.training:
            out_logs = {
                # unweighted, so these stay comparable across uncert settings
                "loss/energy": loss_e.detach(),
                "loss/out_dir": loss_dir.detach(),
                "loss/out_pos": loss_x.detach(),
                "loss/raw": (loss_e + loss_dir + loss_x).detach(),
                **logs,
            }
        return total, out_logs

    @torch.no_grad()
    def _sample_mask(self, ds_t: DataStruct) -> Tensor:
        if not self.rc.mask_conf:
            return ds_t.m.full
        fwd = ds_t.m.full
        inv = (fwd == 0) & (ds_t.am.full == 1)
        inv[:, :self.rc.n_prefix] = 0  # all conditioning scalars stay conditioning
        pick = torch.rand(fwd.shape[0], device=fwd.device) < self.rc.mask_conf.get("p_forward", 0.5)
        return torch.where(pick.unsqueeze(-1), fwd, inv.long())

    def _step(self, ds_t: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        with torch.no_grad():
            ds_t = DataStruct(ds_t.f.full, self._sample_mask(ds_t), ds_t.am.full)
            if self.rc.overflow_delta > 0 or self.rc.edep_overflow_delta > 0:
                ds_t = self._shift_overflow_targets(ds_t)
            if self.sym is not None:
                fwd  = ds_t.m.full[:, self.rc.n_prefix] == 0
                face = self.sym.face_of(ds_t.f.in_cc[..., 0, 4:7])
                ds_t = DataStruct(self._canon_dirs(ds_t.f.full, face, fwd), ds_t.m.full, ds_t.am.full)
            base      = self.gen_base_wrapper(ds_t)
            pdgid_idx = self.convert_pdgids(ds_t.f.pdgids)
            if self.rc.t_dist == "sm_norm":
                t = torch.sigmoid(self.rc.t_dist_scale * torch.randn_like(ds_t.f.d))
            elif self.rc.t_dist == "sd3":
                u = torch.rand_like(ds_t.f.d)
                t = 1 - u + self.rc.t_dist_scale / 3 * ((torch.pi / 2 * u).sin() ** 2 - u)
            else:
                raise ValueError(f"unknown t_dist: {self.rc.t_dist!r}")
            if self.rc.t_zero_frac > 0:
                t = t * (torch.rand_like(t) >= self.rc.t_zero_frac)
            if self.rc.t_dist_shift != 1.0:
                t = t.clamp(min=0) ** (1.0 / self.rc.t_dist_shift)
            if (
                self.reflow_teacher is not None and self.training
                and self.global_step % self.rc.reflow_every == 0
            ):  # reflow: couple base to the teacher's transport of it
                tgt = self.reflow_teacher.solve(ds_t, x_init=base, **self.rc.reflow_kwargs)
                tgt = torch.where((ds_t.m.full == 1).unsqueeze(-1), tgt, ds_t.f.model_in)
                ds_t = DataStruct(
                    torch.cat((tgt, ds_t.f.pdgids), dim=-1), ds_t.m.full, ds_t.am.full,
                )
            ps_ = self.ps.sample(base, ds_t.f.model_in, t)
        step_extras = ({"d": torch.zeros_like(ps_.t)}
                       if getattr(self.model.vf, "step_cond", False) else {})
        v_out = self.model(
            ps_.x_t, ps_.t,
            mask=ds_t.m.full, attn_mask=ds_t.am.full,
            types=self.types_embd, pdgids=pdgid_idx, **step_extras,
        )
        sq = (v_out - ps_.dx_t) ** 2
        loss, logs = self.reduce_loss(sq, ds_t, t=t)
        if logs:
            self.log_dict(logs, on_step=True, on_epoch=False, logger=True, sync_dist=False)

        every = self.rc.one_step_euler_every
        fac   = self.rc.one_step_euler_fac
        lvf   = None
        if fac > 0 and hasattr(self, "lv_flow"):
            # Barrier every step, data term only when the gate fires: the
            # expectations still meet at exp(lv_flow) = E[L_flow], and lv_flow
            # never leaves the graph (DDP find_unused_parameters=False).
            lvf  = self.lv_flow.clamp(min=self.rc.uncert_min).squeeze()
            loss = loss + fac * lvf
        if fac > 0 and (not self.training or self.global_step % every == 0):
            sc = one_step_euler_loss(self, base, ds_t, pdgid_idx)
            w = fac * (every if self.training else 1)
            if lvf is not None:
                w = w * torch.exp(-lvf)
            if self.training:
                logs = {"loss/one_step_euler": sc.detach()}
                if lvf is not None:
                    logs["uncert/var_flow"] = lvf.detach().exp()
                self.log_dict(logs, on_step=True, on_epoch=False, logger=True, sync_dist=False)
            loss = loss + w * sc

        if (ot := self._base_dist_loss) is not None:
            loss = loss + self.rc.base_dist_loss * ot
            if self.training:
                self.log_dict(
                    {"loss/base_dist": ot.detach(),
                     "base/sm_scale": self.gen_base.sm_scale.detach().mean()},
                    on_step=True, on_epoch=False, logger=True, sync_dist=False,
                )
        return loss

    def training_step(self, batch: tuple, _batch_idx: int | Tensor) -> Tensor:
        return self._step(batch, _batch_idx)

    @torch.no_grad()
    def validation_step(self, batch: DataStruct, _batch_idx: int | Tensor) -> Tensor:
        bs = len(batch)
        loss = self._step(batch, _batch_idx)
        self.log("Validation Loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        for name, val in self.val_metrics(self, batch).items():
            self.log(name, val, on_epoch=True, sync_dist=True, batch_size=bs)
        return loss

