"""The flow's base (prior) distribution: the learned per-event head and the wrapper that draws ``x_0``.

Mixin methods rather than a standalone object, because ``lego_eval`` reaches for them on the LightningModule itself:
``lego.gen_base``, ``lego.gen_base_wrapper`` and ``hasattr(lego, "base_head")``. ``base_head`` is also a state_dict
key, so it must stay a direct attribute of ``LEGOLtng`` under that exact name -- which is why ``build_base_head``
returns the module for ``__init__`` to assign rather than owning it here.

Members resolved through ``self`` and owned by ``LEGOLtng``: ``rc``, ``model``, ``gen_base``, ``base_head``,
``pdgids_template``, ``convert_pdgids``, ``_base_head_loss``, ``_make_loader``, ``_train_ds``.
"""

from dataclasses import replace
from itertools import chain, repeat

import torch
from torch import Tensor, nn

from flow_matching.utils.manifolds import Sphere

from legofmt.cfm.geom_encoder import x0_gcm2
from legofmt.compat import load_legacy_base_head
from legofmt.data.struct import DataStruct, _F
from legofmt.rng import species_rank_ids

try:
    from torch_linear_assignment import batch_linear_assignment
except ImportError:
    batch_linear_assignment = None
try:  # skips TLA's NaN / infeasibility .item() checks: our costs are finite and square
    from torch_linear_assignment._triton import batch_linear_assignment as _lap_cuda_unchecked
except ImportError:
    _lap_cuda_unchecked = None

_OT_COUPLING_REQUIRES_LAP = (
    "ot_coupling=True requires `torch-linear-assignment` (a pixi dependency). "
    "Install it or set model_conf.ot_coupling=False."
)

# Conditioning scalars base_head_params reads; no base_head without them.
_BASE_HEAD_SCALARS = ("Z", "A", "Size")
# base_head_lik: uniform-outlier weight (else far-tail units dominate), smallest theta / pi float32 resolves
# (censored below), q bounds, required base_conf flags.
_LIK_EPS, _LIK_W0, _LIK_Q = 1e-3, 3e-4, (0.25, 4.0)
_LIK_FLAGS = ("tanh_theta", "base_head_nout", "ang_shape", "ang_nout", "ang_species")
_HALF_LOG_2PI = float(torch.tensor(2 * torch.pi, dtype=torch.float64).log()) / 2


def _en_logp(u: Tensor, s: Tensor, q: Tensor, eps: float = _LIK_EPS) -> Tensor:
    """Float64 log density of ``u = 1 - tanh(s |z|^q)``, ``z ~ N(0, 1)``, plus an ``eps`` uniform outlier."""
    u, s, q = u.double().clamp(1e-6, 1 - 1e-12), s.double(), q.double()
    y, e = torch.atanh(1 - u) / s, u.new_tensor(eps)  # y = |z|^q
    lp = u.new_tensor(2 / torch.pi).log() / 2 - y.pow(2 / q) / 2 + (1 / q - 1) * y.log() - (q * s * u * (2 - u)).log()
    return torch.logaddexp(lp + (-e).log1p(), e.log())


def _ang_logp(w: Tensor, kap: Tensor, pw: Tensor, eps: float = _LIK_EPS) -> Tensor:
    """Float64 log density of ``w = theta / pi``, ``theta = pi tanh(|z| / kap)^pw`` (``GeomTrafos.sample``), plus an
    ``eps`` uniform outlier. ``w < W0`` (float32 acos rounds it to 0) is scored as ``P(w < W0)``, else exact zeros
    reward ever larger pw."""
    w, kap, pw = w.double(), kap.double(), pw.double()
    wc = w.clamp(_LIK_W0, 1 - 1e-7)
    s = wc.pow(1 / pw).clamp(max=1 - 1e-12)  # tanh(|z| / kap)
    r = kap * torch.atanh(s)                  # |z|
    lp = (0.5 * w.new_tensor(2 / torch.pi).log() - r ** 2 / 2                         # half-normal at |z|
          + kap.log() - pw.log() + (1 / pw - 1) * wc.log() - (-s ** 2).log1p())    # |d|z| / dw|
    cdf0 = torch.erf(kap * torch.atanh(w.new_tensor(_LIK_W0).pow(1 / pw)) / 2 ** 0.5)
    e = w.new_tensor(eps)
    return torch.where(w < _LIK_W0, ((1 - e) * cdf0 + e * _LIK_W0).log(), torch.logaddexp(lp + (-e).log1p(), e.log()))


class BaseHeadLik(nn.Module):
    """``base_conf.base_head_lik``: one head for every base parameter, fitted by likelihood (2026-11 study: a third of
    a transformer's seed spread at equal or better NLL).

    In ``[_base_head_x, _out_set_feats]``: a trunk on [(E, chord, log t) standardised, primary one-hot, out-set] plus
    per-primary slopes, then one hidden layer per output group. Out ``[log sm, log q, mu, log sig, log kappa, log pw]``,
    all but mu and sig per outgoing class. Runs at IEEE fp32: TF32 / bf16 moved E_dep mu by up to 0.8 sigma (wsi16)."""

    def __init__(self, n_cls: int, hidden, sm0: float, kap0: float):
        super().__init__()
        w, w2 = hidden
        n_cont = 4 + n_cls  # (E, chord, log t) + out-set features
        self.n_cls = n_cls
        self.l0 = nn.Linear(n_cont + n_cls, w)
        self.k0 = nn.Linear(n_cls * n_cont, w, bias=False)
        nn.init.uniform_(self.k0.weight, -n_cont ** -0.5, n_cont ** -0.5)  # fan-in of the nonzero block
        self.ge, self.gd, self.ga = (nn.Sequential(nn.Linear(w, w2), nn.Mish()) for _ in range(3))
        self.re, self.rd, self.ra = nn.Linear(w2, 2 * n_cls), nn.Linear(w2, 2), nn.Linear(w2, 2 * n_cls)
        for r in (self.re, self.rd, self.ra):  # zero readouts: start at the priors
            nn.init.zeros_(r.weight)
            nn.init.zeros_(r.bias)
        self.re.bias.data[:n_cls] = torch.tensor(sm0).log()
        self.ra.bias.data[:n_cls] = torch.tensor(kap0).log()
        self.register_buffer("mean", torch.zeros(3))
        self.register_buffer("std", torch.ones(3))   # both set on the training set; inf drops a constant input

    def forward(self, x: Tensor) -> Tensor:
        n, prec = self.n_cls, torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            cont = (x[:, [0, 1 + n, 2 + n]] - self.mean) / self.std
            oh, extra = x[:, 1:1 + n], x[:, 3 + n:]
            kron = (oh.unsqueeze(-1) * torch.cat((cont, extra), -1).unsqueeze(1)).flatten(1)
            h = nn.functional.mish(self.l0(torch.cat((cont, oh, extra), -1)) + self.k0(kron))
            return torch.cat((self.re(self.ge(h)), self.rd(self.gd(h)), self.ra(self.ga(h))), -1)
        finally:
            torch.set_float32_matmul_precision(prec)


def build_base_head(rc, gen_base) -> nn.Module | None:
    """The learned per-event base head, or ``None`` when the config wants none.

    The caller assigns the result as ``LEGOLtng.base_head``; the absence of the
    attribute is what ``gen_base_wrapper`` and ``setup`` test with ``hasattr``.
    """
    bc = rc.config.get("base_conf") or {}
    hs = bc.get("base_head")
    lik = bc.get("base_head_lik", False)
    if lik:  # explicit ask: checked before the gates below return None
        missing = [k for k in _LIK_FLAGS if not bc.get(k, False)]
        if missing or gen_base.scale_dist != "sm_norm":
            raise ValueError(f"base_head_lik needs scale_dist sm_norm and {_LIK_FLAGS} (its two passes); got "
                             f"scale_dist {gen_base.scale_dist!r}, missing {missing}.")
        if hs is None and rc.base_pretrain_batches <= 0:
            raise ValueError("base_head_lik needs base_pretrain_batches > 0 or a saved base_conf.base_head.")
    elif hs is not None and "k0.weight" in hs:
        raise ValueError("base_conf.base_head holds a base_head_lik head; set base_conf.base_head_lik: true.")
    # base_head_params reads Z/A/Size, so there is no head without them: an
    # explicit ask (a saved head that would otherwise vanish silently) is an
    # error, the base_pretrain_batches default is not.
    asked = hs is not None or lik
    if not ((asked or rc.base_pretrain_batches > 0) and gen_base.scale_dist == "sm_norm"):
        return None
    if not all(s in rc.cond_scalars for s in _BASE_HEAD_SCALARS):
        if asked:
            raise ValueError(f"base_head needs the {_BASE_HEAD_SCALARS} conditioning scalars; "
                             f"got cond_scalars={rc.cond_scalars}.")
        return None

    if lik:
        hidden = bc.get("base_head_hidden", [256, 256])
        if len(hidden) != 2:
            raise ValueError(f"base_head_lik needs base_head_hidden [trunk, group], got {hidden}.")
        head = BaseHeadLik(len(rc.pdgids_template) + 1, hidden, float(gen_base.sm_scale), float(gen_base.kappa))
        if hs is None:
            return head
        try:  # a mismatch raises, never silently re-pretrains
            head.load_state_dict(hs)
        except RuntimeError as err:
            raise ValueError("base_conf.base_head does not load into the base_head_lik head (base_head_hidden, "
                             "pdgids?).") from err
        return head.requires_grad_(False)

    n_in = 4 + len(rc.pdgids_template)
    if bc.get("base_head_nout", False):
        n_in += len(rc.pdgids_template) + 2
    n_ang = len(rc.pdgids_template) + 1 if bc.get("ang_species", False) else 1
    n_out = 3 + n_ang * (1 + int(bc.get("ang_shape", False)))
    dims = [n_in, *bc.get("base_head_hidden", [16])]  # hidden widths; 16 under-fits E_dep's prior
    head = nn.Sequential(*(m for a, b in zip(dims, dims[1:]) for m in (nn.Linear(a, b), nn.Mish())),
                         nn.Linear(dims[-1], n_out))
    last = f"{len(head) - 1}.weight"
    with torch.no_grad():
        head[-1].weight.zero_()
        head[-1].bias.copy_(torch.tensor(
            [float(gen_base.sm_scale), 1.0, 1.0]
            + [float(gen_base.kappa)] * n_ang + [1.0] * (n_out - 3 - n_ang)).log())
        if hs is None or hs.keys() != head.state_dict().keys():
            return head  # fresh head: zero-init output layer, priors in the bias
        if any(hs[k].shape != v.shape for k, v in head[:-1].state_dict().items()):
            return head   # base_head_nout / base_head_hidden toggled: re-pretrain, do not mis-load
        if hs[last].shape[0] < head[-1].weight.shape[0]:
            return head   # ang_shape / ang_species toggled on: re-pretrain, do not freeze at pow=1
        if hs[last].shape == head[-1].weight.shape:
            head.load_state_dict(hs)
        else:
            load_legacy_base_head(head, hs)
        head.requires_grad_(False)  # already fitted; the flow never trains the head
    return head


class BaseDist:
    """Base-distribution mixin for :class:`~legofmt.main.modules.LEGOLtng`."""

    def base_head_params(self, ds_t: DataStruct):
        """Per-event ``(sm_scale, edep_mu, edep_sig, kappa, ang_pow, sm_pow)`` from the head's
        inputs: incoming kinematics, species one-hot, cuboid chord,
        ``log(Size*Density/X0)``. ``ang_pow`` is None unless base_conf.ang_shape; ``sm_pow`` is None unless
        base_head_lik, which gives it and ``sm_scale`` per outgoing class."""
        bc      = self.rc.config.get("base_conf") or {}
        n_cls   = len(self.pdgids_template) + 1
        x       = self._base_head_x(ds_t)
        lik     = isinstance(self.base_head, BaseHeadLik)
        ne      = 2 * n_cls if lik else 1   # energy columns, read from the zeroed pass
        if bc.get("base_head_nout", False):
            # Only E_dep's (mu, sig) see the outgoing set. sm_scale and kappa are
            # read from a second pass with those inputs zeroed -- a constant, so
            # the NLL fits them to the marginal over n_out, as without the flag.
            # Conditioning them too moved kappa 5.6..44.7 across branches and cost
            # 2x on direction losses; conditioning E_dep alone kept the gain.
            extra   = torch.cat(self._out_set_feats(ds_t), dim=-1)
            both    = torch.cat((torch.cat((x, extra), -1), torch.cat((x, torch.zeros_like(extra)), -1)))
            with torch.autocast(x.device.type, enabled=False):  # fp32 as at inference and in pretraining
                out_e, out = self.base_head(both).chunk(2)  # one forward for both passes
            ang     = out_e if bc.get("ang_nout", False) else out
            out     = torch.cat((out[..., :ne], out_e[..., ne:ne + 2], ang[..., ne + 2:]), dim=-1)
        else:
            with torch.autocast(x.device.type, enabled=False):
                out = self.base_head(x)
        if lik:
            lsm, lq, mu, lsig, lkap, lpw = out.split((n_cls, n_cls, 1, 1, n_cls, n_cls), -1)
            return (lsm.clamp(-8, 8).exp(), mu, lsig.exp(), lkap.exp().clamp_min(1e-3), lpw.exp().clamp(0.1, 10.0),
                    lq.exp().clamp(*_LIK_Q))
        n_ang   = n_cls if bc.get("ang_species", False) else 1
        return (out[..., 0:1].exp(), out[..., 1:2], out[..., 2:3].exp(),
                out[..., 3:3 + n_ang].exp().clamp_min(1e-3),  # kap: divisor in sample()
                out[..., 3 + n_ang:].exp().clamp(0.1, 10.0) if out.shape[-1] > 3 + n_ang else None, None)

    def _base_head_x(self, ds_t: DataStruct) -> Tensor:
        """The head's inputs ``[E, species one-hot, chord, log t]``, ``(B, 3 + n_cls)``."""
        n_cls   = len(self.pdgids_template) + 1
        species = nn.functional.one_hot(self._species_idx(ds_t.f.in_p[..., 0, -1], n_cls), n_cls).float()
        t       = ds_t.f.cond("Size") * ds_t.f.cond("Density") / x0_gcm2(ds_t.f.cond("Z"), ds_t.f.cond("A"))
        # chord length: full chord through the cuboid along the incoming ray
        # (longest-edge lengths); fwd+bwd -> entry/exit-storage invariant
        inc     = ds_t.f.in_cc[..., 0, 1:7].nan_to_num(1.0)
        u, pos  = inc[..., :3], inc[..., 3:]
        d       = self.cuboid_unit.to(u)
        p       = pos / pos.abs().amax(-1, keepdim=True).clamp_min(1e-8) * d  # pos is dir(x / d)
        ok_u    = u.abs() > 1e-6
        u_safe  = torch.where(ok_u, u, torch.ones_like(u))   # the discarded branch must not divide by 0 (backward)
        tf      = torch.where(ok_u, (u.sign() * d - p) / u_safe, torch.full_like(u, 4.0))
        tb      = torch.where(ok_u, (p + u.sign() * d) / u_safe, torch.full_like(u, 4.0))
        chord   = ((tf.amin(-1) + tb.amin(-1)).clamp(0.0, 3.5) / 2).unsqueeze(-1)
        return torch.cat((ds_t.f.in_cc[..., 0], species, chord, t.log().unsqueeze(-1)), dim=-1)

    def _edep_whiten(self, f: Tensor, ds_t: DataStruct, inverse: bool = False) -> Tensor:
        """``model_conf.edep_whiten``: the generated E_dep row in units of the event's base prior (``inverse``: back).

            y = (logit(u / e_dep_max) - mu) / sig      a draw of the prior becomes N(0, 1)

        No deposit (u <= 0) sits 6 prior sigmas below mu and decodes back to 0 from below 4.5; a real deposit is held
        at 4.4, so it never turns into a pass-through. (mu, sig) come from the frozen head on ``ds_t``, so any
        sub-batch is consistent. Options: ``edep_whiten_global`` (c, s) is one map for every event,
        ``edep_whiten_max_sig`` caps the unit, ``edep_whiten_feed`` writes (mu, unit) on the row's direction spheres.
        Returns a copy; ``f`` may carry a leading time axis."""
        _, mu, sig, _, _, _ = self.base_head_params(ds_t)
        mu, sig = mu.reshape(-1), sig.reshape(-1)
        if self.rc.edep_whiten_global is not None:
            c, s = self.rc.edep_whiten_global
            mu, sig = torch.full_like(mu, c), torch.full_like(sig, s)
        unit = sig if self.rc.edep_whiten_max_sig is None else sig.clamp(max=self.rc.edep_whiten_max_sig)
        sd = sig / unit                            # one prior sigma in whitened units
        emax, e = self.gen_base.e_dep_max, _F(f).edep
        if inverse:
            new = torch.where(e > -4.5 * sd, emax * torch.sigmoid(mu + unit * e), torch.zeros_like(e))
        else:
            y = (torch.logit((e / emax).clamp(1e-6, 1 - 1e-6)) - mu) / unit
            new = torch.where(e > 0, torch.maximum(y, -4.4 * sd), -6.0 * sd)
        gen = ds_t.m.full[:, 1] == 1               # only generated E_dep rows change
        f = f.clone()                              # e still reads the caller's tensor
        _F(f).edep.copy_(torch.where(gen, new, e))
        if self.rc.edep_whiten_feed:
            ones, row = torch.ones_like(mu), f[..., 1, 1:7]
            code = torch.cat([nn.functional.normalize(torch.stack((a, ones, 0 * ones), -1), dim=-1)
                              for a in (mu, unit.log())], -1)
            row.copy_(torch.where(gen.unsqueeze(-1), torch.ones_like(row) if inverse else code, row))
        return f

    def _species_idx(self, pid: Tensor, n_cls: int) -> Tensor:
        """Class index in ``[0, n_cls)`` of raw pdgids (of indices with ``pdgid_is_idx``)."""
        return (pid.long() if self.rc.pdgid_is_idx else self.convert_pdgids(pid)).long().clamp(0, n_cls - 1)

    def _out_set_feats(self, ds_t: DataStruct) -> list[Tensor]:
        """Outgoing multiplicity and per-pdgid composition, both fixed before the
        flow runs. E_dep is bimodal in them, which one logit-normal cannot hold."""
        valid = ds_t.am.out_p
        n_cls = len(self.pdgids_template) + 1
        oh    = nn.functional.one_hot(self._species_idx(ds_t.f.out_p[..., -1], n_cls), n_cls)
        cnt   = (oh * valid.unsqueeze(-1)).sum(-2).float()
        return [valid.sum(-1, keepdim=True).float().log1p(), cnt.log1p()]

    def _base_moment_loss(self, ds_t, fwd, s, mu, sig, kap, pw=None) -> Tensor:
        """Moment-match the prior to the batch: quantile-node mean of the energy scale,
        logit-normal NLL for E_dep, and (``tanh_theta``) the mean opening angle for kappa."""
        z  = 2 ** 0.5 * torch.erfinv(torch.linspace(
            -0.995, 0.995, 100, device=s.device))  # N(0,1) quantile nodes
        v  = (ds_t.am.out_p & fwd.unsqueeze(-1)).float()
        n  = v.sum(-1).clamp(min=1)
        ev = (v.sum(-1) > 0).float()

        u_b = 1 - (z.abs() * s).tanh().mean(-1)
        u_t = (ds_t.f.out_cc[..., 0] * v).sum(-1) / n
        l_scale = ((u_b - u_t) ** 2 * ev).sum() / ev.sum().clamp(min=1)

        ed_t = ds_t.f.edep
        w    = fwd.float() * (ed_t > 0)
        ly   = torch.logit((ed_t.float() / self.gen_base.e_dep_max).clamp(1e-4, 1 - 1e-4))
        mu1, sig1 = mu.squeeze(-1), sig.squeeze(-1)
        l_edep = ((sig1.log() + (ly - mu1) ** 2 / (2 * sig1 ** 2))
                  * w).sum() / w.sum().clamp(min=1)

        if not self.gen_base.tanh_theta:
            return l_scale + l_edep

        tb   = (z.abs() / kap.unsqueeze(-1)).tanh()          # (B, 1, nodes) or (B, K, nodes) per slot
        if pw is not None:
            tb = tb.pow(pw.unsqueeze(-1))
        m1, m2 = tb.mean(-1), tb.square().mean(-1)
        u_i  = nn.functional.normalize(ds_t.f.in_cc[..., 0, 1:4], dim=-1).unsqueeze(1)
        p_i  = nn.functional.normalize(ds_t.f.in_cc[..., 0, 4:7], dim=-1).unsqueeze(1)
        a_m  = (ds_t.f.out_cc[..., 1:4] * u_i).sum(-1).clamp(-1, 1).acos()
        a_p  = (ds_t.f.out_cc[..., 4:7] * p_i).sum(-1).clamp(-1, 1).acos()
        ang  = (a_m + a_p) / 2 / torch.pi
        den  = ev.sum().clamp(min=1)
        if kap.shape[-1] > 1:   # per-slot prior: match each slot's own moments (ang_species)
            vs = v.sum().clamp(min=1)
            l_kappa = (((m1 - ang) ** 2) * v).sum() / vs
            if pw is not None:
                l_kappa = l_kappa + (((m2 - ang.square()) ** 2) * v).sum() / vs
        else:
            l_kappa = ((m1.squeeze(-1) - (ang * v).sum(-1) / n) ** 2 * ev).sum() / den
            if pw is not None:
                l_kappa = l_kappa + (
                    (m2.squeeze(-1) - (ang.square() * v).sum(-1) / n) ** 2 * ev).sum() / den
        l_kappa = l_kappa * (self.rc.config.get("base_conf") or {}).get("ang_weight", 1.0)
        return l_scale + l_edep + l_kappa

    def _base_lik_loss(self, ds_t, fwd, s, mu, sig, kap, pw, q) -> Tensor:
        """Per-unit mean NLLs (3,), float64, on the forward events: energy per continuum slot (u > 0; the atom is the
        flow's), both angles per outgoing slot (momentum / position about the incoming ones), E_dep per deposit event.
        ``s, q, kap, pw`` per slot, ``mu, sig`` per event."""
        f   = ds_t.f
        v   = ds_t.am.out_p & fwd.unsqueeze(-1)
        u   = f.out_cc[..., 0]
        c   = v & (u > 0)
        ok  = fwd & (f.edep > 0)
        u_i = nn.functional.normalize(f.in_cc[..., 0, 1:4], dim=-1).unsqueeze(1)
        p_i = nn.functional.normalize(f.in_cc[..., 0, 4:7], dim=-1).unsqueeze(1)
        wm  = (f.out_cc[..., 1:4] * u_i).sum(-1).clamp(-1, 1).acos() / torch.pi
        wp  = (f.out_cc[..., 4:7] * p_i).sum(-1).clamp(-1, 1).acos() / torch.pi
        le  = _en_logp(u, s, q)
        la  = _ang_logp(wm, kap, pw) + _ang_logp(wp, kap, pw)
        y   = torch.logit((f.edep.float() / self.gen_base.e_dep_max).clamp(1e-4, 1 - 1e-4))
        sig = sig.squeeze(-1)
        ld  = -(sig.log() + (y - mu.squeeze(-1)) ** 2 / (2 * sig ** 2) + _HALF_LOG_2PI)
        sums = torch.stack(((le * c).sum(), (la * v).sum(), (ld * ok).sum().double()))
        return -sums / torch.stack((c.sum(), v.sum(), ok.sum())).double().clamp(min=1)

    def _fit_base_head(self, ds_t, fwd) -> None:
        """Run the head, hand its outputs to the sampler, and cache the moment (``base_head_lik``: likelihood) loss
        while the head is trainable. With grad mode on (density design) the frozen head keeps its graph to the
        inputs, so the draw is reparameterized in them; only a head being fitted hands the sampler detached values."""
        learn = self.model.training and next(self.base_head.parameters()).requires_grad
        keep = (lambda v: v.detach()) if learn else (lambda v: v)
        with torch.set_grad_enabled(learn or torch.is_grad_enabled()):
            s, mu, sig, kap, pw, q = self.base_head_params(ds_t)
            if kap.shape[-1] > 1:   # ang_species: one column per species -> one value per slot
                idx = self._species_idx(ds_t.f.out_p[..., -1], kap.shape[-1])
                kap = kap.gather(1, idx)
                pw  = None if pw is None else pw.gather(1, idx)
                if q is not None:   # base_head_lik: (sm, q) too
                    s, q = s.gather(1, idx), q.gather(1, idx)
            if learn:
                self._base_head_loss = (self._base_moment_loss(ds_t, fwd, s, mu, sig, kap, pw) if q is None
                                        else self._base_lik_loss(ds_t, fwd, s, mu, sig, kap, pw, q).sum().float())
        self.gen_base.sm_scale = keep(s).unsqueeze(-1)
        self.gen_base.edep_mu  = keep(mu)
        self.gen_base.edep_sig = keep(sig)
        self.gen_base.kappa    = keep(kap)
        self.gen_base.ang_pow  = None if pw is None else keep(pw)
        self.gen_base.sm_pow   = 1.0 if q is None else keep(q).unsqueeze(-1)

    def _ot_couple(self, base: Tensor, ds_t, data: Tensor) -> Tensor:
        """Per-event Hungarian assignment of base slots to data slots, blocked across
        pdgids and valid/pad pairs; base samples are pdgid-independent so permuting is free."""
        if batch_linear_assignment is None:
            raise RuntimeError(_OT_COUPLING_REQUIRES_LAP)
        base = base.where(ds_t.am.full.unsqueeze(-1), data)
        pid  = ds_t.f.out_p[..., -1]
        blocked = (
            ds_t.am.out_p.unsqueeze(-1).logical_xor(ds_t.am.out_p.unsqueeze(-2))
            | (pid.unsqueeze(-1) != pid.unsqueeze(-2))
        )
        out = _F(base).out_p
        man = self.rc.manifold
        tgt = ds_t.f.out_cc.unsqueeze(-2).split(man.ambient_dims, dim=-1)
        ref = out.unsqueeze(-3).split(man.ambient_dims, dim=-1)
        cost = sum(
            (((a * b).sum(-1).float().clamp(-1 + 1e-6, 1 - 1e-6).acos()
              if isinstance(mf, Sphere) else (a - b).norm(dim=-1).float()) ** 2)
            for mf, a, b in zip(man.manifolds, tgt, ref)
        ) + blocked * 1e6
        assign = (_lap_cuda_unchecked(cost, validation="off") if cost.is_cuda and _lap_cuda_unchecked is not None
                  else batch_linear_assignment(cost)).long()
        out[:] = torch.take_along_dim(out, assign.unsqueeze(-1), dim=1)
        return base

    def gen_base_wrapper(self, ds_t: "DataStruct | tuple[Tensor, Tensor, Tensor]", all_forward: bool = False,
                         rng=None) -> Tensor:
        """Draw ``x_0``: data on conditioning cells, the (learned) prior on generated
        ones, an isotropic sample for inverse-mask events. ``lego_eval`` calls this. ``rng``
        (legofmt.rng.EventRNG) keys every draw on the event and an outgoing slot's (species, rank)."""
        if not isinstance(ds_t, DataStruct):
            ds_t = DataStruct(*ds_t)
        self._base_head_loss = None
        data  = ds_t.f.model_in
        m     = ds_t.m.full
        fwd   = m[:, self.rc.n_prefix] == 0  # incoming slot conditions => forward event
        noise = None if all_forward or fwd.all() else self.gen_base.iso(m.shape, data.device, rng=rng)
        if not all_forward and not fwd.any():
            return torch.where((m == 1).unsqueeze(-1), noise, data)

        if hasattr(self, "base_head"):
            self._fit_base_head(ds_t, fwd)
        elem = None if rng is None else species_rank_ids(ds_t.f.out_p[..., -1].long())
        base = torch.cat(
            (ds_t.f.non_cc, self.gen_base(ds_t.m.out_p.shape, ds_t.f.in_cc, rng=rng, elem=elem)), dim=1,
        )
        if self.rc.ot_coupling and self.model.training:
            base = self._ot_couple(base, ds_t, data)
        base = self.gen_base.insert_add(base, rng=rng)
        noise = base if noise is None else torch.where(fwd.view(-1, 1, 1), base, noise)
        return torch.where((m == 1).unsqueeze(-1), noise, data)

    def pretrain_base(self, batches, lr: float | None = None, steps: int | None = None) -> float:
        """Fit the base head alone on ``batches`` with Adam, then freeze it: the head
        is never trained alongside the flow. ``steps`` cosine-anneals the lr to lr/100
        over that many steps. Returns the final moment loss; ``lr`` defaults to 1e-2.
        ``base_head_lik`` (2026-11 study recipe): lr 3e-3, ``steps`` required, linear warmup over steps // 50, grad
        clip 1, IEEE fp32; a non-finite loss raises. Returns the final NLL."""
        lik          = isinstance(self.base_head, BaseHeadLik)
        if lik and steps is None:
            raise ValueError("base_head_lik pretraining needs steps (its warmup and cosine).")
        lr           = (3e-3 if lik else 1e-2) if lr is None else lr
        opt          = torch.optim.Adam(self.base_head.parameters(), lr=lr)
        if lik:
            warm  = max(1, steps // 50)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (
                0.01 + 0.99 * 0.5 * (1 + torch.cos(torch.tensor(torch.pi * min(s, steps) / steps)).item())))
        else:
            sched = None if steps is None else torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, lr / 100)
        prec         = torch.get_float32_matmul_precision() if lik else None  # read before any state changes
        was_training = self.model.training
        rc           = self.rc
        self.rc      = replace(rc, ot_coupling=False)
        self.model.train()
        ot = float("nan")
        try:
            if prec:   # the backward too (forward pins itself)
                torch.set_float32_matmul_precision("highest")
            for ds_t in batches:
                if rc.dir_zoom:
                    fwd_z = ds_t.m.full[:, rc.n_prefix] == 0
                    ds_t = DataStruct(self._zoom_dirs(ds_t.f.full, ds_t.am.full, fwd_z, pdgids=ds_t.f.pdgids),
                                      ds_t.m.full, ds_t.am.full)
                opt.zero_grad()
                self.gen_base_wrapper(ds_t)
                if self._base_head_loss is None:
                    continue
                if lik and not torch.isfinite(self._base_head_loss):  # never freeze a NaN prior into the flow
                    raise RuntimeError(f"base_head_lik pretraining: non-finite loss {self._base_head_loss.item()}.")
                self._base_head_loss.backward()
                if lik:
                    nn.utils.clip_grad_norm_(self.base_head.parameters(), 1.0)
                opt.step()
                if sched is not None:
                    sched.step()
                ot = self._base_head_loss.item()
        finally:
            self.model.train(was_training)
            self.rc = rc
            if prec:
                torch.set_float32_matmul_precision(prec)
        self.base_head.requires_grad_(False)
        self._base_head_loss = None
        return ot

    def _pretrain_base_if_needed(self) -> None:
        """Rank 0 pretrains on CPU-resident data moved to its device and broadcasts the
        weights; every rank ends with the head frozen."""
        if (self.rc.base_pretrain_batches and getattr(self, "_trainer", None)
                and hasattr(self, "base_head") and next(self.base_head.parameters()).requires_grad):
            if self.trainer.is_global_zero:
                dev = self.trainer.strategy.root_device
                self.base_head.to(dev)
                lik = isinstance(self.base_head, BaseHeadLik)
                if lik:   # standardise (E, chord, log t) on forward training events
                    ys = []
                    with torch.no_grad():
                        for b in self._make_loader(self._train_ds, shuffle=False, bs=2**17):
                            b = b.to(dev)
                            ys.append(self._base_head_x(b)[b.m.full[:, self.rc.n_prefix] == 0][:, [0, -2, -1]])
                    y  = torch.cat(ys)
                    sd = y.std(0)
                    self.base_head.mean.copy_(y.mean(0))
                    self.base_head.std.copy_(sd.where(sd > 1e-6, torch.inf))
                loader = self._make_loader(
                    self._train_ds, shuffle=True, bs=self.rc.base_pretrain_bs
                )
                final = self.pretrain_base((b.to(dev) for _, b in zip(  # one reshuffled epoch after another
                    range(self.rc.base_pretrain_batches), chain.from_iterable(repeat(loader)))),
                    steps=self.rc.base_pretrain_batches)
                self.base_head.cpu()
                print(f"base_head: pretrained on {self.rc.base_pretrain_batches} "
                      f"batches (final {'NLL' if lik else 'moment loss'} {final:.4f}), frozen")
            sd = self.trainer.strategy.broadcast(
                {k: v.cpu() for k, v in self.base_head.state_dict().items()}, src=0)
            self.base_head.load_state_dict(sd)
            self.base_head.requires_grad_(False)
