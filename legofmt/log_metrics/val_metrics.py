"""Vendored from lego-eval (metrics.py + summary.py); legofmt cannot import
lego_eval (circular). w1_per_feature is the equal-N specialisation.

Two known warts: ``standardize`` pools real+fake statistics and the MMD
bandwidth is re-derived per call, so neither number is comparable across runs;
the event summaries cover only e-/e+/gamma (zero features for hadronic runs).
``edep_by_primary`` exists because of that second one -- see its docstring."""

import torch

from legofmt.data.struct import DataStruct, _F

from legofmt.geometry.geom_trafos import GeomTrafos

_GEOM = GeomTrafos()

KIN_NAMES = ["mom_theta", "mom_phi", "energy", "pos_theta", "pos_phi"]
_TYPES = (11, -11, 22)
_TYPE_TAGS = ("em", "ep", "g")
SUMMARY_FEATURE_NAMES = [
    f"{tag}_{k}_{stat}"
    for tag in _TYPE_TAGS
    for stat in ("mean", "std")
    for k in KIN_NAMES
] + ["E_dep"]

# Incoming guns, for the per-primary E_dep split. Superset of any one dataset;
# species absent from a batch are simply not logged.
PRIMARY_TAGS = {11: "em", -11: "ep", 22: "g", 2212: "p", 2112: "n",
                211: "pip", -211: "pim"}

# Which keys edep_by_primary can emit. Their presence depends on the species and the
# multiplicities a rank's own shard happens to hold, so they must never be logged with
# sync_dist: ranks would issue different numbers of collectives and NCCL would deadlock.
UNSYNCED_PREFIXES = ("val/w1_edep/", "val/edep_ratio_multi/")


def _spherical(mom, e, pos):
    mom_dir = mom / mom.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    pos_dir = pos / pos.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.cat([_GEOM.to_sph(mom_dir), e, _GEOM.to_sph(pos_dir)], dim=-1)


def particle_kinematics(mom, e, pos, active):
    return _spherical(mom, e, pos)[active]


def event_summary(mom, e, pos, pdgid, active, e_dep):
    kin = _spherical(mom, e, pos)
    feats = []
    for pid in _TYPES:
        m = ((pdgid == pid) & active).unsqueeze(-1).to(kin.dtype)
        n = m.sum(1).clamp(min=1)
        mean = (kin * m).sum(1) / n
        var = (((kin - mean.unsqueeze(1)) ** 2) * m).sum(1) / n
        feats += [mean, var.sqrt()]
    return torch.cat([*feats, e_dep.unsqueeze(-1)], dim=-1)


def standardize(a, b):
    """Z-score both sets by their pooled mean/std: an over-dispersed model inflates
    sigma and reports a smaller W1, and each run gets its own normaliser."""
    anchor = torch.cat([a, b], dim=0)
    mu = anchor.mean(dim=0)
    sigma = anchor.std(dim=0).clamp(min=1e-8)
    return (a - mu) / sigma, (b - mu) / sigma


def _median_heuristic(X, Y, max_samples=2000):
    c = torch.cat([X, Y], dim=0)
    if c.shape[0] > max_samples:
        c = c[torch.randperm(c.shape[0], device=c.device)[:max_samples]]
    D = torch.cdist(c, c).square()
    off = ~torch.eye(D.shape[0], dtype=torch.bool, device=c.device)
    return D[off].median().clamp(min=1e-8)


def compute_mmd(X, Y, max_samples=2000):
    # unbiased MMD, Gaussian RBF with median-heuristic bandwidth; subsampled to bound the N^2 kernels
    X, Y = (a[torch.randperm(a.shape[0], device=a.device)[:max_samples]] if a.shape[0] > max_samples else a
            for a in (X, Y))
    bw = _median_heuristic(X, Y)
    k = lambda a, b: torch.exp(-torch.cdist(a, b).square() / bw)
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    n, m = X.shape[0], Y.shape[0]
    mmd_sq = ((Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1)) + (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))
              - 2 * Kxy.mean())
    return mmd_sq.clamp(min=0).sqrt()


def w1_per_feature(X, Y):
    return (X.sort(0).values - Y.sort(0).values).abs().mean(0)


def edep_by_primary(edep_real, edep_fake, pdgid_in, n_out) -> dict:
    """E_dep accuracy split by incoming species: W1 per primary, and the generated / true mean on the events with a
    secondary (``n_out >= 2``; a ratio, not W1, because the error is a scale error and a ratio reads as a percentage).

    Nothing else here resolves the primary: ``event_summary`` tags *outgoing* e-/e+/gamma and pools every gun, so a
    defect of one primary averages away. On rp_fm_had_base_180926 (2026-09-18) E_dep was within 1% for e-/e+/gamma at
    every multiplicity but +15% for protons and +20% for neutrons once ``n_out >= 4``, and no logged metric moved: the
    hadronic error lives where a nuclear interaction happened, which the E_dep prior cannot see."""
    out = {}
    for pid, tag in PRIMARY_TAGS.items():
        sel = pdgid_in == pid
        if not bool(sel.any()):
            continue
        out[f"val/w1_edep/{tag}"] = (edep_real[sel].sort().values - edep_fake[sel].sort().values).abs().mean()
        multi = sel & (n_out >= 2)
        if int(multi.sum()) >= 32:
            out[f"val/edep_ratio_multi/{tag}"] = edep_fake[multi].mean() / edep_real[multi].mean().clamp(min=1e-8)
    return out


def edep_layer_metrics(real: torch.Tensor, fake: torch.Tensor) -> dict:
    """Per-layer E_dep in MeV, ``(B, n_layers)``, from the ``model_conf.edep_ae`` decoder against the truth:
    medians over the layers that saw a deposit of the generated / true mean and spread, and the ratio of the
    mean energy-weighted depth (in layers)."""
    live = real.mean(0) > 0
    l = torch.arange(real.shape[1], device=real.device, dtype=real.dtype)
    depth = lambda E: ((E * l).sum(1) / E.sum(1).clamp(min=1e-12))[E.sum(1) > 0].mean()
    return {
        "val/layers/mean_ratio": (fake.mean(0) / real.mean(0))[live].median(),
        "val/layers/sd_ratio": (fake.std(0) / real.std(0).clamp(min=1e-12))[live].median(),
        "val/layers/depth_ratio": depth(fake) / depth(real),
    }


class ShowerValMetrics:

    def __call__(self, lego, ds_t) -> dict:
        gen = _F(self._generate(lego, ds_t))
        # decode the overflow sentinel as to_mev / denorm_edep do: a generated -overflow_delta
        # is a truth zero, and compared raw it inflated val/mmd_particle 4.5x on plateSi_131026
        gen.out_p[..., 0].clamp_(0.0, 1.0)
        gen.edep.clamp_(0.0, 1.0)
        active, pdg = ds_t.am.out_p, ds_t.f.out_p[..., -1]
        sides = ((ds_t.f.out_cc, ds_t.f.edep), (gen.out_p, gen.edep))  # (outgoing, E_dep) of the truth, generated
        split = lambda cc: (cc[..., 1:4], cc[..., 0:1], cc[..., 4:7])   # mom, e, pos
        reps = {
            "particle": (KIN_NAMES, *(particle_kinematics(*split(cc), active) for cc, _ in sides)),
            "summary": (SUMMARY_FEATURE_NAMES, *(event_summary(*split(cc), pdg, active, ed) for cc, ed in sides)),
        }
        out = {}
        for tag, (names, real, fake) in reps.items():
            real_s, fake_s = standardize(real, fake)
            out[f"val/mmd_{tag}"] = compute_mmd(fake_s, real_s)
            out.update({f"val/w1_{tag}/{n}": w1 for n, w1 in zip(names, w1_per_feature(fake_s, real_s))})
        out.update(edep_by_primary(ds_t.f.edep, gen.edep, ds_t.f.in_p[..., 0, -1].long(), active.sum(-1)))
        if getattr(lego, "edep_ae", None) is not None and ds_t.edep_layers is not None:
            dp = lego._edep_dp
            e_in = dp.pen.from_scalar(ds_t.f.in_cc[:, 0, 1:4], ds_t.f.in_cc[:, 0, 0:1]).norm(dim=-1)
            fake = lego.edep_ae.sample(lego._edep_latents,
                                       dp.denorm_edep_row(gen.edep.float(), e_in.float(), ds_t.f.in_p[:, 0, -1]))
            out.update(edep_layer_metrics(dp.denorm_edep(ds_t.edep_layers.float()), fake))
        return out

    @staticmethod
    def _generate(lego, ds_t):
        """Generate with the ``odeint_conf`` solver settings, canonicalising the
        directions first when ``canon_sym`` is on, exactly as ``Solvers.forward``."""
        cfg = lego.rc.odeint_conf
        sym = getattr(lego, "sym", None)
        if sym is not None:
            fwd  = ds_t.m.full[:, lego.rc.n_prefix] == 0
            face = sym.face_of(ds_t.f.in_cc[..., 0, 4:7])
            ds_t = DataStruct(lego._canon_dirs(ds_t.f.full, face, fwd), ds_t.m.full, ds_t.am.full)
        out = lego.solve(ds_t, x_init=lego.gen_base_wrapper(ds_t), step_size=cfg.get("step_size", 0.04),
                         method=cfg.get("method", "midpoint"))
        if getattr(lego.rc, "dir_zoom", None):
            out = lego._zoom_dirs(out, ds_t.am.full, ds_t.m.full[:, lego.rc.n_prefix] == 0, inverse=True,
                                  pdgids=ds_t.f.pdgids)
        return out if sym is None else lego._canon_dirs(out, face, fwd, inverse=True)
