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
    if X.shape[0] > max_samples:
        X = X[torch.randperm(X.shape[0], device=X.device)[:max_samples]]
    if Y.shape[0] > max_samples:
        Y = Y[torch.randperm(Y.shape[0], device=Y.device)[:max_samples]]
    bw = _median_heuristic(X, Y)
    Kxx = torch.exp(-torch.cdist(X, X).square() / bw)
    Kyy = torch.exp(-torch.cdist(Y, Y).square() / bw)
    Kxy = torch.exp(-torch.cdist(X, Y).square() / bw)
    n, m = X.shape[0], Y.shape[0]
    mmd_sq = (
        (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1))
        + (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))
        - 2 * Kxy.mean()
    )
    return mmd_sq.clamp(min=0).sqrt()


def w1_per_feature(X, Y):
    return (X.sort(0).values - Y.sort(0).values).abs().mean(0)


def edep_by_primary(edep_real, edep_fake, pdgid_in, n_out) -> dict:
    """E_dep accuracy split by incoming species, and again on the events with a
    secondary.

    Nothing else in the validation set resolves the primary: ``event_summary``
    tags *outgoing* e-/e+/gamma and pools every gun together, so a defect that
    lives in one primary averages away. Measured on rp_fm_had_base_180926
    (2026-09-18): E_dep was within 1% for e-/e+/gamma at every multiplicity but
    +15% for protons and +20% for neutrons once ``n_out >= 4``, and none of the
    logged metrics moved. The split on ``n_out >= 2`` is where the hadronic
    error lives -- a nuclear interaction happened, and the E_dep prior cannot
    see that it did.

    Ratios, not W1, for the multi cell: the error is a scale error and a ratio
    reads directly as a percentage.
    """
    out = {}
    for pid, tag in PRIMARY_TAGS.items():
        sel = pdgid_in == pid
        if not bool(sel.any()):
            continue
        r, f = edep_real[sel], edep_fake[sel]
        out[f"val/w1_edep/{tag}"] = (r.sort().values - f.sort().values).abs().mean()
        multi = sel & (n_out >= 2)
        if int(multi.sum()) >= 32:
            rm, fm = edep_real[multi], edep_fake[multi]
            out[f"val/edep_ratio_multi/{tag}"] = fm.mean() / rm.mean().clamp(min=1e-8)
    return out


class ShowerValMetrics:

    def __call__(self, lego, ds_t) -> dict:
        gen = self._generate(lego, ds_t)
        gout = _F(gen).out_p
        active, pdg, rcc = ds_t.am.out_p, ds_t.f.out_p[..., -1], ds_t.f.out_cc
        reps = {
            "particle": (
                KIN_NAMES,
                particle_kinematics(rcc[..., 1:4], rcc[..., 0:1], rcc[..., 4:7], active),
                particle_kinematics(gout[..., 1:4], gout[..., 0:1], gout[..., 4:7], active),
            ),
            "summary": (
                SUMMARY_FEATURE_NAMES,
                event_summary(rcc[..., 1:4], rcc[..., 0:1], rcc[..., 4:7], pdg, active, ds_t.f.edep),
                event_summary(gout[..., 1:4], gout[..., 0:1], gout[..., 4:7], pdg, active, _F(gen).edep),
            ),
        }
        out = {}
        for tag, (names, real, fake) in reps.items():
            real_s, fake_s = standardize(real, fake)
            out[f"val/mmd_{tag}"] = compute_mmd(fake_s, real_s)
            out.update({
                f"val/w1_{tag}/{n}": w1
                for n, w1 in zip(names, w1_per_feature(fake_s, real_s))
            })
        out.update(edep_by_primary(
            ds_t.f.edep, _F(gen).edep, ds_t.f.in_p[..., 0, -1].long(),
            ds_t.am.out_p.sum(-1),
        ))
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
        out = lego.solve(
            ds_t, x_init=lego.gen_base_wrapper(ds_t),
            step_size=cfg.get("step_size", 0.04), method=cfg.get("method", "midpoint"),
        )
        return out if sym is None else lego._canon_dirs(out, face, fwd, inverse=True)
