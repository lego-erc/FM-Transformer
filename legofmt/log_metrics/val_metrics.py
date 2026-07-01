"""Validation distribution metrics for the FM training loop.

Vendored from ``lego-eval`` (``lego_eval/metrics.py`` + ``lego_eval/summary.py``):
``legofmt`` cannot import ``lego_eval`` (it depends on ``legofmt`` -- circular).
Torch + the package's :class:`~legofmt.geometry.geom_trafos.GeomTrafos` for the
spherical conversion. ``compute_mmd`` returns an on-device scalar tensor
(DDP-loggable); ``w1_per_feature`` is the equal-N specialisation (real/generated
share the active mask); multiplicity counts are dropped (pdgid/multiplicity are
conditioned).

``ShowerValMetrics`` is the single entry point: it owns the generation call and
all the slot/column conventions, so ``LEGOLtng`` only connects and logs.
"""

import torch

from legofmt.data.struct import _F

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


def _spherical(mom, e, pos):
    """``(..., 3), (..., 1), (..., 3) -> (..., 5)``: mom (theta, phi), energy, pos (theta, phi).

    ``GeomTrafos.to_sph`` assumes unit vectors (``acos(z)``), so momentum and
    position are direction-normalised first; ``e`` is the bounded energy scalar.
    """
    mom_dir = mom / mom.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    pos_dir = pos / pos.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.cat([_GEOM.to_sph(mom_dir), e, _GEOM.to_sph(pos_dir)], dim=-1)


def particle_kinematics(mom, e, pos, active):
    """Flatten active particles to ``(N_active, 5)`` spherical kinematics."""
    return _spherical(mom, e, pos)[active]


def event_summary(mom, e, pos, pdgid, active, e_dep):
    """Per-event ``(B, 31)``: per-type {e-, e+, g} kinematic mean+std, then E_dep."""
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
    """Joint μ/σ over ``cat([a, b])``; returns standardized ``(a, b)``."""
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


def compute_mmd(X, Y):
    """Unbiased MMD (Gaussian RBF, median-heuristic bandwidth). Returns a scalar tensor."""
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
    """1-D Wasserstein per feature for equal-N samples. Returns ``(D,)``."""
    return (X.sort(0).values - Y.sort(0).values).abs().mean(0)


class ShowerValMetrics:
    """MMD + per-feature W1 between generated and real outgoing showers.

    Stateless. Call with the :class:`~legofmt.main.modules.LEGOLtng` instance
    and a validation ``DataStruct`` batch; returns a flat
    ``{log_name: scalar_tensor}`` dict. Owns the ODE generation and every
    slot/column convention so the module stays a thin connector.
    """

    def __call__(self, lego, ds_t) -> dict:
        gen = self._generate(lego, ds_t)
        gout = _F(gen).out_p  # layout-aware outgoing slice (handles extra cond slots)
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
        return out

    @staticmethod
    def _generate(lego, ds_t):
        cfg = lego.rc.odeint_conf
        step = cfg.get("step_size", 0.04)
        grid = torch.arange(0, 1 + step, step, device=lego.device).clamp_max(1)
        return lego.solve(
            ds_t, x_init=lego.gen_base_wrapper(ds_t),
            step_size=step, method=cfg.get("method", "midpoint"), time_grid=grid,
        )
