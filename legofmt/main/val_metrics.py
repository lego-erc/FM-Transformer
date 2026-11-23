"""Validation metrics of the flow: generated against true events of one batch (``ShowerValMetrics``).

```
particle   every outgoing particle as (mom theta, mom phi, energy, pos theta, pos phi)
summary    per event the mean and std of those five, over its e-, its e+ and its gamma separately, and E_dep
both       standardised by the pooled mean / std, then
             MMD   unbiased, Gaussian kernel exp(-|x - y|**2 / bw), bw = the median squared distance
             W1    per feature, mean |sort(x) - sort(y)| at equal N
E_dep      per incoming species: W1, and the generated / true mean on the events with at least two outgoing
           particles (edep_by_primary)
layers     the edep_ae decoder's E_dep layers against the true ones (edep_layer_metrics)
```

Copied from lego-eval (metrics.py and summary.py): legofmt cannot import lego_eval, the import would be circular.
The metrics have two limits. ``standardize`` pools the true and the generated statistics and the MMD bandwidth is
derived per call, so neither number is comparable across runs. The event summary covers only outgoing e-/e+/gamma
(its features are zero for a hadronic run), which is why ``edep_by_primary`` exists.
"""

import torch
import torch.nn.functional as F

from legofmt.data.struct import CHANNELS, _F
from legofmt.geometry.geom_trafos import GeomTrafos
from legofmt.numerics import masked_mean

_GEOM = GeomTrafos()

KIN_NAMES = ["mom_theta", "mom_phi", "energy", "pos_theta", "pos_phi"]
# The incoming species ``edep_by_primary`` splits E_dep by, with their log tags. A superset of any one dataset's:
# a species that is not in a batch is not logged.
PRIMARY_TAGS = {11: "em", -11: "ep", 22: "g", 2212: "p", 2112: "n",
                211: "pip", -211: "pim"}

_TYPES = (11, -11, 22)   # the outgoing species the event summary covers
SUMMARY_FEATURE_NAMES = [
    f"{PRIMARY_TAGS[pid]}_{k}_{stat}"
    for pid in _TYPES
    for stat in ("mean", "std")
    for k in KIN_NAMES
] + ["E_dep"]

# The prefixes of ``edep_by_primary``'s keys. Which of them a rank logs depends on the species and the multiplicities
# its own shard holds, so they must never be logged with sync_dist: the ranks would issue different numbers of
# collectives and NCCL would deadlock.
UNSYNCED_PREFIXES = ("val/w1_edep/", "val/edep_ratio_multi/")


def particle_kinematics(mom, e, pos, active):
    """Returns the five kinematic features (``KIN_NAMES``) of the ``active`` particles, (N, 5)."""
    return _spherical(mom, e, pos)[active]


def event_summary(mom, e, pos, pdgid, active, e_dep):
    """Returns each event's summary, (B, 31): the kinematics' mean and std per species of ``_TYPES``, then E_dep.

    The mean and std run over the event's ``active`` particles of that species; a species the event lacks gives zeros.
    The features are in the order of ``SUMMARY_FEATURE_NAMES``.
    """
    kin  = _spherical(mom, e, pos).unsqueeze(2)                                        # (B, L, 1, F)
    own  = (pdgid.unsqueeze(-1) == pdgid.new_tensor(_TYPES)) & active.unsqueeze(-1)    # (B, L, S): by species
    own  = own.unsqueeze(-1)
    mean = masked_mean(kin, own, dim=1)                                                # (B, S, F)
    std  = torch.sqrt(masked_mean((kin - mean.unsqueeze(1)) ** 2, own, dim=1))
    return torch.cat((torch.stack((mean, std), dim=2).flatten(1), e_dep.unsqueeze(-1)), dim=-1)

def standardize(a, b):
    """Standardises both sets by their pooled mean and std.

    A model whose samples are too wide raises the pooled std and so lowers its own W1, and every call has its own
    normaliser.
    """
    anchor = torch.cat((a, b), dim=0)
    mu = anchor.mean(0)
    sigma = anchor.std(dim=0).clamp(min=1e-8)
    return (a - mu) / sigma, (b - mu) / sigma


def compute_mmd(X, Y, max_samples=2000):
    """Returns the MMD of two sets under a Gaussian kernel: the root of the unbiased MMD**2, held at 0 or above.

    ```
    MMD**2 = mean_{i != j} k(x_i, x_j) + mean_{i != j} k(y_i, y_j) - 2 mean_{i, j} k(x_i, y_j)
    k(a, b) = exp(-|a - b|**2 / bw),   bw = the median squared distance within the pooled sets
    ```

    Each set is subsampled to ``max_samples`` rows, which bounds the N**2 kernel matrices.
    """
    X, Y = _subsample(X, max_samples), _subsample(Y, max_samples)
    bw = _median_heuristic(X, Y)
    k = lambda a, b: torch.exp(-torch.cdist(a, b) ** 2 / bw)
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    n, m = X.shape[0], Y.shape[0]
    mmd_sq = (
        (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1))
        + (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))
        - 2 * Kxy.mean()
    )
    return torch.sqrt(mmd_sq.clamp(min=0))


def w1_per_feature(X, Y):
    """Returns the 1-Wasserstein distance per feature of two sets of equal size."""
    return torch.abs(X.sort(0).values - Y.sort(0).values).mean(0)


def edep_by_primary(edep_real, edep_fake, pdgid_in, n_out) -> dict:
    """Computes the accuracy of the E_dep token, split by incoming species.

    ```
    val/w1_edep/<tag>            W1 between the generated and the true E_dep tokens
    val/edep_ratio_multi/<tag>   the generated / true mean on the events with n_out >= 2, given at least 32 of them
    ```

    The second is a ratio, not W1, because the error is a scale error and a ratio reads as a percentage. No other
    metric resolves the incoming species: ``event_summary`` covers outgoing e-/e+/gamma and pools every incoming
    species, so an E_dep error of one incoming species moves none of them (measured: +15% for protons and +20% for
    neutrons at ``n_out >= 4``).

    Args:
        edep_real: (B,), the true E_dep tokens.
        edep_fake: (B,), the generated E_dep tokens.
        pdgid_in:  (B,), the incoming pdgids.
        n_out:     (B,), each event's number of outgoing particles.

    Returns:
        The metrics by log name, for the incoming species of ``PRIMARY_TAGS`` that the batch holds.
    """
    out = {}
    for pid, tag in PRIMARY_TAGS.items():
        sel = pdgid_in == pid
        if not bool(sel.any()):
            continue
        out[f"val/w1_edep/{tag}"] = w1_per_feature(edep_real[sel], edep_fake[sel])
        multi = sel & (n_out >= 2)
        if int(multi.sum()) >= 32:   # enough events for a mean
            out[f"val/edep_ratio_multi/{tag}"] = edep_fake[multi].mean() / edep_real[multi].mean().clamp(min=1e-8)
    return out


def edep_layer_metrics(real: torch.Tensor, fake: torch.Tensor) -> dict:
    """Compares the E_dep layers the ``model_conf.edep_ae`` decoder generates with the true ones.

    ```
    val/layers/mean_ratio    the median over the layers of a layer's generated / true mean deposit
    val/layers/sd_ratio      the median over the layers of a layer's generated / true std of the deposit
    val/layers/depth_ratio   the generated / true mean_depth (the depth counted in layers)
    ```

    The medians run over the layers with a true deposit.

    Args:
        real: (B, n_layers), the true E_dep layers [MeV].
        fake: (B, n_layers), the generated E_dep layers [MeV].
    """
    live = real.mean(0) > 0
    l    = torch.arange(real.shape[1], device=real.device, dtype=real.dtype)   # each layer's depth
    return {
        "val/layers/mean_ratio": (fake.mean(0) / real.mean(0))[live].median(),
        "val/layers/sd_ratio": (fake.std(0) / real.std(0).clamp(min=1e-12))[live].median(),
        "val/layers/depth_ratio": mean_depth(fake, l) / mean_depth(real, l),
    }


def mean_depth(E: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """Returns the energy-weighted depth ``sum_l l E_l / sum_l E_l``, averaged over the events with a deposit.

    ``E`` (B, n_layers), ``l`` (n_layers,) the layers' depths.
    """
    return ((E * l).sum(1) / E.sum(1).clamp(min=1e-12))[E.sum(1) > 0].mean()


class ShowerValMetrics:
    """The metrics of the module docstring: a call generates the batch with the model and returns them by log name."""

    def __call__(self, lego, ds_t) -> dict:
        gen = _overflow_decoded(_F(self._generate(lego, ds_t)))
        active, pdg = ds_t.am.out_p, ds_t.f.out_pid
        real, fake  = _mom_e_pos(ds_t.f.out_cc), _mom_e_pos(gen.out_p)
        out = {
            **_compare("particle", KIN_NAMES,
                       particle_kinematics(*real, active), particle_kinematics(*fake, active)),
            **_compare("summary", SUMMARY_FEATURE_NAMES,
                       event_summary(*real, pdg, active, ds_t.f.edep), event_summary(*fake, pdg, active, gen.edep)),
            **edep_by_primary(ds_t.f.edep, gen.edep, ds_t.f.in_pid.long(), active.sum(-1)),
        }
        if getattr(lego, "edep_ae", None) is not None and ds_t.edep_layers is not None:
            out.update(edep_layer_metrics(*_layers_mev(lego, ds_t, gen.edep)))
        return out

    @staticmethod
    def _generate(lego, ds_t):
        """Generates the batch, (B, L, 7), with ``odeint_conf``'s ``step_size`` and ``method`` only.

        Enters and leaves the flow's frame as ``Solvers._forward`` does.
        """
        cfg = lego.rc.odeint_conf
        ds_t, frame = lego._enter_frame(ds_t)
        out = lego.solve(ds_t, x_init=lego.gen_base_wrapper(ds_t), step_size=cfg.get("step_size", 0.04),
                         method=cfg.get("method", "midpoint"))
        return lego._leave_frame(out, ds_t, frame)


# ---- plumbing --------------------------------------------------------------------------------------------------------

def _mom_e_pos(tokens):
    """Returns the momentum, energy and position channels of particle tokens."""
    e, mom, pos, *_ = tokens.tensor_split(CHANNELS, dim=-1)
    return mom, e, pos


def _spherical(mom, e, pos):
    """Returns ``(mom theta, mom phi, energy, pos theta, pos phi)`` per particle."""
    mom_sph = _GEOM.to_sph(F.normalize(mom, dim=-1, eps=1e-8))
    pos_sph = _GEOM.to_sph(F.normalize(pos, dim=-1, eps=1e-8))
    return torch.cat((mom_sph, e, pos_sph), dim=-1)


def _subsample(a, max_samples):
    """Returns at most ``max_samples`` rows of ``a``, a random subset when it has more."""
    return a[torch.randperm(a.shape[0], device=a.device)[:max_samples]] if a.shape[0] > max_samples else a


def _median_heuristic(X, Y, max_samples=2000):
    """Returns the kernel bandwidth: the median squared distance within the pooled (subsampled) sets."""
    c = _subsample(torch.cat((X, Y), dim=0), max_samples)
    D = torch.cdist(c, c) ** 2
    off = ~torch.eye(D.shape[0], dtype=torch.bool, device=c.device)
    return D[off].median().clamp(min=1e-8)


def _compare(tag, names, real, fake) -> dict:
    """Returns the MMD and the per-feature W1 of the standardised sets, by log name."""
    real_s, fake_s = standardize(real, fake)
    return {f"val/mmd_{tag}": compute_mmd(fake_s, real_s),
            **{f"val/w1_{tag}/{n}": w1 for n, w1 in zip(names, w1_per_feature(fake_s, real_s))}}


def _overflow_decoded(gen: _F) -> _F:
    """Holds the generated outgoing energies and E_dep in [0, 1], in place, and returns ``gen``.

    ``to_mev`` and ``denorm_edep_layer`` decode a generated value in the same way. A value near ``-overflow_delta``
    stands for an exact zero of the data: compared raw, such values inflate ``val/mmd_particle`` (4.5x measured).
    """
    gen.out_energy.clamp_(0.0, 1.0)
    gen.edep.clamp_(0.0, 1.0)
    return gen


def _layers_mev(lego, ds_t, gen_edep):
    """Returns ``(true, generated)``, the E_dep layers [MeV], (B, n_layers) each.

    The edep_ae samples the generated ones from the latents the last ``solve`` left and the generated total E_dep.
    """
    dp   = lego._edep_dp
    e_in = dp.pen.from_scalar(ds_t.f.in_mom, ds_t.f.in_energy).norm(dim=-1)
    edep = dp.denorm_edep_margin(gen_edep.float(), e_in.float(), ds_t.f.in_pid)
    fake = lego.edep_ae.sample(lego._edep_latents, edep)
    return dp.denorm_edep_layer(ds_t.edep_layers.float()), fake
