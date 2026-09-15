"""Energy <-> model-scalar conversion.

``to_scalar`` maps MeV to ``log(|p|/cutoff) / log(max_energy/cutoff)`` clamped to
``[0, 1]``. For outgoing rows ``DataPrep.cc_trafo`` then stores the *fractional
log energy loss*

    u = 1 - log(E_out/cutoff) / log(E_in/cutoff)   <=>   E_out = cutoff * (E_in/cutoff)**(1 - u)

-- a ratio of logs, inverted and relative to the incoming particle. It is NOT
``1 - e_out/e_in``, which is what this docstring claimed until 2026-09-15.

``u`` is bounded at both ends but only one end is set by physics: ``cutoff``
pins ``u = 1`` (fully stopped), while ``u -> 0`` (barely interacting) is where
``u`` degenerates. For a small loss ``log(E_in/E_out) ~ dE/E_in``, so ``u`` goes
*linear* in the fractional loss and crushes the entire low end against zero.
``out_log_min`` restores constant relative resolution there; see
:meth:`norm_out`.
"""

import torch
from torch import Tensor


class EnergyProjections:

    def __init__(
        self,
        norm_type: str | bool = "in_frac",
        cutoff_mev: float = 10.0,
        max_energy: float | None = None,
        out_log_min: float | None = None,
    ):
        self.func = getattr(self, norm_type) if isinstance(norm_type, str) else self.identity
        self.cutoff = cutoff_mev
        self.max_energy = max_energy
        self.log_range = torch.tensor(max_energy / cutoff_mev).log().item() if max_energy else None
        # Smallest resolvable fractional log energy loss; None keeps the raw linear u.
        self.out_log_min = out_log_min or None
        self.out_log_range = (
            torch.tensor(1.0 / self.out_log_min).log().item() if self.out_log_min else None
        )

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def to_scalar(self, mom: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
        norm = mom.norm(dim=-1, keepdim=True)
        return mom / norm.clamp_min(eps), self.to_scalar_e(norm, eps)

    def to_scalar_e(self, e_mev: Tensor, eps: float = 1e-8) -> Tensor:
        return (torch.log(e_mev.clamp_min(eps) / self.cutoff) / self.log_range).clamp(0.0, 1.0)

    def from_scalar(self, dir_: Tensor, e: Tensor) -> Tensor:
        norm = self.cutoff * (self.max_energy / self.cutoff) ** e.clamp(0.0, 1.0)
        return dir_ * norm

    def norm_out(self, u: Tensor) -> Tensor:
        """Fractional log energy loss -> model scalar. Identity without ``out_log_min``.

        With it, ``u`` is respread logarithmically over ``[out_log_min, 1]``, so a
        fixed channel error becomes a fixed *relative* error in the loss instead of
        a fixed absolute one -- the move ``DataPrep.norm_edep`` makes for E_dep, for
        the same reason. ``u <= out_log_min`` folds onto 0 and joins the genuine
        no-interaction atom that neutrals produce (which
        ``_shift_overflow_targets`` then shifts to ``-overflow_delta``).

        Choosing it: charged particles never reach 0 -- their smallest real loss is
        u ~ 2.2e-4 in copper, 1.2e-3 in argon -- but neutrons pass through and their
        small-loss tail reaches u ~ 8e-7, so ``out_log_min`` must sit below that or
        it re-clamps them, the exact regression ``edep_log_min`` was added to undo.
        1e-7 is safe; 1e-5 is not.
        """
        if not self.out_log_min:
            return u
        return ((u / self.out_log_min).clamp_min(1.0).log()
                / self.out_log_range).clamp(0.0, 1.0)

    def denorm_out(self, v: Tensor) -> Tensor:
        """Inverse of :meth:`norm_out`. Zero maps to zero either way, so a decoded
        sentinel stays a no-loss particle."""
        if not self.out_log_min:
            return v
        return torch.where(
            v > 0, self.out_log_min * (v * self.out_log_range).exp(), torch.zeros_like(v),
        )

    def to_mev(self, e_model: Tensor, e_in: Tensor) -> Tensor:
        s_out = (1 - self.denorm_out(e_model.clamp(0.0, 1.0))) * e_in.clamp(0.0, 1.0)
        return self.cutoff * (self.max_energy / self.cutoff) ** s_out

    def identity(self, p_x: Tensor) -> Tensor:
        return p_x

    def log(self, p_x: Tensor) -> Tensor:
        p, x = p_x.split(3, -1)
        p_norm = p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.cat((p * (1 - p_norm.log()) / p_norm, x), -1)

    def in_frac(self, p_x: Tensor) -> Tensor:
        p, x = p_x.split(3, -1)
        in_norm = p[:, 0:1].norm(dim=-1, keepdim=True)
        p_normed = torch.cat((p[:, :1], p[:, 1:] / in_norm), dim=1)
        return torch.cat((p_normed, x), -1)

    def in_frac_log(self, p_x: Tensor) -> Tensor:
        p_x = self.in_frac(p_x)
        p_x[:, 1:] = self.log(p_x[:, 1:])
        return p_x
