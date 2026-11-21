"""The base head's network (``base_conf.base_head_lik``): an event's features -> the parameters of its prior."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from legofmt.base.base_dists import BaseParams


class BaseHead(nn.Module):
    """The base head: an MLP from an event's features to every parameter of its prior.

    ```
    x      = [c, s, o]                             c = (E, chord, log t) standardised, s = the primary's one-hot,
                                                   o = the outgoing set
    h(o)   = mish(l0 x + k0 outer(s, [c, o]))      the trunk; k0 holds a slope per input and primary

    energy   (log s, log q)      = re mish(ge h(0))    per outgoing class
    E_dep    (mu, log sig)       = rd mish(gd h(o))    per event
    angles   (log kappa, log pw) = ra mish(ga h(o))    per outgoing class
    ```

    The energy group is blind to the outgoing set: it reads the trunk with ``o`` zeroed, a constant, so the NLL fits
    it to the marginal over n_out. The head is fitted by likelihood (``nn_fit.py``). It is an MLP because that had a
    third of a transformer's seed spread at equal or better NLL (2026-11 study). It runs at IEEE fp32 whatever
    autocast and the matmul precision say: TF32 / bf16 moved E_dep mu by up to 0.8 sigma (wsi16).
    """

    def __init__(self, n_cls: int, hidden, sm0: float, kap0: float):
        super().__init__()
        trunk, group = hidden
        n_in = 3 + 1 + n_cls   # cont + out_set
        self.l0 = nn.Linear(n_in + n_cls, trunk)
        self.k0 = nn.Linear(n_cls * n_in, trunk, bias=False)   # every input times the primary's one-hot
        nn.init.uniform_(self.k0.weight, -n_in**-0.5, n_in**-0.5)  # fan-in of the nonzero block
        self.ge, self.gd, self.ga = (nn.Sequential(nn.Linear(trunk, group), nn.Mish()) for _ in range(3))
        self.re, self.rd, self.ra = nn.Linear(group, 2 * n_cls), nn.Linear(group, 2), nn.Linear(group, 2 * n_cls)
        for r in (self.re, self.rd, self.ra):  # zero readouts: start at the priors
            nn.init.zeros_(r.weight)
            nn.init.zeros_(r.bias)
        self.re.bias.data[:n_cls] = torch.log(torch.tensor(sm0))
        self.ra.bias.data[:n_cls] = torch.log(torch.tensor(kap0))
        self.register_buffer("mean", torch.zeros(3))
        self.register_buffer("std", torch.ones(3))   # both set on the training set; inf drops a constant input

    def forward(self, cont: Tensor, species: Tensor, out_set: Tensor) -> BaseParams:
        """Predicts the parameters of each event's prior.

        Args:
            cont:    (B, 3), (E, chord, log t).
            species: (B, n_cls), the primary's one-hot.
            out_set: (B, 1 + n_cls), the outgoing multiplicity and composition (``BaseDist._head_inputs``).

        Returns:
            ``BaseParams``: the energy and angle parameters (B, n_cls), one column per outgoing class, and E_dep's
            (B,).
        """
        B, n_cls = species.shape
        with ieee_fp32(), torch.autocast(cont.device.type, enabled=False):
            c, s, o = _seen_then_blind((cont - self.mean) / self.std, species, out_set)   # rows [B:]: o zeroed
            h = F.mish(
                self.l0(torch.cat((c, s, o), dim=-1))
                + self.k0(_outer(s, torch.cat((c, o), dim=-1)))
            )
            energy, e_dep, angles = self.re(self.ge(h)), self.rd(self.gd(h)), self.ra(self.ga(h))

        log_sm, log_q     = energy[B:].split(n_cls, dim=-1)   # the energy group: from the blind rows
        mu, log_sig       = e_dep[:B].unbind(-1)
        log_kappa, log_pw = angles[:B].split(n_cls, dim=-1)
        return BaseParams(   # each parameter inside its distribution's range
            sm_scale = torch.exp(log_sm.clamp(-8, 8)),         # s      in [e^-8, e^8]
            sm_pow   = torch.exp(log_q).clamp(0.25, 4.0),      # q      in [1/4, 4]
            kappa    = torch.exp(log_kappa).clamp(min=1e-3),   # kappa  >= 1e-3: a divisor in the sampler
            ang_pow  = torch.exp(log_pw).clamp(0.1, 10.0),     # pw     in [0.1, 10]
            edep_mu  = mu,
            edep_sig = torch.exp(log_sig),
        )


# ---- plumbing --------------------------------------------------------------------------------------------------------

@contextmanager
def ieee_fp32():
    """Runs float32 matmuls at full precision inside the block, whatever ``torch.set_float32_matmul_precision`` says."""
    prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prec)


def _seen_then_blind(c: Tensor, s: Tensor, o: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Stacks the batch twice, the second copy with the outgoing set ``o`` zeroed, so that one pass serves both."""
    return c.repeat(2, 1), s.repeat(2, 1), torch.cat((o, torch.zeros_like(o)))


def _outer(a: Tensor, b: Tensor) -> Tensor:
    """Computes the outer product of ``a`` and ``b`` per row, flattened: (B, i), (B, j) -> (B, i j)."""
    return (a.unsqueeze(-1) * b.unsqueeze(1)).flatten(1)
