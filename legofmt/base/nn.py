"""The base head's network (``base_conf.base_head_lik``): an event's features -> the parameters of its prior."""

from contextlib import contextmanager

import torch
from torch import Tensor, nn

from legofmt.base.base_dists import BaseParams


@contextmanager
def ieee_fp32():
    """Runs float32 matmuls at full precision inside the block, whatever ``torch.set_float32_matmul_precision`` says."""
    prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prec)


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
        self.n_cls = n_cls
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
            (B, 1).
        """
        B = len(cont)
        with ieee_fp32(), torch.autocast(cont.device.type, enabled=False):
            # the batch goes through twice in one pass: [:B] as given, [B:] with the outgoing set zeroed
            cont    = ((cont - self.mean) / self.std).repeat(2, 1)
            species = species.repeat(2, 1)
            out_set = torch.cat((out_set, torch.zeros_like(out_set)))

            slopes = (species.unsqueeze(-1) * torch.cat((cont, out_set), -1).unsqueeze(1)).flatten(1)
            h      = nn.functional.mish(self.l0(torch.cat((cont, species, out_set), -1)) + self.k0(slopes))
            energy, e_dep, angles = self.re(self.ge(h)), self.rd(self.gd(h)), self.ra(self.ga(h))

        log_sm, log_q     = energy[B:].chunk(2, -1)   # the energy group: from the blind pass
        mu, log_sig       = e_dep[:B].chunk(2, -1)
        log_kappa, log_pw = angles[:B].chunk(2, -1)
        return BaseParams(   # each parameter inside its distribution's range
            sm_scale = torch.exp(log_sm.clamp(-8, 8)),         # s      in [e^-8, e^8]
            sm_pow   = torch.exp(log_q).clamp(0.25, 4.0),      # q      in [1/4, 4]
            kappa    = torch.exp(log_kappa).clamp_min(1e-3),   # kappa  >= 1e-3: a divisor in the sampler
            ang_pow  = torch.exp(log_pw).clamp(0.1, 10.0),     # pw     in [0.1, 10]
            edep_mu  = mu,
            edep_sig = torch.exp(log_sig),
        )
