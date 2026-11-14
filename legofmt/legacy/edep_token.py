"""E_dep tokens superseded by the plain ``model_conf.edep_whiten`` (:mod:`legofmt.base.edep_whiten`), kept so older
configs and checkpoints still run unchanged:

    not whitened   the flow carries the deposit itself, and the base draws it as ``e_dep_max sigmoid(mu + sig z)``
    options        ``edep_whiten_global`` (c, s) uses one (mu, sig) for every event, ``edep_whiten_max_sig`` caps the
                   unit, ``edep_whiten_feed`` writes (mu, unit) onto the token's direction spheres for the network

``LegacyEdepToken`` precedes ``BaseDist`` in ``LEGOLtng``'s bases; on a plainly whitened model each override hands
straight to ``BaseDist``. Either way ``gen_base_wrapper`` returns the token in the unit the flow carries it in.
"""

import torch
from torch import Tensor, nn

from legofmt.data.struct import DataStruct, _F


def _plain(rc) -> bool:
    return (rc.edep_whiten and rc.edep_whiten_max_sig is None and rc.edep_whiten_global is None
            and not rc.edep_whiten_feed)


def whiten_edep_options(f: Tensor, generated: Tensor, mu: Tensor, sig: Tensor, e_dep_max: float, rc,
                        inverse: bool = False) -> Tensor:
    """``edep_whiten.whiten_edep`` with the options: ``unit`` in place of ``sig``, and ``sd = sig / unit`` (one prior
    sigma in whitened units) scaling the no-deposit rule."""
    if rc.edep_whiten_global is not None:
        c, s = rc.edep_whiten_global
        mu, sig = torch.full_like(mu, c), torch.full_like(sig, s)
    unit = sig if rc.edep_whiten_max_sig is None else sig.clamp(max=rc.edep_whiten_max_sig)
    sd   = sig / unit
    e    = _F(f).edep
    if inverse:
        back = e_dep_max * torch.sigmoid(mu + unit*e)
        new  = torch.where(e > -4.5*sd, back, torch.zeros_like(e))       # from below -4.5 sd: no deposit
    else:
        frac = (e / e_dep_max).clamp(1e-6, 1 - 1e-6)
        y    = (torch.logit(frac) - mu) / unit
        new  = torch.where(e > 0, torch.maximum(y, -4.4*sd), -6.0*sd)    # no deposit: -6 sd; a deposit: >= -4.4 sd
    f = f.clone()                              # e still reads the caller's tensor
    _F(f).edep.copy_(torch.where(generated, new, e))
    if rc.edep_whiten_feed:
        ones, token = torch.ones_like(mu), f[..., 1, 1:7]
        code = torch.cat([nn.functional.normalize(torch.stack((a, ones, 0 * ones), -1), dim=-1)
                          for a in (mu, torch.log(unit))], -1)
        token.copy_(torch.where(generated.unsqueeze(-1), torch.ones_like(token) if inverse else code, token))
    return f


class LegacyEdepToken:
    """``BaseDist`` methods for an E_dep token that is not whitened or is whitened with an option."""

    def _draw_prior(self, ds_t: DataStruct, rng) -> Tensor:
        """The E_dep draw as a deposit, ``e_dep_max sigmoid(mu + sig z)``, where ``BaseDist`` leaves ``z``."""
        x0 = super()._draw_prior(ds_t, rng)
        if _plain(self.rc):
            return x0
        mu, sig = self.gen_base.edep_mu, self.gen_base.edep_sig
        if torch.is_tensor(mu):
            mu, sig = mu.squeeze(-1), sig.squeeze(-1)
        x0[:, 1, 0] = self.gen_base.e_dep_max * torch.sigmoid(mu + sig * x0[:, 1, 0])
        return x0

    def gen_base_wrapper(self, ds_t, all_forward: bool = False, rng=None) -> Tensor:
        """With an option the deposit that was drawn is whitened like the data."""
        if not isinstance(ds_t, DataStruct):
            ds_t = DataStruct(*ds_t)
        x0 = super().gen_base_wrapper(ds_t, all_forward=all_forward, rng=rng)
        if _plain(self.rc) or not self.rc.edep_whiten:
            return x0
        return self._edep_whiten(x0, ds_t)

    def _edep_whiten(self, f: Tensor, ds_t: DataStruct, inverse: bool = False) -> Tensor:
        """Not whitened: ``f`` as it is."""
        if _plain(self.rc):
            return super()._edep_whiten(f, ds_t, inverse)
        if not self.rc.edep_whiten:
            return f
        _, mu, sig, _, _, _ = self.base_head_params(ds_t)   # by position: a legacy head returns a plain tuple
        return whiten_edep_options(f, ds_t.m.full[:, 1] == 1, mu.reshape(-1), sig.reshape(-1),
                                   self.gen_base.e_dep_max, self.rc, inverse)
