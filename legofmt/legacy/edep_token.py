"""E_dep tokens superseded by the plain ``model_conf.edep_whiten`` (:mod:`legofmt.base.edep_whiten`), kept so older
configs and checkpoints still run unchanged:

    not whitened   the flow carries the deposit itself, and the base samples it as ``e_dep_max sigmoid(mu + sig z)``
    options        ``edep_whiten_global: [c, s]`` uses one ``(mu, sig) = (c, s)`` for every event,
                   ``edep_whiten_max_sig`` caps the unit, ``edep_whiten_feed`` writes ``(mu, log unit)`` onto the
                   token's two direction channels for the field to read

``LegacyEdepToken`` precedes ``BaseDist`` in ``LEGOLtng``'s bases; on a model whitened without an option each
override defers to ``BaseDist``. Either way ``gen_base_wrapper`` returns the token in the unit the flow carries it in.
"""

import torch
from torch import Tensor, nn

from legofmt.data.struct import DataStruct, _F


def _plain(rc) -> bool:
    return (rc.edep_whiten and rc.edep_whiten_max_sig is None and rc.edep_whiten_global is None
            and not rc.edep_whiten_feed)


def whiten_edep_options(f: Tensor, generated: Tensor, mu: Tensor, sig: Tensor, e_dep_max: float, rc,
                        inverse: bool = False) -> Tensor:
    """Whitens the E_dep token of ``f`` as ``edep_whiten.whiten_edep`` does, with the options of ``rc``.

    ``unit = min(sig, edep_whiten_max_sig)`` takes the place of ``sig`` in the map, and the no-deposit values
    (-6, -4.5, -4.4) are scaled by ``sd = sig / unit``, one prior sigma in whitened units. ``edep_whiten_global``
    replaces every event's ``(mu, sig)``. ``edep_whiten_feed`` writes ``(mu, log unit)`` onto the token's direction
    channels (``inverse``: the (1, 1, 1) filler). Arguments and result as in ``whiten_edep``; ``rc`` is the resolved
    config.
    """
    if rc.edep_whiten_global is not None:
        c, s = rc.edep_whiten_global
        mu, sig = torch.full_like(mu, c), torch.full_like(sig, s)
    unit = sig if rc.edep_whiten_max_sig is None else sig.clamp(max=rc.edep_whiten_max_sig)
    sd   = sig / unit
    e    = _F(f).edep
    if inverse:
        back = e_dep_max * torch.sigmoid(mu + unit*e)
        new  = torch.where(e > -4.5*sd, back, torch.zeros_like(e))       # below -4.5 sd: no deposit
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
        """Samples the prior as ``BaseDist`` does, with the E_dep token as a deposit ``e_dep_max sigmoid(mu + sig z)``.

        ``BaseDist`` leaves ``z`` on the token: the unit a model whitened without an option carries.
        """
        x0 = super()._draw_prior(ds_t, rng)
        if _plain(self.rc):
            return x0
        mu, sig = self.gen_base.edep_mu, self.gen_base.edep_sig
        x0[:, 1, 0] = self.gen_base.e_dep_max * torch.sigmoid(mu + sig * x0[:, 1, 0])
        return x0

    def gen_base_wrapper(self, ds_t, all_forward: bool = False, rng=None) -> Tensor:
        """Samples ``x_0`` as ``BaseDist`` does.

        With a whitening option the sampled deposit is then whitened like the data.
        """
        if not isinstance(ds_t, DataStruct):
            ds_t = DataStruct(*ds_t)
        x0 = super().gen_base_wrapper(ds_t, all_forward=all_forward, rng=rng)
        if _plain(self.rc) or not self.rc.edep_whiten:
            return x0
        return self._edep_whiten(x0, ds_t)

    def _edep_whiten(self, f: Tensor, ds_t: DataStruct, inverse: bool = False, drawn: bool = False) -> Tensor:
        """Whitens the generated E_dep token of ``f`` with the options (``whiten_edep_options``).

        A token that is not whitened is returned as it is. Arguments as in ``BaseDist._edep_whiten``; ``drawn`` is
        read only on a model whitened without an option.
        """
        if _plain(self.rc):
            return super()._edep_whiten(f, ds_t, inverse, drawn)
        if not self.rc.edep_whiten:
            return f
        _, mu, sig, _, _, _ = self.base_head_params(ds_t)   # by position: a legacy head returns a plain tuple
        return whiten_edep_options(f, ds_t.m.edep == 1, mu, sig, self.gen_base.e_dep_max, self.rc, inverse)
