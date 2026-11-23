"""Options of the vector field superseded by the recipe (:mod:`legofmt.cfm.cfm_trafo_x`), kept so older configs and
checkpoints still run unchanged:

    time_cond: false   no time: one learned condition for every token (``global_cond``); the one-step models of
                       ``legofmt.legacy.reflow`` are built this way
    per-token t / d    times of shape (B, L) in place of one per event (B, 1)
    dim_in_out         a linear map into and out of the encoder (it has to equal ``h_dim``)
    grad_ckpt          gradient checkpointing of every encoder block

``CFMTrafo_x`` derives from ``LegacyField``: its ``__init__`` ends with ``_init_legacy``, and its ``forward`` hands
a call to ``_legacy_forward`` when the field has ``time_cond: false`` or ``dim_in_out``, or the call's times come
per token. ``grad_ckpt`` only wraps the encoder blocks. The methods read what ``CFMTrafo_x`` builds (``vf``,
``geom``, ``lat_in`` / ``lat_out``, ``_one_hot`` / ``_embed`` / ``_project_out`` / ``_time_emb``).
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def pin_per_token(t: Tensor, d: Tensor | None, mask: Tensor, attn_mask: Tensor) -> tuple[Tensor, Tensor | None]:
    """Pins per-token times on the conditioning tokens: ``t = 1`` and ``d = 0`` (they are not transported).

    ``ProjectModel._call_field`` hands a ``t`` or ``d`` that is not one value per event here. Returns ``(t, d)``,
    each (B, L); a None ``d`` stays None.
    """
    t = torch.where(mask == 1, t.expand_as(attn_mask), 1.0)
    if d is not None:
        d = torch.where(mask == 1, d.expand_as(attn_mask), 0.0)
    return t, d


class LegacyField(nn.Module):
    """The base class of ``CFMTrafo_x``: the options the module docstring lists."""

    def _init_legacy(self, time_cond: bool, dim_in_out: int | None, grad_ckpt: bool) -> None:
        """Applies the options to a field that ``CFMTrafo_x.__init__`` has built.

        Raises:
            ValueError: ``step_cond`` with ``time_cond: false``.
        """
        self.time_cond    = time_cond
        self._legacy_opts = not time_cond or dim_in_out is not None
        if grad_ckpt:
            for _norms, _block, _residual in self.vf.attn_layers.layers:
                _block.forward = partial(checkpoint, _block.forward, use_reentrant=False)
        if not time_cond:
            if self.step_cond:
                raise ValueError("step_cond=True requires time_cond=True")
            del self.freqs, self.mask_freqs
            self.global_cond = nn.Parameter(torch.zeros(1, self.h_dim))

    def _needs_legacy(self, t: Tensor | None, d: Tensor | None) -> bool:
        """Returns whether a call needs ``_legacy_forward``.

        It does with ``time_cond: false``, with ``dim_in_out``, and when ``t`` or ``d`` comes per token.
        """
        return self._legacy_opts or t.shape[-1] != 1 or (d is not None and d.shape[-1] != 1)

    def _legacy_forward(
        self,
        x: Tensor,
        mask: Tensor,
        attn_mask: Tensor,
        types: Tensor,
        pdgids: Tensor | None,
        t: Tensor | None = None,
        d: Tensor | None = None,
        z: Tensor | None = None,
    ) -> Tensor:
        """Predicts the velocity of every token, with any of the options.

        ``t`` / ``d`` per token (B, L) are used as given. Per event (B or 1, 1) they condition the generated
        (``mask == 1``) tokens, and the conditioning tokens get ``t = 1, d = 0``. With ``time_cond: false`` every
        token gets the one learned ``global_cond`` instead. Arguments and result as in ``CFMTrafo_x.forward``.
        """
        B, L = x.shape[:2]
        K = 0 if self.geom is None else self.geom.K
        n_lat = int(self.lat_dim > 0)
        n_pre = n_lat + K          # prepended tokens: [latent, geometry x K], then the sequence
        cond_attn = c_gen = None
        if not self.time_cond:
            cond = self.global_cond.expand(B, -1)
        else:
            c_con = self._time_emb(t.new_ones(1, 1), None if d is None else d.new_zeros(1, 1)).expand(B, 1, -1)
            if t.shape[-1] == 1 and (d is None or d.shape[-1] == 1):
                gen   = (mask == 1).unsqueeze(-1)
                c_gen = self._time_emb(t, d).expand(B, 1, -1)
                cond  = torch.where(gen, c_gen, c_con)
                if self._per_event_cond and 3 != L + n_pre <= self.h_dim:
                    # the selector spans the whole sequence: the latent token is generated, the K geometry tokens
                    # are conditioning tokens
                    sel = F.pad(gen.squeeze(-1).to(cond.dtype), (n_pre, self.h_dim - L - n_pre))
                    if n_lat:
                        sel[:, 0] = 1.0
                    cond_attn = torch.cat((c_con, c_gen, sel.unsqueeze(1)), dim=1)
            else:
                cond = self._time_emb(t, d)

        oh   = self._one_hot(mask, types, pdgids)
        embd = self._embed(x, oh)
        if self.time_cond:
            embd = embd + cond
        if self.geom is not None:
            tok = self.geom().to(embd.dtype).expand(B, -1, -1)
            if self.time_cond:
                c_tok = c_con.expand(B, K, -1)
                tok = tok + c_tok
                cond = torch.cat((c_tok, cond), dim=1)
            embd = torch.cat((tok, embd), dim=1)
            if self.geom_hide:
                attn_mask = attn_mask.clone()
                attn_mask[:, self.geom_scalar_tokens] = False
            attn_mask = torch.cat((attn_mask.new_ones(B, K), attn_mask), dim=1)
        if n_lat:
            tok = self.lat_in(z.to(embd.dtype)).unsqueeze(1)
            if self.time_cond:
                if c_gen is None:  # per-token t: the event's t is that of its generated tokens
                    gen_any = (mask == 1).any(-1, keepdim=True)
                    pick = lambda a, fill: torch.where(
                        gen_any, a.masked_fill(mask != 1, -torch.inf).amax(-1, keepdim=True), fill)
                    c_gen = self._time_emb(pick(t, 1.0), None if d is None else pick(d, 0.0))   # (B, 1, h_dim)
                c_lat = c_gen.expand(B, 1, -1)
                tok = tok + c_lat
                cond = torch.cat((c_lat, cond.expand(B, -1, -1)), dim=1)
            embd = torch.cat((tok, embd), dim=1)
            attn_mask = torch.cat((attn_mask.new_ones(B, 1), attn_mask), dim=1)

        # project_in / project_out: identities unless dim_in_out is set
        embd = self.vf.emb_dropout(self.vf.project_in(embd))
        h = self.vf.project_out(self.vf.attn_layers(
            embd, mask=attn_mask, condition=cond if cond_attn is None else cond_attn))
        v = self._project_out(h[:, n_pre:], mask, oh)
        return (v, self.lat_out(h[:, 0])) if n_lat else v
