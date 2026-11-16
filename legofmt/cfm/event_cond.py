"""The field's time condition, handed to the encoder's adaptive norms once per event instead of once per token.

Within an event the condition takes two values: ``c_gen`` on generated tokens, ``c_con`` on conditioning ones. The
adaptive RMSNorm / layer scale of every block map a token's condition to a gain (``to_gamma``). Per event, that
map runs on the two values instead of on every token, and each token picks its gain:

```
per-event (B, 3, D) = [c_con, c_gen, sel]     sel[i] = 1 on a generated token (its first n_tokens entries)
gamma_i             = to_gamma(c_gen) if sel[i] else to_gamma(c_con)
```

~10% of a training step and of a solve (plate-Si config, H100). ``per_event_gammas`` teaches the norms of an
x-transformers Encoder to read the per-event form, ``per_event_condition`` builds it.
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from x_transformers.x_transformers import AdaptiveLayerScale, AdaptiveRMSNorm


def per_event_gammas(attn_layers: nn.Module) -> bool:
    """Patches the adaptive norms / layer scales of ``attn_layers`` to take a per-event ``(B, 3, D)`` condition.

    Any other condition goes through the original forward.

    Returns:
        Whether it patched: False, with nothing patched, when something else would consume the condition.
    """
    mods = [m for m in attn_layers.modules() if isinstance(m, (AdaptiveRMSNorm, AdaptiveLayerScale))]
    other = any(type(m).__name__.startswith("Adaptive") for m in attn_layers.modules() if m not in mods)
    if not mods or other or type(getattr(attn_layers, "adaptive_mlp", nn.Identity())).__name__ != "Identity":
        return False
    for m in mods:
        m.forward = partial(_per_event_forward, m, m.forward)
    return True


def _per_event_forward(mod, orig, x, *, condition, **kwargs):
    if condition.shape[1] != 3 or x.shape[1] == 3:
        return orig(x, condition=condition, **kwargs)
    sel = condition[:, 2, :x.shape[1], None] > 0.5
    g = mod.to_gamma(condition[:, :2])
    pick = lambda g: torch.where(sel, g[:, 1:2], g[:, :1])
    if isinstance(mod, AdaptiveRMSNorm):  # x_transformers.AdaptiveRMSNorm.forward
        return F.normalize(x, dim=-1) * mod.scale * (pick(g) + 1.)
    g, out = pick(g.sigmoid()), mod.fn(x, **kwargs)  # x_transformers.AdaptiveLayerScale.forward
    return out * g if isinstance(out, Tensor) else (out[0] * g, *out[1:])


def per_event_condition(cond: Tensor, c_con: Tensor, c_gen: Tensor, gen: Tensor) -> Tensor:
    """Builds the per-event condition of a sequence for the patched norms.

    Args:
        cond:  (B, n, D), the condition per token: returned as it is when the per-event form cannot hold the sequence.
        c_con: (B, 1, D), the conditioning tokens' condition.
        c_gen: (B, 1, D), the generated tokens' condition.
        gen:   (B, n) bool, the generated tokens.

    Returns:
        ``(B, 3, D) = [c_con, c_gen, sel]``; ``cond`` when ``n > D`` (the selector does not fit a row) or ``n == 3``
        (a sequence the norms could not tell from the per-event form).
    """
    n, D = gen.shape[1], c_gen.shape[-1]
    if n == 3 or n > D:
        return cond
    sel = F.pad(gen.to(c_gen.dtype), (0, D - n))
    return torch.cat((c_con, c_gen, sel.unsqueeze(1)), dim=1)
