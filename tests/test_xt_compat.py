"""x-transformers upgrade guards: qk-norm scale back-compat and fp32 attention."""

from __future__ import annotations

import pytest
import torch
from x_transformers.attend import Attend

from legofmt.cfm.cfm_trafo_x import CFMTrafo_x
from legofmt.mod_comps import config as cfg_mod
from legofmt.mod_comps.config import resolve_legoltng_config, resolve_mult_config

from test_modules_direct import _tiny_config


def _ckpt(xt_version: str | None):
    c = _tiny_config()
    c["config"]["model_conf"]["model_args"]["attn_qk_norm"] = True
    if xt_version is not None:
        c["config"]["additional"] = {"x_transformers_version": xt_version}
    return c


@pytest.mark.skipif(cfg_mod._XT_VERSION < cfg_mod._XT_QK_NORM_FIX, reason="needs x-transformers >= 2.25.5")
def test_old_checkpoint_gets_cubed_qk_norm_scale():
    rc = resolve_legoltng_config(_ckpt(None))
    assert rc.model_args["attn_qk_norm_scale"] == 1000
    assert rc.config["additional"]["x_transformers_version"] == str(cfg_mod._XT_VERSION)


@pytest.mark.skipif(cfg_mod._XT_VERSION < cfg_mod._XT_QK_NORM_FIX, reason="needs x-transformers >= 2.25.5")
def test_new_checkpoint_keeps_qk_norm_scale():
    c = _ckpt(str(cfg_mod._XT_VERSION))
    c["config"]["model_conf"]["model_args"]["attn_qk_norm_scale"] = 10
    rc = resolve_legoltng_config(c)
    assert rc.model_args["attn_qk_norm_scale"] == 10


def test_no_qk_norm_untouched():
    c = _ckpt(None)
    c["config"]["model_conf"]["model_args"]["attn_qk_norm"] = False
    rc = resolve_legoltng_config(c)
    assert "attn_qk_norm_scale" not in rc.model_args


@pytest.mark.skipif(cfg_mod._XT_VERSION < cfg_mod._XT_QK_NORM_FIX, reason="needs x-transformers >= 2.25.5")
def test_mult_checkpoint_shared_model_args_cubed_once():
    mm = {"attn_qk_norm": True}
    mm_conf = {"model_args": mm, "h_dim": 8, "ptypes": torch.tensor([11, 22]), "ptypes_in": torch.tensor([11]),
               "max_out_particles": 3, "cond_scalars": ("Density",)}
    c = {"state_dict": {}, "config": {"mm_conf": mm_conf, "dl_conf": {}}}
    rc = resolve_mult_config(c)
    assert rc.model_args["attn_qk_norm_scale"] == 1000        # inv_model_args aliases model_args
    assert rc.inv_model_args["attn_qk_norm_scale"] == 1000


@pytest.mark.parametrize("qk_scale, expected", [(1000, torch.float32), (10, torch.bfloat16)])
def test_attention_kernel_dtype_under_autocast(monkeypatch, qk_scale, expected):
    seen, orig = [], Attend.forward
    # fp32_attention wraps the bound forward, so record the dtype the kernel itself receives
    monkeypatch.setattr(Attend, "forward", lambda self, q, k, v, *a, **kw: (seen.append(q.dtype), orig(self, q, k, v, *a, **kw))[1])
    vf = CFMTrafo_x(h_dim=16, nhead=2, max_seq_l=5, ntypes=5, in_dim=7, nlayers=1, npdgids=2, attn_qk_norm=True,
                    attn_qk_norm_scale=qk_scale, use_adaptive_rmsnorm=True, use_adaptive_layerscale=True, dropout=0.0)
    x = torch.randn(2, 5, 7)
    mask = torch.ones(2, 5, dtype=torch.long); am = torch.ones(2, 5, dtype=torch.bool)
    types = torch.arange(5).view(1, -1); pdg = torch.ones(2, 5, dtype=torch.long)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        v = vf(x, mask, am, types, pdg, t=torch.rand(2, 5))
    assert seen and all(d == expected for d in seen)
    assert torch.isfinite(v).all()
