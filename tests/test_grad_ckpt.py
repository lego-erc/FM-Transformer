"""Tests for ``model_args.grad_ckpt`` -- per-block gradient checkpointing in
:class:`legofmt.cfm.cfm_trafo_x.CFMTrafo_x`.

Checkpointing must be a pure memory/speed trade: identical outputs, identical
gradients, identical state_dict (so checkpoints load in both directions), and
it must actually recompute the blocks during backward.
"""

from __future__ import annotations

import torch

from legofmt.cfm.cfm_trafo_x import CFMTrafo_x


def _build(grad_ckpt: bool) -> CFMTrafo_x:
    torch.manual_seed(0)
    return CFMTrafo_x(
        h_dim=32, in_dim=7, nlayers=2, nhead=2, ff_mult=2, nvtypes=2,
        max_seq_l=10, ntypes=4, npdgids=5, dropout=0.0,
        use_adaptive_rmsnorm=True, use_adaptive_layerscale=True,
        grad_ckpt=grad_ckpt,
    )


def _inputs(B: int = 2, L: int = 10):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(B, L, 7, generator=g)
    t = torch.rand(B, L, generator=g)
    return dict(
        x=x, t=t,
        mask=torch.ones(B, L, dtype=torch.long),
        attn_mask=torch.ones(B, L, dtype=torch.bool),
        types=torch.zeros(B, L, dtype=torch.long),
        pdgids=torch.zeros(B, L, dtype=torch.long),
    )


def _fwd(m: CFMTrafo_x, inp: dict) -> torch.Tensor:
    return m(inp["x"], inp["mask"], inp["attn_mask"], inp["types"],
             inp["pdgids"], t=inp["t"])


def test_grad_ckpt_matches_baseline_outputs_and_grads() -> None:
    a, b = _build(False), _build(True)
    b.load_state_dict(a.state_dict())
    a.train(); b.train()
    inp = _inputs()

    out_a, out_b = _fwd(a, inp), _fwd(b, inp)
    assert torch.allclose(out_a, out_b, atol=1e-6), (out_a - out_b).abs().max()

    out_a.square().sum().backward()
    out_b.square().sum().backward()
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.grad is None:
            assert pb.grad is None, n
            continue
        assert torch.allclose(pa.grad, pb.grad, atol=1e-6), n


def test_grad_ckpt_recomputes_blocks_in_backward() -> None:
    calls = {False: 0, True: 0}
    for ckpt in (False, True):
        m = _build(ckpt)
        m.train()
        attn = m.vf.attn_layers.layers[0][1].fn  # first Attention block

        def hook(*_a, _ck=ckpt, **_k):
            calls[_ck] += 1

        h = attn.register_forward_hook(hook)
        _fwd(m, _inputs()).square().sum().backward()
        h.remove()
    assert calls[False] == 1, calls
    assert calls[True] == 2, calls  # forward + recompute during backward


def test_grad_ckpt_state_dict_keys_unchanged() -> None:
    assert set(_build(False).state_dict()) == set(_build(True).state_dict())
