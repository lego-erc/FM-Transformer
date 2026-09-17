"""The conditional token embedding equals its gathered-weight einsum definition.

``CFMTrafo_x`` embeds every token with the sum of three index-selected linear
maps (mask, type, pdgid). The reference below materialises the summed
``(B, L, h_dim, in_dim)`` weight and contracts it, which is the original
formulation; the module must match it exactly (outputs and parameter
gradients) while never building that tensor.
"""

import torch

from legofmt.cfm.cfm_trafo_x import CFMTrafo_x


def _reference(vf, x, mask, types, pdgids, h):
    n = x.shape[1]
    mi, ti, pi = mask.view(-1), types.view(-1)[:n], pdgids.view(-1)
    s3, so, s4 = (-1, n, vf.h_dim), (-1, n, vf.in_dim), (-1, n, vf.h_dim, vf.in_dim)
    embd = (
        torch.einsum("ijl,ijkl->ijk", x,
                     vf.cond_w_mask[mi, 0].view(s4) + vf.cond_w_types[ti, 0] + vf.cond_w_pdgids[pi, 0].view(s4))
        + vf.cond_bi_mask[mi].view(s3) + vf.cond_bi_types[ti].view(s3) + vf.cond_bi_pdgids[pi].view(s3)
    ) / 3
    out = (mask == 1).unsqueeze(-1) * (
        torch.einsum("ijk,ijkl->ijl", h,
                     vf.cond_w_mask[mi, 1].view(s4) + vf.cond_w_types[ti, 1] + vf.cond_w_pdgids[pi, 1].view(s4))
        + vf.cond_bo_mask[mi].view(so) + vf.cond_bo_types[ti].view(so) + vf.cond_bo_pdgids[pi].view(so)
    ) / 3
    return embd, out


def _setup(B=5, L=9, npdgids=6):
    torch.manual_seed(0)
    vf = CFMTrafo_x(h_dim=32, nhead=4, max_seq_l=L, ntypes=4, in_dim=7, nlayers=1, dropout=0.0,
                    npdgids=npdgids, use_adaptive_rmsnorm=True, use_adaptive_layerscale=True,
                    attn_qk_norm=True, attn_qk_norm_scale=10)
    for p in (vf.cond_bi_mask, vf.cond_bi_types, vf.cond_bi_pdgids,
              vf.cond_bo_mask, vf.cond_bo_types, vf.cond_bo_pdgids):
        torch.nn.init.normal_(p)  # zero-init biases would hide indexing errors
    x      = torch.randn(B, L, 7)
    mask   = torch.randint(0, 2, (B, L))
    types  = torch.arange(L).clamp_max(3).view(1, -1)
    pdgids = torch.randint(0, npdgids, (B, L, 1))  # every caller passes _F.pdgids: (B, L, 1)
    return vf, x, mask, types, pdgids


def test_embedding_matches_the_einsum_reference():
    vf, x, mask, types, pdgids = _setup()
    h = torch.randn(x.shape[0], x.shape[1], vf.h_dim)
    embd_ref, out_ref = _reference(vf, x, mask, types, pdgids, h)
    oh = vf._one_hot(mask, types, pdgids, x.shape[0])
    embd, out = vf._embed(x, oh), vf._project_out(h, mask, oh)
    assert torch.allclose(embd, embd_ref, atol=1e-6), (embd - embd_ref).abs().max()
    assert torch.allclose(out, out_ref, atol=1e-6), (out - out_ref).abs().max()


def test_embedding_gradients_match_the_reference():
    vf, x, mask, types, pdgids = _setup()
    h = torch.randn(x.shape[0], x.shape[1], vf.h_dim, requires_grad=True)
    names = [n for n, _ in vf.named_parameters() if n.startswith("cond_")]

    def grads(fn):
        vf.zero_grad(); h.grad = None
        e, o = fn()
        (e.square().sum() + o.square().sum()).backward()
        return {n: p.grad.clone() for n, p in vf.named_parameters() if n in names}, h.grad.clone()

    g_ref, hg_ref = grads(lambda: _reference(vf, x, mask, types, pdgids, h))
    def _new():
        oh = vf._one_hot(mask, types, pdgids, x.shape[0])
        return vf._embed(x, oh), vf._project_out(h, mask, oh)
    g, hg = grads(_new)
    for n in names:
        assert torch.allclose(g[n], g_ref[n], atol=1e-5), (n, (g[n] - g_ref[n]).abs().max())
    assert torch.allclose(hg, hg_ref, atol=1e-5)


def test_full_forward_matches_reference_path():
    """End to end through the transformer: only the embedding changed, so a
    forward with the reference embedding substituted must agree."""
    vf, x, mask, types, pdgids = _setup()
    vf.eval()
    attn = torch.ones_like(mask, dtype=torch.bool)
    t = torch.rand(x.shape[0], x.shape[1])
    with torch.no_grad():
        v = vf(x, mask, attn, types, pdgids, t=t)
        # reference: embed with the einsum, run the same trunk, project with the einsum
        embd_ref, _ = _reference(vf, x, mask, types, pdgids, torch.zeros(*x.shape[:2], vf.h_dim))
        tf = t.unsqueeze(-1) * vf.freqs
        cond = torch.where(vf.mask_freqs.bool(), tf.sin(), tf.cos())
        hh = vf.vf.project_out(vf.vf.attn_layers(vf.vf.project_in(embd_ref + cond), mask=attn, condition=cond))
        _, v_ref = _reference(vf, x, mask, types, pdgids, hh)
    assert torch.allclose(v, v_ref, atol=1e-5), (v - v_ref).abs().max()


def test_state_dict_keys_are_unchanged():
    vf, *_ = _setup()
    keys = {k for k in vf.state_dict() if k.startswith("cond_")}
    assert keys == {
        "cond_w_mask", "cond_bi_mask", "cond_bo_mask",
        "cond_w_types", "cond_bi_types", "cond_bo_types",
        "cond_w_pdgids", "cond_bi_pdgids", "cond_bo_pdgids",
    }
