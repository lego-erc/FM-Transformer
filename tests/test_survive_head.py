"""model_conf.survive_head: a per-event head decides whether the primary survives as the leading
particle; that particle's class is pinned (training: never noised; generation: pre-filled)."""
from __future__ import annotations

import pytest
import torch

from legofmt.data.struct import DataStruct, _F, set_layout
from legofmt.log_metrics.val_metrics import ShowerValMetrics
from legofmt.main.generate import GenerateOut
from legofmt.main.modules import LEGOLtng
from test_modules_direct import _tiny_config
from test_passthrough_gate import _pt_batch


def _cfg() -> dict:
    cfg = _tiny_config()
    cfg["config"]["model_conf"]["survive_head"] = True
    cfg["config"]["model_conf"]["count_head"] = True     # survive_head needs lead_slot, lead_slot needs count_head
    cfg["config"]["model_conf"]["lead_slot"] = True
    return cfg


def _charged_batch(B: int = 8) -> DataStruct:
    ds = _pt_batch(B, passthrough=False)        # rows: 0 density, 1 edep, 2 incoming, 3 outgoing, 4 pad
    f = ds.f.full.clone()
    f[:, 2, 7] = 211.0                           # pi+ primary
    f[:, 3, 7] = 211.0                           # its only outgoing particle: the leading one
    return DataStruct(f, ds.m.full, ds.am.full)


def test_survival_target_pins_the_leading_class_and_trains_the_head() -> None:
    torch.manual_seed(0)
    m = LEGOLtng(_cfg()); m.on_fit_start(); m.train()
    seen = {}
    def hook(mod, args, kwargs, out):
        if kwargs.get("return_species") and "s_t" not in seen:
            seen["s_t"] = kwargs["pdgids"].clone()
    h = m.model.register_forward_hook(hook, with_kwargs=True)
    for _ in range(5):                            # pinned regardless of the sampled noise
        seen.clear()
        loss = m._step(_charged_batch(), 0)
        idx_pi = int(m.convert_pdgids(torch.tensor([211.0])))
        assert (seen["s_t"][:, 3] == idx_pi).all()
    h.remove()
    loss.backward()
    g = sum(p.grad.abs().sum() for p in m.model.vf.sv_head.parameters() if p.grad is not None)
    assert float(g) > 0


def test_generation_pins_a_surviving_primary(tmp_path) -> None:
    torch.manual_seed(0)
    cfg = _cfg()["config"]
    a = LEGOLtng({"state_dict": {}, "config": cfg})
    torch.save({"state_dict": a.model.vf.state_dict(), "config": cfg}, tmp_path / "f.pt")
    gen = GenerateOut(str(tmp_path / "f.pt"), device="cpu")
    set_layout(gen.cond_names)
    cond = torch.zeros(16, gen.n_cond + 7)
    cond[:, 0] = 1.0
    cond[:, gen.n_cond:gen.n_cond + 3] = torch.tensor([0.0, 0.0, 150.0])
    cond[:, gen.n_cond + 3:gen.n_cond + 6] = torch.tensor([0.0, 0.0, -1.0])
    cond[:, -1] = 211
    for bias, expect in ((50.0, True), (-50.0, None)):
        with torch.no_grad():
            gen.model.model.vf.sv_head[-1].bias.fill_(bias)
            gen.model.model.vf.sv_head[-1].weight.zero_()
        sols, _, attn = gen(cond)
        out, occ = _F(sols).out_p, attn[:, gen.n_prefix + 1:]
        has_pi = ((out[..., 7] == 211) & occ).any(-1)
        if expect:
            assert has_pi.all(), "a fired survival head must leave the primary's species in every event"


def test_validation_pins_like_generation() -> None:
    torch.manual_seed(0)
    m = LEGOLtng(_cfg()).eval()
    with torch.no_grad():
        m.model.vf.sv_head[-1].bias.fill_(50.0)
        m.model.vf.sv_head[-1].weight.zero_()
        m.pt_head[-1].bias.fill_(-50.0); m.pt_head[-1].weight.zero_()
    gen, active, pdg = ShowerValMetrics._generate(m, _charged_batch())
    assert ((pdg == 211) & active).any(-1).all()


def test_species_noise_never_shows_the_primary_class() -> None:
    m = LEGOLtng(_cfg())
    prim = torch.tensor([3, 5, 1, 7])
    noise = m._species_noise(torch.zeros(4, 5000, dtype=torch.long), prim)
    K = m.rc.model_args["npdgids"]
    assert (noise != prim[:, None]).all() and noise.min() >= 1 and noise.max() < K   # count mode: 0 is never drawn
    assert noise.unique().numel() == K - 1           # every other particle class still drawn
