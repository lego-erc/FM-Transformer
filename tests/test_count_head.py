"""model_conf.count_head: a per-event head samples the multiplicity; exactly that many slots are generated,
and the species channel only ever chooses among particle classes."""
from __future__ import annotations

import torch

from legofmt.data.struct import DataStruct, _F, set_layout
from legofmt.log_metrics.val_metrics import ShowerValMetrics
from legofmt.main.generate import GenerateOut
from legofmt.main.modules import LEGOLtng
from test_modules_direct import _tiny_config
from test_passthrough_gate import _pt_batch


def _cfg(survive: bool = False, max_seq_l: int | None = None) -> dict:
    cfg = _tiny_config()
    if max_seq_l is not None:
        cfg["config"]["model_conf"]["model_args"]["max_seq_l"] = max_seq_l
    cfg["config"]["model_conf"]["count_head"] = True
    cfg["config"]["model_conf"]["survive_head"] = survive
    if survive:
        cfg["config"]["model_conf"]["lead_slot"] = True  # the resolver refuses a pinned survivor without the leading base draw
    return cfg


def _force_count(model, n: int) -> None:
    with torch.no_grad():
        last = model.model.vf.ct_head[-1]
        last.weight.zero_()
        last.bias.fill_(-50.0)
        last.bias[n] = 50.0


def test_training_generates_only_the_real_particles() -> None:
    torch.manual_seed(0)
    m = LEGOLtng(_cfg()); m.on_fit_start(); m.train()
    seen = {}
    def hook(mod, args, kwargs, out):
        if kwargs.get("return_species") and "mask" not in seen:
            seen["mask"], seen["s_t"] = kwargs["mask"].clone(), kwargs["pdgids"].clone()
    h = m.model.register_forward_hook(hook, with_kwargs=True)
    loss = m._step(_pt_batch(passthrough=False), 0)   # row 3 a particle, row 4 a pad
    h.remove()
    assert (seen["mask"][:, 3] == 1).all() and (seen["mask"][:, 4] == 0).all()
    assert (seen["s_t"][:, 3] != 0).all()
    loss.backward()
    g = sum(p.grad.abs().sum() for p in m.model.vf.ct_head.parameters() if p.grad is not None)
    assert float(g) > 0


def test_generation_emits_exactly_the_sampled_count(tmp_path) -> None:
    torch.manual_seed(0)
    cfg = _cfg()["config"]
    a = LEGOLtng({"state_dict": {}, "config": cfg})
    torch.save({"state_dict": a.model.vf.state_dict(), "config": cfg}, tmp_path / "f.pt")
    gen = GenerateOut(str(tmp_path / "f.pt"), device="cpu")
    with torch.no_grad():
        gen.model.pt_head[-1].bias.fill_(-50.0); gen.model.pt_head[-1].weight.zero_()
    set_layout(gen.cond_names)
    cond = torch.zeros(12, gen.n_cond + 7)
    cond[:, 0] = 1.0
    cond[:, gen.n_cond:gen.n_cond + 3] = torch.tensor([0.0, 0.0, 150.0])
    cond[:, gen.n_cond + 3:gen.n_cond + 6] = torch.tensor([0.0, 0.0, -1.0])
    cond[:, -1] = 211
    for n in (0, 1, 2):
        if n >= gen.max_seq_l - gen.n_prefix - 1 + 1:
            continue
        _force_count(gen.model, n)
        sols, _, attn = gen(cond)
        occ = attn[:, gen.n_prefix + 1:]
        assert occ.sum(-1).eq(n).all(), (n, occ.sum(-1))
        assert (_F(sols).out_p[..., 7][occ] != 0).all()


def test_validation_generates_the_sampled_count() -> None:
    torch.manual_seed(0)
    m = LEGOLtng(_cfg()).eval()
    with torch.no_grad():
        m.pt_head[-1].bias.fill_(-50.0); m.pt_head[-1].weight.zero_()
    _force_count(m, 1)
    gen, active, pdg = ShowerValMetrics._generate(m, _pt_batch(passthrough=False))
    assert active.sum(-1).eq(1).all() and (pdg[active] != 0).all()


def test_count_mode_noise_never_shows_empty_or_the_primary() -> None:
    m = LEGOLtng(_cfg(survive=True))
    prim = torch.tensor([1, 2, 3, 1])
    noise = m._species_noise(torch.zeros(4, 3000, dtype=torch.long), prim)
    assert (noise != 0).all() and (noise != prim[:, None]).all()


def test_count_head_sees_the_survival_target() -> None:
    """n given survival differs by ~3x; the head must be told the survival outcome it is sampled after."""
    for survive, want in ((True, 1.0), (False, 0.0)):   # _pt_batch: gamma in, gamma leads out
        torch.manual_seed(0)
        m = LEGOLtng(_cfg(survive)); m.on_fit_start(); m.train()
        seen = []
        h = m.model.vf.ct_head.register_forward_hook(lambda mod, a, out: seen.append(a[0][:, -1].clone()))
        m._step(_pt_batch(passthrough=False), 0)
        h.remove()
        assert torch.equal(seen[0], torch.full_like(seen[0], want))


def test_out_of_vocabulary_particles_are_not_count_mode_particles() -> None:
    """An ion (class 0) cannot be generated; as a training row its target would be the masked class (CE = inf)."""
    torch.manual_seed(0)
    m = LEGOLtng(_cfg(survive=True)); m.on_fit_start(); m.train()
    ds = _pt_batch(passthrough=False)
    f, am, msk = ds.f.full.clone(), ds.am.full.clone(), ds.m.full.clone()
    f[:, 4] = f[:, 3]; f[:, 4, 7] = 1000010048.0
    am[:, 4] = True; msk[:, 4] = 1
    seen = []
    h = m.model.vf.ct_head.register_forward_hook(lambda mod, a, out: seen.append(out))
    loss = m._step(DataStruct(f, msk, am), 0)
    h.remove()
    assert torch.isfinite(loss)
    assert int(m.convert_pdgids(torch.tensor([1000010048.0]))) == 0


def test_closed_slots_do_not_change_the_field() -> None:
    """Count mode solves each event at the length of its opened slots; that is exact only if closed (masked,
    unattended) trailing rows have no effect on the open ones."""
    torch.manual_seed(0)
    m = LEGOLtng(_cfg(survive=True, max_seq_l=8)).eval()
    ds = _pt_batch(B=4, L=8, passthrough=False)
    f, am, msk = ds.f.full.clone(), ds.am.full.clone(), ds.m.full.clone()
    am[:, 4:] = False; msk[:, 4:] = 0
    x = torch.nan_to_num(f[..., :7], 0.5)
    sp = torch.randint(1, 4, am.shape)
    t = torch.full((4, 1), 0.4)
    kw = lambda L: dict(mask=msk[:, :L], attn_mask=am[:, :L], types=m.types_embd, pdgids=sp[:, :L], return_species=True)
    with torch.no_grad():
        v_full, lg_full = m.model(x, t, **kw(8))
        v_cut, lg_cut = m.model(x[:, :5], t, **kw(5))
    assert torch.allclose(v_full[:, :5], v_cut, atol=1e-5) and torch.allclose(lg_full[:, :5], lg_cut, atol=1e-5)


def test_bucketed_generation_opens_each_events_count(tmp_path, monkeypatch) -> None:
    torch.manual_seed(0)
    cfg = _cfg(survive=True, max_seq_l=23)["config"]     # 20 outgoing slots: every bucket (2, 4, 8, 16, 20) is used
    a = LEGOLtng({"state_dict": {}, "config": cfg})
    torch.save({"state_dict": a.model.vf.state_dict(), "config": cfg}, tmp_path / "f.pt")
    gen = GenerateOut(str(tmp_path / "f.pt"), device="cpu")
    with torch.no_grad():
        gen.model.pt_head[-1].bias.fill_(-50.0); gen.model.pt_head[-1].weight.zero_()
    set_layout(gen.cond_names)
    nmax = gen.max_seq_l - gen.n_prefix - 1
    want = torch.tensor([0, 1, 2, 3, 5, 9, 17, nmax])
    monkeypatch.setattr(gen.model, "sample_count", lambda ds, sv: torch.maximum(want, sv.long()))
    cond = torch.zeros(8, gen.n_cond + 7)
    cond[:, 0] = 1.0
    cond[:, gen.n_cond:gen.n_cond + 3] = torch.tensor([0.0, 0.0, 150.0])
    cond[:, gen.n_cond + 3:gen.n_cond + 6] = torch.tensor([0.0, 0.0, -1.0])
    cond[:, -1] = 211
    sols, _, attn = gen(cond)
    occ = attn[:, gen.n_prefix + 1:]
    sv = _F(sols).out_p[..., 7][:, 0] == 211
    assert torch.equal(occ.sum(-1), torch.maximum(want, (occ.sum(-1) > 0) & sv))
    assert (_F(sols).out_p[..., 7][occ] != 0).all() and not torch.isnan(_F(sols).out_p[..., :7][occ]).any()


def test_lead_slot_moves_the_leader_to_its_own_typed_slot() -> None:
    """lead_slot: slot n0 has a type of its own and holds the most energetic particle in training; here the
    leader is the surviving primary (gamma), so the survival pin must land on n0, not on its data slot."""
    cfg = _cfg(survive=True)
    cfg["config"]["model_conf"]["lead_slot"] = True
    torch.manual_seed(0)
    m = LEGOLtng(cfg); m.on_fit_start(); m.train()
    n0 = m.rc.n_prefix + 1
    assert m.types_embd[0, n0] == m.rc.n_prefix + 1 and (m.types_embd[0, n0 + 1:] == m.rc.n_prefix + 2).all()
    ds = _pt_batch(passthrough=False)
    f, am, msk = ds.f.full.clone(), ds.am.full.clone(), ds.m.full.clone()
    f[:, n0 + 1] = f[:, n0]; f[:, n0 + 1, 0] = 0.1           # gamma, the most energetic (scalar 0 = e_in)
    f[:, n0, 7] = 2212.0; f[:, n0, 0] = 0.7                  # a softer proton in the first data slot
    am[:, n0 + 1] = True; msk[:, n0 + 1] = 1
    seen = {}
    def hook(mod, args, kwargs, out):
        if kwargs.get("return_species") and "pdgids" not in seen:
            seen["pdgids"] = kwargs["pdgids"].clone()
    h = m.model.register_forward_hook(hook, with_kwargs=True)
    torch.manual_seed(1)
    m._step(DataStruct(f, msk, am), 0)
    h.remove()
    g = int(m.convert_pdgids(torch.tensor([22.0])))
    assert (seen["pdgids"][:, n0] == g).all()               # pinned at all t: the leader sits in n0


def test_species_head_mlp_and_settle_temperature() -> None:
    cfg = _cfg(survive=True)
    cfg["config"]["model_conf"]["model_args"].update(sp_layers=2, sp_dim=32)
    torch.manual_seed(0)
    m = LEGOLtng(cfg).eval()
    assert len(m.model.vf.species_head) == 5 and m.model.vf.species_head[-1].out_features == m.rc.model_args["npdgids"]
    lg = torch.zeros(2, 3, m.rc.model_args["npdgids"]); lg[..., 1] = 2.0; lg[..., 2] = 1.9   # two near-tied classes
    assert (m._settle(lg, torch.zeros(2, 3, dtype=torch.long)) == 1).all()
    m.rc.odeint_conf["settle_temp"] = 1.0
    torch.manual_seed(0)
    draws = torch.stack([m._settle(lg[:1].expand(200, 3, -1), torch.zeros(200, 3, dtype=torch.long)) for _ in range(3)])
    assert (draws == 2).any() and (draws != 0).all()          # samples reach the runner-up, never the empty class


def test_species_tau_finalises_the_classes_early() -> None:
    """species_tau = 0.5: no label noise for t >= 0.5 in training; at generation every candidate jumps at the step
    that crosses tau (rate 1) and none afterwards (rate 0)."""
    cfg = _cfg(survive=True)
    cfg["config"]["model_conf"]["species_tau"] = 0.5
    torch.manual_seed(0)
    m = LEGOLtng(cfg).eval()
    B, L, K = 64, 4, m.rc.model_args["npdgids"]
    sp = torch.ones(B, L, dtype=torch.long); sp_gen = torch.ones(B, L, dtype=torch.bool)
    lg = torch.zeros(B, L, K); lg[..., 2] = 50.0
    out = m._species_jump(sp, lg, sp_gen, torch.tensor(0.25), torch.tensor(0.25))   # crosses tau: everyone jumps
    assert (out == 2).all()
    out = m._species_jump(sp, lg, sp_gen, torch.tensor(0.5), torch.tensor(0.25))    # past tau: nobody jumps
    assert (out == 1).all()
    seen = []
    m.train(); m.on_fit_start()
    h = m.model.register_forward_hook(lambda mod, a, kw, o: seen.append(kw["pdgids"].clone()) if kw.get("return_species") else None, with_kwargs=True)
    m._sample_t = lambda ds: torch.full((ds.f.full.shape[0],), 0.75)
    m._step(_pt_batch(passthrough=False), 0)
    h.remove()
    tgt = m.convert_pdgids(_pt_batch(passthrough=False).f.pdgids).squeeze(-1)
    assert torch.equal(seen[0][:, 3], tgt[:, 3])                                     # t = 0.75 >= tau: clean label


def test_species_mask_never_shows_a_wrong_class() -> None:
    """species_mask: corrupted training labels are 0 (unknown), never another class; at generation only masked
    slots jump, once, and an unmasked slot keeps its class through the settle."""
    cfg = _cfg(survive=True)
    cfg["config"]["model_conf"]["species_mask"] = True
    torch.manual_seed(0)
    m = LEGOLtng(cfg).eval()
    assert (m._species_noise(torch.ones(4, 50, dtype=torch.long), torch.tensor([1, 2, 3, 1])) == 0).all()
    K = m.rc.model_args["npdgids"]
    sp = torch.tensor([[0, 2, 0, 3]]).expand(256, 4).clone(); sp_gen = torch.ones(256, 4, dtype=torch.bool)
    lg = torch.zeros(256, 4, K); lg[..., 1] = 50.0
    out = m._species_jump(sp, lg, sp_gen, torch.tensor(0.5), torch.tensor(0.5))      # rate 1: masked -> class 1
    assert (out[:, 0] == 1).all() and (out[:, 2] == 1).all() and (out[:, 1] == 2).all() and (out[:, 3] == 3).all()
    assert torch.equal(m._settle(lg, sp), torch.tensor([[1, 2, 1, 3]]).expand(256, 4))
    seen = []
    m.train(); m.on_fit_start()
    h = m.model.register_forward_hook(lambda mod, a, kw, o: seen.append(kw["pdgids"].clone()) if kw.get("return_species") else None, with_kwargs=True)
    m._step(_pt_batch(passthrough=False), 0)
    h.remove()
    tgt = m.convert_pdgids(_pt_batch(passthrough=False).f.pdgids).squeeze(-1)
    assert ((seen[0][:, 3] == 0) | (seen[0][:, 3] == tgt[:, 3])).all()               # unknown or the truth


def test_lead_slot_base_puts_the_hardest_draw_in_slot_n0() -> None:
    cfg = _cfg(survive=True)
    cfg["config"]["model_conf"]["lead_slot"] = True
    torch.manual_seed(0)
    m = LEGOLtng(cfg).eval()
    ds = _pt_batch(B=16, L=8, passthrough=False)
    am, msk = ds.am.full.clone(), ds.m.full.clone()
    am[:, 4:7] = True; msk[:, 4:7] = 1; am[:, 7] = False; msk[:, 7] = 0
    n0 = m.rc.n_prefix + 1
    base = m.gen_base_wrapper(DataStruct(ds.f.full, msk, am))
    e = base[:, n0:, 0].masked_fill(msk[:, n0:] != 1, float("inf"))
    assert torch.equal(e.argmin(-1), torch.zeros(16, dtype=torch.long))
    m.train()
    cfg2 = _cfg(); torch.manual_seed(0)              # count_head alone: no lead slot, no base rule
    assert not LEGOLtng(cfg2).eval().rc.lead_slot


def test_lead_slot_base_rule_applies_in_training_too() -> None:
    cfg = _cfg(survive=True)
    cfg["config"]["model_conf"]["lead_slot"] = True
    torch.manual_seed(0)
    m = LEGOLtng(cfg); m.on_fit_start(); m.train()
    ds = _pt_batch(B=16, L=8, passthrough=False)
    am, msk = ds.am.full.clone(), ds.m.full.clone()
    am[:, 4:7] = True; msk[:, 4:7] = 1; am[:, 7] = False; msk[:, 7] = 0
    n0 = m.rc.n_prefix + 1
    base = m.gen_base_wrapper(DataStruct(ds.f.full, msk, am))
    e = base[:, n0:, 0].masked_fill(msk[:, n0:] != 1, float("inf"))
    assert torch.equal(e.argmin(-1), torch.zeros(16, dtype=torch.long))
