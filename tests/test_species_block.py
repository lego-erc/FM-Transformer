"""model_conf.species_flow: the species as a continuous block inside the flowed state (one coordinate per
real class, vertices on the sphere of radius 2: fixed one-hot by default, learned with species_learn_vertices),
transported by the field and decoded at t = 1 as the nearest vertex. The block's velocity is parametrised
through the existing species head's posterior."""
from __future__ import annotations

import pytest
import torch

from legofmt.base_dist import base_nn
from legofmt.cfm.cfm_trafo_x import SPECIES_R
from legofmt.data.struct import DataStruct
from legofmt.main.modules import LEGOLtng
from test_count_head import _cfg as _count_cfg
from test_ot_same_pdgid import _BruteLap
from test_passthrough_gate import _pt_batch


def _cfg(**model_conf) -> dict:
    """count_s6-shaped tiny config: count head, survival pin, lead slot, MLP species head, plus species_flow."""
    cfg = _count_cfg(survive=True)
    cfg["config"]["model_conf"].update({"species_flow": True, **model_conf})
    cfg["config"]["model_conf"]["model_args"]["sp_layers"] = 1
    return cfg


def _train_cfg() -> dict:
    """count_s6's loss machinery on the tiny model: Kendall cells, E_dep cell, flow-map ladder."""
    cfg = _cfg(learned_loss_weights=True, edep_cell=True, one_step_euler_fac=1.0, one_step_euler_sections=2)
    cfg["config"]["model_conf"]["model_args"]["step_cond"] = True
    return cfg


def _two_particle_batch(B: int = 4) -> DataStruct:
    """Rows: 0 density, 1 edep, 2 incoming (photon), 3 outgoing photon (the leader: smallest energy scalar),
    4 outgoing pi+. With lead_slot the leader sits in slot n0 = 3 and is the survivor (primary's species)."""
    ds = _pt_batch(B, passthrough=False)
    f  = ds.f.full.clone()
    f[:, 4, 0] = 0.6
    f[:, 4, 1:4] = torch.tensor([0.0, 1.0, 0.0])
    f[:, 4, 4:7] = torch.tensor([0.0, -1.0, 0.0])
    f[:, 4, 7] = 211.0
    f[:, 3, 0] = 0.2
    am = torch.ones(B, 5, dtype=torch.bool)
    m  = am.clone().long()
    m[:, :3] = 0
    return DataStruct(f, m, am)


def _gen_batch(B: int = 4) -> DataStruct:
    """A generation batch: slot 3 pre-filled with the pinned photon, slot 4 open with an unknown species."""
    ds = _two_particle_batch(B)
    f  = ds.f.full.clone()
    f[:, 4, 7] = 0.0
    return DataStruct(f, ds.m.full, ds.am.full)


def _force_species(m, k: int) -> None:
    with torch.no_grad():
        last = m.model.vf.species_head[-1]
        last.weight.zero_(); last.bias.fill_(-50.0); last.bias[k] = 50.0


def _net_call(m, ds: DataStruct, pdgids: torch.Tensor, t: float, z: torch.Tensor, return_species: bool = False):
    """One ProjectModel call on the batch's kinematics with the species block ``z`` appended."""
    x  = torch.cat((ds.f.model_in.nan_to_num(0.0), z), dim=-1)
    tt = torch.full((x.shape[0], 1), t)
    return m.model(x, tt, mask=ds.m.full, attn_mask=ds.am.full, types=m.types_embd,
                   pdgids=pdgids, return_species=return_species)


def test_species_flow_extends_only_the_models_manifold() -> None:
    m = LEGOLtng(_cfg())
    d = m.rc.model_args["npdgids"] - 1
    assert m.rc.species_flow and m.rc.model_args["sp_block"] == d
    assert tuple(m.rc.manifold.ambient_dims) == (1, 3, 3)            # OT coupling / DataPrep split 7 columns by it
    assert tuple(m.model.manifold.ambient_dims) == (1, 3, 3, d)
    assert tuple(m.ps.ambient_dims) == (1, 3, 3, d)


def test_species_flow_needs_count_head_and_refuses_reflow() -> None:
    cfg = _count_cfg()
    cfg["config"]["model_conf"].update({"species_flow": True, "count_head": False})
    with pytest.raises(ValueError, match="species_flow"):
        LEGOLtng(cfg)
    with pytest.raises(ValueError, match="reflow"):
        LEGOLtng(_cfg(reflow_start_epoch=1))


def test_turning_the_flag_off_on_a_saved_config_restores_the_ctmc_model() -> None:
    cfg = LEGOLtng(_cfg()).rc.config      # what a checkpoint saves: the resolved config, sp_block included
    cfg["model_conf"]["species_flow"] = False
    m = LEGOLtng({"state_dict": {}, "config": cfg}); m.eval()
    assert m.model.vf.sp_block == 0 and "sp_block" not in m.rc.model_args
    out, sp = m.solve(_gen_batch(), step_size=0.25, method="euler", return_species=True)
    assert out.shape[-1] == 7 and (sp[:, 4] >= 1).all()


def test_flag_off_training_step_is_unchanged() -> None:
    """Golden values recorded at 76861f9, before species_flow existed: a count_s6-shaped flag-off model's seeded
    _step loss, its grad norm and the next RNG draw. The flag-off path must stay byte-identical."""
    cfg = _train_cfg()
    cfg["config"]["model_conf"].update({"species_flow": False, "species_mask": True})
    torch.manual_seed(1234)
    m = LEGOLtng(cfg); m.on_fit_start(); m.train()
    torch.manual_seed(7)
    loss = m._step(_two_particle_batch(), 0)
    loss.backward()
    g = sum(float(p.grad.norm()) for p in m.model.vf.parameters() if p.grad is not None)
    assert float(loss) == pytest.approx(10.5253219604, rel=1e-5)
    assert g == pytest.approx(54.0328213694, rel=1e-5)
    assert float(torch.rand(1)) == pytest.approx(0.4012463689, abs=1e-9), "the flag-off step drew extra randoms"


def test_count_s6_shaped_weights_warm_start_exactly() -> None:
    cfg = _cfg(); cfg["config"]["model_conf"]["species_flow"] = False
    off = LEGOLtng(cfg)
    on  = LEGOLtng(_cfg())
    with torch.no_grad():
        on.model.vf.species_vertices.normal_()
    on.model.vf.load_state_dict(off.model.vf.state_dict(), strict=True)   # no vertices saved: start at r I
    d = on.rc.model_args["sp_block"]
    assert torch.equal(on.model.vf.vertices(), SPECIES_R * torch.eye(d))


def test_learned_vertices_move_only_through_their_own_loss() -> None:
    """Through the network (soft label, block velocity) the vertices collapse (toy: min distance 2.83 -> 0.05),
    so the network reads them detached; _step's block-posterior loss is their only gradient."""
    torch.manual_seed(0)
    cfg = _train_cfg(); cfg["config"]["model_conf"]["species_learn_vertices"] = True
    m  = LEGOLtng(cfg); m.on_fit_start(); m.train()
    logs = {}
    m.log_dict = lambda d, **kw: logs.update(d)
    ds = _two_particle_batch()
    m._step(ds, 0).backward()
    e = m.model.vf.species_vertices
    assert e.grad is not None and float(e.grad.abs().sum()) > 0
    assert "loss/vertices" in logs and float(logs["species/min_vertex_dist"]) == pytest.approx(SPECIES_R * 2 ** 0.5)
    e.grad = None
    d  = m.rc.model_args["sp_block"]
    pd = m.convert_pdgids(ds.f.pdgids); pd[:, 3:, 0] = -1
    v, lg = _net_call(m, ds, pd, 0.5, torch.randn(4, 5, d), return_species=True)
    (v.sum() + lg.sum()).backward()
    assert e.grad is None


def test_the_vertex_loss_separates_two_close_classes_that_both_occur() -> None:
    torch.manual_seed(0)
    m = LEGOLtng(_cfg(learned_loss_weights=True, species_learn_vertices=True)); m.on_fit_start(); m.train()
    m.log_dict = lambda *a, **k: None
    d = m.rc.model_args["sp_block"]
    g, pi = (int(m.convert_pdgids(torch.tensor([p]))) - 1 for p in (22.0, 211.0))
    with torch.no_grad():                                # pi+ almost on top of the photon
        e = m.model.vf.species_vertices.view(d, d)
        e[pi] = e[g] + 0.05 * e[pi]
    def dist() -> float:
        C = m.model.vf.vertices().detach()
        return float((C[g] - C[pi]).norm())
    d0  = dist()
    opt = torch.optim.Adam([m.model.vf.species_vertices], lr=1e-2)
    for _ in range(20):
        opt.zero_grad(); m._step(_two_particle_batch(16), 0).backward(); opt.step()
    assert dist() > 2 * d0


def test_learned_vertices_get_adamw() -> None:
    cfg = _train_cfg()
    cfg["config"]["opt_conf"] = {"opt": "muon", "lr": 1e-3, "adamw_lr": 1e-3, "weight_decay": 0.007, "adamw_wd": 0.01}
    m = LEGOLtng(cfg)
    e = m.model.vf.species_vertices
    assert any(not g["use_muon"] and any(p is e for p in g["params"]) for g in m.opt.param_groups)


def test_decode_follows_the_learned_vertices() -> None:
    torch.manual_seed(0)
    m = LEGOLtng(_cfg()); m.eval()
    d = m.rc.model_args["sp_block"]
    with torch.no_grad():                                # class k's vertex is r e_(k-2): the largest coordinate
        m.model.vf.species_vertices.copy_(torch.eye(d).roll(1, dims=0).flatten())   # would decode k - 1
    k = 3
    _force_species(m, k)
    ds = _gen_batch()
    _, sp = m.solve(ds, x_init=m.gen_base_wrapper(ds), step_size=0.25, method="euler", return_species=True)
    assert (sp[:, 4] == k).all()


def test_fixed_vertices_are_the_default() -> None:
    torch.manual_seed(0)
    m = LEGOLtng(_cfg(learned_loss_weights=True)); m.on_fit_start(); m.train()
    e = m.model.vf.species_vertices
    d = m.rc.model_args["sp_block"]
    assert not e.requires_grad and torch.equal(m.model.vf.vertices(), SPECIES_R * torch.eye(d))
    m._step(_two_particle_batch(), 0).backward()
    assert e.grad is None


def test_soft_label_is_class_zero_at_t0_and_the_nearest_vertex_near_t1() -> None:
    torch.manual_seed(0)
    m  = LEGOLtng(_cfg()); m.eval()
    ds = _two_particle_batch()
    B, L = ds.m.full.shape
    d  = m.rc.model_args["sp_block"]
    k  = 2                                                     # class index 2 -> vertex row 1
    with torch.no_grad():
        m.model.vf.species_vertices.normal_()                  # learned vertices: any layout
    C  = m.model.vf.vertices()
    lab = m.model.vf._soft_label(C[k - 1].expand(B, L, d), torch.full((B, L), 1.0 - 1e-3), C)
    assert (lab.argmax(-1) == k).all() and (lab[..., k] > 0.99).all()
    # through the model: a sentinel row at t = 0 is conditioned exactly like a class-0 row
    z    = torch.randn(B, L, d)
    hard = m.convert_pdgids(ds.f.pdgids); hard[:, 3:, 0] = 0
    soft = hard.clone();                  soft[:, 3:, 0] = -1
    assert torch.equal(_net_call(m, ds, hard, 0.0, z)[..., :7], _net_call(m, ds, soft, 0.0, z)[..., :7])


def test_block_velocity_is_the_posterior_mean_velocity() -> None:
    torch.manual_seed(0)
    m  = LEGOLtng(_cfg()); m.eval()
    ds = _two_particle_batch()
    B, L = ds.m.full.shape
    d  = m.rc.model_args["sp_block"]
    z  = torch.randn(B, L, d)
    pd = m.convert_pdgids(ds.f.pdgids); pd[:, 3:, 0] = -1
    t  = 0.4
    v, lg = _net_call(m, ds, pd, t, z, return_species=True)
    assert v.shape == (B, L, 7 + d) and lg.shape == (B, L, d + 1)
    want = (torch.softmax(lg[..., 1:].float(), -1) @ m.model.vf.vertices() - z) / (1.0 - t)
    gen  = (pd[..., 0] < 0).unsqueeze(-1).expand_as(want)
    assert torch.allclose(v[..., 7:][gen], want[gen], atol=1e-5)
    assert torch.equal(v[..., 7:][~gen], torch.zeros_like(v[..., 7:][~gen]))
    pinned = pd.clone(); pinned[:, 3, 0] = 1                   # a pinned row has a hard label and no block velocity
    assert torch.equal(_net_call(m, ds, pinned, t, z)[:, 3, 7:], torch.zeros(B, d))


def test_base_draw_carries_a_block_on_generated_rows_only() -> None:
    torch.manual_seed(0)
    lap, orig = _BruteLap(), base_nn.slap
    base_nn.slap = lap
    try:
        m = LEGOLtng(_cfg(ot_coupling=True)); m.on_fit_start(); m.train()
        ds = _two_particle_batch()
        base = m.gen_base_wrapper(ds)
    finally:
        base_nn.slap = orig
    d = m.rc.model_args["sp_block"]
    assert base.shape == (*ds.m.full.shape, 7 + d)
    assert lap.cost is not None and lap.cost.shape[-1] == 2        # OT coupling ran, on the 7 kinematic columns
    blk = base[..., 7:]
    assert torch.equal(blk[:, :3], torch.zeros_like(blk[:, :3]))   # conditioning rows and the E_dep row
    assert (blk[:, 3:].abs().sum(-1) > 0).all()                     # every generated outgoing row
    assert not torch.isnan(base).any()


def test_step_feeds_the_sentinel_and_logs_the_block_terms() -> None:
    torch.manual_seed(0)
    m = LEGOLtng(_train_cfg()); m.on_fit_start(); m.train()
    logs, seen = {}, []
    m.log_dict = lambda d, **kw: logs.update(d)
    h = m.model.register_forward_hook(
        lambda mod, a, kw, out: seen.append((a[0].detach().clone(), kw["pdgids"].clone())), with_kwargs=True)
    loss = m._step(_two_particle_batch(), 0)
    h.remove()
    assert torch.isfinite(loss)
    d = m.rc.model_args["sp_block"]
    k = int(m.convert_pdgids(torch.tensor([22.0])))
    for x, pd in seen:                                   # the plain step and every ladder call
        assert x.shape[-1] == 7 + d
        assert (pd[:, 3] == k).all(), "the pinned survivor keeps its class"
        assert (pd[:, 4] == -1).all(), "a generated row carries the sentinel"
        assert (pd[:, 1] == 0).all(), "the E_dep row keeps class 0"
        assert torch.equal(x[:, 1, 7:], torch.zeros(x.shape[0], d))
    assert {"loss/species", "loss/one_step_euler", "loss/one_step_euler_sp"} <= set(logs)
    assert "loss/vertices" not in logs                   # fixed vertices (the default) have no loss


def test_ladder_kinematic_term_stays_a_seven_channel_mean() -> None:
    """A stub field: teacher calls return 0 (target 0), the trained call 0.5 on the kinematics and 2 on the block,
    so each term is exactly its own channels' mean; the block must not dilute the kinematic term."""
    from legofmt.distill.distill import one_step_euler_loss
    torch.manual_seed(0)
    m  = LEGOLtng(_train_cfg()); m.on_fit_start(); m.train()
    ds = _two_particle_batch()
    nb = m.rc.model_args["sp_block"]
    s1 = m.convert_pdgids(ds.f.pdgids).squeeze(-1)
    ext = DataStruct(torch.cat((ds.f.model_in, torch.zeros(*s1.shape, nb), ds.f.pdgids), -1), ds.m.full, ds.am.full)
    def stub(x, t, d=None, **kw):
        if not torch.is_grad_enabled():
            return torch.zeros_like(x)
        v = torch.full_like(x, 0.5)
        v[..., 7:] = 2.0
        return v.requires_grad_()
    m.model.forward = stub
    kin, sp = one_step_euler_loss(m, m.gen_base_wrapper(ds), ext, s1)
    assert float(kin) == pytest.approx(0.25) and float(sp) == pytest.approx(4.0)


@pytest.mark.parametrize("method", ["euler", "midpoint", "rk4"])
def test_solve_generates_the_forced_class_and_strips_the_block(method: str) -> None:
    torch.manual_seed(0)
    cfg = _cfg(); cfg["config"]["model_conf"]["model_args"]["max_seq_l"] = 6
    m  = LEGOLtng(cfg); m.eval()
    k  = m.rc.model_args["npdgids"] - 1
    _force_species(m, k)
    ds = _gen_batch()
    B  = ds.m.full.shape[0]                              # + a NaN pad slot, as the val loader pads
    ds = DataStruct(torch.cat((ds.f.full, torch.full((B, 1, 8), float("nan"))), 1),
                    torch.cat((ds.m.full, torch.zeros(B, 1, dtype=torch.long)), 1),
                    torch.cat((ds.am.full, torch.zeros(B, 1, dtype=torch.bool)), 1))
    k_pin = int(m.convert_pdgids(torch.tensor([22.0])))
    seen = []
    h = m.model.register_forward_hook(lambda mod, a, kw, out: seen.append(kw["pdgids"].clone()), with_kwargs=True)
    base = m.gen_base_wrapper(ds)
    rng  = torch.get_rng_state()
    out, sp = m.solve(ds, x_init=base, step_size=0.25, method=method, return_species=True)
    h.remove()
    assert torch.equal(torch.get_rng_state(), rng), "the base draw is the only randomness"
    assert out.shape[-1] == 7 and not torch.isnan(out[:, 3:5]).any()
    assert (sp[:, 3] == k_pin).all() and (sp[:, 4] == k).all()
    assert all(torch.equal(p, seen[0]) for p in seen) and (seen[0][:, 4] == -1).all() and (seen[0][:, 3] == k_pin).all()


def test_a_seven_column_x_init_raises() -> None:
    m  = LEGOLtng(_cfg()); m.eval()
    ds = _gen_batch()
    with pytest.raises(RuntimeError):
        m.solve(ds, x_init=ds.f.model_in.nan_to_num(0.0), step_size=0.25, method="euler")


def test_intermediates_are_stripped_and_decoded_from_the_final_state() -> None:
    torch.manual_seed(0)
    m  = LEGOLtng(_cfg()); m.eval()
    ds = _gen_batch(64)
    base = m.gen_base_wrapper(ds)
    out, sp = m.solve(ds, x_init=base, step_size=0.5, method="euler",
                      return_intermediates=True, return_species=True)
    last, sp_last = m.solve(ds, x_init=base, step_size=0.5, method="euler", return_species=True)
    assert out.shape == (3, *ds.m.full.shape, 7)
    assert torch.equal(out[-1], last) and torch.equal(sp, sp_last)


def test_bf16_autocast_generates_the_same_species() -> None:
    torch.manual_seed(0)
    m  = LEGOLtng(_cfg()); m.eval()
    ds = _gen_batch(256)
    base = m.gen_base_wrapper(ds)
    _, sp32 = m.solve(ds, x_init=base, step_size=0.25, method="euler", return_species=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out16, sp16 = m.solve(ds, x_init=base, step_size=0.25, method="euler", return_species=True)
    assert out16.shape[-1] == 7 and not torch.isnan(out16[:, 3:]).any()
    assert (sp16[:, 4] == sp32[:, 4]).float().mean() > 0.9           # bf16 trunk logits flip near-ties only
    _force_species(m, 1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, sp16 = m.solve(ds, x_init=base, step_size=0.25, method="euler", return_species=True)
    assert (sp16[:, 4] == 1).all()


def test_log_likelihood_raises_and_return_base_has_eight_columns() -> None:
    m  = LEGOLtng(_cfg()); m.eval()
    ds = _gen_batch()
    with pytest.raises(RuntimeError):
        m.log_likelihood(ds, log_p0=lambda x: torch.zeros(x.shape[0]))
    m.rc.odeint_conf["return_base"] = True
    try:
        sols, _, _ = m((ds.f.full, ds.m.full, ds.am.full))
    finally:
        m.rc.odeint_conf.pop("return_base")
    assert sols.shape[-1] == 8
