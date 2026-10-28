"""model_conf.edep_ae: per-layer E_dep through ONE latent token carrying the K latents of a frozen autoencoder.

The K standardised latents are an event-level Euclidean state ``z`` of shape (B, K), flown jointly with the particle
state: the vector field embeds it as one extra token (``Linear(K -> h_dim)``) in the same attention and reads its
velocity out of that token (``Linear(h_dim -> K)``, zero-init). The particle sequence keeps its layout and its
length; the AE's factorised stochastic decoder turns the generated latents into the per-layer vector, rescaled to the
event's total E_dep. Design: docs/superpowers/specs/2026-10-20-edep-layer-latents-design.md ("Token version").
"""

from __future__ import annotations

import pytest
import torch

from legofmt.cfm.edep_ae import EdepLayerAE, build_edep_ae
from legofmt.data.struct import DataStruct
from legofmt.main.modules import LEGOLtng
from legofmt.mod_comps.config import resolve_legoltng_config

from test_helpers import _fake_batch, _tiny_config

N_LAYERS, K = 6, 3


def _ae_spec(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    ae = EdepLayerAE(N_LAYERS, K, hidden=8, n_mix=2)
    return {"config": {"n_layers": N_LAYERS, "K": K, "hidden": 8, "n_mix": 2}, "state_dict": ae.state_dict()}


def _config(with_ae: bool = True, step_cond: bool = False, **model_conf) -> dict:
    cfg = _tiny_config()
    mc = cfg["config"]["model_conf"]
    if with_ae:
        mc["edep_ae"] = _ae_spec()
    mc.update(model_conf)
    mc["model_args"]["step_cond"] = step_cond
    return cfg


def _model(with_ae: bool = True, train: bool = False, **kw) -> LEGOLtng:
    torch.manual_seed(0)
    m = LEGOLtng(_config(with_ae, **kw))
    m.on_fit_start()
    m.train(train)
    return m


def _batch() -> DataStruct:
    ds = _fake_batch()
    ds.edep_layers = torch.rand(len(ds), N_LAYERS, generator=torch.Generator().manual_seed(1)) * 0.01
    return ds


def _wake(m: LEGOLtng) -> None:
    """The latent read-out is zero-init (an untrained token moves nothing); give it weights so a test can see it."""
    torch.manual_seed(1)
    torch.nn.init.normal_(m.model.vf.lat_out.weight, std=0.5)


def _vf_args(m: LEGOLtng, ds: DataStruct) -> dict:
    return dict(mask=ds.m.full, attn_mask=ds.am.full, types=m.types_embd[:, :ds.f.full.shape[1]],
                pdgids=m.convert_pdgids(ds.f.pdgids))


def test_sample_sums_to_the_total_and_is_zero_without_a_deposit() -> None:
    ae = build_edep_ae(_ae_spec())
    tot = torch.tensor([0.0, 3.0, 250.0])
    E = ae.sample(torch.randn(3, K), tot)
    assert E.shape == (3, N_LAYERS) and (E >= 0).all()
    torch.testing.assert_close(E.sum(-1), tot)
    assert E[0].abs().max() == 0


def test_nll_is_finite_with_empty_layers() -> None:
    ae = EdepLayerAE(N_LAYERS, K, hidden=8, n_mix=2)
    E = torch.rand(4, N_LAYERS)
    E[:, ::2] = 0.0
    loss = ae.nll(E)
    assert torch.isfinite(loss)
    loss.backward()
    assert ae.enc[0].weight.grad.abs().sum() > 0


def test_one_token_leaves_the_sequence_layout_alone() -> None:
    m, plain = _model(), _model(with_ae=False)
    assert m.rc.n_lat == K and m.rc.model_args["lat_dim"] == K
    torch.testing.assert_close(m.types_embd, plain.types_embd)
    assert m.model.vf.lat_in.in_features == K and m.model.vf.lat_out.out_features == K


def test_field_returns_the_latent_velocity() -> None:
    m = _model()
    ds = _batch()
    v, vz = m.model(ds.f.model_in, torch.tensor(0.3), z=torch.randn(len(ds), K), **_vf_args(m, ds))
    assert v.shape == ds.f.model_in.shape and vz.shape == (len(ds), K)
    assert vz.abs().max() == 0                          # zero-init read-out


def test_latent_token_and_particles_see_each_other() -> None:
    m = _model()
    _wake(m)
    ds = _batch()
    z = torch.randn(len(ds), K)
    v, vz = m.model(ds.f.model_in, torch.tensor(0.3), z=z, **_vf_args(m, ds))
    v2, _ = m.model(ds.f.model_in, torch.tensor(0.3), z=z + 1.0, **_vf_args(m, ds))
    x2 = ds.f.model_in.clone()
    x2[:, 3, 0] += 0.3
    _, vz2 = m.model(x2, torch.tensor(0.3), z=z, **_vf_args(m, ds))
    gen = ds.m.full == 1
    assert not torch.allclose(v[gen], v2[gen])          # particles attend to the latent token
    assert not torch.allclose(vz, vz2)                  # and it attends to them


def test_packed_and_per_token_conditions_agree() -> None:
    """Per-event t takes the packed (B, 3, D) adaptive-norm condition, whose selector must mark the latent token as
    generated; a per-token t equal to it (conditioning tokens at t = 1) takes the unpacked path."""
    m = _model()
    _wake(m)
    ds = _batch()
    vf, z = m.model.vf, torch.randn(2, K)
    x = ds.f.model_in
    a = vf(x, t=torch.full((2, 1), 0.3), z=z, **_vf_args(m, ds))
    b = vf(x, t=torch.where(ds.m.full == 1, 0.3, 1.0), z=z, **_vf_args(m, ds))
    for p, q in zip(a, b):
        torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("flow_map", [False, True])
def test_training_step_reaches_the_latent_token(flow_map: bool) -> None:
    m = _model(train=True, step_cond=flow_map, **({"one_step_euler_fac": 0.5} if flow_map else {}))
    loss = m._step(_batch(), 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.model.vf.lat_out.weight.grad.abs().sum() > 0   # zero-init: its gradient can only come from z's loss
    assert all(p.grad is None for p in m.edep_ae.parameters())   # frozen


def test_latent_cell_has_its_own_loss_weight() -> None:
    m = _model(train=True, learned_loss_weights=True, edep_cell=True)
    assert m.lv.numel() == 5
    m._step(_batch(), 0).backward()
    assert m.lv.grad[-1] != 0


def test_training_step_needs_per_layer_data() -> None:
    m = _model(train=True)
    with pytest.raises(ValueError, match="E_dep_layers"):
        m._step(_fake_batch(), 0)


@pytest.mark.parametrize("method", ["euler", "midpoint", "rk4"])
def test_solve_returns_the_standard_state_and_the_latents(method: str) -> None:
    m = _model()
    _wake(m)
    ds = _batch()
    sols = m.solve(ds, step_size=0.5, method=method)
    assert sols.shape == ds.f.model_in.shape and torch.isfinite(sols).all()
    assert m._edep_latents.shape == (len(ds), K) and torch.isfinite(m._edep_latents).all()
    steps = m.solve(ds, step_size=0.5, method=method, return_intermediates=True)
    assert steps.shape == (3, *ds.f.model_in.shape) and m._edep_latents.shape == (len(ds), K)


def test_solve_splits_and_buckets_the_latents_with_their_events() -> None:
    m = _model()
    _wake(m)
    ds = _batch()
    base = m.gen_base_wrapper(ds)
    torch.manual_seed(4)
    m.solve(ds, x_init=base, step_size=0.5, method="euler")
    whole = m._edep_latents
    torch.manual_seed(4)
    m.solve(ds, x_init=base, step_size=0.5, method="euler", split_size=1, buckets=[2])
    torch.testing.assert_close(m._edep_latents, whole)


def test_val_metrics_report_the_layers() -> None:
    m = _model()
    out = m.val_metrics(m, _batch())
    for k in ("val/layers/mean_ratio", "val/layers/sd_ratio", "val/layers/depth_ratio"):
        assert torch.isfinite(torch.as_tensor(out[k]))


def test_checkpoint_carries_the_autoencoder_and_the_token(tmp_path) -> None:
    torch.manual_seed(0)
    m = LEGOLtng(_config(learned_loss_weights=True, edep_cell=True))
    _wake(m)
    m.rc.config["model_conf"]["loss_weights"] = {"lv": m.lv.detach().clone()}
    path = tmp_path / "flow.pt"
    torch.save({"state_dict": m.model.vf.state_dict(), "config": m.rc.config}, path)
    m2 = LEGOLtng(torch.load(path, weights_only=False))
    for k, v in m.edep_ae.state_dict().items():
        torch.testing.assert_close(m2.edep_ae.state_dict()[k], v)
    torch.testing.assert_close(m2.model.vf.lat_out.weight, m.model.vf.lat_out.weight)
    assert m2.rc.config["model_conf"]["loss_weights"]["lv"].numel() == 5


def test_a_checkpoint_without_the_latent_token_is_refused() -> None:
    """The vf loads with strict=False, so the 16-row prototype (no lat_in / lat_out) would silently generate with a
    zero latent read-out."""
    m = LEGOLtng(_config())
    sd = {k: v for k, v in m.model.vf.state_dict().items() if not k.startswith("lat_")}
    with pytest.raises(ValueError, match="lat_in"):
        LEGOLtng({"state_dict": sd, "config": m.rc.config})


def test_a_path_is_loaded_and_replaced_by_its_spec(tmp_path) -> None:
    torch.save(_ae_spec(), tmp_path / "ae.pt")
    cfg = _config()
    cfg["config"]["model_conf"]["edep_ae"] = {"path": str(tmp_path / "ae.pt")}
    rc = resolve_legoltng_config(cfg)
    assert rc.n_lat == K and "state_dict" in rc.config["model_conf"]["edep_ae"]


@pytest.mark.parametrize("bad", [{"dl": {"len_buckets": [5], "len_split": 2}}, {"mc": {"mask_conf": {"p_forward": 0.5}}}])
def test_unsupported_combinations_are_refused(bad: dict) -> None:
    cfg = _config(**bad.get("mc", {}))
    cfg["config"]["dl_conf"].update(bad.get("dl", {}))
    with pytest.raises(ValueError, match="edep_ae"):
        resolve_legoltng_config(cfg)


def test_without_the_autoencoder_nothing_changes() -> None:
    m = _model(with_ae=False)
    assert m.edep_ae is None and m.rc.n_lat == 0 and not hasattr(m.model.vf, "lat_in")
    ds = _fake_batch()
    assert isinstance(m.model(ds.f.model_in, torch.tensor(0.3), **_vf_args(m, ds)), torch.Tensor)
    sols = m.solve(ds, step_size=0.5, method="euler")
    assert sols.shape == ds.f.model_in.shape and m._edep_latents is None


GEOM = {"layers": [[1.4, 19.3, 74.0, 183.84], [0.3, 2.33, 14.0, 28.0855]], "repeat": 3, "tokens": 2, "dim": 16, "depth": 1,
        "heads": 2}


def _with_geometry() -> LEGOLtng:
    cfg = _config()
    cfg["config"]["model_conf"]["model_args"]["geom_encoder"] = dict(GEOM)
    torch.manual_seed(0)
    m = LEGOLtng(cfg)
    m.on_fit_start()
    _wake(m)
    torch.nn.init.normal_(m.model.vf.geom.out.weight, std=0.5)
    return m


def test_latent_token_with_the_geometry_encoder() -> None:
    """Both prepend tokens: [latent, geometry x K, sequence]. The packed selector marks the latent generated and the
    geometry tokens conditioning, so it must agree with the per-token path; both feed the field."""
    m = _with_geometry()
    m.eval()
    ds = _batch()
    vf, z, x = m.model.vf, torch.randn(2, K), ds.f.model_in
    a = vf(x, t=torch.full((2, 1), 0.3), z=z, **_vf_args(m, ds))
    b = vf(x, t=torch.where(ds.m.full == 1, 0.3, 1.0), z=z, **_vf_args(m, ds))
    for p, q in zip(a, b):
        torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)
    assert a[0].shape == x.shape and a[1].shape == (2, K)
    m.train()
    loss = m._step(_batch(), 0)
    loss.backward()
    assert m.model.vf.lat_out.weight.grad.abs().sum() > 0 and m.model.vf.geom.out.weight.grad.abs().sum() > 0
    m.eval()
    sols = m.solve(ds, step_size=0.5, method="euler")
    assert sols.shape == x.shape and m._edep_latents.shape == (2, K)


def _whitened(train: bool = False) -> LEGOLtng:
    """model_conf.edep_whiten (needs the base head) together with the latent token: both live in solve / _step."""
    from test_helpers import _head_config
    cfg = _head_config(True)
    cfg["config"]["model_conf"].update(edep_whiten=True, edep_ae=_ae_spec())
    cfg["config"]["model_conf"]["model_args"]["ntypes"] = 7   # len(cond_scalars) + 3; _tiny_config's 4 fits one scalar
    torch.manual_seed(0)
    m = LEGOLtng(cfg)
    m.on_fit_start()
    m.train(train)
    return m


def _whitened_batch(B: int = 8) -> DataStruct:
    from test_helpers import _head_batch
    ds = _head_batch(torch.full((B,), 0.5), torch.tensor([1, 2] * (B // 2)))
    ds.edep_layers = torch.rand(B, N_LAYERS, generator=torch.Generator().manual_seed(1)) * 0.01
    return ds


def test_whitened_solve_returns_u_scale_rows_and_latents() -> None:
    # a field pushing the whitened E_dep cell by +1 (one prior sigma) keeps u inside the channel once solve undoes the
    # whitening -- after splitting the latent rows off, which must not reach the E_dep row or the layout
    m = _whitened()
    ds = _whitened_batch()

    def push(x, t, *a, z=None, **k):
        v = torch.zeros_like(x)
        v[:, 1, 0] = 1.0
        return v if z is None else (v, torch.ones_like(z))
    m.model.forward = push
    sols = m.solve(ds, method="euler", step_size=0.25)
    assert sols.shape == ds.f.model_in.shape
    e = sols[:, 1, 0]
    assert ((e > 0) & (e < m.gen_base.e_dep_max)).all(), e
    assert m._edep_latents.shape == (len(ds), K) and torch.isfinite(m._edep_latents).all()


def test_whitened_training_step_reaches_the_latent_token() -> None:
    m = _whitened(train=True)
    loss = m._step(_whitened_batch(), 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.model.vf.lat_out.weight.grad.abs().sum() > 0
