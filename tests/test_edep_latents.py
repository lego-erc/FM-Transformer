"""``model_conf.edep_ae``: the E_dep layers through one latent token that carries the K latents of a frozen autoencoder.

The K standardised latents are a per-event Euclidean state ``z`` (B, K), integrated together with the tokens. The
field embeds ``z`` as one extra token (``lat_in``, ``Linear(K -> h_dim)``) in the same attention and reads its
velocity out of that token (``lat_out``, ``Linear(h_dim -> K)``, zero-initialised). The event's token sequence keeps
its layout and its length. The autoencoder's factorised stochastic decoder turns the generated latents into the
deposits per layer, rescaled to the event's total E_dep.
"""

from __future__ import annotations

import copy

import pytest
import torch

from legofmt.cfm.edep_ae import EdepLayerAE, build_edep_ae
from legofmt.data.struct import DataStruct
from legofmt.main.modules import LEGOLtng
from legofmt.config import resolve_legoltng_config

from test_helpers import GEOM, _fake_batch, _head_batch, _head_config, _tiny_config, _vf_kwargs

N_LAYERS, K = 6, 3


def _ae_spec(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    ae = EdepLayerAE(N_LAYERS, K, hidden=8, n_mix=2)
    return {"config": {"n_layers": N_LAYERS, "K": K, "hidden": 8, "n_mix": 2}, "state_dict": ae.state_dict()}


def _config(with_ae: bool = True, step_cond: bool = False, **model_conf) -> dict:
    if with_ae:
        model_conf["edep_ae"] = _ae_spec()
    return _tiny_config({"step_cond": step_cond}, **model_conf)


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
    """Gives the latent's zero-initialised read-out random weights, so that a test sees the latent token's velocity."""
    torch.manual_seed(1)
    torch.nn.init.normal_(m.model.vf.lat_out.weight, std=0.5)


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
    v, vz = m.model(ds.f.model_in, torch.tensor(0.3), z=torch.randn(len(ds), K), **_vf_kwargs(m, ds))
    assert v.shape == ds.f.model_in.shape and vz.shape == (len(ds), K)
    assert vz.abs().max() == 0                          # zero-init read-out


def test_latent_token_and_particles_see_each_other() -> None:
    m = _model()
    _wake(m)
    ds = _batch()
    z = torch.randn(len(ds), K)
    v, vz = m.model(ds.f.model_in, torch.tensor(0.3), z=z, **_vf_kwargs(m, ds))
    v2, _ = m.model(ds.f.model_in, torch.tensor(0.3), z=z + 1.0, **_vf_kwargs(m, ds))
    x2 = ds.f.model_in.clone()
    x2[:, 3, 0] += 0.3
    _, vz2 = m.model(x2, torch.tensor(0.3), z=z, **_vf_kwargs(m, ds))
    gen = ds.m.full == 1
    assert not torch.allclose(v[gen], v2[gen])          # the generated tokens attend to the latent token
    assert not torch.allclose(vz, vz2)                  # and it attends to them


def test_packed_and_per_token_conditions_agree() -> None:
    """A per-event ``t`` takes the per-event (B, 3, D) adaptive-norm condition, whose selector must mark the latent
    token as generated. The same times given per token (conditioning tokens at t = 1) take the per-token path."""
    m = _model()
    _wake(m)
    ds = _batch()
    vf, z = m.model.vf, torch.randn(2, K)
    x = ds.f.model_in
    a = vf(x, t=torch.full((2, 1), 0.3), z=z, **_vf_kwargs(m, ds))
    b = vf(x, t=torch.where(ds.m.full == 1, 0.3, 1.0), z=z, **_vf_kwargs(m, ds))
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
    """The field loads with ``strict=False``: a checkpoint of the 16-token prototype (no ``lat_in`` / ``lat_out``)
    would generate, silently, with a zero latent read-out."""
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


@pytest.mark.parametrize("bad", [{"dl": {"len_buckets": [5], "len_split": 2}},
                                 {"mc": {"mask_conf": {"p_forward": 0.5}}}])
def test_unsupported_combinations_are_refused(bad: dict) -> None:
    cfg = _config(**bad.get("mc", {}))
    cfg["config"]["dl_conf"].update(bad.get("dl", {}))
    with pytest.raises(ValueError, match="edep_ae"):
        resolve_legoltng_config(cfg)


def test_group_buckets_train_with_the_latent_token() -> None:
    # len_mode group never reorders or splits a batch (the loader groups lengths, the step trims to one bucket), so
    # the per-layer targets stay aligned with their events and the plain field carries z
    cfg = _config()
    cfg["config"]["dl_conf"].update(len_buckets=[4, 5], len_mode="group")
    torch.manual_seed(0)
    m = LEGOLtng(cfg)
    m.on_fit_start()
    m.train()
    ds = _batch()
    ds.m.full[:, 4], ds.am.full[:, 4] = 0, False   # every event one slot shorter: the batch is trimmed to bucket 4
    ds = m.on_before_batch_transfer(ds, 0)
    assert ds.f.full.shape[1] == 4 and ds.edep_layers.shape == (2, N_LAYERS)
    loss = m._step(ds, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.model.vf.lat_out.weight.grad.abs().sum() > 0


def test_without_the_autoencoder_nothing_changes() -> None:
    m = _model(with_ae=False)
    assert m.edep_ae is None and m.rc.n_lat == 0 and not hasattr(m.model.vf, "lat_in")
    ds = _fake_batch()
    assert isinstance(m.model(ds.f.model_in, torch.tensor(0.3), **_vf_kwargs(m, ds)), torch.Tensor)
    sols = m.solve(ds, step_size=0.5, method="euler")
    assert sols.shape == ds.f.model_in.shape and m._edep_latents is None


def _with_geometry() -> LEGOLtng:
    cfg = _config()
    cfg["config"]["model_conf"]["model_args"]["geom_encoder"] = copy.deepcopy(GEOM)
    torch.manual_seed(0)
    m = LEGOLtng(cfg)
    m.on_fit_start()
    _wake(m)
    torch.nn.init.normal_(m.model.vf.geom.out.weight, std=0.5)
    return m


def test_latent_token_with_the_geometry_encoder() -> None:
    """Both options prepend tokens: [latent, geometry tokens, the event's tokens]. The per-event selector marks the
    latent token generated and the geometry tokens conditioning, so the per-event and the per-token path must agree.
    A training step reaches both read-ins, and a solve runs."""
    m = _with_geometry()
    m.eval()
    ds = _batch()
    vf, z, x = m.model.vf, torch.randn(2, K), ds.f.model_in
    a = vf(x, t=torch.full((2, 1), 0.3), z=z, **_vf_kwargs(m, ds))
    b = vf(x, t=torch.where(ds.m.full == 1, 0.3, 1.0), z=z, **_vf_kwargs(m, ds))
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
    """Returns a model with ``model_conf.edep_whiten`` (it needs the base head) and the latent token.

    Both act in ``solve`` and in ``_step``.
    """
    # ntypes = len(cond_scalars) + 3; _tiny_config's 4 fits one scalar
    cfg = _head_config(True, {"ntypes": 7}, edep_whiten=True, edep_ae=_ae_spec())
    torch.manual_seed(0)
    m = LEGOLtng(cfg)
    m.on_fit_start()
    m.train(train)
    return m


def _whitened_batch(B: int = 8) -> DataStruct:
    ds = _head_batch(torch.full((B,), 0.5), torch.tensor([1, 2] * (B // 2)))
    ds.edep_layers = torch.rand(B, N_LAYERS, generator=torch.Generator().manual_seed(1)) * 0.01
    return ds


def test_whitened_solve_returns_u_scale_tokens_and_latents() -> None:
    # A field that pushes the whitened E_dep token by +1 (one sigma of the prior): once solve undoes the whitening,
    # the deposit lies inside (0, e_dep_max). The latent is split off first: it must not reach the E_dep token or the
    # layout.
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


def test_a_step_evaluates_the_head_once() -> None:
    """Sampling the base sets every parameter of the prior; whitening the target's E_dep reads its mu and sig there."""
    m, ds, calls = _whitened(train=True), _whitened_batch(), []
    m.base_head.register_forward_hook(lambda *a: calls.append(1))
    m._step(ds, 0)
    assert len(calls) == 1
    by_head = m._edep_whiten(ds.f.full, ds)                # the solver's call: the head, on the batch it is handed
    m.gen_base_wrapper(ds)
    assert torch.equal(m._edep_whiten(ds.f.full, ds, drawn=True), by_head) and len(calls) == 3


def test_whitened_training_step_reaches_the_latent_token() -> None:
    m = _whitened(train=True)
    loss = m._step(_whitened_batch(), 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.model.vf.lat_out.weight.grad.abs().sum() > 0
