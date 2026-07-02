"""Smoke test for :class:`legofmt.main.generate.GenerateOutDirect`.

Builds a tiny flow checkpoint on disk (no HF download), instantiates
``GenerateOut`` from it, and runs one end-to-end forward on a synthetic
``cond`` tensor.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from legofmt.data.struct import _F
from legofmt.main.generate import GenerateIn, GenerateOutDirect as GenerateOut
from legofmt.main.modules import LEGOLtng as LEGOLtngVelocity, LEGOLtngDirect as LEGOLtng


def _flow_config() -> dict:
    pdgids = torch.tensor([22, 211, 2212], dtype=torch.int64).sort().values
    return {
        "dl_conf": {"lds_args": {"cutoff_mev": 10.0}, "bs": 2, "num_workers": 0},
        "val_conf": {"val_frac": 0.01, "seed": 0},
        "base_conf": {
            "base_range": 3.4,
            "kappa": torch.tensor(8.0),
            "bs_frac": 0.0,
            "base_dist": "poles",
            "scale_dist": "sm_norm",
            "tanh_theta": True,
        },
        "model_conf": {
            "manifold": [
                {"name": "euclidean", "dim": 1},
                {"name": "sphere",    "dim": 3},
                {"name": "sphere",    "dim": 3},
            ],
            "max_energy": 300.0,
            "pdgids": pdgids,
            "model_args": {
                "h_dim": 16,
                "nlayers": 2,
                "nhead": 2,
                "in_dim": 7,
                "max_seq_l": 7,
                "ntypes": 4,
                "nvtypes": 2,
                "npdgids": pdgids.numel() + 1,
                "ff_mult": 1,
                "dropout": 0.0,
                "use_adaptive_rmsnorm": True,
                "use_adaptive_layerscale": True,
                "ff_swish": True,
                "ff_glu": True,
            },
        },
        "opt_conf": {"opt": "schedulefree", "lr": 1e-3},
    }


def _save_flow_ckpt(tmp_path: Path, cls=LEGOLtng, name: str = "flow.pt") -> Path:
    cfg = _flow_config()
    m = cls({"state_dict": {}, "config": cfg})
    path = tmp_path / name
    torch.save({"state_dict": m.model.vf.state_dict(), "config": cfg}, path)
    return path


def _mult_config() -> dict:
    """Checkpoint-style mult config: ptypes/max_count present so
    ``resolve_mult_config`` takes the restore path (no meta.json needed)."""
    return {
        "dl_conf": {"lds_args": {"cutoff_mev": 10.0}, "bs": 2, "num_workers": 0},
        "mm_conf": {
            "ptypes": torch.tensor([22, 211], dtype=torch.int64),
            "ptypes_in": torch.tensor([22, 211, 2212], dtype=torch.int64),
            "max_out_particles": 4,
            "max_count": 4,
            "h_dim": 16,
            "in_dim": 8,
            "n_layers": 1,
            "n_heads": 2,
            "dropout": 0.0,
            "use_abs_pos_emb": False,
            "post_emb_norm": False,
            "model_args": {"use_adaptive_rmsnorm": True},
        },
        "opt_conf": {"opt": "schedulefree", "lr": 1e-3},
    }


def _save_mult_ckpt(tmp_path: Path) -> Path:
    from legofmt.multiplicity.model import MultModel

    cfg = _mult_config()
    m = MultModel({"state_dict": {}, "config": cfg})
    path = tmp_path / "mult.pt"
    torch.save({"state_dict": m.state_dict(), "config": cfg}, path)
    return path


@pytest.fixture(scope="module")
def generator(tmp_path_factory: pytest.TempPathFactory) -> GenerateOut:
    tmp = tmp_path_factory.mktemp("direct_ckpts")
    return GenerateOut(str(_save_flow_ckpt(tmp)), str(_save_mult_ckpt(tmp)), device="cpu")


def _dummy_cond(gen: GenerateOut, batch: int = 2) -> torch.Tensor:
    """``[B, 8]`` cond: density, E-scaled mom (3), entry pos (3), pdgid."""
    cond = torch.zeros(batch, 8)
    cond[:, 0] = 1.0
    cond[:, 1:4] = torch.tensor([0.0, 0.0, 150.0])
    cond[:, 4:7] = torch.tensor([0.0, 0.0, -1.0])
    cond[:, 7] = gen.pdgids[0].item()
    return cond


def test_instantiation(generator: GenerateOut) -> None:
    assert isinstance(generator.model, torch.nn.Module)
    assert generator.pdgids.numel() > 0
    assert generator.max_seq_l > 3


@torch.no_grad()
def test_forward_shapes(generator: GenerateOut) -> None:
    sols, mask, attn_mask = generator(_dummy_cond(generator, batch=2))
    L = generator.max_seq_l
    assert sols.shape == (2, L, 8)
    assert mask.shape == attn_mask.shape == (2, L)
    assert torch.isfinite(sols[attn_mask]).all(), "non-finite values at attended slots"


def _fake_shower(B: int = 2, L: int = 7) -> tuple:
    """Synthetic shower with RAW PDG ids in the last column, as a dataset
    batch would carry them."""
    f = torch.zeros(B, L, 8)
    f[:, 0, 0] = 1.0
    f[:, 1, 1] = 0.1
    _F(f).non_p[..., 1:-1] = 1.0
    f[:, 3:, 0] = 0.5
    g = torch.Generator().manual_seed(0)
    f[:, 3:, 1:4] = torch.nn.functional.normalize(torch.randn(B, L - 3, 3, generator=g), dim=-1)
    f[:, 3:, 4:7] = torch.nn.functional.normalize(torch.randn(B, L - 3, 3, generator=g), dim=-1)
    f[:, 3:, 7] = torch.tensor([22.0, 211.0, 22.0, 211.0])
    m = torch.zeros(B, L, dtype=torch.long)
    m[:, 3:] = 1
    am = torch.ones(B, L, dtype=torch.bool)
    return f, m, am


@torch.no_grad()
def test_generate_in_smoke(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Inverse generation from a raw-PDG-id shower batch (offline)."""
    tmp = tmp_path_factory.mktemp("in_ckpts")
    gen_in = GenerateIn(
        str(_save_flow_ckpt(tmp, LEGOLtngVelocity)), str(_save_mult_ckpt(tmp)), device="cpu",
    )
    in_p = gen_in(_fake_shower())
    assert in_p.shape == (2, 1, 8)
    assert torch.isfinite(in_p).all(), "non-finite incoming features"
    raw_ids = in_p[..., -1].flatten().long()
    assert all(i in gen_in.pdgids for i in raw_ids), raw_ids
