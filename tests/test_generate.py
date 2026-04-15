"""Smoke test: load checkpoints from HF and run GenerateOut end-to-end."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from huggingface_hub import hf_hub_download

from legofmt.main.generate import GenerateOut


HF_REPO = os.environ.get("HF_REPO", "lego-erc/legofmt")
HF_REVISION = os.environ.get("HF_REVISION", "main")
FLOW_CKPT = os.environ.get("HF_FLOW_CKPT", "rp_fm_v4_100426.pt")
MULT_CKPT = os.environ.get("HF_MULT_CKPT", "rp_mult_v1_020426.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def ckpt_paths(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    cache = tmp_path_factory.mktemp("hf")
    token = os.environ.get("HF_TOKEN")  # only needed for private repos
    flow = hf_hub_download(
        repo_id=HF_REPO, filename=FLOW_CKPT, revision=HF_REVISION,
        local_dir=cache, token=token,
    )
    mult = hf_hub_download(
        repo_id=HF_REPO, filename=MULT_CKPT, revision=HF_REVISION,
        local_dir=cache, token=token,
    )
    return Path(flow), Path(mult)


@pytest.fixture(scope="module")
def generator(ckpt_paths: tuple[Path, Path]) -> GenerateOut:
    flow, mult = ckpt_paths
    return GenerateOut(str(flow), str(mult), device=DEVICE)


def _dummy_cond(gen: GenerateOut, batch: int = 2) -> torch.Tensor:
    """Build a valid [B, 8] cond: [density, mom(3), pos(3), pdgid]."""
    pdgid = gen.pdgid_in[0].item()
    cond = torch.zeros(batch, 8, device=DEVICE)
    cond[:, 0] = 1.0                                            # density
    cond[:, 1:4] = torch.tensor([0.0, 0.0, 150.0], device=DEVICE)   # momentum
    cond[:, 4:7] = torch.tensor([0.0, 0.0, -1.0], device=DEVICE)     # entry position
    cond[:, 7] = pdgid
    return cond


def test_instantiation(generator: GenerateOut) -> None:
    assert isinstance(generator.model, torch.nn.Module)
    assert isinstance(generator.gen_mult, torch.nn.Module)
    assert generator.pdgids.numel() > 0
    assert generator.max_seq_l > 3


@torch.no_grad()
def test_forward_shape(generator: GenerateOut) -> None:
    cond = _dummy_cond(generator, batch=2)
    out = generator(cond)
    assert out.ndim == 3
    assert out.shape[0] == cond.shape[0]
    assert out.shape[1] == generator.max_seq_l


@torch.no_grad()
def test_g4_style_api(generator: GenerateOut) -> None:
    n = 3
    pos = torch.tensor([[0.0, 0.0, -1.0]], device=DEVICE)
    mom = torch.tensor([[0.0, 0.0, 1.0]], device=DEVICE)
    energy = torch.tensor([1000.0], device=DEVICE)
    density = torch.tensor([1.0], device=DEVICE)
    size = torch.tensor([1.0], device=DEVICE)
    pdgids = generator.pdgid_in[:1]

    out = generator.gen_model_w_g4_args(n, pos, mom, energy, density, size, pdgids)

    assert set(out) == {"per_event", "per_particle", "per_voxel"}
    assert out["per_particle"]["Incoming"].shape[-1] == 8   # E, mom(3), pdgid, pos(3)
    assert out["per_particle"]["Outgoing"].shape[-1] == 8
    assert torch.isfinite(out["per_particle"]["Outgoing"]).all()
