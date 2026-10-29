"""Shared test builders (pytest puts tests/ on sys.path, so modules ``from test_helpers import ...``)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from legofmt.data.dataloaders import LEGODataset
from legofmt.data.packed import PackedData
from legofmt.data.prep import DataPrep
from legofmt.data.struct import DataStruct, _F
from legofmt.distill.reflow import LEGOLtngDirect

MANIFOLD = [{"name": "euclidean", "dim": 1},
            {"name": "sphere", "dim": 3}, {"name": "sphere", "dim": 3}]


def _tiny_config() -> dict:
    pdgids = torch.tensor([22, 211, 2212], dtype=torch.int64).sort().values
    return {
        "state_dict": {},
        "config": {
            "dl_conf": {
                "lds_args": {"cutoff_mev": 10.0},
                "bs": 2,
                "num_workers": 0,
            },
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
                "manifold": MANIFOLD,
                "max_energy": 300.0,
                "pdgids": pdgids,
                # base_pretrain_batches defaults to 300, which (with
                # scale_dist=sm_norm) builds base_head -- and that reads the
                # Z/A cond scalars this fixture does not carry.
                "base_pretrain_batches": 0,
                "model_args": {
                    "h_dim": 16,
                    "nlayers": 2,
                    "nhead": 2,
                    "in_dim": 7,
                    "max_seq_l": 5,
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
        },
    }


def _fake_batch(B: int = 2, L: int = 5) -> DataStruct:
    """Synthetic batch matching the padded sequence layout: 2 conditioning
    slots, 1 incoming-particle slot, then ``L - 3`` outgoing slots."""
    f = torch.zeros(B, L, 8)
    f[:, 0, 0] = 1.0
    f[:, 1, 0] = 0.1
    _F(f).non_p[..., 1:-1] = 1.0
    f[:, 2, 0] = 0.8
    f[:, 2, 1:4] = torch.tensor([0.0, 0.0, 1.0])
    f[:, 2, 4:7] = torch.tensor([0.0, 0.0, -1.0])
    f[:, 2, 7] = 22.0
    g = torch.Generator().manual_seed(0)
    f[:, 3:, 0] = torch.rand(B, L - 3, generator=g)
    f[:, 3:, 1:4] = torch.nn.functional.normalize(
        torch.randn(B, L - 3, 3, generator=g), dim=-1,
    )
    f[:, 3:, 4:7] = torch.nn.functional.normalize(
        torch.randn(B, L - 3, 3, generator=g), dim=-1,
    )
    f[:, 3:, 7] = 211.0
    m = torch.zeros(B, L, dtype=torch.long)
    m[:, 3:] = 1
    am = torch.ones(B, L, dtype=torch.bool)
    return DataStruct(f, m, am)


def _flow_config() -> dict:
    cfg = _tiny_config()["config"]
    cfg["model_conf"].pop("base_pretrain_batches")
    cfg["model_conf"]["model_args"]["max_seq_l"] = 7
    return cfg


def _save_flow_ckpt(tmp_path: Path, cls=LEGOLtngDirect, name: str = "flow.pt") -> Path:
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
            "train_inverse": True,
            "model_args": {"use_adaptive_rmsnorm": True},
            "inv_model_args": {"use_adaptive_rmsnorm": True},
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




def _head_config(nout: bool) -> dict:
    cfg = _tiny_config()
    pdgids = torch.tensor([11, 22, 2212], dtype=torch.int64).sort().values
    mc = cfg["config"]["model_conf"]
    cfg["config"]["base_conf"]["base_head_nout"] = nout
    mc.update(pdgids=pdgids, cond_scalars=("Density", "Z", "A", "Size"), base_pretrain_batches=1)
    mc["model_args"].update(nlayers=1, max_seq_l=8, npdgids=pdgids.numel() + 1)
    return cfg


def _head_batch(edep: torch.Tensor, n_out: torch.Tensor) -> DataStruct:
    """Rows [Density, edep, Z, A, Size, incoming, out, out]; ``n_out`` in {1, 2}."""
    B = len(edep)
    f = torch.zeros(B, 8, 8)
    for row, val in ((0, 3.0), (2, 18.0), (3, 39.95), (4, 100.0)):
        f[:, row, 0] = val
    _F(f).non_p[..., 1:-1] = 1.0
    f[:, 1, 0] = edep
    f[:, 5, 0] = 1.0
    f[:, 5, 1:4] = torch.tensor([1.0, 0.0, 0.0])
    f[:, 5, 4:7] = torch.tensor([-1.0, 0.0, 0.0])
    f[:, 5, 7] = 2212.0
    g = torch.Generator().manual_seed(0)
    f[:, 6:, 0] = torch.rand(B, 2, generator=g)
    f[:, 6:, 1:4] = torch.nn.functional.normalize(torch.randn(B, 2, 3, generator=g), dim=-1)
    f[:, 6:, 4:7] = torch.nn.functional.normalize(torch.randn(B, 2, 3, generator=g), dim=-1)
    f[:, 6:, 7] = 11.0
    second = n_out == 2
    f[:, 7, 7] = torch.where(second, torch.tensor(11.0), torch.tensor(0.0))
    m = torch.zeros(B, 8, dtype=torch.long)
    m[:, 1] = 1
    m[:, 6:] = 1
    am = torch.ones(B, 8, dtype=torch.bool)
    am[:, 7] = second
    return DataStruct(f, m, am)


_PREPPED_E_OUT = [[50, 50, 50, 50], [50, 1, 50, 1], [1, 1, 1, 1],
                  [50, 1, 1, 1], [50, 50, 50, 1], [1, 50, 1, 50]]


def prepped_padded(cond=("Density",), layers=False, n=6, k=4) -> tuple:
    """A real prepped padded dataset, ``(f, m, am)`` or ``(f, m, am, edep_layers)``, of
    events with 4/2/0/1/3/2 outgoing particles (event 0 full length, event 2 none)."""
    g = torch.Generator().manual_seed(0)
    e_out = torch.tensor(_PREPPED_E_OUT, dtype=torch.float32)[:n, :k]
    def rows(e, shape):
        d = torch.nn.functional.normalize(torch.randn(*shape, 3, generator=g), dim=-1)
        return torch.cat((e[..., None], e[..., None] * d, torch.randn(*shape, 3, generator=g) * 50.0), -1)
    inc = torch.cat((rows(torch.full((n, 1), 200.0), (n, 1)), torch.full((n, 1, 1), 2212.0)), -1)
    out = torch.cat((rows(e_out, (n, k)), torch.full((n, k, 1), 22.0)), -1)
    per_event = {"E_dep": torch.rand(n, generator=g) * 100.0}
    for i, name in enumerate(cond):
        per_event[name] = torch.full((n,), 3.0 + i)
    if layers:
        per_event["E_dep_layers"] = torch.rand(n, 3, generator=g) * 30.0
    cfg = {"cutoff_mev": 10.0, "manifold": MANIFOLD, "proj_ray": False,
           "cond_scalars": list(cond), "energy_kin": True}
    d = LEGODataset(data={"per_particle": {"Incoming": inc, "Outgoing": out}, "per_event": per_event},
                    prep=DataPrep(cfg), cutoff_mev=10.0, min_particles=0).data
    n_prefix = len(cond) + 1
    assert (d.am.full[:, n_prefix + 1:].sum(1) == 0).any(), "no zero-outgoing event survived prep"
    return (d.f.full, d.m.full, d.am.full) + ((d.edep_layers,) if layers else ())


PACKED_COND = ("Density", "Z", "A", "Size")


def packed_prep(**kw) -> DataPrep:
    """The DataPrep the packed-dataset tests load ``write_two_dirs`` with."""
    return DataPrep({"cutoff_mev": 10.0, "manifold": MANIFOLD, "proj_ray": False, "cond_scalars": list(PACKED_COND),
                     "energy_kin": True, "max_energy": 300.0, **kw})


def write_two_dirs(tmp_path: Path, cond=PACKED_COND, layers=True) -> tuple[Path, Path, Path]:
    """The same ``prepped_padded`` events as ``pad/`` (only ``data_prepped.pt``), ``pk/`` (only
    ``data_packed.pt``) and ``both/`` (both files), each with a ``meta.json``."""
    t = prepped_padded(cond, layers)
    packed = PackedData.from_padded(*t[:3], n_prefix=len(cond) + 1, edep_layers=t[3] if layers else None)
    meta = {"ntokens": t[0].shape[1], "particles": [22.0, 2212.0], "particles_in": [2212.0],
            "cutoff_mev": 10.0, "cond_scalars": list(cond), "energy_kin": True, "max_energy": 300.0}
    pad, pk, both = dirs = tuple(tmp_path / n for n in ("pad", "pk", "both"))
    for d in dirs:
        d.mkdir()
        fmt = {} if d == pad else {"format": "packed", "n_events": len(packed)}   # as gen_data writes
        (d / "meta.json").write_text(json.dumps({**meta, **fmt}))
        if d != pk:
            torch.save(t, d / "data_prepped.pt")
        if d != pad:
            torch.save(packed.to_dict(), d / "data_packed.pt")
    return dirs
