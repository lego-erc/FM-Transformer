"""``model_conf.exclude_absorbed``: an absorbed event (no outgoing particle) is left out of training, and generation
gives it its E_dep without the flow.

```
E_dep = E_in                  every incoming species but the e+
E_dep = E_in + 2 m_e          an e+: it annihilates at rest, and both photons are absorbed too
```

At a 0.1 MeV cutoff Geant4 closes these events to 0.1 MeV for e+- / gamma, and exactly for 99.8 % of the stopped
protons. What escapes below the cutoff is missing from E_dep, so the key is for low cutoffs.
"""

from __future__ import annotations

import pytest
import torch

from legofmt.data.struct import _F
from legofmt.main.generate import GenerateOut, absorbed_deposit
from legofmt.main.modules import LEGOLtng
from test_helpers import _gun_cond, _save_flow_ckpt, _save_mult_ckpt, _tiny_config, write_two_dirs
from test_out_margin import _fresh_config

# the tiny configs here have a 10 MeV cutoff, at which the key warns (test_the_key_warns_at_a_high_cutoff)
pytestmark = pytest.mark.filterwarnings("ignore:model_conf.exclude_absorbed:UserWarning")


def test_an_absorbed_event_deposits_everything() -> None:
    e_in = torch.tensor([150.0, 150.0, 150.0, 80.0])
    got = absorbed_deposit(e_in, torch.tensor([11, -11, 22, 2212]))
    assert got.tolist() == pytest.approx([150.0, 151.022, 150.0, 80.0])


@torch.no_grad()
@pytest.mark.parametrize("exclude", [True, False], ids=["key", "no key"])
def test_absorbed_events_skip_the_flow_only_with_the_key(tmp_path, monkeypatch, exclude: bool) -> None:
    """Events 0 and 2 get no outgoing particle. With the key the flow sees only event 1, and the other two deposit
    their E_in; without it the flow generates all three, as it was trained to."""
    gen = GenerateOut(str(_save_flow_ckpt(tmp_path, LEGOLtng, exclude_absorbed=exclude)),
                      str(_save_mult_ckpt(tmp_path)), device="cpu")
    seen = []
    monkeypatch.setattr(gen.model, "forward", lambda batch, *_, rng=None: seen.append(len(batch[0])) or batch)
    sols, _, _ = gen(_gun_cond(22, e_mev=150.0, batch=3), gt_mult=torch.tensor([[0, 0], [1, 0], [0, 0]]))

    assert seen == ([1] if exclude else [3])
    if exclude:
        edep = gen.dp.denorm_edep_margin(_F(sols).edep, torch.full((3,), 150.0), torch.full((3,), 22.0))
        assert edep[[0, 2]].tolist() == pytest.approx([150.0, 150.0], rel=1e-5)


def test_training_leaves_the_absorbed_events_out(tmp_path) -> None:
    """``write_two_dirs`` holds six events, one of them without an outgoing particle."""
    _, pk, _ = write_two_dirs(tmp_path)
    for exclude, n in ((True, 5), (False, 6)):
        model = LEGOLtng(_fresh_config(pk, exclude_absorbed=exclude))
        model.setup()
        assert len(model._train_ds) + len(model._val_ds) == n


def test_the_key_refuses_an_edep_ae() -> None:
    """An edep_ae samples the E_dep layers from the latents the flow generates, which an absorbed event never gets."""
    with pytest.raises(ValueError, match="exclude_absorbed"):
        LEGOLtng(_tiny_config(exclude_absorbed=True, edep_ae={"path": "unused.pt"}))



@pytest.mark.parametrize("cutoff, warns", [(10.0, True), (0.1, False)])
def test_the_key_warns_at_a_high_cutoff(cutoff: float, warns: bool, recwarn) -> None:
    """The particles that escape below the cutoff are missing from an absorbed event's deposit: at a 10 MeV cutoff a
    median 7 MeV (Ar/Cu bricks), at 0.1 MeV at most about 0.1 MeV."""
    cfg = _tiny_config(exclude_absorbed=True)
    cfg["config"]["dl_conf"]["lds_args"]["cutoff_mev"] = cutoff
    LEGOLtng(cfg)
    hits = [w for w in recwarn.list if "exclude_absorbed" in str(w.message)]
    assert bool(hits) == warns
    if warns:
        assert "cutoff_mev=10.0" in str(hits[0].message)
