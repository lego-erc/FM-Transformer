"""``model_conf.energy_border``: the energy of a generated outgoing token never leaves [0, 1] for the field.

The energy's training path runs from a base in [0, 1] to a target in [0, 1]. A state outside was put there by the
step error of a solver or of the flow-map teacher, and no training path lies there. So the field is evaluated with
the energy on the border, and the energy itself moves on the straight line that reaches the border at t = 1.
"""

from __future__ import annotations

import pytest
import torch

from legofmt.data.struct import _F, set_layout
from legofmt.main.modules import LEGOLtng
from test_helpers import _fake_batch, _tiny_config, _vf_kwargs

V_FIELD = 0.25   # the velocity the stand-in field returns in every channel


@pytest.fixture(autouse=True)
def _layout() -> None:
    """Sets the layout of ``_fake_batch``, [Density, E_dep, incoming, outgoing]: a test before may leave another."""
    set_layout(("Density",))


class _Recorder(torch.nn.Module):
    """A stand-in for the field: it keeps the state it is called with and returns ``V_FIELD`` everywhere."""
    step_cond = False

    def forward(self, x, mask, attn_mask, types, pdgids, t=None, d=None):
        self.seen = x.clone()
        return torch.full_like(x, V_FIELD)


def _flow(border: bool) -> LEGOLtng:
    """Returns a tiny flow whose field is a ``_Recorder``."""
    m = LEGOLtng(_tiny_config(energy_border=border))
    m.model.vf = _Recorder()
    return m


def _state():
    """Returns a batch of two events with two outgoing tokens each, and a state on it.

    Event 0's outgoing energies are outside [0, 1] (-0.2 and 1.3), event 1's inside (0.4 and 0.9). The E_dep token is
    generated too, and it, the conditioning scalar and the incoming energy are outside [0, 1] as well.
    """
    ds = _fake_batch(edep_generated=True)
    x  = ds.f.model_in.clone()
    _F(x).out_energy.copy_(torch.tensor([[-0.2, 1.3], [0.4, 0.9]]))
    _F(x).edep.fill_(-0.5)
    _F(x).d.fill_(2.0)
    _F(x).in_energy.fill_(1.7)
    return ds, x


def _field(m: LEGOLtng, ds, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return m.model(x, t, **_vf_kwargs(m, ds))


def _is(a: torch.Tensor, b) -> bool:
    return torch.allclose(a, torch.as_tensor(b, dtype=a.dtype), atol=1e-6)


def test_the_field_sees_the_outgoing_energy_on_the_border() -> None:
    """Only the energy of the generated outgoing tokens is held: the E_dep token, the conditioning scalar, the
    incoming energy and every direction reach the field as without the option."""
    ds, x = _state()
    on, off = _flow(True), _flow(False)
    t = torch.full((2, 1), 0.5)
    _field(on, ds, x, t), _field(off, ds, x, t)

    held = on.model.vf.seen
    assert _is(_F(held).out_energy, [[0.0, 1.0], [0.4, 0.9]])
    as_given = off.model.vf.seen.clone()
    assert _is(_F(as_given).out_energy, [[-0.2, 1.3], [0.4, 0.9]]), "off: the state as it is"
    _F(as_given).out_energy.copy_(_F(held).out_energy)
    assert torch.equal(held, as_given)


def test_an_outgoing_token_that_is_given_keeps_its_energy() -> None:
    """An outgoing token the flow does not generate (``m = 0``) is conditioning: its value is not the flow's to move."""
    ds, x = _state()
    ds.m.full[0, 3:] = 0
    on = _flow(True)
    v = _field(on, ds, x, torch.full((2, 1), 0.5))

    assert _is(_F(on.model.vf.seen).out_energy[0], [-0.2, 1.3])
    assert _is(_F(v).out_energy[0], [V_FIELD, V_FIELD])


def test_outside_the_border_the_energy_moves_straight_to_it() -> None:
    """``v = (border - e) / (1 - t)`` where the energy is outside; inside, and in every other channel, the field's."""
    ds, x = _state()
    on, off = _flow(True), _flow(False)
    t = torch.tensor([[0.25], [0.5]])
    v, v_off = _field(on, ds, x, t), _field(off, ds, x, t)

    assert _is(_F(v).out_energy, [[0.2 / 0.75, -0.3 / 0.75], [V_FIELD, V_FIELD]])
    _F(v_off).out_energy.copy_(_F(v).out_energy)
    assert torch.equal(v, v_off)


@pytest.mark.parametrize("h", [0.25, 0.1])
def test_the_velocity_is_the_same_all_the_way_to_the_border(h: float) -> None:
    """After a step of any size the energy is still on its line, and the velocity there is the one it left with. So
    that velocity is also the mean velocity of a flow-map step of any ``d``."""
    ds, x = _state()
    on = _flow(True)
    t  = torch.full((2, 1), 0.25)
    v  = _F(_field(on, ds, x, t)).out_energy[0]

    x_h = x.clone()
    _F(x_h).out_energy[0] += h * v
    assert _is(v, [0.2 / 0.75, -0.3 / 0.75])
    assert _is(_F(_field(on, ds, x_h, t + h)).out_energy[0], v)


def test_the_step_over_the_time_left_ends_on_the_border() -> None:
    ds, x = _state()
    t = torch.full((2, 1), 0.25)
    v = _F(_field(_flow(True), ds, x, t)).out_energy[0]

    assert _is(_F(x).out_energy[0] + 0.75 * v, [0.0, 1.0])


@pytest.mark.parametrize("t", [torch.tensor(0.25), torch.full((2, 5), 0.25)], ids=["scalar", "per token"])
def test_the_time_may_be_a_scalar_or_one_per_token(t: torch.Tensor) -> None:
    """A solver hands one time for the batch, a legacy caller one per token: the velocity is the per-event one."""
    ds, x = _state()
    on = _flow(True)

    assert torch.equal(_field(on, ds, x, t), _field(on, ds, x, torch.full((2, 1), 0.25)))


def test_the_border_can_be_differentiated_through() -> None:
    """A solve may be differentiated (the density design): outside the border the energy's velocity has the slope
    ``-1 / (1 - t)`` in the energy, and no gradient is NaN."""
    ds, x = _state()
    on = _flow(True)
    on.model.no_detach = True
    x.requires_grad_(True)
    v = on.model(x, torch.full((2, 1), 0.5), **_vf_kwargs(on, ds))
    (grad,) = torch.autograd.grad(_F(v).out_energy.sum(), x)

    assert torch.isfinite(grad).all()
    assert _is(_F(grad).out_energy, [[-2.0, -2.0], [0.0, 0.0]])


def test_the_flow_map_term_trains_with_the_border() -> None:
    """The teacher's two half steps call the same field: the term stays finite and its backward runs."""
    m = LEGOLtng(_tiny_config({"step_cond": True}, one_step_euler_fac=1.0, one_step_euler_sections=4,
                              energy_border=True))
    m.on_fit_start()
    m.train()
    loss = m._step(_fake_batch(), 0)
    loss.backward()

    assert torch.isfinite(loss)


def test_the_velocity_is_finite_when_no_time_is_left() -> None:
    """rk4's last stage and an adaptive solver evaluate the field at t = 1, where ``1 / (1 - t)`` has no value."""
    ds, x = _state()
    v = _field(_flow(True), ds, x, torch.ones(2, 1))

    assert torch.isfinite(v).all()
    assert (_F(v).out_energy[0] * torch.tensor([1.0, -1.0]) > 0).all(), "still towards the border"


def test_a_solve_that_the_field_pushes_out_ends_on_the_border() -> None:
    """Four Euler steps of a field that only lowers the energy: without the option the energy ends below 0, with it
    on the border, which decodes to the incoming energy."""
    ds, _ = _state()
    x0 = ds.f.model_in.clone()
    _F(x0).out_energy.fill_(0.5)
    ends = {}
    for border in (True, False):
        m = _flow(border)
        m.model.vf.forward = lambda x, *a, t=None, d=None: torch.full_like(x, -1.3)
        ends[border] = _F(m.solve(ds, x_init=x0, method="euler", step_size=0.25)).out_energy

    assert _is(ends[False], [[-0.8, -0.8]] * 2)
    assert _is(ends[True], [[0.0, 0.0]] * 2)


def test_the_training_step_is_the_same_with_the_option() -> None:
    """Every point of a training path is inside [0, 1], so the velocity loss never meets the border."""
    plain, border = LEGOLtng(_tiny_config()), LEGOLtng(_tiny_config(energy_border=True))
    border.model.load_state_dict(plain.model.state_dict())
    losses = []
    for m in (plain, border):
        m.on_fit_start()
        m.train()
        torch.manual_seed(0)
        losses.append(m._step(_fake_batch(), 0))

    assert torch.equal(*losses)


def test_the_option_is_off_by_default() -> None:
    assert not LEGOLtng(_tiny_config()).model.energy_border
    assert LEGOLtng(_tiny_config(energy_border=True)).model.energy_border


def test_the_option_needs_overflow_delta_zero() -> None:
    """``overflow_delta`` puts targets at ``-overflow_delta``: on a training path, yet outside [0, 1]."""
    with pytest.raises(ValueError, match="overflow_delta"):
        LEGOLtng(_tiny_config(energy_border=True, overflow_delta=0.1))
