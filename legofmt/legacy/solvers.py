"""Solver methods superseded by the hand-unrolled manifold steps (:mod:`legofmt.cfm.solvers`), kept so older configs
still run unchanged: every ``method`` other than ``euler`` / ``midpoint`` / ``rk4`` (e.g. ``dopri5``) is integrated by
flow_matching's Euclidean ``ODESolver``, over the bare time grid ``[0, 1]``.
"""

from flow_matching.solver import ODESolver
from torch import Tensor


def odesolver_sample(model, x_init: Tensor, time_grid: Tensor, step_size: float, method: str,
                     return_intermediates: bool, z: list, **extras) -> Tensor:
    """Integrates ``x_init`` along ``time_grid`` with flow_matching's ``ODESolver``.

    ``Solvers._solve`` hands every method that is not one of its own here.

    Args:
        model:                the field (``ProjectModel``).
        x_init:               (B, L, 7), the start.
        time_grid:            the grid.
        step_size:            the step of a fixed-step method.
        method:               a torchdiffeq method.
        return_intermediates: return the state at every grid point.
        z:                    the ``model_conf.edep_ae`` latent, in a list; it has to be empty.
        extras:               the field's keyword inputs.

    Returns:
        The final state (B, L, 7), or every grid point's (T, B, L, 7).

    Raises:
        NotImplementedError: there is a latent.
    """
    if z:
        raise NotImplementedError(f"method {method!r} does not integrate the model_conf.edep_ae latent.")
    return ODESolver(velocity_model=model).sample(
        x_init=x_init, time_grid=time_grid, step_size=step_size, method=method,
        return_intermediates=return_intermediates, **extras,
    )
