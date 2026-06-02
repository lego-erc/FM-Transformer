r"""Configuration resolution for :class:`~legofmt.main.modules.LEGOLtng`.

This module owns three pieces of construction logic that previously lived
inline in :meth:`~legofmt.main.modules.LEGOLtng.__init__`:

* Manifold construction (a small registry; replaces ``eval()`` of string
  manifold specs in both :mod:`legofmt.main.modules` and
  :mod:`legofmt.data.prep`).
* Dispatch between the two construction paths -- fresh training (driven by
  ``meta.json`` on disk) and checkpoint restore (driven by a serialized
  ``state_dict`` and config).
* Back-compatibility shims for older checkpoint formats.

The public entry point is :func:`resolve_legoltng_config`. It performs a
deep copy of its input, so callers may rely on their ``full_config`` dict
being untouched after the call. The returned :class:`ResolvedLEGOConfig`
exposes :attr:`~ResolvedLEGOConfig.serializable`, a snapshot of the
post-resolution config suitable for :func:`torch.save`.
"""
from __future__ import annotations

import copy
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from flow_matching.utils.manifolds import Euclidean, Sphere

from legofmt.geometry.path_sample_mult import ProductManifold

# Registry of factor manifolds available to the structured spec form.
_MANIFOLDS: dict[str, type] = {
    "euclidean": Euclidean,
    "sphere": Sphere,
}

# Allow-list for the legacy string-based spec form. The eval namespace is
# restricted to these names; ``__builtins__`` is cleared at the call site.
_MANIFOLD_EVAL_NS: dict[str, Any] = {
    "ProductManifold": ProductManifold,
    "Euclidean": Euclidean,
    "Sphere": Sphere,
}


def build_manifold(spec: str | list) -> ProductManifold:
    r"""Constructs a :class:`ProductManifold` from a structured spec.

    Two input forms are accepted for :attr:`spec`:

    * **List of factor dicts** (preferred). Each dict has a ``"name"`` key
      (looked up case-insensitively in the manifold registry) and a
      ``"dim"`` key (the factor's ambient dimension passed to
      :class:`ProductManifold`).
    * **Legacy string**. A Python expression evaluated in a restricted
      namespace exposing only :class:`ProductManifold`, :class:`Euclidean`,
      and :class:`Sphere`. Retained so historical checkpoints continue to
      load.

    Args:
        spec (list or str): either a list of ``{"name": str, "dim": int}``
            factor descriptions, or a legacy Python-expression string of
            the form ``"ProductManifold([Euclidean(), Sphere()], (3, 3))"``.

    Returns:
        ProductManifold: the constructed product manifold.

    Raises:
        ValueError: if :attr:`spec` is neither a list nor a string.
        KeyError: if a factor's ``"name"`` is not in the manifold registry.

    .. warning::
        The legacy string form emits :class:`DeprecationWarning`. New
        configs should use the list form.

    Example::

        >>> m = build_manifold([
        ...     {"name": "euclidean", "dim": 3},
        ...     {"name": "sphere",    "dim": 3},
        ... ])
        >>> m.ambient_dims
        (3, 3)
    """
    if isinstance(spec, str):
        warnings.warn(
            "String manifold specs are deprecated; use a list of factor dicts "
            "([{'name': 'euclidean', 'dim': 3}, ...]).",
            DeprecationWarning,
            stacklevel=2,
        )
        return eval(spec, {"__builtins__": {}}, _MANIFOLD_EVAL_NS)

    if isinstance(spec, list):
        manifolds = [_MANIFOLDS[p["name"].lower()]() for p in spec]
        dims = tuple(p["dim"] for p in spec)
        return ProductManifold(manifolds, dims)

    raise ValueError(f"Cannot build manifold from spec: {spec!r}")


@dataclass(frozen=True)
class ResolvedLEGOConfig:
    r"""Fully-resolved configuration consumed by :class:`LEGOLtng`.

    Every attribute :meth:`LEGOLtng.__init__` sets on the module is derived
    from one of the fields below. The class is frozen; callers must not
    mutate it after construction. Build via :func:`resolve_legoltng_config`.

    Attributes:
        max_seq_l (int): maximum sequence length, including conditioning slots.
        pdgids_template (Tensor): sorted ``int64`` tensor of known PDG ids;
            registered as a buffer by :class:`LEGOLtng`.
        manifold (ProductManifold): product manifold over momentum and
            position factors.
        model_args (dict): keyword arguments to splat into
            :class:`~legofmt.cfm.cfm_trafo_x.CFMTrafo_x`.
        t_dist (str): name of the time-sampling distribution
            (e.g. ``"uniform"``, ``"sd3"``, ``"sd3_grid"``).
        t_dist_scale (float): scale parameter for the time distribution.
        ot_coupling (bool): if ``True``, use optimal-transport coupling
            during training. Requires ``torch_lap_cuda_lib`` to be
            importable; :class:`LEGOLtng` validates this at construction.
        ot_e_only (bool): if ``True``, the LAP cost uses pairwise ``|mom|``
            magnitude differences rather than 6-D ``cdist``. Gives strict
            energy-ordered pairing. Ignored when :attr:`ot_coupling` is
            ``False``.
        proj_en_out (bool): if ``True``, apply energy projection to model
            outputs.
        pdgid_is_idx (bool): if ``True``, treat the PDG-id field of inputs
            as an integer index rather than a raw PDG id.
        loss_sc_fac (float): scalar multiplier for the auxiliary loss term.
        cond_cube (bool): if ``True``, project the conditioning position
            onto the cube before each forward pass.
        dl_conf (dict): dataloader sub-config, passed through to the
            dataset constructor.
        opt_conf (dict): optimizer sub-config, consumed by
            :func:`~legofmt.mod_comps.optimizers.build_optimizer`.
        odeint_conf (dict): ODE-solver sub-config used during sampling.
        val_conf (dict): validation sub-config consumed by
            :meth:`LEGOLtng.setup` (held-out split ``val_frac`` and ``seed``).
        config (dict): snapshot of the post-resolution inner config
            (including any field migrations applied during resolution).
            Handed to :class:`~legofmt.geometry.gen_base.GenerateBase`
            (which expects a dict rather than a structured object) and
            safe to :func:`torch.save` for later round-trip via
            :func:`resolve_legoltng_config`.
        state_dict (dict or None): ``None`` for fresh training; otherwise
            the model state dict to load into ``LEGOLtng.model.vf``.
    """

    max_seq_l: int
    pdgids_template: torch.Tensor
    manifold: ProductManifold
    model_args: dict[str, Any]

    t_dist: str
    t_dist_scale: float
    ot_coupling: bool
    ot_e_only: bool
    proj_en_out: bool
    pdgid_is_idx: bool
    loss_sc_fac: float
    cond_cube: bool

    dl_conf: dict
    opt_conf: dict
    odeint_conf: dict
    val_conf: dict
    config: dict

    state_dict: dict | None

    # Reflow: when ``reflow_path`` points at a velocity-model checkpoint,
    # :class:`legofmt.main.modules.LEGOLtngDirect` loads it as a frozen
    # teacher and uses ``teacher.solve(base)`` as the per-batch training
    # target. ``reflow_kwargs`` is forwarded to ``teacher.solve``.
    reflow_path: str | None
    reflow_kwargs: dict


def resolve_legoltng_config(full_config: dict) -> ResolvedLEGOConfig:
    r"""Resolves a raw :class:`LEGOLtng` config into a typed form.

    Dispatches on the presence of a ``"state_dict"`` key in
    :attr:`full_config`:

    * If absent, the **fresh-training** path runs. It reads ``meta.json``
      from the dataset directory referenced by
      ``full_config["dl_conf"]["lds_args"]["data"]`` and populates
      ``npdgids``, ``max_seq_l``, and ``pdgids`` on the resolved copy.
    * If present, the **checkpoint-restore** path runs. It reads those
      fields back from ``full_config["config"]`` and applies any field
      migrations required by older checkpoint formats.

    Args:
        full_config (dict): either a flat dict (fresh-training form, no
            ``"state_dict"`` key) or a checkpoint dict of the shape
            ``{"state_dict": ..., "config": {...}}``.

    Returns:
        ResolvedLEGOConfig: the fully-resolved configuration.

    Raises:
        ValueError: if the fresh-training path receives a ``.pt`` file
            where it expected a dataset directory, or if the manifold spec
            cannot be parsed.
        FileNotFoundError: if the fresh-training path cannot locate
            ``meta.json`` under the configured dataset directory.
        KeyError: if a required config key is missing.

    .. note::
        :attr:`full_config` is deep-copied up front and is not mutated by
        this function. The mutated copy is exposed on the returned
        :class:`ResolvedLEGOConfig` as :attr:`~ResolvedLEGOConfig.config`.
    """
    full = copy.deepcopy(full_config)
    state_dict = full.get("state_dict")
    config = full.get("config", full)

    if state_dict is None:
        return _resolve_fresh(config)
    return _resolve_from_checkpoint(config, state_dict)


def _resolve_fresh(config: dict) -> ResolvedLEGOConfig:
    r"""Resolves a fresh-training config by reading dataset metadata.

    Reads ``meta.json`` under the dataset directory and writes the derived
    ``npdgids``, ``max_seq_l``, ``ntypes``, ``pdgids``, and ``data_path``
    onto the local config copy. Existing values for ``max_seq_l`` and
    ``ntypes`` are preserved (``setdefault`` semantics).

    Args:
        config (dict): the deep-copied inner config dict produced by
            :func:`resolve_legoltng_config`.

    Returns:
        ResolvedLEGOConfig: the resolved config with
            :attr:`~ResolvedLEGOConfig.state_dict` set to ``None``.

    Raises:
        ValueError: if ``dl_conf.lds_args.data`` points at a ``.pt`` file
            rather than a directory containing ``meta.json``.
        FileNotFoundError: if ``meta.json`` does not exist under the
            dataset directory.
    """
    model_conf = config["model_conf"]
    model_args = model_conf["model_args"]
    dpath = config["dl_conf"]["lds_args"]["data"]

    if dpath.endswith(".pt"):
        raise ValueError(
            "Fresh-training path expects a directory containing meta.json; "
            f"got a .pt file: {dpath}"
        )

    config["dl_conf"].setdefault("data_path", f"{dpath}/data_prepped.pt")

    meta = json.loads(Path(dpath, "meta.json").read_text())
    max_seq_l = meta["ntokens"]
    pdgids = (
        torch.tensor(meta["particles"], dtype=torch.int64).sort().values.contiguous()
    )

    model_args["npdgids"] = pdgids.shape[0] + 1
    model_args.setdefault("max_seq_l", max_seq_l)
    model_args.setdefault("ntypes", 4)
    # ``pdgids`` lives at model_conf scope (one level above model_args) so
    # it is preserved by the manual torch.save round-trip in scripts/train.py.
    model_conf["pdgids"] = pdgids

    return _build_resolved(
        config,
        model_conf,
        model_args,
        max_seq_l,
        pdgids,
        state_dict=None,
    )


def _resolve_from_checkpoint(config: dict, state_dict: dict) -> ResolvedLEGOConfig:
    r"""Resolves a config loaded from a saved checkpoint.

    Reads ``max_seq_l`` and ``pdgids`` from the serialized config and
    applies any field migrations required by older checkpoint formats
    (see :func:`_apply_legacy_projection_in_out` and the
    ``ntokens`` -> ``max_seq_l`` rename below).

    Args:
        config (dict): the deep-copied inner config dict produced by
            :func:`resolve_legoltng_config`.
        state_dict (dict): the model state dict from the checkpoint. Used
            both to detect legacy layouts and to be loaded into
            ``LEGOLtng.model.vf`` after construction.

    Returns:
        ResolvedLEGOConfig: the resolved config with
            :attr:`~ResolvedLEGOConfig.state_dict` populated.
    """
    model_conf = config["model_conf"]
    model_args = model_conf["model_args"]

    # Legacy field rename: pre-refactor checkpoints used "ntokens" for what
    # is now called "max_seq_l".
    if "ntokens" in model_args:
        model_args["max_seq_l"] = model_args.pop("ntokens")

    _apply_legacy_projection_in_out(model_args, state_dict)

    max_seq_l = model_args["max_seq_l"]
    pdgids = model_conf["pdgids"]
    return _build_resolved(
        config,
        model_conf,
        model_args,
        max_seq_l,
        pdgids,
        state_dict=state_dict,
    )


def _apply_legacy_projection_in_out(model_args: dict, state_dict: dict) -> None:
    r"""Re-enables ``project_in`` / ``project_out`` layers for legacy checkpoints.

    Older checkpoints were saved with ``vf.project_in.*`` and
    ``vf.project_out.*`` linear layers in the state dict. The current
    :class:`~legofmt.cfm.cfm_trafo_x.CFMTrafo_x` only constructs those
    layers when its :attr:`dim_in_out` argument is non-``None``. When such
    keys are detected in the incoming :attr:`state_dict`, this function
    forces ``model_args["dim_in_out"] = model_args["h_dim"]`` so that
    :class:`CFMTrafo_x` rebuilds matching layers and the subsequent
    :meth:`~torch.nn.Module.load_state_dict` succeeds.

    Args:
        model_args (dict): the ``model_args`` sub-dict on the local config
            copy. Modified in place if legacy keys are detected.
        state_dict (dict): the model state dict from the checkpoint.
    """
    if any(k.startswith("vf.project_in.") for k in state_dict):
        model_args["dim_in_out"] = model_args["h_dim"]


def _build_resolved(
    config: dict,
    model_conf: dict,
    model_args: dict,
    max_seq_l: int,
    pdgids: torch.Tensor,
    state_dict: dict | None,
) -> ResolvedLEGOConfig:
    r"""Assembles a :class:`ResolvedLEGOConfig` from the resolved inputs.

    Pulled out of the two path-specific resolvers so the field mapping
    lives in one place. Applies default values for all optional
    ``model_conf`` keys via :meth:`dict.get` so the defaulting policy is
    consistent across construction paths.

    Args:
        config (dict): the (mutated) local inner config dict.
        model_conf (dict): ``config["model_conf"]``, passed explicitly to
            avoid re-indexing.
        model_args (dict): ``config["model_conf"]["model_args"]``, ditto.
        max_seq_l (int): maximum sequence length.
        pdgids (Tensor): sorted ``int64`` tensor of known PDG ids.
        state_dict (dict or None): the state dict for checkpoint restore,
            or ``None`` for fresh training.

    Returns:
        ResolvedLEGOConfig: the assembled, frozen configuration.
    """
    return ResolvedLEGOConfig(
        max_seq_l=max_seq_l,
        pdgids_template=pdgids.contiguous(),
        manifold=build_manifold(model_conf["manifold"]),
        model_args=model_args,
        t_dist=model_conf.get("t_dist", "uniform"),
        t_dist_scale=model_conf.get("t_dist_scale", 1.4),
        ot_coupling=model_conf.get("ot_coupling", False),
        ot_e_only=model_conf.get("ot_e_only", False),
        proj_en_out=model_conf.get("proj_en_out", False),
        pdgid_is_idx=model_conf.get("pdgid_is_idx", False),
        loss_sc_fac=model_conf.get("loss_sc", 0.0),
        cond_cube=model_conf.get("cond_cube", False),
        dl_conf=config["dl_conf"],
        opt_conf=config["opt_conf"],
        odeint_conf=config.get("odeint_conf", {}),
        val_conf=config.get("val_conf", {}),
        config=config,
        state_dict=state_dict,
        reflow_path=model_conf.get("reflow_path"),
        reflow_kwargs=model_conf.get("reflow_kwargs", {}),
    )
