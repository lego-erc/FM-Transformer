"""Superseded dataset inputs: the padded tensors and the raw particles.

A packed file (``legofmt.data.load.packed``) holds a dataset without its padding. Before it, a dataset was the padded
tensors ``(f, m, am[, edep_layers])``: a ``data_prepped.pt`` file, the same tuple in memory, or a raw dict that is cut
and prepped on the spot. ``PaddedData`` holds them and answers the same calls as a ``PackedData``, which is what
``load_dataset`` returns for these inputs.
"""

import torch
from torch import Tensor

from legofmt.data.write.cut import GetLEGOData
from legofmt.data.struct import DataStruct, cond_slot, n_prefix


class PaddedData(DataStruct):
    """The padded tensors of a whole dataset, with the calls of a ``PackedData``."""

    @classmethod
    def read(cls, data: dict | tuple, prep, frac: float | None, device, *, inplace: bool, **cut_args) -> "PaddedData":
        """Constructs the dataset from the padded tensors or from raw particles.

        Args:
            data:     the padded tensors ``(f, m, am[, edep_layers])``, stored in MeV; or a raw dict, which is cut
                (``GetLEGOData``, taking ``cut_args``) and prepped here.
            prep:     the ``DataPrep``: its ``norm_e`` puts padded tensors on the model scale, the whole of it preps
                raw particles. None: the padded tensors are read as stored.
            frac:     keep a fixed random fraction of the events.
            device:   of the raw particles' result and of ``frac``'s generator.
            inplace:  the tensors are a fresh load nobody else holds: normalise them in place.

        Raises:
            ValueError: a raw dict without a ``prep``.
        """
        if isinstance(data, dict):   # raw particles
            if prep is None:
                raise ValueError(
                    "load_dataset(dict, ...) requires `prep` (e.g. DataPrep(config)); "
                    "GetLEGOData yields pre-format_add layout that DataStruct misaligns."
                )
            data = prep(GetLEGOData(device=device, **cut_args)(data))
        elif prep is not None:
            data = prep.norm_e(data, inplace=inplace) if hasattr(prep, "norm_e") else prep(data)
        self = cls(*data)
        if frac and frac < 1.0:   # the same events every run
            n = int(len(self) * frac)
            self = self[torch.randperm(len(self), device=device,
                                       generator=torch.Generator(device=device).manual_seed(0))[:n]]
        return self

    @property
    def data(self) -> "PaddedData":
        return self

    @property
    def lengths(self) -> Tensor:
        return self.am.full.sum(1)

    @property
    def n_out(self) -> Tensor:
        return self.am.out_p.sum(-1)

    @property
    def prefix(self) -> Tensor:
        return self.f.full[:, :n_prefix(), 0]

    @property
    def edep(self) -> Tensor:
        return self.prefix[:, 1]

    def cond(self, name: str) -> Tensor:
        return self.prefix[:, cond_slot(name)]

    @property
    def incoming(self) -> Tensor:
        return self.f.in_p[..., 0, :]

    def species_counts(self, ptypes: Tensor) -> Tensor:
        pid = self.f.out_p[..., -1]  # one species at a time: the broadcast compare made an (N, max_out, S) int64
        return torch.stack([(pid == p).sum(1) for p in ptypes], -1)
