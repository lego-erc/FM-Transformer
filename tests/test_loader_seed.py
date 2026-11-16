"""Under DDP every rank must build the same shuffled batch list.

Lightning wraps ``make_loader``'s batch sampler in ``DistributedSamplerWrapper``,
which hands rank ``r`` every W-th batch of the list *that rank* built. With the
shuffle drawn from each rank's own global RNG (unseeded runs, or rank 0 after
pretraining the base head), the shards overlapped: on 2 GPUs a quarter of every
epoch was never seen and another quarter was seen twice.
"""

import pytest
import torch
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper

from legofmt.data.dataloaders import loader_seed, make_loader


@pytest.mark.parametrize("wrapper_shuffle", [False, True])
def test_ddp_ranks_partition_every_epoch(wrapper_shuffle: bool) -> None:
    N, W, bs = 4096, 2, 64
    seen = [[], []]  # per epoch
    for rank in range(W):
        torch.manual_seed(1000 + rank)  # per-rank global RNG state
        loader = make_loader(range(N), bs=bs, shuffle=True, num_workers=0, batched_sampler=True, seed=3)
        sampler = DistributedSamplerWrapper(loader.sampler, num_replicas=W, rank=rank, shuffle=wrapper_shuffle)
        for epoch in range(2):
            sampler.set_epoch(epoch)
            seen[epoch] += [i for b in sampler for i in b]
    for epoch, s in enumerate(seen):
        assert sorted(s) == list(range(N)), f"epoch {epoch}: overlapping shards"


def test_loader_seed_resolution(monkeypatch) -> None:
    monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)
    assert loader_seed({}) == 0
    monkeypatch.setenv("PL_GLOBAL_SEED", "7")
    assert loader_seed({}) == 7 and loader_seed({"seed": 2}) == 2
