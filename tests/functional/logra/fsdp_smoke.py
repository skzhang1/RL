"""Run with torchrun --standalone --nproc_per_node=2; no model downloads."""

import copy
import os
import tempfile

import torch.distributed.checkpoint as dcp

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

from nemo_rl.models.automodel.logra import (
    LoGRAConfig,
    LoGRAOptimizer,
    row_adam_direction,
)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13, 7)
        self.q_proj = nn.Linear(7, 11, bias=False)
        self.o_proj = nn.Linear(11, 7, bias=False)
        self.head = nn.Linear(7, 13, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, ids):
        return self.head(self.o_proj(self.q_proj(self.embedding(ids)).tanh()))


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, size = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(19)
    model = TinyPolicy().cuda()
    reference = copy.deepcopy(model)
    mesh = init_device_mesh("cuda", (size,))
    fully_shard(model.q_proj, mesh=mesh)
    fully_shard(model.o_proj, mesh=mesh)
    fully_shard(model, mesh=mesh)
    optimizer = LoGRAOptimizer(
        model, LoGRAConfig(rank=3, update_ratio=0.01, update_factor_dtype="float32")
    )
    ids = torch.tensor([[1, 2, 3], [4, 5, 6], [3, 7, 5], [2, 8, 9]], device="cuda")
    reference(ids).square().sum().div(ids.numel()).backward()
    local = ids.chunk(size)[rank]
    model(local).square().sum().mul(size / ids.numel()).backward()
    optimizer.prepare_update(process_group=dist.group.WORLD, normalization_tokens=1)
    expected_directions = []
    for layer, direction in zip(optimizer.layers, optimizer.directions):
        original = reference.get_submodule(layer.name)
        expected_sketch = original.weight.grad @ layer.projection.T
        torch.testing.assert_close(layer.sketch, expected_sketch, rtol=2e-5, atol=1e-6)
        assert layer.module.weight.grad is None
        expected = row_adam_direction(
            expected_sketch,
            torch.zeros_like(layer.second_moment),
            step=1,
            beta2=0.95,
            epsilon=1e-8,
        )
        expected_directions.append(expected @ layer.projection)
    ratio = 0.01 * torch.sqrt(
        sum(
            reference.get_submodule(x.name).weight.square().sum()
            for x in optimizer.layers
        )
        / sum(x.square().sum() for x in expected_directions)
    )
    with torch.no_grad():
        for layer, delta in zip(optimizer.layers, expected_directions):
            reference.get_submodule(layer.name).weight.add_(delta, alpha=-ratio.item())
    optimizer.apply_update(1.0)
    with torch.no_grad():
        torch.testing.assert_close(model(ids), reference(ids), rtol=3e-5, atol=2e-6)
    print(
        f"RANK {rank}: FSDP sketch and weight update match dense reference", flush=True
    )
    optimizer.mismatch = 0.000013
    optimizer_state = optimizer
    paths = [tempfile.mkdtemp(prefix="logra-dcp-") if rank == 0 else None]
    dist.broadcast_object_list(paths, src=0)
    saved_moments = [x.second_moment.clone() for x in optimizer.layers]
    saved_projections = [x.projection.clone() for x in optimizer.layers]
    dcp.save({"optimizer": optimizer_state}, checkpoint_id=paths[0])
    optimizer.param_groups[0]["logra_step"] = 0
    optimizer.mismatch = 0.0
    for layer in optimizer.layers:
        layer.second_moment.zero_()
    optimizer._refresh()
    dcp.load({"optimizer": optimizer_state}, checkpoint_id=paths[0])
    assert optimizer.update_count == 1
    assert optimizer.mismatch == 0.000013
    for layer, moment, projection in zip(
        optimizer.layers, saved_moments, saved_projections
    ):
        torch.testing.assert_close(layer.second_moment, moment, rtol=0, atol=0)
        torch.testing.assert_close(layer.projection, projection, rtol=0, atol=0)
    print(f"RANK {rank}: LoGRA DCP state restored exactly", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
