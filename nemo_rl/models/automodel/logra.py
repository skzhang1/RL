# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Low-rank gradient accumulation primitives for the optional LoGRA backend.

The frozen weight participates in the ordinary forward and input derivative.
Only G @ A.T is accumulated; no full weight gradient is constructed. Projection
seeds and RowAdam arithmetic follow the original RPGA implementation.
"""

import hashlib
import math
from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, overload

import torch
import torch.distributed as dist
from pydantic import BaseModel, Field
from torch import Tensor, nn
from torch.distributed.tensor import DTensor, Shard
from torch.utils.hooks import RemovableHandle


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class LoGRAConfig(BaseModel, extra="forbid"):
    """Opt-in configuration; absence of this block leaves dense training intact."""

    rank: int = Field(default=256, gt=0)
    seed: int = 42
    distribution: Literal["rademacher", "gaussian"] = "rademacher"
    refresh: bool = True
    # Historical main runs round the final factor to BF16 before merging it.
    update_factor_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    beta2: float = Field(default=0.95, ge=0, lt=1)
    epsilon: float = Field(default=1e-8, gt=0)
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    # Ratio of the proposed update norm to the selected weight norm, before KL scaling.
    update_ratio: float = Field(default=6e-5, gt=0)
    kl_budget: float = Field(default=2e-4, gt=0)
    kl_budget_final: float = Field(default=2e-5, gt=0)
    schedule_steps: int = Field(default=300, gt=0)
    alpha_max: float = Field(default=3.0, gt=0)
    probe_responses: int = Field(default=2, gt=0)
    probe_max_tokens: int = Field(default=4096, gt=0)
    mismatch_subtraction: bool = True
    mismatch_ema_weight: float = Field(default=0.3, gt=0, le=1)
    mismatch_budget_floor: float = Field(default=0.5, gt=0, le=1)


def projection_seed(base_seed: int, update: int, name: str) -> int:
    """Generate the original implementation's stable per-layer seed."""
    wrappers = {"_checkpoint_wrapped_module", "_fsdp_wrapped_module", "_orig_mod"}
    name = ".".join(part for part in name.split(".") if part not in wrappers)
    digest = int.from_bytes(
        hashlib.blake2b(name.encode(), digest_size=8).digest(), "little"
    )
    return (base_seed * 1_000_003 + update * 7_919 + digest) & ((1 << 62) - 1)


def make_projection(
    rank: int,
    width: int,
    seed: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    distribution: str,
) -> Tensor:
    """Draw on CPU so the same seed describes the same matrix on every device."""
    if rank <= 0 or width <= 0:
        raise ValueError("Projection dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed & ((1 << 62) - 1))
    if distribution == "rademacher":
        bits = torch.randint(
            0, 2, (rank, width), generator=generator, dtype=torch.uint8
        )
        result = bits.to(dtype).mul_(2).sub_(1).mul_(1 / math.sqrt(rank))
    elif distribution == "gaussian":
        result = torch.randn(rank, width, generator=generator, dtype=torch.float64)
        result = result.mul_(1 / math.sqrt(rank)).to(dtype)
    else:
        raise ValueError(f"Unsupported projection distribution: {distribution}")
    return result.to(device)


@torch.no_grad()
def row_adam_direction(
    sketch: Tensor,
    second_moment: Tensor,
    *,
    step: int,
    beta2: float,
    epsilon: float,
) -> Tensor:
    """Update row statistics and preserve each layer's original sketch norm.

    The second moment is indexed by output row, so it survives projection refresh.
    There is no first-moment buffer. This includes the original implementation's
    RMS normalization of the inverse row scale before layer-norm restoration.
    """
    if step < 1 or not 0 <= beta2 < 1 or epsilon <= 0:
        raise ValueError("Invalid RowAdam step, decay, or epsilon")
    if second_moment.shape != (sketch.shape[0], 1):
        raise ValueError("RowAdam requires one second moment per output row")
    if not torch.isfinite(sketch).all():
        raise FloatingPointError("Nonfinite gradient sketch")
    tiny = torch.finfo(sketch.dtype).tiny
    second_moment.mul_(beta2).add_(
        sketch.square().mean(1, keepdim=True), alpha=1 - beta2
    )
    root = (second_moment / (1 - beta2**step)).sqrt()
    denominator = root + (epsilon * root.mean()).clamp_min(tiny)
    # Dividing min(denominator) by denominator is proportional to its inverse,
    # but avoids squaring a near-float-max inverse for an exactly zero sketch.
    scale = denominator.min() / denominator
    scale = scale / scale.square().mean().sqrt().clamp_min(tiny)
    direction = scale * sketch
    return direction * (sketch.norm() / direction.norm().clamp_min(tiny))


def predicted_kl(logits: Tensor, tangent: Tensor, *, temperature: float) -> Tensor:
    """Second-order KL per prediction from a simultaneous logit direction."""
    if logits.shape != tangent.shape or temperature <= 0:
        raise ValueError(
            "KL prediction requires matching shapes and positive temperature"
        )
    probabilities = (logits.double() / temperature).softmax(-1)
    direction = tangent.double() / temperature
    centered = direction - (probabilities * direction).sum(-1, keepdim=True)
    return 0.5 * (probabilities * centered.square()).sum(-1)


def kl_step_scale(prediction: float, *, budget: float, alpha_max: float) -> float:
    """Scale to a predicted KL budget; invalid predictions must not silently update."""
    if not math.isfinite(prediction) or prediction < 0:
        raise FloatingPointError("Invalid predicted KL")
    if budget <= 0 or alpha_max <= 0 or not math.isfinite(budget + alpha_max):
        raise ValueError("KL budget and alpha cap must be positive and finite")
    if prediction == 0:
        return 1.0  # Matches the source controller's zero-direction fallback.
    return min(alpha_max, math.sqrt(budget / prediction))


@dataclass
class SketchState:
    """Per-layer state stored outside autograd and explicitly reduced by the trainer."""

    name: str
    module: nn.Linear
    projection: Tensor
    sketch: Tensor
    second_moment: Tensor
    handle: RemovableHandle | None

    def forward_hook(
        self, module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor
    ) -> None:
        if not torch.is_grad_enabled() or not output.requires_grad:
            return
        # Non-reentrant checkpoint replay reconstructs backward intermediates;
        # its outputs are not differentiated a second time.
        if torch._C._current_graph_task_id() != -1:
            return
        with torch.no_grad():
            activation = inputs[0]
            projected = (
                activation.reshape(-1, activation.shape[-1])
                @ self.projection.to(activation.dtype).T
            )

        def accumulate(gradient: Tensor) -> None:
            with torch.no_grad():
                contribution = gradient.reshape(-1, gradient.shape[-1]).T @ projected
                self.sketch.add_(contribution.float())

        output.register_hook(accumulate)


def install_sketches(model: nn.Module, config: LoGRAConfig) -> list[SketchState]:
    """Install sketches on explicit linear targets, leaving the model state_dict clean.

    Supported initial scope is dense Hugging Face linear layers and non-reentrant
    activation checkpointing. Call before freezing unrelated parameters. A frozen
    embedding output must be made differentiable by the training integration.
    """
    selected = [
        (name, module)
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in config.target_modules
    ]
    if not selected:
        raise ValueError("LoGRA target_modules matched no modules")
    for name, module in selected:
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"LoGRA requires nn.Linear targets, got {name}: {type(module)}"
            )
    states = []
    for name, module in selected:
        weight = module.weight
        projection = make_projection(
            config.rank,
            module.in_features,
            projection_seed(config.seed, 0, name),
            device=weight.device,
            dtype=torch.float32,
            distribution=config.distribution,
        )
        state = SketchState(
            name,
            module,
            projection,
            torch.zeros(
                module.out_features,
                config.rank,
                device=weight.device,
                dtype=torch.float32,
            ),
            torch.zeros(
                module.out_features, 1, device=weight.device, dtype=torch.float32
            ),
            None,
        )
        weight.requires_grad_(False)
        state.handle = module.register_forward_hook(state.forward_hook)
        states.append(state)
    return states


def local_weight_rows(weight: Tensor) -> tuple[Tensor, int]:
    """Return the local weight and global row offset, rejecting other sharding."""
    if not isinstance(weight, DTensor):
        return weight, 0
    if any(isinstance(p, Shard) and p.dim != 0 for p in weight.placements):
        raise ValueError("LoGRA supports FSDP row sharding, not tensor parallelism")
    # PyTorch's helper accounts for uneven and empty FSDP shards.
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    _, offset = compute_local_shape_and_global_offset(
        weight.shape, weight.device_mesh, weight.placements
    )
    return weight.to_local(), offset[0]


class LoGRAOptimizer(torch.optim.Optimizer):
    """RowAdam over replicated sketches with row-sharded FP32 weight updates.

    The worker prepares a direction after backward, estimates its predicted KL,
    and applies the scaled update. Only row moments and the update counter persist.
    """

    def __init__(self, model: "PreTrainedModel", config: LoGRAConfig) -> None:
        self.config = config
        self.layers = install_sketches(model, config)
        weights = [layer.module.weight for layer in self.layers]
        for weight in weights:
            local, _ = local_weight_rows(weight)
            if local.dtype != torch.float32:
                raise ValueError("LoGRA requires FP32 storage with BF16 computation")
        super().__init__(
            weights,
            {
                "lr": 1.0,
                "logra_step": 0,
                "logra_config": config.model_dump(),
                "logra_mismatch": 0.0,
            },
        )
        model.requires_grad_(False)
        self.anchor_handle = model.get_input_embeddings().register_forward_hook(
            self._anchor
        )
        self.directions: list[Tensor] = []
        self.prepared = False
        self.mismatch = 0.0
        for layer in self.layers:
            self.state[layer.module.weight]["row_second_moment"] = layer.second_moment

    @staticmethod
    def _anchor(
        module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor
    ) -> Tensor:
        if torch.is_grad_enabled():
            output.requires_grad_(True)
        return output

    @property
    def update_count(self) -> int:
        return self.param_groups[0]["logra_step"]

    @property
    def mismatch(self) -> float:
        return self.param_groups[0]["logra_mismatch"]

    @mismatch.setter
    def mismatch(self, value: float) -> None:
        self.param_groups[0]["logra_mismatch"] = value

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        if self.prepared:
            raise RuntimeError("Cannot discard a prepared LoGRA update")
        for layer in self.layers:
            layer.sketch.zero_()

    @torch.no_grad()
    def prepare_update(
        self, *, process_group: dist.ProcessGroup | None, normalization_tokens: int
    ) -> float:
        """Average DP sketches and normalize the proposed update to a weight ratio."""
        if normalization_tokens < 1:
            raise ValueError("LoGRA normalization requires valid generated tokens")
        if self.prepared:
            raise RuntimeError("LoGRA update already prepared")
        distributed = dist.is_initialized()
        world_size = dist.get_world_size(process_group) if distributed else 1
        self.directions = []
        totals = torch.zeros(
            3, device=self.layers[0].sketch.device, dtype=torch.float64
        )
        for layer in self.layers:
            if distributed:
                dist.all_reduce(layer.sketch, group=process_group)
                # NeMo RL multiplies microbatch losses by DP size before backward.
                layer.sketch.div_(world_size)
            direction = row_adam_direction(
                layer.sketch / normalization_tokens,
                layer.second_moment,
                step=self.update_count + 1,
                beta2=self.config.beta2,
                epsilon=self.config.epsilon,
            )
            self.directions.append(direction)
            local, offset = local_weight_rows(layer.module.weight)
            totals[0] += local.double().square().sum()
            totals[2] += layer.sketch.double().square().sum() / world_size
            for start in range(0, local.shape[0], 256):
                stop = min(start + 256, local.shape[0])
                update = direction[offset + start : offset + stop] @ layer.projection
                totals[1] += update.double().square().sum()
        if distributed:
            dist.all_reduce(totals, group=process_group)
        if not torch.isfinite(totals).all():
            raise FloatingPointError("Nonfinite LoGRA update or weight norm")
        if totals[1] > 0:
            factor = self.config.update_ratio * (totals[0] / totals[1]).sqrt().item()
            for direction in self.directions:
                direction.mul_(factor)
        self.prepared = True
        return totals[2].sqrt().item()

    @torch.no_grad()
    def apply_update(self, alpha: float) -> None:
        """Apply a scaled low-rank update, then refresh the projection."""
        if not self.prepared:
            raise RuntimeError("prepare_update must precede apply_update")
        if not math.isfinite(alpha) or alpha < 0:
            raise FloatingPointError("Invalid LoGRA update scale")
        for layer, direction in zip(self.layers, self.directions):
            scaled = (
                (direction * alpha)
                .to(getattr(torch, self.config.update_factor_dtype))
                .float()
            )
            local, offset = local_weight_rows(layer.module.weight)
            for start in range(0, local.shape[0], 256):
                stop = min(start + 256, local.shape[0])
                update = scaled[offset + start : offset + stop] @ layer.projection
                local[start:stop].add_(update, alpha=-1.0)
        self.param_groups[0]["logra_step"] += 1
        if self.config.refresh:
            self._refresh()
        self.directions = []
        self.prepared = False

    @overload
    def step(self, closure: None = None, *, alpha: float | None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float], *, alpha: float | None = None) -> float: ...

    def step(
        self, closure: Callable[[], float] | None = None, *, alpha: float | None = None
    ) -> float | None:
        if closure is not None or alpha is None:
            raise ValueError("LoGRA step requires an explicit controller scale")
        self.apply_update(alpha)

    def _refresh(self) -> None:
        for layer in self.layers:
            layer.projection.copy_(
                make_projection(
                    self.config.rank,
                    layer.module.in_features,
                    projection_seed(self.config.seed, self.update_count, layer.name),
                    device=layer.projection.device,
                    dtype=layer.projection.dtype,
                    distribution=self.config.distribution,
                )
            )

    def state_dict(self) -> dict:
        if self.prepared:
            raise RuntimeError("Checkpoint only after a completed LoGRA update")
        # DCP preserves optimizer state and param_groups, not arbitrary top-level keys.
        return super().state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        if state_dict["param_groups"][0]["logra_config"] != self.config.model_dump():
            raise ValueError("LoGRA checkpoint configuration does not match")
        super().load_state_dict(state_dict)
        for layer in self.layers:
            layer.second_moment = self.state[layer.module.weight]["row_second_moment"]
        if self.config.refresh:
            self._refresh()
