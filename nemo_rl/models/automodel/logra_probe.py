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

"""Simultaneous factorized logit JVP for LoGRA's predicted-KL controller."""

import math
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import torch
import torch.autograd.forward_ad as forward_ad
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from nemo_rl.models.automodel.logra import LoGRAOptimizer, kl_step_scale, predicted_kl


if TYPE_CHECKING:
    from transformers import PreTrainedModel


@contextmanager
def factorized_direction(optimizer: LoGRAOptimizer) -> Iterator[None]:
    """Inject all layer directions in the same forward AD pass, without changing weights."""
    handles = []

    def make_hook(direction: Tensor, projection: Tensor):
        def hook(
            module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor
        ) -> Tensor:
            primal, tangent = forward_ad.unpack_dual(output)
            activation = forward_ad.unpack_dual(inputs[0]).primal
            # The historical probe computes factor products in FP32, then
            # casts the tangent to the layer output dtype. Preserve this even
            # when the HF forward runs inside an autocast context.
            with torch.autocast(device_type=activation.device.type, enabled=False):
                delta = -((activation.float() @ projection.T) @ direction.T)
            delta = delta.to(primal.dtype)
            tangent = delta if tangent is None else tangent + delta
            return forward_ad.make_dual(primal, tangent)

        return hook

    try:
        for layer, direction in zip(optimizer.layers, optimizer.directions):
            handles.append(
                layer.module.register_forward_hook(
                    make_hook(direction, layer.projection)
                )
            )
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def predict_batch_kl(
    model: "PreTrainedModel",
    optimizer: LoGRAOptimizer,
    batch: dict[str, Tensor],
    *,
    temperature: float,
    compute_dtype: torch.dtype,
    process_group: dist.ProcessGroup | None,
) -> float:
    """Predict average KL on generated positions of the first probe responses per rank.

    The same count is used on every rank so FSDP forward collectives stay aligned.
    HF eager attention is restored even if the forward fails. Invalid probes raise
    rather than silently substituting an uncontrolled update.
    """
    count = torch.tensor(
        min(optimizer.config.probe_responses, batch["input_ids"].shape[0]),
        device=batch["input_ids"].device,
    )
    if dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.MIN, group=process_group)
    total = torch.zeros(2, device=count.device, dtype=torch.float64)
    previous_attention = model.config._attn_implementation
    training = model.training
    try:
        model.eval()
        model.set_attn_implementation("eager")
        with (
            sdpa_kernel([SDPBackend.MATH]),
            factorized_direction(optimizer),
            forward_ad.dual_level(),
        ):
            for index in range(int(count)):
                length = min(
                    int(batch["input_lengths"][index]),
                    optimizer.config.probe_max_tokens,
                )
                ids = batch["input_ids"][index : index + 1, :length]
                with torch.autocast("cuda", dtype=compute_dtype):
                    output = model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        use_cache=False,
                    )
                logits, tangent = forward_ad.unpack_dual(output.logits)
                if tangent is None:
                    raise RuntimeError("LoGRA JVP was lost in a model operation")
                mask = batch["token_mask"][index, 1:length].bool()
                mask &= batch["sample_mask"][index].bool()
                positions = mask.nonzero().flatten()
                for chunk in positions.split(64):
                    values = predicted_kl(
                        logits[0, chunk], tangent[0, chunk], temperature=temperature
                    )
                    total[0] += values.sum()
                    total[1] += values.numel()
                del output, logits, tangent
    finally:
        model.set_attn_implementation(previous_attention)
        model.train(training)
    if dist.is_initialized():
        dist.all_reduce(total, group=process_group)
    if total[1] == 0:
        raise RuntimeError("No valid generated-token positions in the LoGRA probe")
    prediction = (total[0] / total[1]).item()
    if not math.isfinite(prediction) or prediction < 0:
        raise FloatingPointError("Invalid LoGRA predicted KL")
    return prediction


def scheduled_budget(optimizer: LoGRAOptimizer) -> float:
    """Match the historical main-experiment schedule: endpoints at updates 1 and H."""
    cfg = optimizer.config
    progress = min(optimizer.update_count / max(cfg.schedule_steps - 1, 1), 1.0)
    return cfg.kl_budget_final + 0.5 * (cfg.kl_budget - cfg.kl_budget_final) * (
        1 + math.cos(math.pi * progress)
    )


@torch.no_grad()
def control_step(
    model: "PreTrainedModel",
    optimizer: LoGRAOptimizer,
    batch: dict[str, Tensor],
    *,
    temperature: float,
    compute_dtype: torch.dtype,
    process_group: dist.ProcessGroup | None,
    eos_token_ids: set[int],
) -> dict[str, float]:
    """Estimate KL and apply the scheduled, mismatch-adjusted budget."""
    budget = scheduled_budget(optimizer)
    if optimizer.config.mismatch_subtraction:
        # NeMo RL stores both log-prob arrays aligned with their target tokens.
        ids = batch["input_ids"]
        last = ids.gather(1, (batch["input_lengths"] - 1).unsqueeze(1)).squeeze(1)
        completed = torch.zeros_like(last, dtype=torch.bool)
        for eos in eos_token_ids:
            completed |= last == eos
        mask = (
            batch["token_mask"].bool()
            & completed[:, None]
            & batch["sample_mask"][:, None].bool()
        )
        difference = batch["generation_logprobs"] - batch["prev_logprobs"]
        totals = torch.stack(
            (difference.masked_select(mask).double().sum(), mask.sum().double())
        )
        if dist.is_initialized():
            dist.all_reduce(totals, group=process_group)
        if totals[1] > 0:
            observed = (totals[0] / totals[1]).item()
            if not math.isfinite(observed):
                raise FloatingPointError("Nonfinite rollout/trainer mismatch")
            observed = max(0.0, observed)
            weight = optimizer.config.mismatch_ema_weight
            optimizer.mismatch = (1 - weight) * optimizer.mismatch + weight * observed
            budget = max(
                budget - optimizer.mismatch,
                budget * optimizer.config.mismatch_budget_floor,
            )
    prediction = predict_batch_kl(
        model,
        optimizer,
        batch,
        temperature=temperature,
        compute_dtype=compute_dtype,
        process_group=process_group,
    )
    alpha = kl_step_scale(
        prediction, budget=budget, alpha_max=optimizer.config.alpha_max
    )
    optimizer.step(alpha=alpha)
    return {
        "logra/predicted_kl": prediction,
        "logra/predicted_kl_scaled": prediction * alpha**2,
        "logra/alpha": alpha,
        "logra/effective_budget": budget,
        "logra/mismatch": optimizer.mismatch,
    }
