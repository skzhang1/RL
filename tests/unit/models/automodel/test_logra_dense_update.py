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

"""Validate that each Dense recipe retains gradients and actually updates weights."""

from pathlib import Path

import pytest
import torch
import yaml
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm


@pytest.mark.parametrize("seed", [1, 2])
def test_dense_recipe_preserves_gradients_and_updates_weights(seed: int) -> None:
    root = Path(__file__).resolve().parents[4]
    config = yaml.safe_load(
        (root / f"research/logra/configs/dense-s{seed}.yaml").read_text()
    )
    policy = config["policy"]
    # AutoModel interprets zero as clipping every gradient to zero; None disables it.
    assert policy["max_grad_norm"] is None
    model = torch.nn.Linear(2, 1, bias=False)
    before = model.weight.detach().clone()
    gradient = torch.tensor([[3.0, 4.0]])
    model.weight.grad = gradient.clone()
    optimizer = torch.optim.AdamW(model.parameters(), **policy["optimizer"]["kwargs"])
    scale_grads_and_clip_grad_norm(
        policy["max_grad_norm"],
        [model],
        pp_enabled=False,
        num_label_tokens=1,
        dp_group_size=4,
    )
    torch.testing.assert_close(model.weight.grad, gradient, rtol=0, atol=0)
    optimizer.step()
    assert torch.all(model.weight.detach() < before)
