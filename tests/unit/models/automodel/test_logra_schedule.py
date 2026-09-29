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

"""Exercise the production class loader, schedule values, and state restoration."""

import copy
import math

import pytest
import torch
from hydra.utils import get_class


def test_controlled_schedule_load_values_and_resume() -> None:
    scheduler_class = get_class("research.logra.schedule.ControlledAdamScheduler")
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [scheduler_class(optimizer, steps=300, warmup=9, min_lr_ratio=0.1)],
        milestones=[],
    )
    for index in range(301):
        expected = (
            (index + 1) / 9
            if index < 9
            else 0.1 + 0.9 * (1 + math.cos(math.pi * min((index - 8) / 291, 1))) / 2
        )
        assert optimizer.param_groups[0]["lr"] == pytest.approx(
            1e-6 * expected, rel=1e-13
        )
        if index == 137:
            optimizer_state = copy.deepcopy(optimizer.state_dict())
            scheduler_state = copy.deepcopy(scheduler.state_dict())
        optimizer.step()
        scheduler.step()
    resumed_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-6)
    resumed_scheduler = torch.optim.lr_scheduler.SequentialLR(
        resumed_optimizer,
        [scheduler_class(resumed_optimizer, steps=300, warmup=9, min_lr_ratio=0.1)],
        milestones=[],
    )
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)
    for _ in range(137, 301):
        resumed_optimizer.step()
        resumed_scheduler.step()
    assert resumed_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
