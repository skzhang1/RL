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

"""The frozen controlled-math-v2 Dense schedule, including its first-step convention."""

import math
import torch


class ControlledAdamScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Historical schedule exposed as a class for NeMo's Hydra loader."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        steps: int,
        warmup: int,
        min_lr_ratio: float,
    ) -> None:
        if not 0 < warmup < steps or not 0 <= min_lr_ratio <= 1:
            raise ValueError("Invalid controlled-math schedule")

        def factor(index: int) -> float:
            if index < warmup:
                return (index + 1) / warmup
            progress = min((index - warmup + 1) / (steps - warmup), 1.0)
            return (
                min_lr_ratio
                + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
            )

        super().__init__(optimizer, factor)
