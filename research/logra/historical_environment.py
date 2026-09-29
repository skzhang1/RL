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

"""Benchmark-only adapter for the frozen LoGRA main-experiment reward function."""

from typing import Any
import ray
import torch
from nemo_rl.environments.math_environment import BaseMathEnvironment
from nemo_rl.environments.interfaces import EnvironmentReturn
from nemo_rl.environments.utils import chunk_list_to_workers
from research.logra.historical_math_grader import score_response


@ray.remote  # pragma: no cover
class HistoricalVerifyWorker:
    def verify(self, records: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
        """Use the original query/prompt/label contract, including answer fallbacks."""
        return [
            score_response(query, prompt, answer) for query, prompt, answer in records
        ]


@ray.remote(max_restarts=0)  # pragma: no cover
class HistoricalMathEnvironment(BaseMathEnvironment):
    WORKER_CLASS_DICT = {"math": HistoricalVerifyWorker}

    def step(
        self,
        message_log_batch: list,
        metadata: list,
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn:
        records = []
        for messages, info in zip(message_log_batch, metadata):
            prompt = "".join(str(m["content"]) for m in messages if m["role"] == "user")
            response = "".join(
                str(m["content"]) for m in messages if m["role"] == "assistant"
            )
            records.append((prompt + response, prompt, info["ground_truth"]))
        chunks = chunk_list_to_workers(records, self.num_workers)
        start = next(self._worker_counter)
        futures = [
            self.workers[(start + i) % self.num_workers].verify.remote(chunk)
            for i, chunk in enumerate(chunks)
        ]
        results = [result for chunk in ray.get(futures) for result in chunk]
        rewards = torch.tensor(
            [result["reward"] for result in results], dtype=torch.float32
        )
        return EnvironmentReturn(
            observations=[
                {
                    "role": "environment",
                    "content": "Environment: correct"
                    if r
                    else "Environment: incorrect",
                }
                for r in rewards
            ],
            metadata=metadata,
            next_stop_strings=[None] * len(results),
            rewards=rewards,
            terminateds=torch.ones_like(rewards),
            answers=[result["prediction"] for result in results]
            if return_extracted_answer
            else None,
        )
