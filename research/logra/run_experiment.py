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

"""Run NeMo RL with the historical LoGRA verifier, without changing default environments."""

from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.utils import register_env
from examples.run_grpo import main

if __name__ == "__main__":
    fqn = "research.logra.historical_environment.HistoricalMathEnvironment"
    ACTOR_ENVIRONMENT_REGISTRY[fqn] = PY_EXECUTABLES.SYSTEM
    register_env("historical_math", fqn)
    main()
