import pytest
import torch
from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    RunningBaselineAdvantageEstimator,
)


def test_running_baseline_uses_only_previous_batches():
    config = AdvEstimatorConfig(name="reinforce_running_baseline")
    estimator = RunningBaselineAdvantageEstimator(config)
    rewards = torch.tensor([1.0, 0.0])
    mask = torch.ones(2, 3)
    first = estimator.compute_advantage(None, rewards, mask)
    torch.testing.assert_close(first[:, 0], torch.tensor([1.0, -0.01]))
    assert estimator.baseline == pytest.approx(0.0495)
    before = estimator.baseline
    second = estimator.compute_advantage(None, rewards, mask)
    torch.testing.assert_close(second[:, 0], torch.tensor([1.0, -0.01]) - before)
    assert second[0, 0] != 0


def test_invalid_rows_do_not_change_running_baseline():
    estimator = RunningBaselineAdvantageEstimator(
        AdvEstimatorConfig(name="reinforce_running_baseline")
    )
    estimator.compute_advantage(
        None, torch.tensor([1.0, float("nan")]), torch.tensor([[1, 1], [0, 0]])
    )
    assert estimator.baseline == pytest.approx(0.1)


def test_running_baseline_factory_with_real_config():
    from pathlib import Path
    from omegaconf import OmegaConf
    from nemo_rl.algorithms.grpo import MasterConfig, _create_advantage_estimator
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    path = Path(__file__).resolve().parents[4] / "examples/configs/grpo_math_1B.yaml"
    cfg = load_config(path)
    cfg.grpo.adv_estimator.name = "reinforce_running_baseline"
    master = MasterConfig(**OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(
        _create_advantage_estimator(master), RunningBaselineAdvantageEstimator
    )
    master.data_plane["enabled"] = True
    with pytest.raises(ValueError, match="synchronous"):
        _create_advantage_estimator(master)
