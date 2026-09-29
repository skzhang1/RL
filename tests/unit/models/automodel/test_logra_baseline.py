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


def test_length_weighted_reinforce_matches_historical_policy_gradient():
    from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    estimator = RunningBaselineAdvantageEstimator(
        AdvEstimatorConfig(
            name="reinforce_running_baseline", response_length_reference=1024.0
        )
    )
    estimator.baseline = 0.2
    mask = torch.tensor(
        [[0, 1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1]], dtype=torch.float32
    )
    rewards = torch.tensor([1.0, 0.0])
    advantages = estimator.compute_advantage(None, rewards, mask)
    assert estimator.baseline == pytest.approx(0.2295)
    logprobs = torch.linspace(-2, -1, 12).reshape(2, 6).requires_grad_()
    prev = torch.cat([torch.zeros(2, 1), logprobs.detach()], dim=1)
    ratios = torch.tensor([0.5, 3.0]).unsqueeze(-1)
    data = BatchedDataDict(
        {
            "input_ids": torch.ones(2, 7, dtype=torch.long),
            "token_mask": mask,
            "sample_mask": torch.ones(2),
            "advantages": advantages,
            "prev_logprobs": prev,
            "generation_logprobs": prev - ratios.log(),
        }
    )
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            disable_ppo_ratio=True,
            reference_policy_kl_penalty=0,
            use_importance_sampling_correction=True,
            truncated_importance_sampling_type="tis",
            truncated_importance_sampling_ratio=2.0,
        )
    )
    loss, _ = loss_fn(logprobs, data, torch.tensor(2.0), mask.sum())
    loss.backward()
    # Frozen historical rpga_cs + molt token mean + detached TIS, including
    # unequal lengths, past-batch baseline, failure penalty, and padding.
    expected = -(torch.tensor([1.0, -0.01]) - 0.2).unsqueeze(-1)
    expected = expected * (1024.0 / mask.sum(-1, keepdim=True))
    expected = expected * ratios.clamp_max(2) * mask[:, 1:] / mask.sum()
    torch.testing.assert_close(logprobs.grad, expected)
