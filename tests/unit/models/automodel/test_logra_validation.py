from unittest.mock import MagicMock, patch
import pytest


def test_validation_sampling_restores_training_after_failure():
    from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration

    generation = VllmGeneration.__new__(VllmGeneration)
    generation.cfg = {
        "vllm_cfg": {"async_engine": False},
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "val_temperature": 0.7,
        "val_top_p": 0.7,
        "val_top_k": None,
    }
    generation.worker_group = MagicMock()
    with patch("nemo_rl.models.generation.vllm.vllm_generation.ray.get"):
        with pytest.raises(RuntimeError, match="test exception"):
            with generation.validation_sampling():
                raise RuntimeError("test exception")
    calls = generation.worker_group.run_all_workers_single_data.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["temperature"] == 0.7
    assert calls[0].kwargs["top_p"] == 0.7
    assert calls[1].kwargs["temperature"] == 1.0
    assert calls[1].kwargs["top_p"] == 1.0


def test_default_validation_does_not_touch_workers():
    from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration

    generation = VllmGeneration.__new__(VllmGeneration)
    generation.cfg = {
        "vllm_cfg": {"async_engine": False},
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "val_temperature": 1.0,
        "val_top_p": 1.0,
        "val_top_k": None,
    }
    generation.worker_group = MagicMock()
    with generation.validation_sampling():
        pass
    generation.worker_group.run_all_workers_single_data.assert_not_called()
