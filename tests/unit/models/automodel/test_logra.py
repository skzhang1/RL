import copy
import math

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from nemo_rl.models.automodel.logra import (
    LoGRAConfig,
    install_sketches,
    kl_step_scale,
    make_projection,
    predicted_kl,
    projection_seed,
    row_adam_direction,
)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_accumulated_sketch_matches_dense_gradient(checkpointed):
    torch.manual_seed(4)
    dense = nn.Sequential(
        nn.Linear(7, 11, bias=False), nn.Tanh(), nn.Linear(11, 5, bias=False)
    )
    compressed = copy.deepcopy(dense)
    config = LoGRAConfig(rank=3, target_modules=("0", "2"))
    before_keys = set(compressed.state_dict())
    states = install_sketches(compressed, config)
    assert set(compressed.state_dict()) == before_keys
    for _ in range(3):
        x = torch.randn(2, 4, 7, requires_grad=True)
        x_sketch = x.detach().clone().requires_grad_()
        target = torch.randn(2, 4, 5)
        loss = (dense(x) * target).sum()
        out = (
            checkpoint(compressed, x_sketch, use_reentrant=False)
            if checkpointed
            else compressed(x_sketch)
        )
        (out * target).sum().backward()
        loss.backward()
        torch.testing.assert_close(x.grad, x_sketch.grad)
    for state in states:
        torch.testing.assert_close(
            state.sketch,
            dense.get_submodule(state.name).weight.grad @ state.projection.T,
        )
        assert state.module.weight.grad is None


@pytest.mark.parametrize("distribution", ["rademacher", "gaussian"])
def test_projection_matches_original_reference(distribution):
    from pathlib import Path
    import importlib.util
    import os

    # Exact original code snapshot is optional outside the transfer workspace.
    source = (
        Path(os.environ["LOGRA_REFERENCE_ROOT"]) / "rpga/projection.py"
        if "LOGRA_REFERENCE_ROOT" in os.environ
        else Path(__file__).resolve().parents[5] / "reference/rpga/projection.py"
    )
    if not source.exists():
        pytest.skip("Original LoGRA snapshot not available")
    spec = importlib.util.spec_from_file_location("original_projection", source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    for update in (0, 1, 200):
        seed = projection_seed(42, update, "model.layers.0.self_attn.q_proj")
        assert seed == reference.layer_seed(
            42, update, "model.layers.0.self_attn.q_proj"
        )
        actual = make_projection(
            4,
            19,
            seed,
            device=torch.device("cpu"),
            dtype=torch.float32,
            distribution=distribution,
        )
        assert torch.equal(
            actual, reference.make_projection(4, 19, seed, kind=distribution)
        )


def test_rowadam_matches_original_arithmetic_across_updates():
    torch.manual_seed(8)
    moment = torch.zeros(7, 1)
    reference_moment = moment.clone()
    for step in range(1, 5):
        sketch = torch.randn(7, 3) * torch.arange(1, 8).unsqueeze(1)
        reference_moment.mul_(0.95).add_(
            sketch.square().mean(1, keepdim=True), alpha=0.05
        )
        root = (reference_moment / (1 - 0.95**step)).sqrt()
        inverse = 1 / (
            root + (1e-8 * root.mean()).clamp_min(torch.finfo(root.dtype).tiny)
        )
        inverse = inverse / inverse.square().mean().sqrt()
        expected = inverse * sketch
        expected *= sketch.norm() / expected.norm()
        actual = row_adam_direction(sketch, moment, step=step, beta2=0.95, epsilon=1e-8)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(moment, reference_moment)


def test_zero_rowadam_and_invalid_sketch():
    zero = torch.zeros(5, 3)
    assert torch.equal(
        row_adam_direction(zero, torch.zeros(5, 1), step=1, beta2=0.95, epsilon=1e-8),
        zero,
    )
    with pytest.raises(FloatingPointError):
        row_adam_direction(
            zero + math.nan, torch.zeros(5, 1), step=1, beta2=0.95, epsilon=1e-8
        )


def test_kl_prediction_matches_small_actual_policy_change():
    torch.manual_seed(5)
    logits = torch.randn(3, 7, dtype=torch.float64)
    direction = torch.randn_like(logits)
    temperature = 0.7
    scale = 1e-4
    p_log = (logits / temperature).log_softmax(-1)
    q_log = ((logits + scale * direction) / temperature).log_softmax(-1)
    actual = (p_log.exp() * (p_log - q_log)).sum(-1)
    prediction = predicted_kl(logits, direction, temperature=temperature) * scale**2
    torch.testing.assert_close(actual, prediction, rtol=2e-4, atol=1e-12)
    alpha = kl_step_scale(0.008, budget=0.0002, alpha_max=3.0)
    assert alpha**2 * 0.008 == pytest.approx(0.0002)
    assert kl_step_scale(1e-12, budget=0.0002, alpha_max=3.0) == 3.0
    with pytest.raises(FloatingPointError):
        kl_step_scale(math.nan, budget=0.0002, alpha_max=3.0)


def test_default_linear_remains_unchanged_and_invalid_targets_fail():
    model = nn.Sequential(nn.Linear(5, 3))
    model(torch.ones(2, 5)).sum().backward()
    assert model[0].weight.grad is not None
    with pytest.raises(ValueError, match="matched no"):
        install_sketches(model, LoGRAConfig())


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13, 7)
        self.q_proj = nn.Linear(7, 11, bias=False)
        self.o_proj = nn.Linear(11, 7, bias=False)
        self.head = nn.Linear(7, 13, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, ids):
        return self.head(self.o_proj(self.q_proj(self.embedding(ids)).tanh()))


def test_optimizer_merge_and_resume():
    from nemo_rl.models.automodel.logra import LoGRAOptimizer

    torch.manual_seed(3)
    model = TinyPolicy()
    cfg = LoGRAConfig(rank=3, update_ratio=0.01, update_factor_dtype="float32")
    opt = LoGRAOptimizer(model, cfg)
    inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
    model(inputs).square().mean().backward()
    before = [layer.module.weight.detach().clone() for layer in opt.layers]
    before_projection = opt.layers[0].projection.clone()
    assert opt.prepare_update(process_group=None, normalization_tokens=1) > 0
    deltas = [
        direction @ layer.projection
        for direction, layer in zip(opt.directions, opt.layers)
    ]
    assert math.sqrt(
        sum(d.square().sum().item() for d in deltas)
        / sum(w.square().sum().item() for w in before)
    ) == pytest.approx(0.01, rel=1e-5)
    opt.apply_update(0.7)
    for state, original, delta in zip(opt.layers, before, deltas):
        torch.testing.assert_close(state.module.weight, original - 0.7 * delta)
        assert state.module.weight.grad is None
    assert not torch.equal(before_projection, opt.layers[0].projection)
    restored = TinyPolicy()
    restored.load_state_dict(model.state_dict())
    restored_opt = LoGRAOptimizer(restored, cfg)
    restored_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    for current_model, current_opt in [(model, opt), (restored, restored_opt)]:
        current_opt.zero_grad()
        current_model(inputs).square().mean().backward()
        current_opt.prepare_update(process_group=None, normalization_tokens=1)
        current_opt.apply_update(0.8)
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(opt.layers, restored_opt.layers):
        torch.testing.assert_close(a.second_moment, b.second_moment, rtol=0, atol=0)
        torch.testing.assert_close(a.projection, b.projection, rtol=0, atol=0)


@pytest.mark.parametrize("factor_dtype", ["bfloat16", "float32"])
def test_full_updates_match_historical_optimizer(factor_dtype):
    import os
    import sys
    from pathlib import Path

    if "LOGRA_REFERENCE_ROOT" not in os.environ:
        pytest.skip("Set LOGRA_REFERENCE_ROOT to the historical experiment code")
    sys.path.insert(0, str(Path(os.environ["LOGRA_REFERENCE_ROOT"])))
    from rpga.optimizer import RPGAConfig, RPGALayer, RPGAOptimizer
    from nemo_rl.models.automodel.logra import LoGRAOptimizer

    torch.manual_seed(11)
    model = TinyPolicy()
    reference = copy.deepcopy(model).requires_grad_(False)
    new = LoGRAOptimizer(
        model,
        LoGRAConfig(
            rank=3, seed=42, update_ratio=0.001, update_factor_dtype=factor_dtype
        ),
    )
    old_layers = [
        RPGALayer(
            x.name,
            reference.get_submodule(x.name).weight,
            None,
            3,
            projection_seed(42, 0, x.name),
            "rademacher",
            need_m=False,
            need_v=False,
        )
        for x in new.layers
    ]
    old = RPGAOptimizer(
        old_layers,
        RPGAConfig(
            rank=3,
            lr=1.0,
            base_seed=42,
            optimizer="rowadam",
            max_sketch_norm=0.0,
            max_update_ratio=0.001,
            update_mode="target",
            merge_precision="fp32",
            preserve_layer_scale=True,
            u_wire_dtype=torch.bfloat16 if factor_dtype == "bfloat16" else None,
        ),
    )
    for tokens in (33, 71, 11):
        for a, b in zip(new.layers, old.layers):
            sketch = torch.randn_like(a.sketch)
            a.sketch.copy_(sketch)
            b.S.copy_(sketch)
        old.tokens = tokens
        old.merge()
        new.prepare_update(process_group=None, normalization_tokens=tokens)
        new.apply_update(1.0)
        for a, b in zip(new.layers, old.layers):
            torch.testing.assert_close(a.module.weight, b.W, rtol=2e-6, atol=2e-8)
            torch.testing.assert_close(a.second_moment, b.v_row, rtol=2e-6, atol=1e-10)
            torch.testing.assert_close(a.projection, b.A, rtol=0, atol=0)


def test_projection_seed_is_independent_of_checkpoint_wrappers():
    bare = "model.layers.0.self_attn.q_proj"
    wrapped = "_orig_mod.model.layers.0._checkpoint_wrapped_module.self_attn.q_proj"
    assert projection_seed(42, 3, bare) == projection_seed(42, 3, wrapped)


def test_sketch_saved_activation_lifetime_and_repeated_backward():
    import weakref

    model = nn.Sequential(nn.Linear(7, 11, bias=False))
    states = install_sketches(model, LoGRAConfig(rank=3, target_modules=["0"]))
    x = torch.randn(2, 4, 7, requires_grad=True)
    output = model(x)
    projected = weakref.ref(output.grad_fn.saved_tensors[0])
    loss = output.square().sum()
    loss.backward(retain_graph=True)
    first = states[0].sketch.clone()
    assert projected() is not None
    loss.backward()
    torch.testing.assert_close(states[0].sketch, 2 * first)
    # NeMo keeps loss tensors in its microbatch results after backward. Their
    # graph objects must not keep every microbatch's projected activations alive.
    assert projected() is None
    assert loss.grad_fn is not None


def test_bfloat16_probe_factor_products_match_frozen_reference():
    import importlib.util
    import os
    from pathlib import Path
    from types import SimpleNamespace
    from torch.autograd import forward_ad
    from nemo_rl.models.automodel.logra_probe import factorized_direction

    if "LOGRA_REFERENCE_ROOT" not in os.environ:
        pytest.skip("Original LoGRA snapshot not available")
    source = Path(os.environ["LOGRA_REFERENCE_ROOT"]) / "rpga/kl_probe.py"
    spec = importlib.util.spec_from_file_location("original_kl_probe", source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.manual_seed(12)
    model = nn.Linear(19, 11, bias=False, dtype=torch.bfloat16)
    x = torch.randn(2, 3, 19, dtype=torch.bfloat16)
    projection = torch.randn(4, 19)
    direction = torch.randn(11, 4)
    legacy_layer = SimpleNamespace(A=projection, last_u=direction)
    handle = model.register_forward_hook(
        reference._make_tangent_hook(legacy_layer, 1.0, {})
    )
    with torch.no_grad(), forward_ad.dual_level():
        expected = forward_ad.unpack_dual(model(x)).tangent.clone()
    handle.remove()
    optimizer = SimpleNamespace(
        layers=[SimpleNamespace(module=model, projection=projection)],
        directions=[direction],
    )
    with (
        torch.no_grad(),
        forward_ad.dual_level(),
        factorized_direction(optimizer),
        torch.autocast("cpu", dtype=torch.bfloat16),
    ):
        actual = forward_ad.unpack_dual(model(x)).tangent.clone()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
