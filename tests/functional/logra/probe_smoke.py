"""GPU numerical test of factorized JVP through a Hugging Face transformer."""

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from nemo_rl.models.automodel.logra import LoGRAConfig, LoGRAOptimizer
from nemo_rl.models.automodel.logra_probe import predict_batch_kl, control_step


def main():
    torch.manual_seed(3)
    config = Qwen2Config(
        vocab_size=43,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).cuda()
    opt = LoGRAOptimizer(
        model, LoGRAConfig(rank=4, update_ratio=1e-3, mismatch_subtraction=False)
    )
    ids = torch.randint(3, 43, (2, 16), device="cuda")
    batch = {
        "input_ids": ids,
        "input_lengths": torch.full((2,), 16, device="cuda"),
        "token_mask": torch.ones_like(ids),
        "sample_mask": torch.ones(2, device="cuda"),
    }
    model(ids).logits.square().mean().backward()
    opt.prepare_update(process_group=None, normalization_tokens=32)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    prediction = predict_batch_kl(
        model,
        opt,
        batch,
        temperature=1.0,
        compute_dtype=torch.float32,
        process_group=None,
    )
    for n, p in model.named_parameters():
        torch.testing.assert_close(p, before[n], rtol=0, atol=0)
    with torch.no_grad():
        lp_before = model(ids).logits[:, :-1].double().log_softmax(-1)
    opt.apply_update(1.0)
    with torch.no_grad():
        lp_after = model(ids).logits[:, :-1].double().log_softmax(-1)
        actual = (lp_before.exp() * (lp_before - lp_after)).sum(-1).mean().item()
    print(
        {"predicted_kl": prediction, "actual_kl": actual, "ratio": actual / prediction},
        flush=True,
    )
    assert abs(actual / prediction - 1) < 0.03
    # A second step exercises real mixed precision and the controller's application.
    opt.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(ids).logits.square().mean().backward()
    opt.prepare_update(process_group=None, normalization_tokens=32)
    metrics = control_step(
        model,
        opt,
        batch,
        temperature=1.0,
        compute_dtype=torch.bfloat16,
        process_group=None,
        eos_token_ids={2},
    )
    assert metrics["logra/predicted_kl"] > 0
    print("BF16_CONTROLLER_OK", metrics, flush=True)


if __name__ == "__main__":
    main()
