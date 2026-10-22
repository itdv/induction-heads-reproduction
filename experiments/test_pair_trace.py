"""Check the decomposition on real forward passes with nonzero biases."""

import pytest
import torch

from pair_data import make_pair_batch
from pair_model import read_run
from pair_trace import trace_run


@pytest.mark.parametrize("early_head,late_head", [(0, 1), (1, 0)])
def test_trace_reconstructs_the_run_without_changing_it(early_head: int, late_head: int) -> None:
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    torch.set_num_threads(2)
    config = HookedTransformerConfig(
        n_layers=2, d_model=16, n_heads=2, d_head=8, n_ctx=256,
        d_vocab=200, attn_only=True, normalization_type=None,
        positional_embedding_type="shortformer", seed=7, device="cpu",
    )
    model = HookedTransformer(config).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".b_" in name:
                parameter.copy_(torch.randn_like(parameter) * .1)

    data = make_pair_batch(200, 0, [0], batch=2, pairs=3, repeats=3, seed=3)
    cache = read_run(model, data)
    original_cache = {name: value.clone() for name, value in cache.items()}
    original_weights = {name: value.clone() for name, value in model.state_dict().items()}
    original_tokens = data.tokens.clone()

    trace = trace_run(model, cache, data, early_head, late_head)
    keys, output = trace["keys"], trace["output"]
    assert keys["key_reconstruction_error"] < 1e-4
    assert keys["score_reconstruction_error"] < 1e-4
    assert output["value_reconstruction_error"] < 1e-4
    assert output["head_reconstruction_error"] < 1e-4
    assert output["head_logit"] == pytest.approx(
        output["matching_sources_logit"] + output["other_sources_logit"], abs=1e-5,
    )
    assert sum(output["matching_value_parts"].values()) == pytest.approx(
        output["matching_sources_logit"], abs=1e-5,
    )
    assert output["head_logit"] + output["rest_logit"] == pytest.approx(
        output["target_logit"], abs=1e-5,
    )
    for edge in keys["edges"]:
        assert edge["source"] < edge["query"]
        same_trigger = data.tokens[edge["row"], edge["source"] - 1] == data.tokens[edge["row"], edge["query"]]
        assert edge["matching"] == bool(same_trigger)

    for name, value in original_cache.items():
        torch.testing.assert_close(cache[name], value, rtol=0, atol=0)
    for name, value in original_weights.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(data.tokens, original_tokens, rtol=0, atol=0)
