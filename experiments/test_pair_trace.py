"""Check the decomposition on real forward passes with nonzero biases."""

import pytest
import torch

from pair_data import make_pair_batch
from pair_model import read_run
from pair_trace import trace_run


@pytest.mark.parametrize("read_previous", [False, True])
def test_value_read_labels(read_previous: bool) -> None:
    """A single known read must land in the right source group, with bias separate."""
    from types import SimpleNamespace
    from pair_trace import trace_value_reads

    data = make_pair_batch(20, 0, [0], batch=1, pairs=1, repeats=2, seed=4)
    event = data.last_occurrences()[0]
    source = event.source_positions[0]
    length = data.tokens.shape[1]
    early = torch.zeros(1, 1, length, length)
    late = torch.zeros_like(early)
    early[0, 0, source, source - int(read_previous)] = 1
    late[0, 0, event.trigger_position, source] = 1
    values = torch.zeros(1, length, 1, 1)
    values[0, source, 0, 0] = 3  # content 2 plus bias 1
    values[0, source - 1, 0, 0] = 5  # content 4 plus bias 1
    model = SimpleNamespace(
        W_O=torch.ones(2, 1, 1, 1), W_V=torch.ones(2, 1, 1, 1),
        b_V=torch.ones(2, 1, 1),
    )
    cache = {("pattern", 0): early, ("pattern", 1): late, ("v", 0): values}
    expected = 5.0 if read_previous else 3.0
    path = trace_value_reads(model, cache, data, 0, 0, torch.ones(1, 1), torch.tensor([expected]))
    groups = path["read_contributions"]
    assert groups["source_itself"] == (0 if read_previous else 2)
    assert groups["token_before_source"] == (4 if read_previous else 0)
    assert groups["other_tokens"] == 0
    assert groups["early_value_bias"] == 1
    assert path["reconstruction_error"] == 0


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
    path = output["early_value_path"]
    assert path["reconstruction_error"] < 1e-4
    assert sum(path["read_contributions"].values()) == pytest.approx(
        output["matching_value_parts"][f"L0H{path['early_head']}"], abs=1e-5,
    )
    for record in path["queries"]:
        assert len(record["token_ids"]) == len(record["token_logit_contributions"])
        assert len(record["token_ids"]) == max(record["sources"]) + 1
    for edge in keys["edges"]:
        assert edge["source"] < edge["query"]
        same_trigger = data.tokens[edge["row"], edge["source"] - 1] == data.tokens[edge["row"], edge["query"]]
        assert edge["matching"] == bool(same_trigger)

    for name, value in original_cache.items():
        torch.testing.assert_close(cache[name], value, rtol=0, atol=0)
    for name, value in original_weights.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(data.tokens, original_tokens, rtol=0, atol=0)
