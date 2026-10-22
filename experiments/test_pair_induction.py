"""Check pair sources and attention indexing without pretrained weights."""

import pytest
import torch

from pair_data import make_pair_batch
from pair_measurements import measure_attention, select_heads


def sample(seed=1):
    return make_pair_batch(
        200, 0, [0, 199], batch=4, pairs=4, repeats=4, seed=seed,
    )


def test_recorded_sources_are_all_and_only_the_earlier_matching_values():
    data = sample()
    for row, mapping in enumerate(data.pairs):
        assert len({token for pair in mapping for token in pair}) == 8
        for pair_index, (trigger, target) in enumerate(mapping):
            events = [
                event for event in data.occurrences
                if event.sequence_index == row and event.pair_index == pair_index
            ]
            positions = [event.trigger_position for event in events]
            real_tokens = data.tokens[row, :data.lengths[row]]
            assert (real_tokens == trigger).nonzero().flatten().tolist() == positions
            assert (real_tokens == target).nonzero().flatten().tolist() == [
                position + 1 for position in positions
            ]
            for index, event in enumerate(events):
                assert event.source_positions == tuple(p + 1 for p in positions[:index])

    assert len(data.last_occurrences()) == 16
    assert torch.equal(data.tokens, sample().tokens)
    assert data.to_dict() != sample(2).to_dict()
    assert all(199 not in row for row in data.tokens.tolist())


def test_attention_counts_past_sources_and_ignores_padding():
    data = sample()
    size = data.tokens.shape[1]
    pattern = torch.zeros(4, 2, size, size)

    for event in data.last_occurrences():
        for source in event.source_positions:
            pattern[event.sequence_index, 0, event.trigger_position, source] = 1 / 3
        # Head 1 reads the current B, which must not count as a past source.
        pattern[event.sequence_index, 1, event.trigger_position, event.trigger_position + 1] = 1

    for row, length in enumerate(data.lengths):
        positions = torch.arange(1, length)
        pattern[row, 1, positions, positions - 1] = .25
        # Deliberately large padding scores must not affect the mean.
        padding = torch.arange(length, size)
        pattern[row, 1, padding, padding - 1] = 100

    scores = measure_attention([pattern, pattern], data)
    assert scores[1]["matching_source_mass"] == pytest.approx([1, 0])
    assert scores[1]["top_source_match_rate"] == pytest.approx([1, 0])
    assert scores[0]["previous_token_mass"][1] == pytest.approx(.25)


def test_head_selection_uses_the_right_layer_and_score():
    scores = [
        {"previous_token_mass": [.2, .8], "matching_source_mass": [.9, .1]},
        {"previous_token_mass": [.1, .9], "matching_source_mass": [.7, .3]},
    ]
    assert select_heads(scores) == {"early_head": 1, "late_head": 0}


def test_run_cache_matches_a_normal_forward_pass():
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from pair_model import read_run

    config = HookedTransformerConfig(
        n_layers=2, d_model=16, n_heads=2, d_head=8, n_ctx=256,
        d_vocab=200, attn_only=True, normalization_type=None,
        positional_embedding_type="shortformer", seed=7, device="cpu",
    )
    model = HookedTransformer(config).eval()
    data = make_pair_batch(200, 0, [0], batch=2, pairs=2, repeats=3, seed=3)
    cache = read_run(model, data)
    with torch.no_grad():
        _, full_cache = model.run_with_cache(data.tokens)

    for name, value in cache.items():
        torch.testing.assert_close(value, full_cache[name])
