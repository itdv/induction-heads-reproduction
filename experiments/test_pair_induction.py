import pytest
import torch

from pair_data import make_pair_batch, remove_pair_history, rotate_pair_values, best_fixed_offset
from pair_measurements import attention_metrics
from pair_trace import project_components, stack_components


def sample(seed=1):
    return make_pair_batch(200, 0, [0, 199], batch=8, pairs=4, repeats=4, seed=seed)


def test_pairs_are_consistent_and_have_no_accidental_triggers():
    data = sample()
    for row, mapping in enumerate(data.pairs):
        for pair, (a, b) in enumerate(mapping):
            events = [event for event in data.events if event.row == row and event.pair == pair]
            positions = [event.query for event in events]
            assert (data.tokens[row, :data.lengths[row]] == a).nonzero().flatten().tolist() == positions
            for i, event in enumerate(events):
                assert data.tokens[row, event.query + 1] == b
                assert event.previous_values == tuple(q + 1 for q in positions[:i])
    assert torch.equal(data.tokens, sample().tokens)
    assert not torch.equal(data.tokens, sample(2).tokens)
    lags = {event.query - event.previous_values[-1] for event in data.queries()}
    assert len(lags) > 8  # Not one global repeated-copy lag.
    assert all(199 not in row for row in data.tokens.tolist())
    assert best_fixed_offset(data) > 0


def test_history_control_preserves_values_and_queries_but_removes_prior_A():
    data = sample()
    control = remove_pair_history(data)
    assert not torch.equal(data.tokens, control.tokens)
    for event in data.queries():
        a = data.pairs[event.row][event.pair][0]
        assert control.tokens[event.row, event.query] == a
        assert a not in control.tokens[event.row, :event.query]
        assert torch.equal(control.tokens[event.row, list(event.previous_values)],
                           data.tokens[event.row, list(event.previous_values)])


def test_value_rotation_changes_answers_without_changing_trigger_or_background_positions():
    data = sample()
    rotated = rotate_pair_values(data)
    changed = (data.tokens != rotated.tokens).nonzero().tolist()
    assert {tuple(index) for index in changed} == {(event.row, event.query + 1) for event in data.events}
    for before, after in zip(data.events, rotated.events):
        assert before.target != after.target
        assert rotated.tokens[after.row, after.query + 1] == after.target


def test_matching_attention_sums_all_prior_successors_and_excludes_future_or_current_B():
    data = sample()
    size = data.tokens.shape[1]
    pattern = torch.zeros(8, 2, size, size)
    for event in data.queries():
        for source in event.previous_values:
            pattern[event.row, 0, event.query, source] = 1 / len(event.previous_values)
        pattern[event.row, 1, event.query, event.query + 1] = 1
    cache = {f'blocks.{layer}.attn.hook_pattern':pattern for layer in range(2)}
    metrics = attention_metrics(cache, data)
    assert metrics[1]['matching_mass'] == pytest.approx([1, 0])


def test_qk_decomposition_includes_all_cross_terms_and_biases():
    torch.manual_seed(1)
    components = {'one':torch.randn(2, 3, 5), 'two':torch.randn(2, 3, 5)}
    rows, positions = torch.tensor([0, 1]), torch.tensor([1, 2])
    wq, wk = torch.randn(5, 4), torch.randn(5, 4)
    bq, bk = torch.randn(4), torch.randn(4)
    q = project_components(components, rows, positions, wq, bq)
    k = project_components(components, rows, positions, wk, bk)
    direct = sum(components.values())[rows, positions]
    expected = ((direct @ wq + bq) * (direct @ wk + bk)).sum(-1)
    terms = torch.einsum('eid,ejd->eij', stack_components(q), stack_components(k))
    torch.testing.assert_close(terms.sum((1, 2)), expected, atol=1e-4, rtol=1e-4)


def test_passive_trace_reconstructs_a_real_shortformer_with_nonzero_biases():
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from pair_model import clean_run, validate_model
    from pair_trace import trace_key_composition, trace_output
    cfg = HookedTransformerConfig(n_layers=2, d_model=16, n_heads=2, d_head=8,
        n_ctx=256, d_vocab=200, attn_only=True, normalization_type=None,
        positional_embedding_type='shortformer', seed=7, device='cpu')
    model = HookedTransformer(cfg).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if '.b_' in name or name == 'unembed.b_U':
                parameter.copy_(torch.randn_like(parameter) * .1)
    validate_model(model)
    data = make_pair_batch(200, 0, [0], batch=2, pairs=3, repeats=3, seed=3)
    residual, cache = clean_run(model, data, capture=True)
    original_tokens = data.tokens.clone()
    original_residual = residual.clone()
    keys = trace_key_composition(model, cache, data, previous_head=0, induction_head=1)
    output = trace_output(model, cache, residual, data, induction_head=1)
    assert keys['recorded_score_max_error'] < 1e-4
    assert output['matching_source_value_decomposition']['max_write_reconstruction_error'] < 1e-4
    torch.testing.assert_close(data.tokens, original_tokens)
    torch.testing.assert_close(residual, original_residual)
