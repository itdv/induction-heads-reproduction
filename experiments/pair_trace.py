"""Exact additive decompositions of the UNMODIFIED run (no hooks changing activations).

Legacy detailed analysis, retained for the previous reference results. The current
experiment uses the simpler weight-only tests in paper_circuits.py instead.

QK terms are contributions to pre-softmax scores, not additive attention probabilities.
Logit terms are direct contributions, not causal effects or probabilities.
The algebra is restricted to the architecture checked in pair_model.validate_model.
"""
from dataclasses import dataclass

import torch


def head_write(model, cache, layer, head):
    return cache[f'blocks.{layer}.attn.hook_z'][:, :, head] @ model.W_O[layer, head]


def early_components(model, cache):
    """Shortformer adds position to Q/K inputs, but NOT to the value/residual stream."""
    embedding = cache['hook_embed']
    components = {'token_embedding':embedding,
                  'position_embedding':cache['hook_pos_embed'].expand_as(embedding),
                  'early_output_bias':model.b_O[0].expand_as(embedding)}
    for head in range(model.cfg.n_heads):
        components[f'L0H{head}'] = head_write(model, cache, 0, head)
    return components


def project_components(components, rows, positions, weight, bias):
    projected = {name:values[rows, positions] @ weight for name, values in components.items()}
    projected['projection_bias'] = bias.expand(len(rows), -1)
    return projected


def stack_components(components):
    return torch.stack(list(components.values()), dim=1)


def edge_indices(data):
    """For every final query, include all earlier planted B positions as matched controls."""
    edges = []
    for query_index, query in enumerate(data.queries()):
        for source in data.events:
            if source.row == query.row and source.query + 1 < query.query:
                edges.append({'row':query.row, 'query':query.query, 'source':source.query + 1,
                              'query_index':query_index, 'matching':source.pair == query.pair})
    return edges


def mean_groups(values, matching):
    correct = values[matching].mean(0)
    wrong = values[~matching].mean(0)
    return {'matching':correct.tolist(), 'other_pairs':wrong.tolist(),
            'matching_minus_other':(correct - wrong).tolist()}


@torch.no_grad()
def trace_key_composition(model, cache, data, previous_head, induction_head):
    """Trace early-head -> late-key contributions in the original forward pass.

    The steps below only organize the analysis. Each calculation has its own
    function so that selecting positions, tracing information and writing the
    report can be read separately.
    """
    edges = edge_indices(data)
    positions = edge_positions(edges)
    qk = decompose_qk(model, cache, positions, induction_head)
    scale = model.blocks[1].attn.attn_scale

    key_scores = score_key_components(qk['query'], qk['key_parts'], scale)
    head_column = list(qk['key_parts']).index(f'L0H{previous_head}')
    selected_head_scores = key_scores[:, head_column]

    previous_token = trace_previous_token(
        model, cache, data, positions, qk['query'], previous_head, induction_head)

    add_edge_measurements(
        edges, cache, positions, induction_head,
        qk['scores'], selected_head_scores, previous_token)
    return summarize_key_trace(edges, positions.matching, qk, key_scores, previous_token)


@dataclass(frozen=True)
class EdgePositions:
    """Parallel index tensors: one row per query-to-source edge being analyzed."""
    rows: torch.Tensor
    queries: torch.Tensor
    sources: torch.Tensor
    matching: torch.Tensor


def edge_positions(edges):
    """Convert the recorded edge locations to batch indices for the cached tensors."""
    return EdgePositions(
        rows=torch.tensor([edge['row'] for edge in edges]),
        queries=torch.tensor([edge['query'] for edge in edges]),
        sources=torch.tensor([edge['source'] for edge in edges]),
        matching=torch.tensor([edge['matching'] for edge in edges]),
    )


def reconstruction_error(reconstructed, observed):
    """Check our decomposition, then report its largest numerical discrepancy."""
    torch.testing.assert_close(reconstructed, observed, atol=3e-4, rtol=3e-4)
    return float((reconstructed - observed).abs().max())


def project_qk_components(model, cache, positions, head):
    """Apply W_Q and W_K separately to each input contribution, including biases."""
    components = early_components(model, cache)
    query_parts = project_components(
        components, positions.rows, positions.queries, model.W_Q[1, head], model.b_Q[1, head])
    key_parts = project_components(
        components, positions.rows, positions.sources, model.W_K[1, head], model.b_K[1, head])
    return query_parts, key_parts


def decompose_qk(model, cache, positions, head):
    """Read actual Q/K, decompose them, and verify all Q-part x K-part scores."""
    query = cache['blocks.1.attn.hook_q'][positions.rows, positions.queries, head]
    key = cache['blocks.1.attn.hook_k'][positions.rows, positions.sources, head]
    recorded_scores = cache['blocks.1.attn.hook_attn_scores'][
        positions.rows, head, positions.queries, positions.sources]
    query_parts, key_parts = project_qk_components(model, cache, positions, head)
    scale = model.blocks[1].attn.attn_scale
    score_terms = qk_cross_terms(query_parts, key_parts, scale)
    scores = matching_score(query, key, scale)
    errors = {
        'q_max_reconstruction_error':reconstruction_error(stack_components(query_parts).sum(1), query),
        'k_max_reconstruction_error':reconstruction_error(stack_components(key_parts).sum(1), key),
        'score_max_reconstruction_error':reconstruction_error(score_terms.sum((1, 2)), scores),
        'recorded_score_max_error':reconstruction_error(scores, recorded_scores),
    }
    return {'query':query, 'key_parts':key_parts, 'scores':scores,
            'component_names':list(query_parts), 'score_terms':score_terms, 'errors':errors}


def matching_score(query, key, scale):
    """Scaled dot product; this is a raw score before softmax, not a probability."""
    return (query * key).sum(-1) / scale


def qk_cross_terms(query_parts, key_parts, scale):
    """For each edge, multiply every Q component by every K component.

    e = edge, i = Q component, j = K component, d = head dimension.
    Summing over i and j recovers the full matching score.
    """
    return torch.einsum(
        'eid,ejd->eij', stack_components(query_parts), stack_components(key_parts)) / scale


def score_key_components(query, key_parts, scale):
    """Hold the actual full Q fixed and score each contribution to K separately."""
    return matching_score(query[:, None], stack_components(key_parts), scale)


def previous_token_writes(model, cache, data, positions, head):
    """Separate the early head's preceding-token write from its other source reads.

    Return both the cached V path (including b_V) and the embedding-only path.
    """
    rows, sources = positions.rows, positions.sources
    attention = cache['blocks.0.attn.hook_pattern'][rows, head, sources, sources - 1]
    value = cache['blocks.0.attn.hook_v'][rows, sources - 1, head]
    write = attention[:, None] * (value @ model.W_O[0, head])

    token_ids = data.tokens[rows, sources - 1]
    content_value = model.W_E[token_ids] @ model.W_V[0, head]
    content_write = attention[:, None] * (content_value @ model.W_O[0, head])
    return attention, write, content_write


def trace_previous_token(model, cache, data, positions, query, early_head, late_head):
    """Follow previous token -> early-head write -> late key -> matching score."""
    attention, write, content_write = previous_token_writes(
        model, cache, data, positions, early_head)
    key_weight = model.W_K[1, late_head]
    scale = model.blocks[1].attn.attn_scale
    key_contribution = write @ key_weight
    content_key_contribution = content_write @ key_weight
    return {'attention':attention,
            'edge_score':matching_score(query, key_contribution, scale),
            'content_score':matching_score(query, content_key_contribution, scale)}


def add_edge_measurements(edges, cache, positions, head, scores, head_scores, previous_token):
    """Attach numbers to each recorded edge; no model calculations happen here."""
    late_attention = cache['blocks.1.attn.hook_pattern'][
        positions.rows, head, positions.queries, positions.sources]
    for i, edge in enumerate(edges):
        edge.update(total_qk_score=float(scores[i]),
                    early_head_key_score=float(head_scores[i]),
                    previous_token_edge_key_score=float(previous_token['edge_score'][i]),
                    previous_token_content_key_score=float(previous_token['content_score'][i]),
                    early_previous_attention=float(previous_token['attention'][i]),
                    late_attention=float(late_attention[i]))


def summarize_key_trace(edges, matching, qk, key_scores, previous_token):
    """Preserve the report schema and compare matching sources with other pairs."""
    return {'component_names':qk['component_names'], **qk['errors'],
            'full_qk_scores':mean_groups(qk['scores'], matching),
            'qk_component_matrix':mean_groups(qk['score_terms'], matching),
            'key_component_scores_with_actual_q':mean_groups(key_scores, matching),
            'previous_token_edge_key_score':mean_groups(previous_token['edge_score'], matching),
            'previous_token_content_key_score':mean_groups(previous_token['content_score'], matching),
            'edges':edges}


def trace_matching_values(model, cache, data, induction_head, expected_write, gold_direction, margin_direction):
    """Within the retrieved B sources, distinguish B embeddings from earlier head outputs."""
    components = early_components(model, cache)
    del components['position_embedding']  # Shortformer does not put position into V.
    pattern = cache['blocks.1.attn.hook_pattern'][:, induction_head]
    writes = {name:[] for name in components}
    writes['value_projection_bias'] = []
    for event in data.queries():
        sources = list(event.previous_values)
        weights = pattern[event.row, event.query, sources]
        for name, stream in components.items():
            weighted = (weights[:, None] * stream[event.row, sources]).sum(0)
            writes[name].append(weighted @ model.W_V[1, induction_head] @ model.W_O[1, induction_head])
        writes['value_projection_bias'].append(
            weights.sum() * (model.b_V[1, induction_head] @ model.W_O[1, induction_head]))
    writes = {name:torch.stack(values) for name, values in writes.items()}
    reconstructed = torch.stack(list(writes.values())).sum(0)
    torch.testing.assert_close(reconstructed, expected_write, atol=3e-4, rtol=3e-4)
    return {'components':{name:{'target_logit':float((write * gold_direction).sum(-1).mean()),
                                'target_vs_other_values_margin':float((write * margin_direction).sum(-1).mean())}
                          for name, write in writes.items()},
            'max_write_reconstruction_error':float((reconstructed - expected_write).abs().max())}


@torch.no_grad()
def trace_output(model, cache, residual, data, induction_head):
    """Add up all direct target-logit and target-vs-other-values margin contributions."""
    events = data.queries()
    rows = torch.tensor([event.row for event in events])
    positions = torch.tensor([event.query for event in events])
    targets = torch.tensor([event.target for event in events])
    candidate_ids = torch.tensor([[value for _, value in data.pairs[event.row]] for event in events])
    gold_direction = model.W_U[:, targets].T
    other_directions = model.W_U[:, candidate_ids].permute(1, 2, 0)
    other_mask = candidate_ids != targets[:, None]
    direction = gold_direction - (other_directions * other_mask[:, :, None]).sum(1) / other_mask.sum(1)[:, None]
    streams = {'token_embedding':cache['hook_embed'][rows, positions]}
    for layer in range(2):
        streams[f'L{layer}_output_bias'] = model.b_O[layer].expand(len(events), -1)
        for head in range(model.cfg.n_heads):
            streams[f'L{layer}H{head}'] = head_write(model, cache, layer, head)[rows, positions]
    reconstructed = torch.stack(list(streams.values())).sum(0)
    observed = residual[rows, positions]
    torch.testing.assert_close(reconstructed, observed, atol=3e-4, rtol=3e-4)
    bias_gold = model.b_U[targets]
    bias_margin = bias_gold - (model.b_U[candidate_ids] * other_mask).sum(1) / other_mask.sum(1)
    parts = {name:{'target_logit':float((values * gold_direction).sum(-1).mean()),
                  'target_vs_other_values_margin':float((values * direction).sum(-1).mean())}
             for name, values in streams.items()}
    parts['unembedding_bias'] = {'target_logit':float(bias_gold.mean()),
                                'target_vs_other_values_margin':float(bias_margin.mean())}
    actual_margin = (observed * direction).sum(-1) + bias_margin
    sum_margin = sum((values * direction).sum(-1) for values in streams.values()) + bias_margin
    torch.testing.assert_close(sum_margin, actual_margin, atol=3e-4, rtol=3e-4)

    # Split the selected late head output into matching B sources and everything else.
    pattern = cache['blocks.1.attn.hook_pattern'][:, induction_head]
    values = cache['blocks.1.attn.hook_v'][:, :, induction_head]
    reconstructed_z = pattern @ values
    observed_z = cache['blocks.1.attn.hook_z'][:, :, induction_head]
    torch.testing.assert_close(reconstructed_z, observed_z, atol=3e-4, rtol=3e-4)
    matching_writes = []
    for event in events:
        sources = list(event.previous_values)
        weighted = pattern[event.row, event.query, sources, None] * values[event.row, sources]
        matching_writes.append(weighted.sum(0) @ model.W_O[1, induction_head])
    matching_writes = torch.stack(matching_writes)
    late_write = streams[f'L1H{induction_head}']
    partition = {}
    for name, write in [('matching_sources',matching_writes), ('other_sources',late_write - matching_writes)]:
        partition[name] = {'target_logit':float((write * gold_direction).sum(-1).mean()),
                           'target_vs_other_values_margin':float((write * direction).sum(-1).mean())}
    value_trace = trace_matching_values(model, cache, data, induction_head, matching_writes, gold_direction, direction)
    return {'direct_components':parts, 'selected_head_source_partition':partition,
            'matching_source_value_decomposition':value_trace,
            'residual_max_reconstruction_error':float((reconstructed - observed).abs().max()),
            'margin_max_reconstruction_error':float((sum_margin - actual_margin).abs().max()),
            'z_max_reconstruction_error':float((reconstructed_z - observed_z).abs().max()),
            'actual_mean_target_vs_other_values_margin':float(actual_margin.mean()),
            'interpretation':'Direct additive logit contributions in the observed clean run; not ablation effects.'}
