"""Exact additive decompositions of the UNMODIFIED run (no hooks changing activations).

QK terms are contributions to pre-softmax scores, not additive attention probabilities.
Logit terms are direct contributions, not causal effects or probabilities.
The algebra is restricted to the architecture checked in pair_model.validate_model.
"""
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
    """Reconstruct every selected QK score, including embeddings, positions and biases."""
    edges = edge_indices(data)
    rows = torch.tensor([edge['row'] for edge in edges])
    queries = torch.tensor([edge['query'] for edge in edges])
    sources = torch.tensor([edge['source'] for edge in edges])
    matching = torch.tensor([edge['matching'] for edge in edges])
    components = early_components(model, cache)
    q_parts = project_components(components, rows, queries,
                                 model.W_Q[1, induction_head], model.b_Q[1, induction_head])
    k_parts = project_components(components, rows, sources,
                                 model.W_K[1, induction_head], model.b_K[1, induction_head])
    actual_q = cache['blocks.1.attn.hook_q'][rows, queries, induction_head]
    actual_k = cache['blocks.1.attn.hook_k'][rows, sources, induction_head]
    q_sum, k_sum = stack_components(q_parts).sum(1), stack_components(k_parts).sum(1)
    torch.testing.assert_close(q_sum, actual_q, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(k_sum, actual_k, atol=3e-4, rtol=3e-4)
    scale = model.blocks[1].attn.attn_scale
    score_terms = torch.einsum('eid,ejd->eij', stack_components(q_parts), stack_components(k_parts)) / scale
    actual_scores = (actual_q * actual_k).sum(-1) / scale
    torch.testing.assert_close(score_terms.sum((1, 2)), actual_scores, atol=3e-4, rtol=3e-4)
    recorded_scores = cache['blocks.1.attn.hook_attn_scores'][rows, induction_head, queries, sources]
    torch.testing.assert_close(actual_scores, recorded_scores, atol=3e-4, rtol=3e-4)

    # Keep the actual full Q fixed. These K-component scores sum to the full score.
    key_terms = (stack_components(k_parts) * actual_q[:, None]).sum(-1) / scale
    selected_key = key_terms[:, list(k_parts).index(f'L0H{previous_head}')]

    # Isolate the actual earlier head edge (A -> following B), including its V bias.
    previous_attention = cache['blocks.0.attn.hook_pattern'][rows, previous_head, sources, sources - 1]
    previous_v = cache['blocks.0.attn.hook_v'][rows, sources - 1, previous_head]
    edge_write = previous_attention[:, None] * (previous_v @ model.W_O[0, previous_head])
    edge_key = edge_write @ model.W_K[1, induction_head]
    edge_key_score = (actual_q * edge_key).sum(-1) / scale

    # The content-only part excludes b_V, rather than calling all of the V vector "A".
    previous_tokens = data.tokens[rows, sources - 1]
    content_v = model.W_E[previous_tokens] @ model.W_V[0, previous_head]
    content_write = previous_attention[:, None] * (content_v @ model.W_O[0, previous_head])
    content_score = (actual_q * (content_write @ model.W_K[1, induction_head])).sum(-1) / scale
    for i, edge in enumerate(edges):
        edge.update(total_qk_score=float(actual_scores[i]),
                    early_head_key_score=float(selected_key[i]),
                    previous_token_edge_key_score=float(edge_key_score[i]),
                    previous_token_content_key_score=float(content_score[i]),
                    early_previous_attention=float(previous_attention[i]),
                    late_attention=float(cache['blocks.1.attn.hook_pattern'][rows[i], induction_head, queries[i], sources[i]]))
    report = {'component_names':list(q_parts),
              'q_max_reconstruction_error':float((q_sum - actual_q).abs().max()),
              'k_max_reconstruction_error':float((k_sum - actual_k).abs().max()),
              'score_max_reconstruction_error':float((score_terms.sum((1, 2)) - actual_scores).abs().max()),
              'recorded_score_max_error':float((actual_scores - recorded_scores).abs().max()),
              'full_qk_scores':mean_groups(actual_scores, matching),
              'qk_component_matrix':mean_groups(score_terms, matching),
              'key_component_scores_with_actual_q':mean_groups(key_terms, matching),
              'previous_token_edge_key_score':mean_groups(edge_key_score, matching),
              'previous_token_content_key_score':mean_groups(content_score, matching),
              'edges':edges}
    return report


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
