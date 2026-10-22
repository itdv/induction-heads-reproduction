"""Trace keys and output contributions in one unchanged forward pass.

All attention, Q, K, V and Z values come from that run's cache.
The decomposition is specific to this two-layer attention-only Shortformer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import torch

from pair_data import PairBatch

if TYPE_CHECKING:
    from transformer_lens import ActivationCache, HookedTransformer


class SourceMeans(TypedDict):
    matching: float
    other_pairs: float | None
    difference: float | None


class EdgeRecord(TypedDict):
    row: int
    query: int
    source: int
    matching: bool
    score: float
    early_head_key_score: float
    preceding_token_key_score: float
    early_attention_to_preceding_token: float
    late_attention: float


class KeyTrace(TypedDict):
    total_score: SourceMeans
    key_parts: dict[str, SourceMeans]
    preceding_token_key_score: SourceMeans
    key_reconstruction_error: float
    score_reconstruction_error: float
    edges: list[EdgeRecord]


class ValueReadRecord(TypedDict):
    row: int
    query: int
    sources: list[int]
    token_ids: list[int]
    token_logit_contributions: list[float]
    value_bias_logit: float


class EarlyValuePath(TypedDict):
    early_head: int
    read_contributions: dict[str, float]
    reconstruction_error: float
    queries: list[ValueReadRecord]


class OutputTrace(TypedDict):
    target_logit: float
    head_logit: float
    rest_logit: float
    matching_sources_logit: float
    other_sources_logit: float
    matching_value_parts: dict[str, float]
    early_value_path: EarlyValuePath
    value_reconstruction_error: float
    head_reconstruction_error: float
    query_head_logits: list[float]
    query_matching_source_logits: list[float]


class RunTrace(TypedDict):
    keys: KeyTrace
    output: OutputTrace


@dataclass
class SourceEdges:
    """One entry per final A and earlier pair-B position in the same sequence."""
    rows: torch.Tensor
    queries: torch.Tensor
    sources: torch.Tensor
    matching: torch.Tensor


def source_edges(data: PairBatch) -> SourceEdges:
    rows, queries, sources, matching = [], [], [], []
    for query in data.last_occurrences():
        for event in data.occurrences:
            if (event.sequence_index == query.sequence_index
                    and event.trigger_position + 1 < query.trigger_position):
                rows.append(query.sequence_index)
                queries.append(query.trigger_position)
                sources.append(event.trigger_position + 1)
                matching.append(event.pair_index == query.pair_index)
    return SourceEdges(
        torch.tensor(rows), torch.tensor(queries), torch.tensor(sources),
        torch.tensor(matching, dtype=torch.bool),
    )


def source_means(values: torch.Tensor, matching: torch.Tensor) -> SourceMeans:
    correct = float(values[matching].mean())
    other = float(values[~matching].mean()) if (~matching).any() else None
    return {
        "matching": correct, "other_pairs": other,
        "difference": correct - other if other is not None else None,
    }


def check_sum(reconstructed: torch.Tensor, cached: torch.Tensor) -> float:
    """Check our arithmetic against the model's actual cached result."""
    torch.testing.assert_close(reconstructed, cached, atol=3e-4, rtol=3e-4)
    return float((reconstructed - cached).abs().max())


def early_writes(
    model: HookedTransformer, cache: ActivationCache,
) -> dict[str, torch.Tensor]:
    """The actual pieces of the residual stream after layer 0; no position here."""
    embedding = cache["embed"]
    parts = {
        "token_embedding": embedding,
        "early_output_bias": model.b_O[0].expand_as(embedding),
    }
    for head in range(model.cfg.n_heads):
        parts[f"L0H{head}"] = cache["z", 0][:, :, head] @ model.W_O[0, head]
    return parts


@torch.no_grad()
def trace_keys(
    model: HookedTransformer, cache: ActivationCache, data: PairBatch,
    early_head: int, late_head: int,
) -> KeyTrace:
    """Hold the real Q fixed and split the real K into its input contributions."""
    edges = source_edges(data)
    rows, queries, sources = edges.rows, edges.queries, edges.sources
    query = cache["q", 1][rows, queries, late_head]
    key = cache["k", 1][rows, sources, late_head]
    scale = model.blocks[1].attn.attn_scale
    key_weight = model.W_K[1, late_head]

    inputs = early_writes(model, cache)
    inputs["position_embedding"] = cache["pos_embed"].expand_as(cache["embed"])
    key_parts = {
        name: values[rows, sources] @ key_weight for name, values in inputs.items()
    }
    key_parts["key_bias"] = model.b_K[1, late_head].expand_as(key)
    key_error = check_sum(torch.stack(list(key_parts.values())).sum(0), key)

    part_scores = {
        name: (query * part).sum(-1) / scale for name, part in key_parts.items()
    }
    scores = cache["attn_scores", 1][rows, late_head, queries, sources]
    score_error = check_sum(torch.stack(list(part_scores.values())).sum(0), scores)

    # Follow the selected early head's actual read from the token before each B.
    early_attention = cache["pattern", 0][rows, early_head, sources, sources - 1]
    previous_value = cache["v", 0][rows, sources - 1, early_head]
    previous_write = early_attention[:, None] * (previous_value @ model.W_O[0, early_head])
    previous_key = previous_write @ key_weight
    previous_score = (query * previous_key).sum(-1) / scale
    early_score = part_scores[f"L0H{early_head}"]
    late_attention = cache["pattern", 1][rows, late_head, queries, sources]

    records: list[EdgeRecord] = [
        {
            "row": int(rows[i]), "query": int(queries[i]), "source": int(sources[i]),
            "matching": bool(edges.matching[i]), "score": float(scores[i]),
            "early_head_key_score": float(early_score[i]),
            "preceding_token_key_score": float(previous_score[i]),
            "early_attention_to_preceding_token": float(early_attention[i]),
            "late_attention": float(late_attention[i]),
        }
        for i in range(len(rows))
    ]
    return {
        "total_score": source_means(scores, edges.matching),
        "key_parts": {name: source_means(values, edges.matching)
                      for name, values in part_scores.items()},
        "preceding_token_key_score": source_means(previous_score, edges.matching),
        "key_reconstruction_error": key_error,
        "score_reconstruction_error": score_error,
        "edges": records,
    }


@torch.no_grad()
def trace_value_reads(
    model: HookedTransformer, cache: ActivationCache, data: PairBatch,
    early_head: int, late_head: int, target_direction: torch.Tensor,
    expected_logits: torch.Tensor,
) -> EarlyValuePath:
    """Follow input token -> early read -> late read -> B's logit.

    Both attention weights come from the same run. Separate the early value
    bias so it is not mistaken for information from a particular token.
    """
    value_to_output = (
        model.W_O[0, early_head]
        @ model.W_V[1, late_head]
        @ model.W_O[1, late_head]
    )
    directions = target_direction @ value_to_output.T
    groups: dict[str, list[torch.Tensor]] = {
        "source_itself": [], "token_before_source": [],
        "other_tokens": [], "early_value_bias": [],
    }
    records: list[ValueReadRecord] = []
    for i, event in enumerate(data.last_occurrences()):
        row, query = event.sequence_index, event.trigger_position
        sources = list(event.source_positions)
        late_weights = cache["pattern", 1][row, late_head, query, sources]
        early_weights = cache["pattern", 0][row, early_head, sources, :]
        content_values = cache["v", 0][row, :, early_head] - model.b_V[0, early_head]
        token_scores = content_values @ directions[i]

        # Each entry follows one original token through one earlier B position.
        contributions = late_weights[:, None] * early_weights * token_scores[None, :]
        source_index = torch.arange(len(sources))
        same = contributions[source_index, sources].sum()
        before = contributions[source_index, [s - 1 for s in sources]].sum()
        bias = late_weights.sum() * (model.b_V[0, early_head] @ directions[i])
        groups["source_itself"].append(same)
        groups["token_before_source"].append(before)
        groups["other_tokens"].append(contributions.sum() - same - before)
        groups["early_value_bias"].append(bias)

        prefix_length = max(sources) + 1
        records.append({
            "row": row, "query": query, "sources": sources,
            "token_ids": data.tokens[row, :prefix_length].tolist(),
            "token_logit_contributions": contributions.sum(0)[:prefix_length].tolist(),
            "value_bias_logit": float(bias),
        })

    parts = {name: torch.stack(values) for name, values in groups.items()}
    error = check_sum(torch.stack(list(parts.values())).sum(0), expected_logits)
    return {
        "early_head": early_head,
        "read_contributions": {name: float(values.mean()) for name, values in parts.items()},
        "reconstruction_error": error,
        "queries": records,
    }


@torch.no_grad()
def trace_output(
    model: HookedTransformer, cache: ActivationCache, data: PairBatch, late_head: int,
) -> OutputTrace:
    """Split the late head's actual write by source and by value input component."""
    events = data.last_occurrences()
    rows = torch.tensor([event.sequence_index for event in events])
    positions = torch.tensor([event.trigger_position for event in events])
    targets = torch.tensor([event.target_token for event in events])

    # A write dotted with this vector gives its contribution to B's logit.
    target_direction = model.W_U[:, targets].T

    attention = cache["pattern", 1][rows, late_head, positions]
    values = cache["v", 1][rows, :, late_head]
    cached_z = cache["z", 1][rows, positions, late_head]
    head_error = check_sum(torch.einsum("es,esd->ed", attention, values), cached_z)
    head_write = cached_z @ model.W_O[1, late_head]

    inputs = early_writes(model, cache)

    # Read the full B logit directly. Everything else is total minus this head.
    residual = cache["resid_post", 1][rows, positions]
    target_logit = (residual * target_direction).sum(-1) + model.b_U[targets]

    matching_writes = []
    value_parts: dict[str, list[torch.Tensor]] = {name: [] for name in inputs}
    value_parts["value_bias"] = []
    for event in events:
        row, position = event.sequence_index, event.trigger_position
        sources = list(event.source_positions)
        weights = cache["pattern", 1][row, late_head, position, sources]
        source_values = cache["v", 1][row, sources, late_head]
        matching_writes.append((weights @ source_values) @ model.W_O[1, late_head])
        for name, stream in inputs.items():
            weighted_input = weights @ stream[row, sources]
            value_parts[name].append(
                weighted_input @ model.W_V[1, late_head] @ model.W_O[1, late_head]
            )
        value_parts["value_bias"].append(
            weights.sum() * (model.b_V[1, late_head] @ model.W_O[1, late_head])
        )

    matching_write = torch.stack(matching_writes)
    parts = {name: torch.stack(writes) for name, writes in value_parts.items()}
    value_error = check_sum(torch.stack(list(parts.values())).sum(0), matching_write)
    head_logit = (head_write * target_direction).sum(-1)
    matching_logit = (matching_write * target_direction).sum(-1)
    value_logits = {
        name: (write * target_direction).sum(-1) for name, write in parts.items()
    }
    # Continue through the early head with the largest value-path contribution.
    early_value_head = max(
        range(model.cfg.n_heads), key=lambda head: float(value_logits[f"L0H{head}"].mean()),
    )
    early_value_path = trace_value_reads(
        model, cache, data, early_value_head, late_head, target_direction,
        value_logits[f"L0H{early_value_head}"],
    )
    return {
        "target_logit": float(target_logit.mean()),
        "head_logit": float(head_logit.mean()),
        "rest_logit": float((target_logit - head_logit).mean()),
        "matching_sources_logit": float(matching_logit.mean()),
        "other_sources_logit": float((head_logit - matching_logit).mean()),
        "matching_value_parts": {
            name: float(values.mean()) for name, values in value_logits.items()
        },
        "early_value_path": early_value_path,
        "value_reconstruction_error": value_error,
        "head_reconstruction_error": head_error,
        "query_head_logits": head_logit.tolist(),
        "query_matching_source_logits": matching_logit.tolist(),
    }


def trace_run(
    model: HookedTransformer, cache: ActivationCache, data: PairBatch,
    early_head: int, late_head: int,
) -> RunTrace:
    """Trace this run without changing any activation or model weight."""
    if not (model.cfg.n_layers == 2 and model.cfg.attn_only
            and model.cfg.normalization_type is None
            and model.cfg.positional_embedding_type == "shortformer"):
        raise ValueError("This decomposition needs a two-layer attention-only Shortformer without normalization.")
    return {
        "keys": trace_keys(model, cache, data, early_head, late_head),
        "output": trace_output(model, cache, data, late_head),
    }
