"""Measure attention and choose candidate heads. No prediction or interventions."""

import torch


def measure_attention(patterns, data):
    """patterns[layer] has shape [sequence, head, query, source]."""
    queries = data.last_occurrences()
    layer_scores = []

    for pattern in patterns:
        previous_token_scores = []
        matching_source_scores = []
        top_source_matches = []

        # At each real position, measure attention to the preceding token.
        for row, length in enumerate(data.lengths):
            positions = torch.arange(1, length)
            weights = pattern[row, :, positions, positions - 1]
            previous_token_scores.append(weights.mean(dim=-1))

        # At each final A, measure attention to all earlier matching B positions.
        for query in queries:
            weights = pattern[query.sequence_index, :, query.trigger_position]
            sources = torch.tensor(query.source_positions, dtype=torch.long)
            matching_source_scores.append(weights[:, sources].sum(dim=-1))
            top_source_matches.append(
                torch.isin(weights.argmax(dim=-1), sources).float()
            )

        layer_scores.append({
            "previous_token_mass": torch.stack(previous_token_scores).mean(0).tolist(),
            "matching_source_mass": torch.stack(matching_source_scores).mean(0).tolist(),
            "top_source_match_rate": torch.stack(top_source_matches).mean(0).tolist(),
        })

    return layer_scores


def select_heads(discovery_scores):
    """Choose an early previous-token head and a late matching-source head."""
    early_scores = discovery_scores[0]["previous_token_mass"]
    late_scores = discovery_scores[1]["matching_source_mass"]
    return {
        "early_head": max(range(len(early_scores)), key=early_scores.__getitem__),
        "late_head": max(range(len(late_scores)), key=late_scores.__getitem__),
    }
