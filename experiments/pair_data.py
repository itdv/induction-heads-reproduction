"""Build random sequences with recurring A -> B pairs."""

from collections.abc import Collection
from dataclasses import asdict, dataclass
import random

import torch


@dataclass(frozen=True)
class PairOccurrence:
    sequence_index: int
    pair_index: int
    occurrence_index: int
    trigger_position: int
    target_token: int
    source_positions: tuple[int, ...]  # Earlier B positions for this pair only.


@dataclass
class PairBatch:
    tokens: torch.Tensor
    lengths: list[int]  # Real lengths, before right padding.
    pairs: list[list[tuple[int, int]]]
    occurrences: list[PairOccurrence]
    repeats: int

    def last_occurrences(self) -> list[PairOccurrence]:
        return [
            event for event in self.occurrences
            if event.occurrence_index == self.repeats - 1
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "tokens": self.tokens.tolist(),
            "lengths": self.lengths,
            "pairs": self.pairs,
            "occurrences": [asdict(event) for event in self.occurrences],
            "repeats": self.repeats,
        }


def make_pair_batch(
    vocab: int, bos: int, special_ids: Collection[int], *,
    batch: int = 32, pairs: int = 4, repeats: int = 4,
    min_gap: int = 1, max_gap: int = 9, seed: int = 42,
) -> PairBatch:
    """Each round contains every pair once, with new order and random gaps."""
    if batch < 1 or pairs < 1 or repeats < 2 or not 1 <= min_gap <= max_gap:
        raise ValueError("Use batch>=1, pairs>=1, repeats>=2 and 1<=min_gap<=max_gap.")

    allowed_tokens = [
        token for token in range(vocab)
        if token != bos and token not in special_ids
    ]
    if len(allowed_tokens) <= 2 * pairs:
        raise ValueError("The vocabulary must leave room for background tokens.")

    rng = random.Random(seed)
    sequences: list[list[int]] = []
    all_pairs: list[list[tuple[int, int]]] = []
    occurrences: list[PairOccurrence] = []

    for sequence_index in range(batch):
        pair_tokens = rng.sample(allowed_tokens, 2 * pairs)
        mapping = list(zip(pair_tokens[:pairs], pair_tokens[pairs:]))
        background = [token for token in allowed_tokens if token not in pair_tokens]
        sequence = [bos]
        source_history: list[list[int]] = [[] for _ in mapping]

        for occurrence_index in range(repeats):
            pair_order = rng.sample(range(pairs), pairs)
            for pair_index in pair_order:
                gap_length = rng.randint(min_gap, max_gap)
                sequence.extend(rng.choices(background, k=gap_length))
                trigger, target = mapping[pair_index]
                trigger_position = len(sequence)

                # Record past sources before adding the current B.
                occurrences.append(PairOccurrence(
                    sequence_index, pair_index, occurrence_index,
                    trigger_position, target, tuple(source_history[pair_index]),
                ))
                sequence.extend([trigger, target])
                source_history[pair_index].append(trigger_position + 1)

        sequences.append(sequence)
        all_pairs.append(mapping)

    lengths = [len(sequence) for sequence in sequences]
    tokens = torch.full((batch, max(lengths)), bos, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        tokens[row, :len(sequence)] = torch.tensor(sequence)

    return PairBatch(tokens, lengths, all_pairs, occurrences, repeats)
