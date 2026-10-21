"""Random background with recurring A->B pairs, without a repeated whole sequence."""
from dataclasses import asdict, dataclass, replace
import random

import torch


@dataclass(frozen=True)
class PairEvent:
    row: int
    pair: int
    occurrence: int
    query: int
    target: int
    previous_values: tuple[int, ...]


@dataclass
class PairBatch:
    tokens: torch.Tensor
    lengths: list[int]
    pairs: list[list[tuple[int, int]]]
    reserves: list[list[int]]
    events: list[PairEvent]
    repeats: int

    def queries(self, occurrence=-1):
        """Choose one occurrence per pair; -1 means the last occurrence."""
        occurrence = self.repeats - 1 if occurrence == -1 else occurrence
        return [event for event in self.events if event.occurrence == occurrence]

    def serializable(self):
        return {"tokens": self.tokens.tolist(), "lengths": self.lengths,
                "pairs": self.pairs, "reserves": self.reserves,
                "events": [asdict(event) for event in self.events], "repeats": self.repeats}


def make_pair_batch(vocab, bos, special_ids, *, batch=32, pairs=4, repeats=4,
                    min_gap=1, max_gap=9, seed=42):
    """Shuffle pair order anew in each round and put fresh random noise before each pair.

    Pair IDs and reserved control IDs are excluded from noise, preventing accidental
    A->other continuations. Sampling uses a local RNG; model weights are untouched.
    Rounds balance exposure counts but their token sequences and lengths differ.
    """
    if batch < 1 or pairs < 2 or repeats < 2 or not 1 <= min_gap < max_gap:
        raise ValueError("Require batch>=1, pairs>=2, repeats>=2, 1<=min_gap<max_gap")
    allowed = [token for token in range(vocab) if token not in special_ids and token != bos]
    if len(allowed) <= 3 * pairs:
        raise ValueError("Vocabulary too small for distinct pairs, reserves and background")
    rng = random.Random(seed)
    sequences, all_pairs, reserves, events = [], [], [], []
    for row in range(batch):
        chosen = rng.sample(allowed, 3 * pairs)
        mapping = list(zip(chosen[:pairs], chosen[pairs:2 * pairs]))
        reserved = chosen[2 * pairs:]
        excluded = set(chosen)
        background = [token for token in allowed if token not in excluded]
        sequence = [bos]
        history = [[] for _ in mapping]
        for occurrence in range(repeats):
            # Each round contains every pair once, in a newly shuffled order.
            order = rng.sample(range(pairs), pairs)
            for pair in order:
                # Sample a fresh background length and fresh background tokens.
                sequence.extend(rng.choices(background, k=rng.randint(min_gap, max_gap)))
                trigger, target = mapping[pair]
                # The next position will contain A, whose next-token target is B.
                query = len(sequence)
                # Snapshot only PAST B positions; the current B must not be a source.
                events.append(PairEvent(row, pair, occurrence, query, target, tuple(history[pair])))
                sequence.extend([trigger, target])
                history[pair].append(query + 1)
        sequences.append(sequence)
        all_pairs.append(mapping)
        reserves.append(reserved)
    lengths = [len(sequence) for sequence in sequences]
    tokens = torch.full((batch, max(lengths)), bos, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        tokens[row, :len(sequence)] = torch.tensor(sequence)
    return PairBatch(tokens, lengths, all_pairs, reserves, events, repeats)


def remove_pair_history(data):
    """Change only earlier A tokens to reserved IDs, preserving B tokens and distances.

    The final-round queries still contain original A. None of these A tokens has
    appeared earlier in its own prefix. This is a control input, not a head ablation.
    """
    tokens = data.tokens.clone()
    for event in data.events:
        if event.occurrence < data.repeats - 1:
            tokens[event.row, event.query] = data.reserves[event.row][event.pair]
    return replace(data, tokens=tokens)


def rotate_pair_values(data):
    """Keep trigger positions and background, cyclically permute B identities."""
    mappings = [[(trigger, mapping[(i + 1) % len(mapping)][1])
                 for i, (trigger, _) in enumerate(mapping)] for mapping in data.pairs]
    tokens = data.tokens.clone()
    events = []
    for event in data.events:
        target = mappings[event.row][event.pair][1]
        tokens[event.row, event.query + 1] = target
        events.append(replace(event, target=target))
    return replace(data, tokens=tokens, pairs=mappings, events=events)


def best_fixed_offset(data):
    """Choose a fixed lag using discovery only, as a positional baseline."""
    queries = data.queries()
    hits = [sum(event.query - lag in event.previous_values for event in queries)
            for lag in range(1, data.tokens.shape[1])]
    lag = max(range(1, len(hits) + 1), key=lambda value: hits[value - 1])
    return lag


def fixed_offset_accuracy(data, lag):
    queries = data.queries()
    return sum(event.query - lag in event.previous_values for event in queries) / len(queries)
