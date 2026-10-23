"""Check B's prediction at every occurrence of a pair, as a reference for the final one.

At the first occurrence the pair has not been seen, so in-context copying is impossible.
Run from the repository root: python experiments/occurrence_baseline.py
"""

from __future__ import annotations

import json

import torch

from induction_pairs import Settings, build_data
from pair_config import ROOT
from pair_measurements import measure_predictions
from pair_model import load_model, read_run

OUTPUT = ROOT / "work/induction/occurrence-baseline-01"


@torch.no_grad()
def main() -> None:
    settings = Settings()
    torch.set_num_threads(settings.threads)
    OUTPUT.mkdir(parents=True, exist_ok=False)

    model = load_model()
    data = build_data(model, settings, settings.seed)
    cache = read_run(model, data)

    rows = []
    for occurrence in range(settings.repeats):
        events = [e for e in data.occurrences if e.occurrence_index == occurrence]
        scores = measure_predictions(model, cache, data, events)
        rows.append({
            "occurrence": occurrence + 1,
            "correct": scores["correct"],
            "total": scores["total"],
            "mean_target_probability": scores["mean_target_probability"],
        })

    report = {"seed": settings.seed, "vocabulary": model.cfg.d_vocab, "occurrences": rows}
    (OUTPUT / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(f"occurrence {row['occurrence']}: {row['correct']}/{row['total']} top-1, "
              f"mean P(B) = {row['mean_target_probability']:.4f}")


if __name__ == "__main__":
    main()
