"""Find candidate heads and trace information through an actual forward pass.

Run from the repository root: python experiments/induction_pairs.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import torch

from pair_config import ROOT, MODEL, REVISION, TOKENIZER, TOKENIZER_REVISION
from pair_data import PairBatch, make_pair_batch
from pair_measurements import AttentionScores, SelectedHeads, measure_attention, select_heads
from pair_model import load_model, read_attention, read_run
from pair_trace import RunTrace, trace_run

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


class ExperimentReport(TypedDict):
    selected_heads: SelectedHeads
    query_count: int
    discovery_attention: list[AttentionScores]
    evaluation_attention: list[AttentionScores]
    trace: RunTrace


@dataclass(frozen=True)
class Settings:
    batch: int = 32
    pairs: int = 4
    repeats: int = 4
    min_gap: int = 1
    max_gap: int = 9
    seed: int = 20261021  # Evaluation uses seed + 1.
    threads: int = 2
    output: str = "work/induction/trace-01"


def parse_settings() -> Settings:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Settings()
    for name in ("batch", "pairs", "repeats", "min_gap", "max_gap", "seed", "threads"):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(defaults, name),
        )
    parser.add_argument("--output", default=defaults.output)
    settings = Settings(**vars(parser.parse_args()))
    if settings.threads < 1:
        parser.error("--threads must be positive")
    if settings.pairs < 2:
        parser.error("--pairs must be at least 2 for output tracing")
    return settings


def build_data(model: HookedTransformer, settings: Settings, seed: int) -> PairBatch:
    return make_pair_batch(
        model.cfg.d_vocab, model.tokenizer.bos_token_id,
        model.tokenizer.all_special_ids,
        batch=settings.batch, pairs=settings.pairs, repeats=settings.repeats,
        min_gap=settings.min_gap, max_gap=settings.max_gap, seed=seed,
    )


@torch.no_grad()
def run_experiment(
    model: HookedTransformer, settings: Settings,
) -> tuple[ExperimentReport, dict[str, dict[str, object]]]:
    # 1. Choose heads using one set of random sequences.
    discovery = build_data(model, settings, settings.seed)
    discovery_scores = measure_attention(read_attention(model, discovery), discovery)
    selected = select_heads(discovery_scores)

    # 2. Save the actual evaluation run and measure its attention.
    evaluation = build_data(model, settings, settings.seed + 1)
    cache = read_run(model, evaluation)
    patterns = [cache["pattern", layer] for layer in range(model.cfg.n_layers)]
    evaluation_scores = measure_attention(patterns, evaluation)

    # 3. Follow key and output contributions in that same, unchanged run.
    trace = trace_run(model, cache, evaluation, selected["early_head"], selected["late_head"])

    report: ExperimentReport = {
        "selected_heads": selected,
        "query_count": len(evaluation.last_occurrences()),
        "discovery_attention": discovery_scores,
        "evaluation_attention": evaluation_scores,
        "trace": trace,
    }
    inputs = {"discovery": discovery.to_dict(), "evaluation": evaluation.to_dict()}
    return report, inputs


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_summary(path: Path, report: ExperimentReport) -> None:
    selected = report["selected_heads"]
    early, late = selected["early_head"], selected["late_head"]
    attention = report["evaluation_attention"][1]
    lines = [
        "# Information flow in the actual run",
        "",
        f"Selected heads: L0H{early} and L1H{late}.",
        f"Evaluation queries: {report['query_count']}.",
        f"Matching-source attention: {attention['matching_source_mass'][late]:.4f}.",
        f"Top source match rate: {attention['top_source_match_rate'][late]:.4f}.",
        "",
        "## Key contributions to raw attention scores",
        "",
        "Matching and other sources are earlier pair-B positions in the same inputs.",
        "Values are means over query-source edges, before softmax.",
        "",
        "| Key component | Matching sources | Other pair sources | Difference |",
        "|---|---:|---:|---:|",
    ]
    keys = report["trace"]["keys"]
    score_rows = {"Total cached score": keys["total_score"], **keys["key_parts"]}
    score_rows["Preceding-token read within selected early head"] = keys["preceding_token_key_score"]
    for name, scores in score_rows.items():
        other = scores["other_pairs"]
        difference = scores["difference"]
        other_text = "n/a" if other is None else f"{other:.4f}"
        difference_text = "n/a" if difference is None else f"{difference:.4f}"
        lines.append(f"| {name} | {scores['matching']:.4f} | {other_text} | {difference_text} |")
    output = report["trace"]["output"]
    lines.extend([
        "",
        "The preceding-token row is part of the early-head row; do not add it again.",
        "",
        "## Output contributions",
        "",
        "A margin is the target logit minus the mean logit of the other pair values.",
        "These are direct contributions from the selected late head, not prediction accuracy.",
        "",
        f"Selected head margin: {output['head_margin']:.4f}.",
        f"From matching sources: {output['matching_sources_margin']:.4f}.",
        f"From all other sources: {output['other_sources_margin']:.4f}.",
        "",
        "| Value component within matching sources | Margin contribution |",
        "|---|---:|",
    ])
    for name, margin in output["matching_value_parts"].items():
        lines.append(f"| {name} | {margin:.4f} |")
    errors = [keys["key_reconstruction_error"], keys["score_reconstruction_error"],
              output["value_reconstruction_error"], output["head_reconstruction_error"]]
    lines.extend([
        "",
        f"Largest reconstruction error: {max(errors):.8f}.",
        "No activations were changed. Contributions are not intervention effects.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    settings = parse_settings()
    output = ROOT / settings.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(settings.threads)

    print("Loading the model.", flush=True)
    model = load_model()
    print("Measuring attention and tracing the cached run.", flush=True)
    report, inputs = run_experiment(model, settings)

    save_json(output / "protocol.json", {
        "model": MODEL, "model_revision": REVISION,
        "tokenizer": TOKENIZER, "tokenizer_revision": TOKENIZER_REVISION,
        "settings": asdict(settings), "training": False,
        "analysis": "Additive key and output contributions in the actual cached run",
    })
    save_json(output / "inputs.json", inputs)
    save_json(output / "results.json", report)
    write_summary(output / "summary.md", report)
    print((output / "summary.md").read_text(encoding="utf-8"), flush=True)
    print(f"Saved results to {output}", flush=True)


if __name__ == "__main__":
    main()
