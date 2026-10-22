"""Find candidate heads, measure their attention, and test their OV/QK weights.

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
from pair_model import load_model, read_attention
from paper_circuits import CircuitAnalysis, analyze_circuits
from plot_circuits import save_circuit_plot

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


class ExperimentReport(TypedDict):
    selected_heads: SelectedHeads
    query_count: int
    discovery_attention: list[AttentionScores]
    evaluation_attention: list[AttentionScores]
    circuits: CircuitAnalysis


@dataclass(frozen=True)
class Settings:
    batch: int = 32
    pairs: int = 4
    repeats: int = 4
    min_gap: int = 1
    max_gap: int = 9
    seed: int = 20261021  # Evaluation uses seed + 1.
    threads: int = 2
    output: str = "work/induction/mechanism-01"


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

    # 2. Measure attention on a separate set of random sequences.
    evaluation = build_data(model, settings, settings.seed + 1)
    evaluation_scores = measure_attention(read_attention(model, evaluation), evaluation)

    # 3. Test OV and composed QK for every late head, using the selected early head.
    circuits = analyze_circuits(model, selected["early_head"])

    report: ExperimentReport = {
        "selected_heads": selected,
        "query_count": len(evaluation.last_occurrences()),
        "discovery_attention": discovery_scores,
        "evaluation_attention": evaluation_scores,
        "circuits": circuits,
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
        "# Mechanistic analysis",
        "",
        f"Selected heads: L0H{early} and L1H{late}.",
        f"Evaluation queries: {report['query_count']}.",
        f"Matching-source attention: {attention['matching_source_mass'][late]:.4f}.",
        f"Top source match rate: {attention['top_source_match_rate'][late]:.4f}.",
        "",
        "| Late head | OV score | QK score |",
        "|---|---:|---:|",
    ]
    for head in report["circuits"]["heads"]:
        ov = head["ov_copying"]["positivity"]
        qk = head["qk_same_matching"]["positivity"]
        ov_text = "undefined" if ov is None else f"{ov:.4f}"
        qk_text = "undefined" if qk is None else f"{qk:.4f}"
        lines.append(f"| L1H{head['head']} | {ov_text} | {qk_text} |")
    lines.extend([
        "",
        "Scores describe weights, not prediction accuracy.",
        "The maps omit biases, positions and other paths in the model.",
        "High scores support the proposed circuit; they do not prove it on every input.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    settings = parse_settings()
    output = ROOT / settings.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(settings.threads)

    print("Loading the model.", flush=True)
    model = load_model()
    print("Measuring attention and testing OV/QK weights.", flush=True)
    report, inputs = run_experiment(model, settings)

    save_json(output / "protocol.json", {
        "model": MODEL, "model_revision": REVISION,
        "tokenizer": TOKENIZER, "tokenizer_revision": TOKENIZER_REVISION,
        "settings": asdict(settings), "training": False,
    })
    save_json(output / "inputs.json", inputs)
    save_json(output / "results.json", report)
    write_summary(output / "summary.md", report)
    save_circuit_plot(report["circuits"], report["selected_heads"]["late_head"],
                      output / "circuits.png")
    print((output / "summary.md").read_text(encoding="utf-8"), flush=True)
    print(f"Saved results to {output}", flush=True)


if __name__ == "__main__":
    main()
