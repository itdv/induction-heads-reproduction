"""Draw the OV/QK scores. Plotting is separate from the analysis."""

import os
from pathlib import Path

from pair_config import ROOT
from paper_circuits import CircuitAnalysis


def save_circuit_plot(circuits: CircuitAnalysis, selected_head: int, path: Path) -> None:
    cache_dir = ROOT / "work/mpl-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6, 5))
    for result in circuits["heads"]:
        ov = result["ov_copying"]["positivity"]
        qk = result["qk_same_matching"]["positivity"]
        if ov is None or qk is None:
            continue
        selected = result["head"] == selected_head
        axes.scatter(ov, qk, color="tab:red" if selected else "tab:gray")
        offset = (-42, 8) if selected else (5, -9)
        axes.annotate(
            f"L1H{result['head']}", (ov, qk),
            xytext=offset, textcoords="offset points",
        )

    axes.set(
        xlim=(-1.1, 1.15), ylim=(-1.1, 1.15),
        xlabel="OV score: copying", ylabel="QK score: same-token matching",
        title=f"Weight tests through L0H{circuits['early_head']}",
    )
    axes.axhline(0, color="lightgray", linewidth=.5)
    axes.axvline(0, color="lightgray", linewidth=.5)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
