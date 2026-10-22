"""Test copying through OV and same-token matching through composed QK.

These are the two weight tests in Elhage et al. (2021).
Weights use TransformerLens's row-vector convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


class Eigenvalue(TypedDict):
    real: float
    imag: float


class EigenvalueSummary(TypedDict):
    positivity: float | None
    eigenvalues: list[Eigenvalue]


class HeadCircuitScores(TypedDict):
    head: int
    ov_copying: EigenvalueSummary
    qk_same_matching: EigenvalueSummary


class CircuitAnalysis(TypedDict):
    early_head: int
    late_layer: int
    score: str
    assumptions: str
    source: str
    heads: list[HeadCircuitScores]


def ov_factors(
    model: HookedTransformer, head: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """left @ right maps a source token to output token scores."""
    embedding = model.W_E.double()
    left = embedding @ model.W_V[1, head].double()
    right = model.W_O[1, head].double() @ model.W_U.double()
    return left, right


def matching_factors(
    model: HookedTransformer, early_head: int, late_head: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """left @ right matches a current token to an earlier preceding token."""
    embedding = model.W_E.double()

    # The current token supplies the late head's query.
    query = embedding @ model.W_Q[1, late_head].double()

    # The early head writes information about the preceding token.
    early_write = (
        embedding
        @ model.W_V[0, early_head].double()
        @ model.W_O[0, early_head].double()
    )
    key = early_write @ model.W_K[1, late_head].double()
    scale = model.blocks[1].attn.attn_scale
    return query, key.T / scale


def eigenvalue_summary(
    left: torch.Tensor, right: torch.Tensor,
) -> EigenvalueSummary:
    """Compute the paper's score without building a vocabulary-sized map.

    The nonzero eigenvalues of left @ right equal those of right @ left.
    Here right @ left is only 64 by 64.
    """
    eigenvalues = torch.linalg.eigvals(right @ left)
    denominator = eigenvalues.abs().sum()
    score = float(eigenvalues.real.sum() / denominator) if denominator > 0 else None
    return {
        "positivity": score,
        "eigenvalues": [
            {"real": float(value.real), "imag": float(value.imag)}
            for value in eigenvalues
        ],
    }


@torch.no_grad()
def analyze_circuits(model: HookedTransformer, early_head: int) -> CircuitAnalysis:
    """Apply the same two tests to every head in the second layer."""
    heads: list[HeadCircuitScores] = []
    for head in range(model.cfg.n_heads):
        ov = eigenvalue_summary(*ov_factors(model, head))
        qk = eigenvalue_summary(*matching_factors(model, early_head, head))
        heads.append({"head": head, "ov_copying": ov, "qk_same_matching": qk})

    return {
        "early_head": early_head,
        "late_layer": 1,
        "score": "real(sum(eigenvalues)) / sum(abs(eigenvalues))",
        "assumptions": "Weight maps only; no biases, positions or other paths.",
        "source": "https://transformer-circuits.pub/2021/framework/index.html",
        "heads": heads,
    }
