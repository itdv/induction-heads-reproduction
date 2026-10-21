"""Test the low-rank spectral calculation and which token maps are being analyzed."""
from types import SimpleNamespace

import pytest
import torch

from paper_circuits import eigenvalue_summary, ov_factors, matching_factors


def test_low_rank_score_matches_full_matrix_including_complex_eigenvalues():
    # Eigenvalues are i, -i and 2; an extra zero eigenvalue must have no effect.
    left = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 2.], [0., 0., 0.]], dtype=torch.float64)
    right = torch.eye(4, dtype=torch.float64)[:3]
    measured = eigenvalue_summary(left, right)
    full = torch.linalg.eigvals(left @ right)
    assert measured['positivity'] == pytest.approx(float(full.real.sum() / full.abs().sum()))
    assert measured['positivity'] == pytest.approx(.5)  # Includes imaginary magnitudes.


def test_zero_map_has_undefined_score_rather_than_spurious_copying():
    assert eigenvalue_summary(torch.zeros(5, 2), torch.zeros(2, 5))['positivity'] is None


def toy_model():
    rng = torch.Generator().manual_seed(7)
    def random(*shape):
        return torch.randn(*shape, generator=rng, dtype=torch.float64)
    return SimpleNamespace(W_E=random(7, 5), W_U=random(5, 7),
        W_V=random(2, 2, 5, 3), W_O=random(2, 2, 3, 5),
        W_Q=random(2, 2, 5, 3), W_K=random(2, 2, 5, 3),
        blocks=[None, SimpleNamespace(attn=SimpleNamespace(attn_scale=3 ** .5))])


def test_ov_rows_are_source_tokens_and_columns_are_output_logits():
    model = toy_model()
    left, right = ov_factors(model, head=1)
    source_token = 2
    value = model.W_E[source_token] @ model.W_V[1, 1]
    expected_logits = (value @ model.W_O[1, 1]) @ model.W_U
    torch.testing.assert_close((left @ right)[source_token], expected_logits)


def test_matching_rows_are_queries_and_columns_are_previous_source_tokens():
    model = toy_model()
    left, right = matching_factors(model, early_head=0, late_head=1)
    current_token, earlier_token = 2, 4
    query = model.W_E[current_token] @ model.W_Q[1, 1]
    early_write = (model.W_E[earlier_token] @ model.W_V[0, 0]) @ model.W_O[0, 0]
    key = early_write @ model.W_K[1, 1]
    expected_score = (query @ key) / model.blocks[1].attn.attn_scale
    assert float((left @ right)[current_token, earlier_token]) == pytest.approx(float(expected_score))
