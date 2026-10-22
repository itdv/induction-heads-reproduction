# Information flow in the actual run

Selected heads: L0H3 and L1H5.
Queries: 128.
Correct top-1 predictions: 27/128.
Mean probability of B: 0.0912.
Matching-source attention: 0.6027.
Top source match rate: 0.5547.

## Key contributions to raw attention scores

Matching and other sources are earlier pair-B positions in the same inputs.
Values are means over query-source edges, before softmax.

| Key component | Matching sources | Other pair sources | Difference |
|---|---:|---:|---:|
| Total cached score | 13.4412 | 6.0746 | 7.3665 |
| token_embedding | -0.0498 | -0.1393 | 0.0895 |
| early_output_bias | -0.5275 | -0.5282 | 0.0007 |
| L0H0 | 0.0043 | 0.0050 | -0.0007 |
| L0H1 | 0.1706 | 0.1773 | -0.0067 |
| L0H2 | -0.0783 | -0.0563 | -0.0220 |
| L0H3 | 7.3815 | 0.1953 | 7.1862 |
| L0H4 | 0.0549 | 0.0433 | 0.0116 |
| L0H5 | 0.2363 | 0.0774 | 0.1589 |
| L0H6 | 0.1927 | 0.2021 | -0.0093 |
| L0H7 | 0.2847 | 0.3475 | -0.0628 |
| position_embedding | -1.3360 | -1.3593 | 0.0233 |
| key_bias | 7.1077 | 7.1097 | -0.0020 |
| Preceding-token read within selected early head | 7.3579 | 0.1695 | 7.1884 |

The preceding-token row is part of the early-head row; do not add it again.

## Output contributions

All contributions below are to B's raw logit. No other-token baseline is used.
The prediction check above uses the full vocabulary; logit contributions are not probabilities.

| Component | Mean contribution to B's logit | Percent of mean total |
|---|---:|---:|
| Full model | 5.9779 | 100% |
| Selected head | 7.5742 | 126.7% |
| Everything else | -1.5963 | -26.7% |

Percentages are signed ratios of means, not probabilities or positive-support shares.
They can be negative or exceed 100%; they are undefined when the mean total is zero.

From matching sources: 7.5288.
From all other sources: 0.0454.

| Value component within matching sources | Contribution to B's logit |
|---|---:|
| token_embedding | 4.5476 |
| early_output_bias | -0.0646 |
| L0H0 | 0.0016 |
| L0H1 | 0.0375 |
| L0H2 | 3.0524 |
| L0H3 | -0.1323 |
| L0H4 | 0.1037 |
| L0H5 | -0.0640 |
| L0H6 | 0.0574 |
| L0H7 | -0.0106 |
| value_bias | 0.0000 |

## Input reads through L0H2 and the late head

The early head is selected by its mean value-path contribution above.
Each input read is weighted by both heads' actual attention weights.

| Early read | Contribution to B's logit |
|---|---:|
| source_itself | 3.0422 |
| token_before_source | 0.0068 |
| other_tokens | 0.0034 |
| early_value_bias | 0.0000 |

source_itself means the earlier continuation position; token_before_source means its trigger.
These rows split the selected early-head value contribution; do not add them to it again.

Largest reconstruction error: 0.00000572.
No activations were changed. Contributions are not intervention effects.
