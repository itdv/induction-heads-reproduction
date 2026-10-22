# Information flow in the actual run

Selected heads: L0H3 and L1H5.
Queries: 128.
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

A margin is the target logit minus the mean logit of the other pair values.
These are direct contributions from the selected late head, not prediction accuracy.

Selected head margin: 7.4430.
From matching sources: 7.4805.
From all other sources: -0.0375.

| Value component within matching sources | Margin contribution |
|---|---:|
| token_embedding | 4.5118 |
| early_output_bias | -0.0069 |
| L0H0 | 0.0026 |
| L0H1 | 0.0059 |
| L0H2 | 3.0210 |
| L0H3 | -0.0917 |
| L0H4 | 0.0911 |
| L0H5 | -0.0482 |
| L0H6 | 0.0104 |
| L0H7 | -0.0155 |
| value_bias | 0.0000 |

Largest reconstruction error: 0.00000572.
No activations were changed. Contributions are not intervention effects.
