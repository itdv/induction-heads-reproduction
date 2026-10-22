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

Full model target logit: 5.9779.
Selected head contribution: 7.5742.
From matching sources: 7.5288.
From all other sources: 0.0454.

| Model component | Contribution to B's logit |
|---|---:|
| token_embedding | -0.0670 |
| early_output_bias | -0.1324 |
| L0H0 | -0.0119 |
| L0H1 | 0.0119 |
| L0H2 | 0.0254 |
| L0H3 | -0.0114 |
| L0H4 | -0.0008 |
| L0H5 | -0.0487 |
| L0H6 | 0.0790 |
| L0H7 | -0.0091 |
| L1H0 | -0.0111 |
| L1H1 | 0.3533 |
| L1H2 | 0.6890 |
| L1H3 | 0.1183 |
| L1H4 | -0.0341 |
| L1H5 | 7.5742 |
| L1H6 | 0.0400 |
| L1H7 | -0.0104 |
| late_output_bias | -2.4915 |
| unembedding_bias | -0.0850 |

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

Largest reconstruction error: 0.00000572.
No activations were changed. Contributions are not intervention effects.
