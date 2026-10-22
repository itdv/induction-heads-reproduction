# Results: prediction and direct support for B

Run on 2026-10-22, on the same 32 sequences and seed as the previous traced run.
One model forward pass supplies both head selection and the analysis. No training
or activation interventions.

## Does the model predict B?

B is the top prediction over the full vocabulary in **27/128 queries (21.09%)**.
The mean probability of B is **9.12%**. Per-query target IDs, predicted IDs and
B probabilities are saved in results.json.

This is partial success. A positive contribution to B's logit does not guarantee
that B wins against the full vocabulary.

## Attention and keys

Selected heads: L0H3 and L1H5. Matching-source attention is 60.27%.
The top attention source matches in 71/128 queries (55.47%).

| Raw attention-score contribution | Matching sources | Other pair sources | Difference |
|---|---:|---:|---:|
| Cached score | 13.4412 | 6.0746 | 7.3665 |
| L0H3 contribution through K | 7.3815 | 0.1953 | 7.1862 |
| Within L0H3: preceding-token read | 7.3579 | 0.1695 | 7.1884 |

These key statistics describe existing sources in the same input. The final row
is contained in the L0H3 row.

## Direct contributions to B

The output analysis now measures B's raw logit directly. It no longer subtracts
other pair values or any other-token baseline.

| Component | Mean contribution to B's logit |
|---|---:|
| Full model B logit | 5.9779 |
| L1H5 | 7.5742 |
| Within L1H5: matching sources | 7.5288 |
| Within L1H5: all other sources | 0.0454 |
| Within matching sources: token embedding | 4.5476 |
| Within matching sources: L0H2 writes | 3.0524 |

L1H5 is the largest positive contributor on average. Negative components, notably
the final layer's output bias (-2.4915), make the full logit smaller than L1H5's
contribution. Do not interpret contributions as percentages of probability.

The source and value rows are nested parts, not extra terms to add to the full head.
All model-component contributions are included in the saved report, along with
per-query predictions and selected-head contributions.

## Verification and files

The largest reconstruction error is 0.00000572. Six tests passed, including
nonzero biases, full-logit reconstruction, prediction checks against a normal
forward pass, and unchanged cache/weights.

- [Readable tables](results/target-reference/summary.md).
- [Predictions and contributions](results/target-reference/results.json).
- [Exact inputs](results/target-reference/inputs.json).
- [Settings and revisions](results/target-reference/protocol.json).

These are additive contributions in the actual run, not ablation effects.
Older result folders record earlier metrics and are not the current output analysis.
