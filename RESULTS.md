# Results: prediction and direct support for B

Run on 2026-10-23, on the same 32 sequences and seed as the earlier traced run.
The inputs, key trace, predictions and existing output contributions match the saved results.
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

Each write is projected onto B's output direction to measure its raw-logit contribution.

| Component | Mean contribution to B's logit |
|---|---:|
| Full model B logit | 5.9779 |
| L1H5 | 7.5742 |
| Everything else together | -1.5963 |
| Within L1H5: matching sources | 7.5288 |
| Within L1H5: all other sources | 0.0454 |
| Within matching sources: token embedding | 4.5476 |
| Within matching sources: L0H2 writes | 3.0524 |

The selected head contributes 126.7% of the mean total logit; everything else
contributes -26.7%. These are signed ratios of means, not probability shares.
Ratios can exceed 100% because other components reduce the total. They are
undefined when the mean total is zero. No positive-support denominator is used.

The source and value rows are nested parts, not extra terms to add to the full head.

## Following L0H2 back to the input

L0H2 has the largest mean early-head contribution through the late value path.
Its contribution of 3.0524 splits as follows:

| Original read by L0H2 at an earlier B position | Contribution to B's logit through L1H5 |
|---|---:|
| B at that same position | 3.0422 |
| A immediately before it | 0.0068 |
| All other positions | 0.0034 |
| Early value bias | 0.0000 |

Each entry includes both heads' observed attention weights and their value/output
maps. In this run, L0H2 mainly carries information from B itself into the late
head's value path. L0H3 carries information from the preceding A into its key path.
The per-query records also show the contribution from each original input position.
The new split reconstructs the existing L0H2 contribution with a maximum error
of 0.00000143.

## Verification and files

The largest reconstruction error is 0.00000572. Eight tests passed, including
nonzero biases, prediction checks against a normal
forward pass, and unchanged cache/weights, and known reads with separate value bias.

- [Readable tables](results/value-path-reference/summary.md).
- [Predictions and contributions](results/value-path-reference/results.json).
- [Exact inputs](results/value-path-reference/inputs.json).
- [Settings and revisions](results/value-path-reference/protocol.json).

These are additive contributions in the actual run, not ablation effects.
Older result folders record earlier metrics and are not the current output analysis.
