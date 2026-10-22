# Results: information flow in the actual run

Run on 2026-10-22. No training, input changes or activation interventions.
Inputs match the preceding mechanism reference; this is not a new independent sample.

## Attention

Discovery selects L0H3 and L1H5. Evaluation has 128 queries in 32 sequences.

L1H5 puts 60.27% of its attention on earlier matching B sources.
Its top attention position is a matching source in 71/128 queries (55.47%).
These are attention measurements, not next-token prediction accuracy.

## Key contributions

Hold the actual L1H5 Q fixed. Decompose its cached K, including positions and biases.

| Raw-score contribution | Matching sources | Other pair sources | Difference |
|---|---:|---:|---:|
| Total cached attention score | 13.4412 | 6.0746 | 7.3665 |
| Key contribution from L0H3 | 7.3815 | 0.1953 | 7.1862 |
| Within L0H3: read from preceding token | 7.3579 | 0.1695 | 7.1884 |

The last row is contained in the L0H3 row. It must not be added again.
The comparison groups existing sources in the same unchanged input.

## Output contributions

Report direct contributions to the correct target logit minus the mean logit of
the other pair values in that context.

| Contribution | Margin |
|---|---:|
| Full L1H5 write | 7.4430 |
| L1H5 reads from matching sources | 7.4805 |
| L1H5 reads from all other sources | -0.0375 |
| Within matching sources: token embeddings | 4.5118 |
| Within matching sources: L0H2 writes | 3.0210 |

The embedding and L0H2 rows are parts of the matching-source row.
This is a direct logit decomposition, not prediction accuracy or an ablation effect.

## Reconstruction

The largest discrepancy when reconstructing cached K, attention scores, Z and
matching-source writes is 0.00000572. The same code passed tests on small real
Shortformer runs with nonzero biases, leaving all cached tensors and weights unchanged.

These results support a context-dependent path from L0H3 into L1H5's key and from
matching sources into its output. They do not establish that this is the only
path, that two heads alone suffice, or that all contributions are causal effects.

## Files

- [Readable tables](results/run-reference/summary.md).
- [All contributions and per-edge records](results/run-reference/results.json).
- [Exact inputs](results/run-reference/inputs.json).
- [Settings and revisions](results/run-reference/protocol.json).

Six tests passed. The full pretrained-model run completed.
