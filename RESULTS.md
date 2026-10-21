# Reference experiment

Recorded on 2026-10-22 using the pinned pretrained model, without training. This run repeats the same inputs as the local development experiment; it is not an independent seed.

## Protocol

- Discovery seed: 20261021; evaluation seed: 20261022.
- 32 sequences in each set, four pairs per sequence, four occurrences per pair.
- Freshly shuffled pair order in each round; 1–9 background tokens before each pair.
- Four final-occurrence queries per evaluation sequence: 128 queries in total.
- Candidate heads selected on discovery data: previous-token head L0H3; later matching head L1H5. Head indices start at zero.

## Prediction

Top-1 predictions are selected from the full vocabulary, without a restricted answer list.

| Condition | Mean cross-entropy (nats) | Correct top-1 predictions |
|---|---:|---:|
| First occurrence of each pair | 14.861 | 0/128 |
| Fourth occurrence, consistent pair history | 6.142 | 28/128 (21.875%) |
| Earlier triggers replaced; B identities and positions preserved | 13.875 | 0/128 |
| B identities rotated; scored against the new answers | 6.252 | 24/128 (18.75%) |
| Same rotated inputs; scored against the old answers | 13.980 | 0/128 |

The model uses contextual associations, but it does **not** solve the full task reliably.

## Attention

L1H5 assigned mean attention mass 0.5886 to the three earlier matching B positions. The uniform-attention reference mass was 0.0296. Its attention argmax selected a matching source in 75/128 queries (58.594%). A fixed lag selected on discovery data found a matching source in 7/128 queries (5.469%). Attention-location accuracy is not prediction accuracy.

## Passive key tracing

These terms are computed from the unmodified run. Matching sources are compared with earlier B sources belonging to other pairs.

| Pre-softmax score term | Matching sources | Other-pair sources | Difference |
|---|---:|---:|---:|
| Full QK score of L1H5 | 13.226 | 5.862 | 7.364 |
| L0H3 contribution to K, dotted with the actual full Q | 7.319 | -0.017 | 7.335 |
| Previous-token embedding through L0H3 to K, excluding V bias | 7.304 | -0.048 | 7.352 |

The full Q-component by K-component matrix, including position and bias terms, is also saved. Reconstructed scores match the scores captured from the model to numerical precision.

## Passive output tracing

The reported margin is the raw logit of the correct B minus the average logit of the three other paired B values. It is a descriptive margin, not candidate-restricted prediction accuracy.

| Component | Mean direct margin contribution |
|---|---:|
| Actual final output margin | 7.381 |
| Entire L1H5 output | 7.287 |
| Within L1H5: matching B sources | 7.355 |
| Within L1H5: all other sources | -0.069 |
| Within matching sources: token embeddings through the value path | 4.478 |
| Within matching sources: L0H2 output through the value path | 2.860 |

These components are nested: the last two rows are parts of the matching-source row and must not be added to it again. Other head outputs and biases complete the exact decomposition. The results support distinct contributions through keys and values; they do not establish that only two heads implement the task.

## Separate ablations

All 16 heads were ablated individually. Mean loss increased by 7.177 for L1H5, 7.039 for L0H3, and 5.641 for L0H2. Top-1 accuracy fell to zero after either L1H5 or L0H3 was ablated. All head results are retained, not only these examples.

## Verification and limits

Six tests passed, including reconstruction on a small Shortformer with nonzero biases. Q/K components, attention scores, head value mixtures, final residuals and output margins were reconstructed within approximately 1e-5 or less on the pretrained model. Removing ablation hooks restored the original recorded predictions and losses.

The additive decompositions describe raw attention scores and output logits. They are not additive decompositions of softmax probabilities and are not causal effects. This is one data seed in one model, on a controlled synthetic task. Four queries sharing a sequence are not fully independent observations. There is no claim of general natural-language understanding, a minimum sufficient circuit, or a new discovery relative to the original paper.

## Inspect the complete records

- [Exact inputs and pair-event metadata](results/reference/inputs.json).
- [English input and prediction tables](results/reference/examples.md).
- [Full passive trace and per-edge records](results/reference/passive_trace.json).
- [Prediction metrics, input controls and all ablations](results/reference/results.json).
- [Protocol](results/reference/protocol.json).
