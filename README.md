# Induction heads: information flow in an actual run

This is a small CPU experiment with a pretrained two-layer attention-only model.
There is no training. The analysis reads cached activations from an unchanged
forward pass. It does not use eigenvalue scores, token-map tests, input controls
or head ablations.

## Read the code

1. [induction_pairs.py](experiments/induction_pairs.py): start with `run_experiment()`.
   It runs one batch once, then selects heads and traces that same cached run.
2. [pair_data.py](experiments/pair_data.py): build random pairs and record earlier sources.
3. [pair_measurements.py](experiments/pair_measurements.py): measure attention and select heads.
4. [pair_trace.py](experiments/pair_trace.py): read `trace_keys()` and `trace_output()`.
5. [pair_model.py](experiments/pair_model.py): load the model; `read_run()` saves the needed activations.

Functions have type hints. Results use `TypedDict` so the editor can show their keys.
Names, comments and output use plain English.

## Inputs

Each sequence has four different A-to-B pairs. Each pair occurs four times.
Every round has a new pair order. Each pair has 1–9 random background tokens
before it. The background excludes pair tokens. These are token IDs, not stories.

There is one batch of 32 sequences and one model run. Its cached attention
selects an early head by attention to the previous token, and a late head by
attention to earlier matching B positions. The trace uses the same cache and the
last appearance of every pair: 128 queries, each with three matching sources.

The default seed is 20261022, preserving the earlier traced inputs.
The model's own causal attention mask handles future positions.

## What is traced?

### Information entering the key

For each final A, inspect the earlier pair-B positions in that same sequence.
Read the actual late-head Q, K and attention score from the cache.

Split K into the contributions from token embedding, position, the eight early
heads and biases. Keep the actual Q fixed:

```python
key_part = early_head_write @ W_K_late
score_part = (actual_query * key_part).sum() / attention_scale
```

The early-head write comes from its cached Z and output weight, so it already
includes what the head read in that context.

Also follow the selected early head's actual read from the token immediately
before each B. This is part of that head's contribution, not an extra term.

Report means for matching pair sources and other pair sources in the same
unchanged input. This is not a modified-input control.

### Information entering the output

Reconstruct the selected late head's output from its actual attention and V.
Split its write into matching B sources and all other sources.

Within matching sources, split V into the token embedding, early-head writes and
biases. Shortformer positions are not added to V.

Project each write onto B's unembedding vector to get its direct contribution
to B's raw logit. There is no comparison with the other pair values. Read the
full B logit from the final cached residual and subtract the selected head's
contribution to get everything else together. No loop over the other late heads
is needed for this comparison.

The summary also shows signed percentages: mean contribution divided by mean
total logit. These may be negative or exceed 100%, and are undefined when the
mean total is zero. They are not probability shares or positive-support shares.

Separately, use the cached final residual to compute next-token probabilities
over the full vocabulary. Report whether B is the top prediction and its probability
for each query. This uses the same run, without another transformer forward pass.

The code checks that the pieces sum back to the cached K, attention scores, Z
and matching-source write. No activation is removed or changed. These are additive
contributions in the observed run, not intervention effects or a full proof of causality.

This decomposition needs two attention-only layers, no normalization, and
Shortformer positions. It is not a general decomposition for arbitrary models.

## Run

Use Python 3.11 and run from the repository root:

```bash
python -m venv .venv
```

Activate in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Then:

```bash
python -m pip install -r requirements.txt
python experiments/download_model.py
python experiments/induction_pairs.py
```

Only the download needs internet. The analysis uses local files, CPU and two threads.

Default output: `work/induction/trace-01`. Existing output folders are not overwritten:

```bash
python experiments/induction_pairs.py --output work/induction/trace-02
```

Use `--help` for the settings.
`INDUCTION_HF_HOME` can point to an existing Hugging Face cache.

## Output

- `summary.md`: attention and readable key/output contribution tables.
- `results.json`: predictions, B probabilities, logit contributions, per-edge key records and reconstruction errors.
- `inputs.json`: exact inputs for the single batch.
- `protocol.json`: settings and pinned model/tokenizer revisions.

Current results: [results/target-reference](results/target-reference). See [RESULTS.md](RESULTS.md).
Other result folders record earlier versions and are not the current analysis.
There is no eigenvalue analysis in the active code.

## Model and source

[NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr](https://huggingface.co/NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr):
two layers, eight heads per layer, width 512, head width 64.

Model revision: `438f8e124b85bde1b6e64c63e9445e31b45f5be6`.

Tokenizer: `EleutherAI/gpt-neox-20b`, revision
`c292233c833e336628618a88a648727eb3dff0a7`. Only tokenizer files are downloaded
from that repository, not its model weights.

The circuit idea follows [Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html).
The pair inputs and this activation report are our implementation in a different
public model. This does not reproduce the original weights or every paper experiment.

## Tests

```bash
python -m pytest -q
```

Six tests cover data, source indexing, head selection, cache reading, and real
forward-pass decompositions with nonzero biases, and predictions against normal
model logits. They also check that tracing
leaves the cache, input tokens and model weights unchanged. No pretrained model
download is needed for tests.
