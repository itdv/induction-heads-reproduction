# Induction heads: a small mechanistic analysis

This project follows two weight tests from [Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html): **OV copying** and **QK matching through an earlier head**.

It uses a pretrained two-layer attention-only model. There is no training.

## What the script does

1. Build random sequences with four recurring token pairs.
2. Find an early head that reads the previous token, and a late head that reads earlier matching B positions.
3. Measure attention on a separate set of sequences.
4. Test the OV and composed QK weights of all eight late heads.
5. Save a short table, a plot and the exact inputs.

There are no input controls, prediction scores, head ablations or detailed activation decompositions in the current code.

## Read the code in this order

| File | What to read |
|---|---|
| [induction_pairs.py](experiments/induction_pairs.py) | Start with `run_experiment()`. Its three steps show the whole analysis. `main()` loads the model and saves the output. |
| [pair_data.py](experiments/pair_data.py) | Build the sequences. Each occurrence records the earlier matching B positions. |
| [pair_measurements.py](experiments/pair_measurements.py) | Measure attention to the previous token and to matching sources; choose heads. |
| [paper_circuits.py](experiments/paper_circuits.py) | Build the two weight maps and compute their eigenvalue scores. |
| [pair_model.py](experiments/pair_model.py) | Load local weights and read only attention patterns from the cache. |
| [plot_circuits.py](experiments/plot_circuits.py) | Draw the scores. This file does not do the analysis. |
| [pair_config.py](experiments/pair_config.py) | Model names, pinned revisions and cache paths. |

Names, comments and output are in plain English.

Functions have input and return types. Result dictionaries use `TypedDict`, so
the editor can show their keys and follow the types between functions.

## Inputs

Each sequence contains four distinct pairs, such as `A -> B` and `C -> D`. Each pair occurs four times. Every round has a new pair order. Each pair has 1–9 random background tokens before it.

The letters stand for randomly sampled token IDs, not words in a story. Background excludes the pair tokens. The full sequence is never duplicated.

The analysis uses the fourth occurrence of every pair. Each A then has three earlier matching B sources. There are 32 discovery sequences and 32 separate evaluation sequences, with 128 evaluation queries. The model's own causal attention mask prevents reading future tokens.

## The two weight maps

The code uses row vectors. Let `E` be the token embedding, `U` the unembedding, `p` the early head and `h` a late head. Take each weight from its head's layer.

```python
# A source token changes output token scores through OV.
M_OV = E @ W_V[h] @ W_O[h] @ U

# The current token supplies Q; the early head supplies information for K.
Q = E @ W_Q[h]
K = E @ W_V[p] @ W_O[p] @ W_K[h]
M_QK = Q @ K.T / attention_scale
```

The early head moves information about A into the following B position. The composed QK map tests whether a new A matches that information. OV tests whether the late head can copy the B it reads.

For each map, the score is:

```text
real(sum(eigenvalues)) / sum(abs(eigenvalues))
```

A score near +1 supports copying or same-token matching. It is not prediction accuracy or the fraction of positive eigenvalues. These maps omit biases, positions and other paths in the model. They support a circuit interpretation, but do not prove the full computation.

The maps are stored as two factors, `left` and `right`. Their nonzero eigenvalues equal those of `right @ left`, which is only 64 by 64. The full vocabulary-sized maps are never built.

## Run

Use Python 3.11. Run from the repository root:

```bash
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Then run:

```bash
python -m pip install -r requirements.txt
python experiments/download_model.py
python experiments/induction_pairs.py
```

The download step needs internet. The analysis loads cached files and uses CPU with two threads.

The default output is `work/induction/mechanism-01`. Existing output folders are not overwritten. To use a new folder:

```bash
python experiments/induction_pairs.py --output work/induction/mechanism-02
```

Use `--help` for the data settings. `INDUCTION_HF_HOME` can point to an existing Hugging Face cache.

## Output

- `summary.md`: selected heads, attention scores and the OV/QK table.
- `circuits.png`: OV score versus QK score for each late head.
- `results.json`: all attention scores, map scores and eigenvalues.
- `inputs.json`: exact token IDs, pair mappings and source positions.
- `protocol.json`: settings and pinned model/tokenizer revisions.

The current reference is [results/mechanism-reference](results/mechanism-reference). See [RESULTS.md](RESULTS.md). The older `results/reference` and `results/paper-reference` folders record earlier versions; their extra analyses are no longer run.

Removing tokens reserved for old controls changed the generated sequences, even with the same seeds. The new reference records the new inputs. The model and weight tests are unchanged.

## Model

[NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr](https://huggingface.co/NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr): two layers, eight heads per layer, width 512, head width 64, Shortformer positions, no MLP or normalization.

Model revision: `438f8e124b85bde1b6e64c63e9445e31b45f5be6`.

Tokenizer: `EleutherAI/gpt-neox-20b`, revision `c292233c833e336628618a88a648727eb3dff0a7`. Only tokenizer files are downloaded from that repository, not its model weights.

The pair inputs are our variant. The paper also shows a complete random sequence repeated three times. We use a different public model, not Anthropic's original weights.

## Tests

```bash
python -m pytest -q
```

Eight tests check source positions, attention indexing, head selection, cache reading, the small-matrix eigenvalue calculation and the direction of the two maps. They use small local examples, with no downloaded model.
