# Induction heads: recurring pairs and paper-style circuit tests

A small, CPU-based seminar experiment inspired by [Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html).

The experiment uses a pretrained, two-layer attention-only transformer to predict recurring token pairs inside random background. Its mechanistic analysis follows the two weight-based tests in the paper's **Checking the Mechanistic Theory** section: an OV copying map and a same-token QK matching map composed through an early head. There is no training or fine-tuning.

The input task is our recurring-pair variant; the paper also illustrates a complete random sequence repeated three times. The mechanistic tests follow the paper, in a different public model. This does not reproduce Anthropic's original weights or the entire paper.

## 1. What is the task?

Each sequence contains four distinct pairs, such as `A -> B` and `C -> D`. Each pair occurs four times. Pair order is shuffled in every round, and 1–9 random background tokens are inserted before each pair:

```text
BOS  x y A B  z C D  ...  w C D  x A B  ...  z A [predict B]
```

The actual inputs are integer token IDs, not these letters or natural-language sentences. Background excludes all pair tokens, preventing accidental conflicting continuations. Rounds balance exposure counts; the full sequence is never duplicated. Distances between occurrences vary.

On the last occurrence of each trigger, the model predicts its paired next token using three earlier examples. All four pairs are evaluated: 32 sequences produce 128 queries. A separate set of 32 sequences is used to select candidate heads. A candidate's attention score is not a declaration that it implements the mechanism.

The model is autoregressive. Its built-in causal attention mask prevents a query from reading future tokens, including the supplied next-token answer and right padding. No extra prefix audit is part of the experiment.

## 2. Model

- Model repository: [`NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr`](https://huggingface.co/NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr).
- Pinned model revision: `438f8e124b85bde1b6e64c63e9445e31b45f5be6`.
- Architecture: two attention layers, eight heads per layer, model width 512, head width 64, Shortformer positions, no MLP and no normalization.
- Tokenizer: `EleutherAI/gpt-neox-20b`, revision `c292233c833e336628618a88a648727eb3dff0a7`. Only tokenizer files are downloaded from that repository, **not** its model weights.

The weights belong to the upstream model repository and are downloaded separately. This repository does not redistribute model weights.

## 3. Install and run

Use Python 3.11. Run commands from the repository root.

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
source .venv/bin/activate
```

Install the tested dependencies, download the pinned files once, and run:

```bash
python -m pip install -r requirements.txt
python experiments/download_model.py
python experiments/induction_pairs.py
```

The download step requires internet access. The experiment itself loads local cached files and runs on CPU using two threads. The first download includes a pretrained model; tests do not need to download it.

The default output directory is `work/induction/paper-01`. Existing output directories are deliberately not overwritten. For another run:

```bash
python experiments/induction_pairs.py --output work/induction/run-02
```

To run the weight-based circuit tests and input controls without head ablations:

```bash
python experiments/induction_pairs.py --skip-ablations --output work/induction/circuits-only
```

Other options include `--batch`, `--pairs`, `--repeats`, `--min-gap`, `--max-gap`, `--seed` and `--threads`. Defaults and meanings are in `Settings` in the entry script. `INDUCTION_HF_HOME` can point to an existing compatible Hugging Face cache; otherwise files are cached under `work/hf-cache`.

## 4. Reading the code

Start with `main()` and then `run_experiment()` in the entry script.

| File | Responsibility |
|---|---|
| [`experiments/induction_pairs.py`](experiments/induction_pairs.py) | Read settings, coordinate the experiment, save results and English example tables. |
| [`experiments/pair_data.py`](experiments/pair_data.py) | Generate inputs, record true source positions, construct input controls and a fixed-offset baseline. |
| [`experiments/pair_measurements.py`](experiments/pair_measurements.py) | Measure next-token prediction and attention, select candidate heads on discovery data, and ablate individual heads. |
| [`experiments/paper_circuits.py`](experiments/paper_circuits.py) | The main mechanistic analysis: OV copying, composed-QK matching, eigenvalue scores and the two-axis plot. |
| [`experiments/pair_model.py`](experiments/pair_model.py) | Load cached weights and tokenizer, check architecture assumptions, and capture activations. |
| [`experiments/pair_config.py`](experiments/pair_config.py) | Keep paths and pinned repository revisions in one place. |
| [`experiments/download_model.py`](experiments/download_model.py) | Explicitly download only the required pretrained files. |
| [`experiments/test_pair_induction.py`](experiments/test_pair_induction.py) | Test input construction, controls, attention indexing and passive decompositions. |
| [`experiments/test_paper_circuits.py`](experiments/test_paper_circuits.py) | Check low-rank eigenvalues and the direction of the token-to-token maps. |
| [`experiments/pair_trace.py`](experiments/pair_trace.py) | Archived detailed activation decomposition; no longer called by the main experiment. |

Start the mechanistic analysis with `ov_factors()` and `matching_factors()` in `paper_circuits.py`. Each forms one token map in a few matrix multiplications. `eigenvalue_summary()` computes the paper's summary score. `analyze_circuits()` applies the same two tests to every second-layer head.

## 5. Measurements and controls

The main prediction metrics are cross-entropy and top-1 accuracy over the **entire vocabulary**. No candidate answer list is supplied to the model.

The attention metric sums attention to every earlier B that followed the queried A. Source positions come from recorded pair occurrences, not an assumed fixed offset. Attention argmax accuracy and a uniform-attention reference are reported separately from prediction accuracy.

Controls:

1. **First occurrence:** predict B before any earlier A-B example exists.
2. **No pair history:** replace earlier A tokens with reserved tokens, retaining B identities, B positions and the final-round queries.
3. **Rotated values:** cyclically change which B belongs to each A, keeping triggers, background and positions. Score against both the new and old answers.
4. **Fixed offset:** select one best distance on discovery data and measure its source-location accuracy on evaluation data.
5. **Head ablations:** zero each of the 16 head outputs separately and measure the change in prediction loss. This is separate from the paper-style weight tests.

## 6. The two mechanistic tests

The code uses TransformerLens's row-vector convention. Let `E = W_E`, `U = W_U`, `p` be the earlier previous-token head, and `h` a later head.

**Copying through OV:** form the token-to-logit map

```text
M_OV = (E @ W_V[h]) @ (W_O[h] @ U)
```

Its rows describe which token is read; its columns describe which output logits it supports.

**Same-token matching through K-composition:** form

```text
Q = E @ W_Q[h]
K = E @ W_V[p] @ W_O[p] @ W_K[h]
M_QK = Q @ K.T / attention_scale
```

Here the row is the current token; the column is the earlier token whose information the previous-token head moves into the following position. Matching the same token supports the induction pattern `A B ... A -> B`.

For both maps, following the paper's figure, the summary is `real(sum(eigenvalues)) / sum(abs(eigenvalues))`. Values near +1 indicate a spectrum strongly biased toward positive real eigenvalues. This is **not** the fraction of positive eigenvalues, a probability, or prediction accuracy. As the authors emphasize, positive eigenvalues support the interpretation but are not sufficient proof of copying.

The vocabulary-wide maps would have over 2.5 billion entries. We do not construct them: if a map is `L @ R`, its nonzero eigenvalues equal those of the small matrix `R @ L`. The code therefore diagonalizes a 64-by-64 matrix. This identity is valid for eigenvalues, not for singular values or Frobenius norms.

These are weight-only token maps, omitting biases and positional components as in the paper's simplified analysis. They test particular OV and K-composition terms, not the complete context-dependent attention calculation. No detailed per-activation reconstruction is needed in the main run.

## 7. Tests and recorded results

```bash
python -m pytest -q
```

Ten tests cover the data/controls, the small-vs-full eigenvalue identity (including complex eigenvalues), token-map orientation, and the retained legacy decomposition. No pretrained weights or network access are needed for these tests.

The current reference run, including exact inputs, is in [`results/paper-reference`](results/paper-reference). See [`RESULTS.md`](RESULTS.md). The earlier detailed analysis remains under `results/reference` for historical comparison; it uses the same inputs and is not an independent data seed.

The primary result is **28/128 correct next-token predictions (21.875%)**, compared with 0/128 after removing pair history. This is evidence of context-dependent prediction, not a complete solution of the task. Attention-location accuracy is a different metric and must not be presented as prediction accuracy. This is one data seed, and four queries from one sequence are not independent observations.

## 8. Output files

| File | Contents |
|---|---|
| `protocol.json` | Settings and pinned model revision. |
| `inputs.json` | Exact discovery/evaluation token IDs, pair maps and occurrence positions. |
| `examples.md` | English tables showing inputs, pair positions and predictions. |
| `baseline.json` | Candidate selection, clean-run prediction and attention metrics. |
| `mechanistic_analysis.json` | Weight-only OV and K-composed QK eigenvalue scores for all eight late heads. |
| `mechanistic_analysis.png` | Paper-style plot: OV positivity against QK positivity, highlighting the attention-selected candidate. |
| `results.json` | Clean metrics, input controls and optional ablations. |

## References

- Elhage et al. (2021). [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html).
- Neel Nanda. [*Attn-Only-2L512W-Shortformer-6B-big-lr*, pretrained model repository](https://huggingface.co/NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr).
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).
