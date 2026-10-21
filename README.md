# Induction heads: recurring pairs and passive circuit tracing

A small, CPU-based seminar experiment inspired by [Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html).

The experiment uses a pretrained, two-layer attention-only transformer to predict recurring token pairs inside random background. It measures attention, runs input controls and head ablations, and traces additive contributions through keys and values in the **unmodified** forward pass. There is no training or fine-tuning.

This is a reproduction of a mechanism in a different public model, not a reproduction of Anthropic's original model weights or the entire paper.

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

The default output directory is `work/induction/pairs-01`. Existing output directories are deliberately not overwritten. For another run:

```bash
python experiments/induction_pairs.py --output work/induction/run-02
```

To run passive tracing and input controls without head ablations:

```bash
python experiments/induction_pairs.py --skip-ablations --output work/induction/passive-only
```

Other options include `--batch`, `--pairs`, `--repeats`, `--min-gap`, `--max-gap`, `--seed` and `--threads`. Defaults and meanings are in `Settings` in the entry script. `INDUCTION_HF_HOME` can point to an existing compatible Hugging Face cache; otherwise files are cached under `work/hf-cache`.

## 4. Reading the code

Start with `main()` and then `run_experiment()` in the entry script.

| File | Responsibility |
|---|---|
| [`experiments/induction_pairs.py`](experiments/induction_pairs.py) | Read settings, coordinate the experiment, save results and English example tables. |
| [`experiments/pair_data.py`](experiments/pair_data.py) | Generate inputs, record true source positions, construct input controls and a fixed-offset baseline. |
| [`experiments/pair_measurements.py`](experiments/pair_measurements.py) | Measure next-token prediction and attention, select candidate heads on discovery data, and ablate individual heads. |
| [`experiments/pair_trace.py`](experiments/pair_trace.py) | Decompose the observed QK scores and direct output logits without changing activations. |
| [`experiments/pair_model.py`](experiments/pair_model.py) | Load cached weights and tokenizer, check architecture assumptions, and capture activations. |
| [`experiments/pair_config.py`](experiments/pair_config.py) | Keep paths and pinned repository revisions in one place. |
| [`experiments/download_model.py`](experiments/download_model.py) | Explicitly download only the required pretrained files. |
| [`experiments/test_pair_induction.py`](experiments/test_pair_induction.py) | Test input construction, controls, attention indexing and passive decompositions. |

## 5. Measurements and controls

The main prediction metrics are cross-entropy and top-1 accuracy over the **entire vocabulary**. No candidate answer list is supplied to the model.

The attention metric sums attention to every earlier B that followed the queried A. Source positions come from recorded pair occurrences, not an assumed fixed offset. Attention argmax accuracy and a uniform-attention reference are reported separately from prediction accuracy.

Controls:

1. **First occurrence:** predict B before any earlier A-B example exists.
2. **No pair history:** replace earlier A tokens with reserved tokens, retaining B identities, B positions and the final-round queries.
3. **Rotated values:** cyclically change which B belongs to each A, keeping triggers, background and positions. Score against both the new and old answers.
4. **Fixed offset:** select one best distance on discovery data and measure its source-location accuracy on evaluation data.
5. **Head ablations:** zero each of the 16 head outputs separately and measure the change in prediction loss. This is separate from passive tracing.

## 6. What does passive tracing mean?

The clean forward pass is captured once. For this model, the first layer's residual stream is exactly the token embedding plus the sum of head writes and the output bias. Shortformer adds position embeddings to Q and K inputs, but not to V inputs.

The code reconstructs:

- Q and K from all earlier head outputs, token embeddings, position embeddings and biases;
- every Q-component × K-component term in the attention score;
- the earlier head's contribution to a later key, evaluated against the actual query;
- the specific previous-token-to-current-position edge, with its token-content and bias terms distinguished;
- direct target logits and target-versus-other-pair-value margins from every residual component;
- the later head's writes from matching B sources versus other sources;
- within matching sources, the contribution of B embeddings, earlier head outputs and value-projection bias.

Each decomposition is checked numerically against the actual model calculation. These are additive decompositions of **pre-softmax attention scores and raw output logits**, not additive decompositions of attention probabilities or causal effects.

## 7. Tests and recorded results

```bash
python -m pytest -q
```

The tests include a randomly initialized, tiny Shortformer with nonzero biases. This checks that the algebra does not merely work because a bias happens to be zero. No pretrained weights or network access are needed for the tests.

The full reference run, including exact inputs, is in [`results/reference`](results/reference). See [`RESULTS.md`](RESULTS.md) for the main numbers and their limits.

The primary result is **28/128 correct next-token predictions (21.875%)**, compared with 0/128 after removing pair history. This is evidence of context-dependent prediction, not a complete solution of the task. Attention-location accuracy is a different metric and must not be presented as prediction accuracy. This is one data seed, and four queries from one sequence are not independent observations.

## 8. Output files

| File | Contents |
|---|---|
| `protocol.json` | Settings and pinned model revision. |
| `inputs.json` | Exact discovery/evaluation token IDs, pair maps and occurrence positions. |
| `examples.md` | English tables showing inputs, pair positions and predictions. |
| `baseline.json` | Candidate selection, clean-run prediction and attention metrics. |
| `passive_trace.json` | Full additive decompositions, per-edge records and reconstruction checks. |
| `results.json` | Clean metrics, input controls and optional ablations. |

## References

- Elhage et al. (2021). [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html).
- Neel Nanda. [*Attn-Only-2L512W-Shortformer-6B-big-lr*, pretrained model repository](https://huggingface.co/NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr).
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).
