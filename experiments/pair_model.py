"""Load the pinned model from local files and read its attention patterns."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch

from pair_config import (
    CACHE_DIR, MODEL, REVISION, TOKENIZER, TOKENIZER_REVISION, snapshot_path,
)
from pair_data import PairBatch

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


def load_model() -> HookedTransformer:
    """Run download_model.py once before using this offline loader."""
    os.environ["HF_HOME"] = str(CACHE_DIR)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    model_path = snapshot_path(MODEL, REVISION)
    tokenizer_path = snapshot_path(TOKENIZER, TOKENIZER_REVISION)
    if not model_path.is_dir() or not tokenizer_path.is_dir():
        raise FileNotFoundError(
            "Missing model files. Run: python experiments/download_model.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, add_bos_token=True,
    )

    # TransformerLens asks for a repository file list even when offline.
    # Give it the names from the pinned local snapshot instead.
    local_files = [path.name for path in model_path.iterdir() if path.is_file()]
    with patch("huggingface_hub.HfApi.list_repo_files", return_value=local_files):
        model = HookedTransformer.from_pretrained(
            MODEL, revision=REVISION, tokenizer=tokenizer, device="cpu",
            fold_ln=False, center_writing_weights=False, center_unembed=False,
        )

    return model.eval()


@torch.no_grad()
def read_attention(model: HookedTransformer, data: PairBatch) -> list[torch.Tensor]:
    """Run the model once. Save only attention patterns; do not compute logits."""
    if data.tokens.shape[1] > model.cfg.n_ctx:
        raise ValueError("The generated sequence is longer than the model context.")

    _, cache = model.run_with_cache(
        data.tokens,
        return_type=None,
        names_filter=lambda name: name.endswith("attn.hook_pattern"),
    )
    return [cache["pattern", layer] for layer in range(model.cfg.n_layers)]
