"""Pinned, local-only model loading and clean-run activation capture."""
import os
from unittest.mock import patch

from pair_config import ROOT, CACHE_DIR, MODEL, REVISION, TOKENIZER, TOKENIZER_REVISION, snapshot_path

os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
temporary = ROOT / 'work/finetune-temp'
temporary.mkdir(parents=True, exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(temporary)

import torch


def load_model():
    """Use only prepared local files; download_model.py is the explicit download step."""
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer
    snapshot = snapshot_path(MODEL, REVISION)
    tokenizer_snapshot = snapshot_path(TOKENIZER, TOKENIZER_REVISION)
    if not snapshot.is_dir() or not tokenizer_snapshot.is_dir():
        raise FileNotFoundError('Pinned model files are missing. Run: python experiments/download_model.py')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_snapshot, local_files_only=True, add_bos_token=True)
    files = [path.name for path in snapshot.iterdir() if path.is_file()]
    # TransformerLens 2.15 lists repository files even in offline mode. The local
    # snapshot is authoritative here, so supply its filenames without a network call.
    with patch('huggingface_hub.HfApi.list_repo_files', return_value=files):
        model = HookedTransformer.from_pretrained(
            MODEL, revision=REVISION, tokenizer=tokenizer, device='cpu', fold_ln=False,
            center_writing_weights=False, center_unembed=False,
        ).eval()
    validate_model(model)
    return model


def validate_model(model):
    """The passive algebra below relies on this exact architecture."""
    cfg = model.cfg
    assert cfg.n_layers == 2 and cfg.attn_only
    assert cfg.normalization_type is None and not cfg.final_rms
    assert cfg.positional_embedding_type == 'shortformer'
    assert cfg.attn_scores_soft_cap <= 0
    assert cfg.attention_dir == 'causal'


def capture_name(name):
    return name in ('hook_embed', 'hook_pos_embed') or name.endswith((
        'hook_resid_post', 'hook_q', 'hook_k', 'hook_v', 'hook_z', 'hook_pattern', 'hook_attn_scores'))


@torch.no_grad()
def clean_run(model, data, *, capture=False, hooks=()):
    """Return residuals, projecting to vocabulary only at measured query positions later."""
    if data.tokens.shape[1] > model.cfg.n_ctx:
        raise ValueError("Generated sequence exceeds model context")
    with model.hooks(fwd_hooks=list(hooks)):
        if capture:
            return model.run_with_cache(data.tokens, stop_at_layer=2, names_filter=capture_name)
        return model(data.tokens, stop_at_layer=2), None
