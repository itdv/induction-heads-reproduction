"""Shared paths and pinned Hugging Face revisions; importing this does not load a model."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(os.environ.get('INDUCTION_HF_HOME', ROOT / 'work/hf-cache'))
MODEL = 'NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr'
REVISION = '438f8e124b85bde1b6e64c63e9445e31b45f5be6'
TOKENIZER = 'EleutherAI/gpt-neox-20b'
TOKENIZER_REVISION = 'c292233c833e336628618a88a648727eb3dff0a7'


def snapshot_path(repo_id: str, revision: str) -> Path:
    """Find a pinned snapshot in the standard Hugging Face cache layout."""
    return CACHE_DIR / 'hub' / ('models--' + repo_id.replace('/', '--')) / 'snapshots' / revision
