"""Download only the pinned experiment weights and tokenizer, then run offline.

Usage: python experiments/download_model.py
The tokenizer repository belongs to a 20B model, but NO 20B model weights are fetched.
"""
from pair_config import CACHE_DIR, MODEL, REVISION, TOKENIZER, TOKENIZER_REVISION


def main():
    from huggingface_hub import snapshot_download

    cache = str(CACHE_DIR / 'hub')
    snapshot_download(MODEL, revision=REVISION, cache_dir=cache,
                      allow_patterns=['config.json', 'model_final.pth'])
    snapshot_download(TOKENIZER, revision=TOKENIZER_REVISION, cache_dir=cache,
                      allow_patterns=['tokenizer.json', 'tokenizer_config.json',
                                      'special_tokens_map.json', 'added_tokens.json',
                                      'vocab.json', 'merges.txt'])
    print('Pinned files are ready. Run: python experiments/induction_pairs.py')


if __name__ == '__main__':
    main()
