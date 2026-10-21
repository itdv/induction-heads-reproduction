r"""Readable induction experiment: recurring pairs, passive tracing, optional ablations.

Run: .venv\Scripts\python.exe experiments/induction_pairs.py
"""
import argparse
from dataclasses import asdict, dataclass
import json

from pair_model import ROOT, MODEL, REVISION, load_model, clean_run
from pair_data import make_pair_batch, remove_pair_history, rotate_pair_values
from pair_data import best_fixed_offset, fixed_offset_accuracy
from pair_measurements import prediction_metrics, attention_metrics, select_heads, ablation_sweep
from pair_trace import trace_key_composition, trace_output
import torch


@dataclass(frozen=True)
class Settings:
    batch: int = 32
    pairs: int = 4
    repeats: int = 4
    min_gap: int = 1
    max_gap: int = 9
    seed: int = 20261021
    threads: int = 2
    ablations: bool = True
    output: str = 'work/induction/pairs-01'


def parse_settings():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('batch', 'pairs', 'repeats', 'min_gap', 'max_gap', 'seed', 'threads'):
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=getattr(Settings(), name))
    parser.add_argument('--output', default=Settings().output)
    parser.add_argument('--skip-ablations', action='store_true')
    args = vars(parser.parse_args())
    args['ablations'] = not args.pop('skip_ablations')
    if args['threads'] < 1:
        parser.error('threads must be positive')
    return Settings(**args)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def build_data(model, settings, seed):
    return make_pair_batch(model.cfg.d_vocab, model.tokenizer.bos_token_id,
                           model.tokenizer.all_special_ids, batch=settings.batch,
                           pairs=settings.pairs, repeats=settings.repeats,
                           min_gap=settings.min_gap, max_gap=settings.max_gap, seed=seed)


def discover_heads(model, discovery):
    _, cache = clean_run(model, discovery, capture=True)
    attention = attention_metrics(cache, discovery)
    return select_heads(attention), attention, best_fixed_offset(discovery)


def evaluate_controls(model, data):
    results = {}
    for name, control in [('no_pair_history', remove_pair_history(data)),
                          ('rotated_values', rotate_pair_values(data))]:
        residual, _ = clean_run(model, control)
        results[name] = prediction_metrics(model, residual, control)
        if name == 'rotated_values':
            results['rotated_values_scored_against_original_answers'] = prediction_metrics(model, residual, data)
    return results


def write_examples(path, data, predictions):
    """Write exact token IDs, pair locations and predictions for manual inspection."""
    lines = ['# Inputs and predictions: recurring pairs in random background\n',
             'Numbers are token IDs. These synthetic sequences are not natural-language sentences.\n']
    for row, length in enumerate(data.lengths):
        lines.extend([f'## Sequence {row}\n', '```text',
                      ' '.join(map(str, data.tokens[row, :length].tolist())), '```',
                      '| Trigger A | Target B | Positions of A |', '|---|---|---|'])
        for pair, (a, b) in enumerate(data.pairs[row]):
            positions = [event.query for event in data.events if event.row == row and event.pair == pair]
            lines.append(f'|{a}|{b}|{positions}|')
        lines.extend(['\n| Query position | Correct token | Predicted token |', '|---|---|---|'])
        for prediction in predictions['rows']:
            if prediction['row'] == row:
                lines.append(f"|{prediction['query']}|{prediction['target']}|{prediction['prediction']}|")
    path.write_text('\n'.join(lines), encoding='utf-8')


@torch.no_grad()
def run_experiment(model, settings, out):
    discovery = build_data(model, settings, settings.seed)
    data = build_data(model, settings, settings.seed + 1)
    save(out / 'inputs.json', {'discovery':discovery.serializable(), 'evaluation':data.serializable()})
    selected, discovery_attention, lag = discover_heads(model, discovery)
    print('Selected on discovery: ' + json.dumps(selected), flush=True)
    residual, cache = clean_run(model, data, capture=True)
    baseline = prediction_metrics(model, residual, data)
    report = {'selected':selected, 'discovery_attention':discovery_attention,
              'attention':attention_metrics(cache, data), 'final_occurrence':baseline,
              'first_occurrence':prediction_metrics(model, residual, data, occurrence=0),
              'positional_baseline':{'lag_selected_on_discovery':lag,
                                     'evaluation_hit_rate':fixed_offset_accuracy(data, lag)}}
    save(out / 'baseline.json', report)
    key_trace = trace_key_composition(model, cache, data, selected['previous'], selected['induction'])
    output_trace = trace_output(model, cache, residual, data, selected['induction'])
    save(out / 'passive_trace.json', {'key_composition':key_trace, 'output':output_trace})
    del cache, residual
    report['controls'] = evaluate_controls(model, data)
    if settings.ablations:
        report['ablations'] = ablation_sweep(model, data, baseline)
    restored, _ = clean_run(model, data)
    restored_metrics = prediction_metrics(model, restored, data)
    assert restored_metrics['rows'] == baseline['rows'], 'Clean predictions changed after interventions'
    report['hook_cleanup_verified'] = True
    save(out / 'results.json', report)
    write_examples(out / 'examples.md', data, baseline)
    print('FINAL ' + json.dumps({k:v for k,v in baseline.items() if k != 'rows'}), flush=True)
    return report


def main():
    settings = parse_settings()
    out = ROOT / settings.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(settings.threads)
    save(out / 'protocol.json', {'model':MODEL, 'revision':REVISION, **asdict(settings),
         'trained':False, 'passive_trace':'actual clean run; additive QK scores and direct logits',
         'selection':'separate discovery data; early previous head, late matching head'})
    model = load_model()
    run_experiment(model, settings, out)


if __name__ == '__main__':
    main()
