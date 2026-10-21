"""Metrics use known pair occurrences, never a presumed fixed distance."""
import torch
import torch.nn.functional as F

from pair_model import clean_run


@torch.no_grad()
def prediction_metrics(model, residual, data, occurrence=-1):
    events = data.queries(occurrence)
    row = torch.tensor([event.row for event in events])
    position = torch.tensor([event.query for event in events])
    targets = torch.tensor([event.target for event in events])
    scores = residual[row, position] @ model.W_U + model.b_U
    candidates = torch.tensor([[value for _, value in data.pairs[event.row]] for event in events])
    candidate_scores = scores.gather(1, candidates)
    gold = scores.gather(1, targets[:, None]).squeeze(1)
    other = candidate_scores.masked_fill(candidates == targets[:, None], 0).sum(1) / (candidates.shape[1] - 1)
    losses = F.cross_entropy(scores, targets, reduction='none')
    predictions = scores.argmax(-1)
    records = [{'row':event.row, 'pair':event.pair, 'query':event.query,
                'target':event.target, 'prediction':int(pred), 'loss':float(loss)}
               for event, pred, loss in zip(events, predictions, losses)]
    return {'n':len(events), 'loss':float(losses.mean()),
            'top1':float((predictions == targets).float().mean()),
            'target_vs_other_values_margin':float((gold - other).mean()), 'rows':records}


@torch.no_grad()
def attention_metrics(cache, data):
    """Sum attention over all previous B positions belonging to the current A."""
    events = data.queries()
    layers = []
    for layer in range(2):
        pattern = cache[f'blocks.{layer}.attn.hook_pattern']
        matching, top_matches, uniform, previous = [], [], [], []
        for event in events:
            weights = pattern[event.row, :, event.query]
            matching.append(weights[:, list(event.previous_values)].sum(-1))
            top_matches.append(torch.isin(weights.argmax(-1), torch.tensor(event.previous_values)).float())
            uniform.append(len(event.previous_values) / (event.query + 1))
        for row, length in enumerate(data.lengths):
            q = torch.arange(1, length)
            previous.append(pattern[row, :, q, q - 1].mean(-1))
        layers.append({'matching_mass':torch.stack(matching).mean(0).tolist(),
                       'attention_argmax_matches':torch.stack(top_matches).mean(0).tolist(),
                       'uniform_matching_mass':sum(uniform) / len(uniform),
                       'previous_token_mass':torch.stack(previous).mean(0).tolist()})
    return layers


def select_heads(discovery_attention):
    """Enforce early-layer previous-token head and late-layer retrieval head."""
    previous = max(range(len(discovery_attention[0]['previous_token_mass'])),
                   key=lambda h:discovery_attention[0]['previous_token_mass'][h])
    induction = max(range(len(discovery_attention[1]['matching_mass'])),
                    key=lambda h:discovery_attention[1]['matching_mass'][h])
    return {'previous':previous, 'induction':induction,
            'selection_rule':'argmax on discovery only; candidates, not a success declaration'}


def zero_head(head):
    def erase(z, hook):
        result = z.clone()
        result[:, :, head] = 0
        return result
    return erase


@torch.no_grad()
def ablation_sweep(model, data, baseline):
    records = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            residual, _ = clean_run(model, data, hooks=[
                (f'blocks.{layer}.attn.hook_z', zero_head(head))])
            measured = prediction_metrics(model, residual, data)
            records.append({'layer':layer, 'head':head, 'loss':measured['loss'],
                            'delta_loss':measured['loss'] - baseline['loss'], 'top1':measured['top1']})
    return records
