"""The two weight-only tests in Elhage et al. (2021), Checking the Mechanistic Theory.

OV should copy tokens. QK, composed through an early head's key path, should match
the same token. Following the paper, these maps exclude biases and positions.
TransformerLens uses row-vector weights, the transpose convention of the paper.
"""
import torch


def ov_factors(model, head):
    """Token -> late-head value -> output logits. A row belongs to a source token."""
    embedding = model.W_E.double()
    left = embedding @ model.W_V[1, head].double()
    right = model.W_O[1, head].double() @ model.W_U.double()
    return left, right


def matching_factors(model, early_head, late_head):
    """Current token -> Q; earlier token -> early OV -> late K.

    left @ right has rows = current token, columns = token preceding a source.
    Diagonal structure therefore describes same-token matching through K.
    """
    embedding = model.W_E.double()
    query = embedding @ model.W_Q[1, late_head].double()
    early_write = embedding @ model.W_V[0, early_head].double() @ model.W_O[0, early_head].double()
    key = early_write @ model.W_K[1, late_head].double()
    scale = model.blocks[1].attn.attn_scale
    return query, key.T / scale


def eigenvalue_summary(left, right):
    """Paper's score: real(sum(lambda)) / sum(abs(lambda)).

    If M = left @ right, its nonzero eigenvalues equal those of right @ left.
    Thus we diagonalize a head-size matrix, not a vocabulary-size matrix.
    The extra zero eigenvalues do not change the score. This identity applies
    to eigenvalues, not to singular values or Frobenius norms.
    """
    eigenvalues = torch.linalg.eigvals(right @ left)
    denominator = eigenvalues.abs().sum()
    positivity = float(eigenvalues.real.sum() / denominator) if denominator > 0 else None
    return {'positivity':positivity,
            'eigenvalues':[{'real':float(value.real), 'imag':float(value.imag)} for value in eigenvalues]}


@torch.no_grad()
def analyze_circuits(model, early_head):
    """Run the same OV and composed-QK tests for every second-layer head."""
    heads = []
    for head in range(model.cfg.n_heads):
        ov = eigenvalue_summary(*ov_factors(model, head))
        qk = eigenvalue_summary(*matching_factors(model, early_head, head))
        heads.append({'head':head, 'ov_copying':ov, 'qk_same_matching':qk})
    return {'method':'Elhage et al. (2021): OV and K-composed QK eigenvalue positivity',
            'source':'https://transformer-circuits.pub/2021/framework/index.html',
            'early_head':early_head, 'late_layer':1,
            'assumptions':'Weight-only token maps; excludes biases, positions and other composition paths.',
            'score':'real(sum(eigenvalues)) / sum(abs(eigenvalues)); not a fraction of positive eigenvalues.',
            'interpretation':'A positive spectral bias supports copying/matching; it is not sufficient proof.',
            'heads':heads}


def save_circuit_plot(report, selected_head, path):
    """Plot all late heads on the paper's OV-positivity / QK-positivity axes."""
    import os
    from pair_config import ROOT
    plot_cache = ROOT / 'work/mpl-cache'
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(plot_cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6, 5))
    for result in report['heads']:
        x = result['ov_copying']['positivity']
        y = result['qk_same_matching']['positivity']
        if x is None or y is None:
            continue
        color = 'tab:red' if result['head'] == selected_head else 'tab:gray'
        axes.scatter(x, y, color=color)
        label_offset = (-42, 8) if result['head'] == selected_head else (5, -9)
        axes.annotate(f"L1H{result['head']}", (x, y), xytext=label_offset, textcoords='offset points')
    axes.set(xlim=(-1.1, 1.15), ylim=(-1.1, 1.15),
             xlabel='OV eigenvalue positivity (copying)',
             ylabel='K-composed QK eigenvalue positivity (same matching)',
             title=f"Weight-only circuit tests via L0H{report['early_head']}")
    axes.axhline(0, color='lightgray', linewidth=.5)
    axes.axvline(0, color='lightgray', linewidth=.5)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
