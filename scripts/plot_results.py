import argparse
import json
from pathlib import Path


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.grid': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 0.7,
                         'xtick.direction': 'out', 'ytick.direction': 'out', 'savefig.bbox': 'tight'})
    return plt


def rq2(intensity, recurrence, output, layout):
    plt = style()
    methods = list(next(iter(intensity['conditions'].values()))['methods'])
    palettes = {'Backbone': ('#4C4C4C', 'o', '-'), 'Spotlighting': ('#7D7D7D', '^', ':'),
                'PromptGuard2-Record': ('#999999', 's', '--'), 'Random-Matched': ('#B8B8B8', 'D', '-.'),
                'CREST': ('#82B3D8', 'o', '-')}
    rows = 2 if layout == 'four' else 1
    figure, axes = plt.subplots(rows, 2, figsize=(8.3, 5.2 if rows == 2 else 3.0), squeeze=False)
    legend = {}
    for column, (source, xlabel) in enumerate(((intensity, 'Nominal intensity (%)'), (recurrence, 'Affected decisions (%)'))):
        levels = sorted(source['conditions'], key=float)
        positions = list(range(len(levels)))
        upper = axes[0, column]
        lower = axes[1, column] if rows == 2 else upper.twinx()
        for method in methods:
            color, marker, line = palettes.get(method, ('#777777', 'o', '-'))
            for axis, metric in ((upper, 'gmax'), (lower, 'ndcg')):
                values = [source['conditions'][key]['methods'][method]['metrics'][metric] for key in levels]
                if any(value is None for value in values):
                    raise ValueError('Cannot plot incomplete method comparisons')
                plot = axis.plot(positions, values, color=color, marker=marker, markersize=4,
                                 markerfacecolor=color if method == 'CREST' else 'white',
                                 markeredgecolor='#427EA9' if method == 'CREST' else color,
                                 linewidth=1.6 if method == 'CREST' else 1.0,
                                 linestyle=line if rows == 2 else '-' if metric == 'gmax' else '--',
                                 label=method.replace('PromptGuard2', 'Prompt Guard 2'))[0]
                if metric == 'gmax':
                    legend[method] = plot
        upper.axhline(source['eta'], color='#888888', linestyle=':', linewidth=0.9)
        upper.set_ylabel(r'$G_{\max}$')
        lower.set_ylabel('NDCG@5 (%)')
        for axis in (upper, lower):
            axis.set_xticks(positions, [f'{100 * float(key):g}' for key in levels])
            axis.grid(False)
        lower.set_xlabel(xlabel)
        if rows == 2:
            upper.text(0.02, 0.96, f'({chr(97 + column)})', transform=upper.transAxes, va='top')
            lower.text(0.02, 0.96, f'({chr(99 + column)})', transform=lower.transAxes, va='top')
        else:
            upper.set_xlabel(xlabel)
            upper.text(0.02, 0.96, f'({chr(97 + column)})', transform=upper.transAxes, va='top')
    handles = list(legend.values())
    if rows == 1:
        from matplotlib.lines import Line2D
        handles.extend([Line2D([], [], color='#444444', linestyle='-', label=r'$G_{\max}$ (left)'),
                        Line2D([], [], color='#444444', linestyle='--', label='NDCG@5 (right)')])
    figure.legend(handles, [handle.get_label() for handle in handles],
                  loc='lower center', ncol=3, frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, 0.12 if rows == 2 else 0.17, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def errorbar(axis, xs, rows, color='#82B3D8'):
    available = [(x, row) for x, row in zip(xs, rows) if row['mean'] is not None]
    if not available:
        raise ValueError('No defined estimates are available for this figure')
    x = [pair[0] for pair in available]
    y = [pair[1]['mean'] for pair in available]
    axis.plot(x, y, color=color, marker='o', markeredgecolor='#427EA9', markersize=4, linewidth=1.5)
    for position, row in available:
        interval = row.get('interval')
        if interval is not None:
            if interval[0] > interval[1]:
                raise ValueError('Reversed interval bounds')
            axis.vlines(position, interval[0], interval[1], color=color, linewidth=1.0)
            axis.plot([position, position], interval, linestyle='none', marker='_', color=color, markersize=5)


def rq3(values, output):
    plt = style()
    figure, axes = plt.subplots(1, 3, figsize=(9.0, 2.7))
    alpha, eta = values['alpha'], values['eta']
    alpha_positions = list(range(len(alpha)))
    eta_positions = list(range(len(eta)))
    errorbar(axes[0], alpha_positions, alpha)
    axes[0].plot(alpha_positions, [row['nominal_percent'] for row in alpha], ':', color='#888888', linewidth=1.0)
    errorbar(axes[1], eta_positions, [row['mean_gmax'] for row in eta])
    axes[1].plot(eta_positions, [row['x'] for row in eta], ':', color='#888888', linewidth=1.0)
    errorbar(axes[2], eta_positions, [row['ndcg_percent'] for row in eta])
    for index, (axis, rows, xlabel, ylabel) in enumerate(zip(axes, [alpha, eta, eta],
            [r'$\alpha$', r'$\eta$', r'$\eta$'], ['Miscoverage (%)', r'$G_{\max}$', 'NDCG@5 (%)'])):
        axis.set_xticks(range(len(rows)), [f"{row['x']:g}" for row in rows])
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.text(0.03, 0.96, f'({chr(97 + index)})', transform=axis.transAxes, va='top')
        axis.grid(False)
    figure.tight_layout(w_pad=1.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate figures from computed result summaries.')
    commands = parser.add_subparsers(dest='figure', required=True)
    second = commands.add_parser('rq2')
    second.add_argument('--intensity', type=Path, required=True)
    second.add_argument('--recurrence', type=Path, required=True)
    second.add_argument('--layout', choices=['four', 'dual'], default='four')
    third = commands.add_parser('rq3')
    third.add_argument('--values', type=Path, required=True)
    for command in (second, third):
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.figure == 'rq2':
        rq2(json.loads(args.intensity.read_text()), json.loads(args.recurrence.read_text()), args.output, args.layout)
    else:
        rq3(json.loads(args.values.read_text()), args.output)


if __name__ == '__main__':
    main()
