"""Shows the progress of a training run as charts, from the log that train.py writes.

    /usr/bin/python3 src/plot_log.py                 the model set in train.py
    /usr/bin/python3 src/plot_log.py small           another model
    /usr/bin/python3 src/plot_log.py medium out.png  save a picture instead of opening a window

Works while the training is running; start it again to see the newest rounds.
"""
import csv
import re
import sys

import matplotlib
import matplotlib.pyplot as plt

import Tic_Tac_Toe_NN as NN
import train

# colours
DEPTH_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
GOOD, NEUTRAL, BAD = '#2a78d6', '#c3c2b7', '#eb6834'      # win / tie / loss
SURFACE, INK, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'


def read_log(path):
    """The log as a list of rows; every value a number, or None where the cell is empty."""
    with open(path, newline='') as file:
        return [{key: float(value) if value not in ('', None) else None for key, value in row.items()}
                for row in csv.DictReader(file)]


def column(rows, name):
    """(iterations, values) of one column, without the rounds where it is empty."""
    pairs = [(row['iteration'], row[name]) for row in rows if row.get(name) is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def style(axis, title, ylabel=None, percent=False):
    axis.set_title(title, loc='left', fontsize=11, color=INK, pad=8)
    axis.set_xlabel('round', color=MUTED, fontsize=9)
    if ylabel:
        axis.set_ylabel(ylabel, color=MUTED, fontsize=9)
    axis.set_facecolor(SURFACE)
    axis.grid(axis='y', color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ('top', 'right', 'left'):
        axis.spines[side].set_visible(False)
    axis.spines['bottom'].set_color(AXIS)
    axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if percent:
        axis.set_ylim(0, 1)
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def legend(axis, **options):
    axis.legend(frameon=False, fontsize=8, labelcolor=MUTED, handlelength=1.4, **options)


def plot_loss(axis, rows, name, title):
    x, y = column(rows, name)
    axis.plot(x, y, color=DEPTH_COLORS[0], linewidth=2)
    style(axis, title)


def plot_self_play(axis, rows):
    """How the self-play games of each round ended, as shares."""
    x = [row['iteration'] for row in rows]
    totals = [row['x_wins'] + row['o_wins'] + row['ties'] for row in rows]
    shares = [[row[name] / total for row, total in zip(rows, totals)]
              for name in ('x_wins', 'ties', 'o_wins')]
    axis.stackplot(x, shares, colors=[GOOD, NEUTRAL, BAD], labels=['X wins', 'ties', 'O wins'],
                   edgecolor=SURFACE, linewidth=1)
    style(axis, 'Self-play games', percent=True)
    axis.margins(x=0)
    legend(axis, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)


def plot_scores(axis, rows, depths):
    """Share of points against minimax (win = 1, tie = 0.5), one line per depth."""
    for depth in depths:
        x, y = column(rows, f'vs_minimax_{depth}')
        color = DEPTH_COLORS[depths.index(depth) % len(DEPTH_COLORS)]
        axis.plot(x, y, color=color, linewidth=2, marker='o', markersize=5,
                  markeredgecolor=SURFACE, markeredgewidth=1, label=f'depth {depth}',
                  clip_on=False, zorder=3)
    axis.axhline(0.5, color=AXIS, linewidth=1, linestyle=(0, (4, 3)))
    style(axis, 'Points against minimax (win = 1, tie = 0.5)', percent=True)
    legend(axis, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(depths))


def plot_split(axis, rows, depth):
    """Wins, ties and losses against minimax of one depth, one bar per test."""
    tests = [row for row in rows if row.get(f'wins_{depth}') is not None]
    x = [row['iteration'] for row in tests]
    width = 0.8 * (min(b - a for a, b in zip(x, x[1:])) if len(x) > 1 else 1)
    bottom = [0.0] * len(tests)
    for name, color, label in (('wins', GOOD, 'wins'), ('ties', NEUTRAL, 'ties'), ('losses', BAD, 'losses')):
        totals = [row[f'wins_{depth}'] + row[f'ties_{depth}'] + row[f'losses_{depth}'] for row in tests]
        share = [row[f'{name}_{depth}'] / total for row, total in zip(tests, totals)]
        axis.bar(x, share, width, bottom=bottom, color=color, label=label, edgecolor=SURFACE, linewidth=1)
        bottom = [b + s for b, s in zip(bottom, share)]
    style(axis, f'Against minimax depth {depth}', percent=True)
    legend(axis, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)


def plot(model, save_as=None):
    log_file = train.paths(model)[2]
    rows = read_log(log_file)
    if not rows:
        raise SystemExit(f'{log_file} has no rounds yet.')
    names = rows[0].keys()
    depths = sorted(int(n.rsplit('_', 1)[1]) for n in names
                    if re.fullmatch(r'vs_minimax_\d+', n) and column(rows, n)[0])
    split_depths = [d for d in depths if f'wins_{d}' in names and column(rows, f'wins_{d}')[0]]

    columns = max(3, len(split_depths))
    figure = plt.figure(figsize=(4.6 * columns, 11 if split_depths else 7.5), facecolor=SURFACE)
    grid = figure.add_gridspec(3 if split_depths else 2, columns)
    plot_loss(figure.add_subplot(grid[0, 0]), rows, 'policy_loss', 'Policy loss (lower = moves better predicted)')
    plot_loss(figure.add_subplot(grid[0, 1]), rows, 'value_loss', 'Value loss (lower = results better predicted)')
    plot_self_play(figure.add_subplot(grid[0, 2:]), rows)
    if depths:
        plot_scores(figure.add_subplot(grid[1, :]), rows, depths)
    for i, depth in enumerate(split_depths):
        plot_split(figure.add_subplot(grid[2, i]), rows, depth)
    figure.suptitle(f"Training of the '{model}' network: {len(rows)} rounds, "
                    f"{int(rows[-1]['games'])} games", x=0.02, ha='left', fontsize=13, color=INK)
    figure.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.5, w_pad=2.5)
    if save_as:
        figure.savefig(save_as, dpi=130)
        print(f'Saved {save_as}')
    else:
        plt.show()


if __name__ == '__main__':
    arguments = sys.argv[1:]
    model = arguments[0] if arguments else train.MODEL
    if model not in NN.MODEL_VARIANTS:
        raise SystemExit(f"Unknown model '{model}'. Use one of {list(NN.MODEL_VARIANTS)}.")
    plot(model, arguments[1] if len(arguments) > 1 else None)
