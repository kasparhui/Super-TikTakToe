"""Versions of a network play against a finished network, or all against each other.

    /usr/bin/python3 src/test_tournament.py        play all games that are missing, then print and draw the result
    /usr/bin/python3 src/test_tournament.py 60     stop after about 60 minutes (the next run continues)
    /usr/bin/python3 src/test_tournament.py 0      only print and draw the result of the saved games

The players are snapshots that train.py has kept in models/snapshots/ (copy them
there from Google Drive first), the newest network, and any finished networks
named in ALSO. With PAIRS = 'against_also' every version only plays against the
networks in ALSO; with 'all' everyone plays everyone. Every pair plays every
opening twice, once from each side, both with the same number of simulations per
move and without randomness. The openings are the same for all pairs.

Every finished game is added to  models/alphazero_<model>_tournament.csv . A run
only plays what is missing, so it can be stopped (Ctrl+C) and continued, and
players can be added later: start it again when the training has made new snapshots.
The chart (.png next to the file) shows the points over the training iterations,
or with 'all' who scored how much against whom and a rating for every player.
On Google Colab (colab_test.ipynb) it can run in a second session beside the training.
On one machine, do not run it while a training or another test is running.
"""
import csv
import os
import re
import subprocess
import sys
import time

import numpy as np

import game
import test_networks

# ------------------------------- Settings -------------------------------
MODEL = 'large'                 # the network whose versions are compared
ITERATIONS = 'all'              # 'all': every snapshot that is there, or a list like (40, 80, 120)
EVERY = 40                      # with 'all': only snapshots whose iteration is a multiple of this
NEWEST = False                  # also the newest network (models/alphazero_<MODEL>.pt)
ALSO = ['medium']               # finished networks of other sizes that play too (their newest version)
PAIRS = 'against_also'          # 'against_also': the versions only play the networks in ALSO; 'all': everyone
SIMULATIONS = (500, 2000)       # simulations per move, the same for both players; every size is a full test
OPENINGS = 128                  # openings (2 random moves, always the same ones); 2 games per opening and pair
PARALLEL_GAMES = 256            # how many games run at the same time (256 for Colab's L4; 64 on a Mac)
OPEN_CHART = True
# ------------------------------------------------------------------------

FIELDS = ['a', 'b', 'simulations', 'a_plays', 'result', 'moves', 'opening']
POINTS = {'win': 1.0, 'tie': 0.5, 'loss': 0.0}


def result_file(model):
    import Tic_Tac_Toe_NN as NN
    return os.path.join(NN.MODEL_DIR, f'alphazero_{model}_tournament.csv')


def players():
    """[(name, model, iteration or None for the newest)], the versions of MODEL in order of training."""
    import Tic_Tac_Toe_NN as NN
    folder = os.path.dirname(NN.snapshot_path(MODEL, 0))
    there = sorted(int(m.group(1)) for name in (os.listdir(folder) if os.path.isdir(folder) else [])
                   for m in [re.fullmatch(rf'alphazero_{MODEL}_iteration(\d+)\.pt', name)] if m)
    if ITERATIONS == 'all':
        chosen = [i for i in there if i % EVERY == 0]
    else:
        chosen = list(ITERATIONS)
        missing = [i for i in chosen if i not in there]
        if missing:
            raise SystemExit(f'There is no snapshot of iteration {missing} in {folder}.')
    result = [(f'{MODEL} {i}', MODEL, i) for i in chosen]
    if NEWEST and os.path.exists(NN.checkpoint_path(MODEL)):
        newest = NN.saved_iteration(MODEL)
        if newest not in chosen:
            result.append((f'{MODEL} {newest}', MODEL, None))
    for other in ALSO:
        if os.path.exists(NN.checkpoint_path(other)):
            result.append((f'{other} {NN.saved_iteration(other)}', other, None))
        else:
            print(f"(No finished '{other}' network in the models folder, so it does not play.)")
    return result


def openings():
    """Always the same openings (the first ones are those of the check during the training)."""
    random, result = np.random.RandomState(0), []
    for _ in range(OPENINGS):
        board, opening = game.Board(playerturn=1), []
        for _ in range(2):
            moves = board.possible_moves()
            opening.append(moves[random.randint(len(moves))])
            board.execute(opening[-1])
        result.append(opening)
    return result


def saved_games(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as file:
        return list(csv.DictReader(file))


def run(minutes=None):
    import MCTS
    import Tic_Tac_Toe_NN as NN

    device = NN.choose_device()
    entrants = players()
    if len(entrants) < 2:
        raise SystemExit('A tournament needs at least two players. Copy snapshots into models/snapshots/ '
                         'or change the settings at the top of this file.')
    path = result_file(MODEL)
    done = {(row['a'], row['b'], row['simulations'], row['opening'], row['a_plays']) for row in saved_games(path)}
    pairs = [(a, b) for i, a in enumerate(entrants) for b in entrants[i + 1:]
             if PAIRS == 'all' or (a[1] == MODEL) != (b[1] == MODEL)]
    todo = []
    for simulations in SIMULATIONS:
        for a, b in pairs:
            games = [(opening, side) for opening in openings() for side in (1, -1)
                     if (a[0], b[0], str(simulations), test_networks.opening_text(opening),
                         'X' if side == 1 else 'O') not in done]
            if games:
                todo.append((simulations, a, b, games))
    missing = sum(len(games) for _, _, _, games in todo)
    print(f"{len(entrants)} players ({', '.join(name for name, _, _ in entrants)}), {len(pairs)} pairs, "
          f"{', '.join(map(str, SIMULATIONS))} simulations per move on {device}: {missing} games are missing",
          flush=True)
    models, start, played = {}, time.perf_counter(), 0
    for simulations, a, b, games in todo:
        for name, model, iteration in (a, b):
            if name not in models:
                models[name] = NN.load_model(model, device, iteration)
        parallel = min(PARALLEL_GAMES, len(games))
        search_a = MCTS.BatchedSearch(models[a[0]], device, parallel, simulations)
        search_b = MCTS.BatchedSearch(models[b[0]], device, parallel, simulations)
        for first in range(0, len(games), parallel):
            if minutes is not None and played and time.perf_counter() - start > minutes * 60:
                print('The time is over; the next run plays the games that are left.')
                return
            batch = games[first:first + parallel]
            batch_start = time.perf_counter()
            results = test_networks.play_batch(search_a, search_b, batch)
            new = not os.path.exists(path)
            with open(path, 'a', newline='') as file:
                writer = csv.writer(file)
                if new:
                    writer.writerow(FIELDS)
                writer.writerows([a[0], b[0], simulations, 'X' if side == 1 else 'O', result, moves,
                                  test_networks.opening_text(opening)]
                                 for (opening, side), (result, moves) in zip(batch, results))
            played += len(batch)
            count = [sum(result == r for result, _ in results) for r in test_networks.RESULTS]
            print(f'{simulations} simulations, {a[0]} against {b[0]}: {len(batch)} games in '
                  f'{time.perf_counter() - batch_start:.0f}s, {count[0]}-{count[1]}-{count[2]} '
                  f'| {missing - played} games left', flush=True)
        del search_a, search_b
    print(f'{played} games in {(time.perf_counter() - start) / 60:.1f} minutes, saved to {path}')


def ratings(names, points, games):
    """Elo-like ratings from the results of all pairs (a difference of 100 = about 64% of the points).
    points[i][j]: points of i against j, games[i][j]: their number of games."""
    points, games = np.array(points, dtype=float), np.array(games, dtype=float)
    # half a win and half a loss extra per pair, so that a 100% result does not give an endless rating
    points, games = points + 0.5 * (games > 0), games + (games > 0)
    rating = np.zeros(len(names))
    for _ in range(2000):
        expected = 1 / (1 + 10 ** ((rating[None, :] - rating[:, None]) / 400))
        step = (points - games * expected).sum(axis=1) / np.maximum(games.sum(axis=1), 1)
        rating += 200 * step
    return rating


def summary():
    """Prints the result of the saved games and draws the chart(s). Returns the path of the last chart."""
    saved, order = saved_games(result_file(MODEL)), [name for name, _, _ in players()]
    sizes = sorted({int(row['simulations']) for row in saved})
    if not sizes:
        print('No saved games yet.')
        return None
    picture, lines = None, {}
    for simulations in sizes:
        rows = [row for row in saved if int(row['simulations']) == simulations]
        names = [n for n in order if any(n in (row['a'], row['b']) for row in rows)]
        if len(names) < 2:
            continue
        index = {name: i for i, name in enumerate(names)}
        points = np.zeros((len(names), len(names)))
        games = np.zeros((len(names), len(names)))
        for row in rows:
            if row['a'] in index and row['b'] in index:
                a, b = index[row['a']], index[row['b']]
                points[a, b] += POINTS[row['result']]
                points[b, a] += 1 - POINTS[row['result']]
                games[a, b] += 1
                games[b, a] += 1
        others = [i for i, name in enumerate(names) if not name.startswith(MODEL + ' ')]
        print(f'\n{simulations} simulations per move, {int(games.sum() / 2)} games:')
        for i, name in enumerate(names):
            if i in others:
                continue
            for j in others:
                if games[i, j]:
                    wins = sum(1 for row in rows if (row['a'], row['b'], row['result']) in
                               ((name, names[j], 'win'), (names[j], name, 'loss')))
                    ties = sum(1 for row in rows if {row['a'], row['b']} == {name, names[j]}
                               and row['result'] == 'tie')
                    print(f'  {name:>12} against {names[j]}: {points[i, j] / games[i, j]:.0%} of the points '
                          f'({wins}-{ties}-{int(games[i, j]) - wins - ties}) in {int(games[i, j])} games')
                    lines.setdefault((simulations, names[j]), []).append(
                        (int(name.split()[1]), points[i, j] / games[i, j], games[i, j]))
        if PAIRS == 'all':
            rating = ratings(names, points, games)
            anchor = others[0] if others else 0
            rating -= rating[anchor]
            print(f'  ratings ({names[anchor]} = 0, 100 more = about 64% of the points against it): '
                  + ', '.join(f'{name} {r:+.0f}' for name, r in zip(names, rating)))
            picture = plot(names, points, games, rating, anchor, simulations)
    if PAIRS != 'all' and lines:
        picture = plot_against(lines)
    return picture


def plot_against(lines):
    """Points of the versions of MODEL against the finished networks, over the training iterations.
    lines: {(simulations, opponent): [(iteration, share of points, games)]}"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_log import DEPTH_COLORS, SURFACE, INK, MUTED, AXIS, style, legend

    figure, axis = plt.subplots(figsize=(10, 5.8), facecolor=SURFACE)
    for number, ((simulations, opponent), values) in enumerate(sorted(lines.items())):
        values.sort()
        x, y, n = zip(*values)
        color = DEPTH_COLORS[number % len(DEPTH_COLORS)]
        error = [np.sqrt(max(share * (1 - share), 0.05) / (2 * count)) for share, count in zip(y, n)]
        axis.fill_between(x, np.array(y) - error, np.array(y) + error, color=color, alpha=0.12, linewidth=0)
        axis.plot(x, y, color=color, linewidth=2, marker='o', markersize=6, markeredgecolor=SURFACE,
                  markeredgewidth=1, label=f'{simulations} simulations, against {opponent}', zorder=3)
        for xi, yi in zip(x, y):
            axis.annotate(f'{yi:.0%}', (xi, yi), textcoords='offset points', xytext=(0, 9),
                          ha='center', fontsize=8, color=MUTED)
    axis.axhline(0.5, color=AXIS, linewidth=1, linestyle=(0, (4, 3)))
    style(axis, f"Points of the '{MODEL}' network against the finished network (win = 1, tie = 0.5)\n"
                f'{2 * OPENINGS} games per point, the same openings everywhere; '
                f'the band shows the uncertainty', percent=True)
    axis.set_xlabel('training iteration', color=MUTED)
    axis.set_xticks(sorted({v[0] for values in lines.values() for v in values}))
    legend(axis, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2)
    figure.tight_layout()
    path = result_file(MODEL)[:-4] + '.png'
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(f'Chart: {path}')
    return path


def plot(names, points, games, rating, anchor, simulations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from plot_log import GOOD, BAD, SURFACE, INK, MUTED, AXIS, style

    n = len(names)
    figure, (left, right) = plt.subplots(1, 2, figsize=(max(11, 5.5 + 1.1 * n), max(5.5, 2.5 + 0.55 * n)),
                                         facecolor=SURFACE, gridspec_kw={'width_ratios': [1.15, 1]})
    share = np.divide(points, games, out=np.full_like(points, np.nan), where=games > 0)
    colors = LinearSegmentedColormap.from_list('score', [BAD, SURFACE, GOOD])
    left.imshow(share, cmap=colors, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if games[i, j]:
                left.text(j, i, f'{share[i, j]:.0%}', ha='center', va='center', fontsize=9,
                          color='white' if abs(share[i, j] - 0.5) > 0.3 else INK)
    left.set_xticks(range(n))
    left.set_xticklabels(names, rotation=45, ha='right', fontsize=9, color=MUTED)
    left.set_yticks(range(n))
    left.set_yticklabels(names, fontsize=9, color=MUTED)
    left.set_title('Points of the row player against the column player', loc='left', fontsize=11, color=INK)
    left.set_facecolor(SURFACE)
    left.tick_params(length=0)
    for side in left.spines.values():
        side.set_visible(False)

    own = [i for i, name in enumerate(names) if name.startswith(MODEL + ' ')]
    x = [int(names[i].split()[1]) for i in own]
    right.plot(x, rating[own], color=GOOD, linewidth=2, marker='o', markersize=6,
               markeredgecolor=SURFACE, markeredgewidth=1, label=f"'{MODEL}' network", zorder=3)
    for i in range(n):
        if i not in own:
            right.axhline(rating[i], color=BAD, linewidth=1.5, linestyle=(0, (4, 3)), label=names[i])
    style(right, f'Rating ({names[anchor]} = 0, +100 = about 64% of the points)')
    right.set_xlabel('training iteration', color=MUTED)
    right.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower right')
    figure.suptitle(f'Tournament with {simulations} simulations per move, {int(games.sum() / 2)} games',
                    x=0.02, ha='left', fontsize=13, color=INK)
    figure.tight_layout(rect=(0, 0, 1, 0.95), w_pad=3)
    path = result_file(MODEL)[:-4] + f'_{simulations}.png'
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(f'Chart: {path}')
    return path


if __name__ == '__main__':
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else None
    if minutes != 0:
        try:
            run(minutes)
        except KeyboardInterrupt:
            print('\nStopped. The games that were still running are not counted; '
                  'the next run plays the ones that are missing.')
    picture = summary()
    if picture and OPEN_CHART and sys.platform == 'darwin':
        subprocess.run(['open', picture])
