"""Lets two versions of a network play against each other, for several search sizes.

    /usr/bin/python3 src/test_networks.py        play all games that are missing, then print the totals
    /usr/bin/python3 src/test_networks.py 60     stop after about 60 minutes (the next run continues)
    /usr/bin/python3 src/test_networks.py 0      only print the totals of the saved games and draw the chart

NETWORK_A and NETWORK_B are training iterations of the same model: the newest
network, or a snapshot that train.py has kept in models/snapshots/. For every
search size in SIMULATIONS both get that many simulations per move, and every
opening is played twice, once with A as X and once as O.

The openings are the ones of the test against minimax (test_vs_minimax.py), so
all tests use the same starting positions; without that file they are random.
Two networks without randomness always play the same game from the same opening,
so every opening is played only once per side.

Every finished game is added to  models/results/alphazero_<model>_networks.csv . A run
only plays the games that are not in the file yet, so it can be stopped (Ctrl+C)
and continued; when every opening is played with every search size, it is finished.
After a run the totals are printed and drawn as a chart (.png next to the file). It does not touch the training log, the network or the snapshots.
Do not run it while a training or another test is running.
"""
import csv
import os
import subprocess
import sys
import time

import numpy as np

import game

# ------------------------------- Settings -------------------------------
MODEL = 'medium'                    # which network: 'small', 'medium' or 'large'
NETWORK_A = None                    # None: the newest network; a number: the snapshot of that training iteration
NETWORK_B = 132
SIMULATIONS = (500, 2000, 4000)     # search sizes per move; both networks get the same
PARALLEL_GAMES = 64                 # how many games run at the same time
RANDOM_OPENINGS = 88                # number of openings, if there is no test against minimax to take them from
RANDOM_OPENING_MOVES = 2
OPEN_CHART = True                   # open the chart when the run is over
# ------------------------------------------------------------------------

# a / b: training iteration of the two networks. result: 'win', 'tie' or 'loss' of A.
FIELDS = ['model', 'a', 'b', 'simulations', 'a_plays', 'result', 'moves', 'opening']
RESULTS = ('win', 'tie', 'loss')


def result_file(model):
    import Tic_Tac_Toe_NN as NN
    return NN.data_path('results', f'alphazero_{model}_networks.csv')


def chart_file(model, a, b):
    return result_file(model)[:-4] + f'_{a}_vs_{b}.png'


def opening_text(opening):
    return ' '.join(f'{int(big)}-{int(little)}' for big, little in opening)


def openings_to_play(model):
    """The openings of the test against minimax, in their order; random ones if there are none."""
    import test_vs_minimax
    rounds = test_vs_minimax.saved_rounds(test_vs_minimax.result_file(model), model)
    openings = {opening_text(opening): opening for _, one in sorted(rounds.items()) for opening in one['openings']}
    if openings:
        return list(openings.values()), 'the openings of the test against minimax'
    test_vs_minimax.RANDOM_OPENING_MOVES = RANDOM_OPENING_MOVES
    return test_vs_minimax.random_openings(RANDOM_OPENINGS), 'random openings'


def saved_games(path, model, a, b):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as file:
        return [row for row in csv.DictReader(file)
                if row['model'] == model and row['a'] == str(a) and row['b'] == str(b)]


def save_games(path, rows):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        if new:
            writer.writerow(FIELDS)
        writer.writerows(rows)


def _result(board, side):
    """'win', 'tie' or 'loss' of the network that plays this side in a finished game."""
    if not game.game_is_win(board.won_games)[0]:
        return 'tie'
    return 'win' if -board.playerturn == side else 'loss'     # the player who moved last has won


def play_batch(search_a, search_b, games):
    """games: list of (opening, side of A: 1 = X, -1 = O), all played at the same time.
    Returns one (result of A, number of moves) per game."""
    boards = []
    for opening, _ in games:
        board = game.Board(playerturn=1)
        for move in opening:
            board.execute(move)
        boards.append(board)
    active = np.zeros(search_a.n_games, dtype=bool)
    while not all(board.game_is_over for board in boards):
        for search, sign in ((search_a, 1), (search_b, -1)):
            active[:] = False
            for g, board in enumerate(boards):
                if not board.game_is_over and board.playerturn == sign * games[g][1]:
                    active[g] = True
                    search.set_board(g, board)
            if active.any():
                counts, _ = search.search(active)
                for g in np.flatnonzero(active):
                    boards[g].execute(int(np.argmax(counts[g])))
    return [(_result(board, side), len(board.log)) for (_, side), board in zip(games, boards)]


def run(minutes=None):
    import MCTS
    import Tic_Tac_Toe_NN as NN

    device = NN.choose_device()
    a = NN.saved_iteration(MODEL) if NETWORK_A is None else NETWORK_A
    b = NN.saved_iteration(MODEL) if NETWORK_B is None else NETWORK_B
    if a == b:
        raise SystemExit(f'NETWORK_A and NETWORK_B are both iteration {a}. Choose two different ones.')
    model_a = NN.load_model(MODEL, device, NETWORK_A)
    model_b = NN.load_model(MODEL, device, NETWORK_B)
    path = result_file(MODEL)
    openings, source = openings_to_play(MODEL)
    done = {(row['simulations'], row['opening'], row['a_plays']) for row in saved_games(path, MODEL, a, b)}
    print(f"'{MODEL}' network after iteration {a} (A) against iteration {b} (B) with "
          f"{', '.join(map(str, SIMULATIONS))} simulations per move, {len(openings)} openings from both "
          f"sides ({source}) on {device}", flush=True)
    start, played = time.perf_counter(), 0
    for simulations in SIMULATIONS:
        todo = [(opening, side) for opening in openings for side in (1, -1)
                if (str(simulations), opening_text(opening), 'X' if side == 1 else 'O') not in done]
        if not todo:
            continue
        search_a = MCTS.BatchedSearch(model_a, device, PARALLEL_GAMES, simulations)
        search_b = MCTS.BatchedSearch(model_b, device, PARALLEL_GAMES, simulations)
        for first in range(0, len(todo), PARALLEL_GAMES):
            if minutes is not None and played and time.perf_counter() - start > minutes * 60:
                print('The time is over; the next run plays the games that are left.')
                return
            games = todo[first:first + PARALLEL_GAMES]
            print(f'{simulations} simulations: playing games {first + 1} to {first + len(games)} '
                  f'of the {len(todo)} that are missing ...', flush=True)
            batch_start = time.perf_counter()
            results = play_batch(search_a, search_b, games)
            save_games(path, [[MODEL, a, b, simulations, 'X' if side == 1 else 'O', result, moves,
                               opening_text(opening)] for (opening, side), (result, moves) in zip(games, results)])
            played += len(games)
            count = [sum(result == r for result, _ in results) for r in RESULTS]
            print(f'{simulations} simulations: these {len(games)} games took '
                  f'{time.perf_counter() - batch_start:.0f}s and are saved | A wins-ties-losses '
                  f'{count[0]}-{count[1]}-{count[2]} | {len(todo) - first - len(games)} games left '
                  f'with {simulations} simulations', flush=True)
        del search_a, search_b
    print(f'{played} games in {(time.perf_counter() - start) / 60:.1f} minutes, saved to {path}')


def summary():
    import Tic_Tac_Toe_NN as NN
    a = NN.saved_iteration(MODEL) if NETWORK_A is None else NETWORK_A
    b = NN.saved_iteration(MODEL) if NETWORK_B is None else NETWORK_B
    games = saved_games(result_file(MODEL), MODEL, a, b)
    if not games:
        print(f'No saved games of iteration {a} against iteration {b} yet.')
        return None
    print(f"\n'{MODEL}' network after iteration {a} (A) against iteration {b} (B), "
          f'both with the same number of simulations per move:')
    for simulations in sorted({int(row['simulations']) for row in games}):
        rows = [row for row in games if int(row['simulations']) == simulations]
        parts = []
        for label, side in (('all games', None), ('A as X', 'X'), ('A as O', 'O')):
            c = [sum(row['result'] == r and side in (None, row['a_plays']) for row in rows) for r in RESULTS]
            parts.append(f'{label}: {c[0]}-{c[1]}-{c[2]}')
            if side is None:
                points = (c[0] + 0.5 * c[1]) / sum(c)
        print(f'  {simulations} simulations, {len(rows)} games: A scores {points:.0%} of the points'
              f' | wins-ties-losses of A  ' + ', '.join(parts))
    return plot(games, a, b)


def plot(games, a, b):
    """One bar per search size: how the games ended for network A. Returns the path of the chart."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_log import GOOD, NEUTRAL, BAD, SURFACE, INK, AXIS, style, legend

    sizes = sorted({int(row['simulations']) for row in games})
    counts = {size: {side: [sum(row['result'] == r and int(row['simulations']) == size
                                and side in (None, row['a_plays']) for row in games) for r in RESULTS]
                     for side in (None, 'X', 'O')} for size in sizes}
    figure, axis = plt.subplots(figsize=(max(8.5, 2.2 + 2.6 * len(sizes)), 6.2), facecolor=SURFACE)
    bottom = np.zeros(len(sizes))
    for i, (color, label, ink) in enumerate(((GOOD, f'iteration {a} wins', 'white'), (NEUTRAL, 'ties', INK),
                                              (BAD, f'iteration {b} wins', 'white'))):
        shares = np.array([counts[size][None][i] / sum(counts[size][None]) for size in sizes])
        axis.bar(range(len(sizes)), shares, 0.62, bottom=bottom, color=color, label=label,
                 edgecolor=SURFACE, linewidth=2)
        for x, (share, size, low) in enumerate(zip(shares, sizes, bottom)):
            if share >= 0.06:
                axis.text(x, low + share / 2, str(counts[size][None][i]), ha='center', va='center',
                          color=ink, fontsize=10)
        bottom = bottom + shares
    axis.axhline(0.5, color=AXIS, linewidth=1, linestyle=(0, (4, 3)))
    labels = []
    for size in sizes:
        c, x, o = counts[size][None], counts[size]['X'], counts[size]['O']
        labels.append(f'{size} simulations\n{(c[0] + 0.5 * c[1]) / sum(c):.0%} of points for iteration {a}\n'
                      f'{sum(c)} games\nas X {x[0]}-{x[1]}-{x[2]}, as O {o[0]}-{o[1]}-{o[2]}')
    style(axis, f"'{MODEL}' network after iteration {a} against iteration {b}\n"
                f'both with the same number of simulations per move', percent=True)
    axis.set_xlabel('')
    axis.set_xticks(range(len(sizes)))
    axis.set_xticklabels(labels, fontsize=8.5)
    axis.set_xlim(-0.7, max(len(sizes), 3) - 0.3)      # the bars keep their width while settings are missing
    legend(axis, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    figure.tight_layout()
    path = chart_file(MODEL, a, b)
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
