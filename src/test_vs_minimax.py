"""Measures how strong a trained network is against minimax, for several search sizes.

    /usr/bin/python3 src/test_vs_minimax.py        play for MINUTES, then draw the chart
    /usr/bin/python3 src/test_vs_minimax.py 10     play for 10 minutes instead
    /usr/bin/python3 src/test_vs_minimax.py 0      only draw the chart from the saved games
    /usr/bin/python3 src/test_vs_minimax.py time   measure the real thinking time per move

Every finished game is added to  models/alphazero_<model>_vs_minimax.csv  (one line
per game), so several runs add up. The chart is saved next to it as a .png; it is
redrawn every CHART_EVERY rounds and opened when the run is over.

In one round the network plays, with every search size in SIMULATIONS, against
every depth in DEPTHS. Every random opening is played twice, once with the network
as X and once as O, and all of them get the same openings in a round.
If a run is stopped in the middle of a round (Ctrl+C), the next run first plays
the missing part of that round with the same openings, so that every search size
has played exactly the same starting positions.

'time' plays TIMING_GAMES games one at a time, like in main.py, and saves how long
the network and minimax need per move. The chart shows these times. It only
measures the settings that have no time yet, so run it once, and again after
adding a new entry to SIMULATIONS or DEPTHS.

This is separate from the tests during training: it does not touch the training
log, the network or the snapshots. Do not run it while a training is running,
both would get slow.
"""
import csv
import os
import signal
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

import fast_minimax
import game

# ------------------------------- Settings -------------------------------
MODEL = 'medium'                    # which network: 'small', 'medium' or 'large'
SIMULATIONS = (500, 2000, 4000)     # search sizes of the network per move
DEPTHS = (12, 13, 14, 15)           # minimax opponents; each searches its full depth, no time limit
MINUTES = 30                # how long to play (the round that is running is finished first)
GAMES_PER_ROUND = 8         # games per search size against each depth in a round (must be even)
CHART_EVERY = 2             # after this many rounds the totals are printed and the chart is redrawn
OPEN_CHART = True           # open the chart when the run is over
RANDOM_OPENING_MOVES = 2    # random moves at the start, so that the games differ
MINIMAX_PROCESSES = 8       # minimax moves of different games are computed in parallel
TIMING_GAMES = 2            # games played one at a time for 'time'
# ------------------------------------------------------------------------

# network_seconds / minimax_seconds: average thinking time per move in that game.
# Many games run at the same time here, so for the network it is its share of the
# time (playing one game alone it needs longer per move), and minimax is somewhat
# slower than alone because several searches share the processor.
# round / opening: the number of the round and the random opening moves of the game
# (big field-little field for every move), used to finish a round that was stopped.
FIELDS = ['model', 'iteration', 'simulations', 'depth', 'network_plays', 'result', 'moves',
          'network_seconds', 'minimax_seconds', 'round', 'opening']
RESULTS = ('win', 'tie', 'loss')


def result_file(model):
    import Tic_Tac_Toe_NN as NN
    return os.path.join(NN.MODEL_DIR, f'alphazero_{model}_vs_minimax.csv')


def timing_file(model):
    return result_file(model)[:-4] + '_times.csv'


def chart_file(model):
    return result_file(model)[:-4] + '.png'


def _warm_up():
    signal.signal(signal.SIGINT, signal.SIG_IGN)            # Ctrl+C is handled by the main process
    fast_minimax.best_move(game.Board(playerturn=1), 2)     # compiles the search in this process


def _minimax_move(job):
    board, depth = job
    start = time.perf_counter()
    move = fast_minimax.best_move(board, depth)[1]
    return move, time.perf_counter() - start


def random_openings(count):
    openings = []
    for _ in range(count):
        board, opening = game.Board(playerturn=1), []
        for _ in range(RANDOM_OPENING_MOVES):
            moves = board.possible_moves()
            opening.append(moves[np.random.randint(len(moves))])
            board.execute(opening[-1])
        openings.append(opening)
    return openings


# ----------------------------------------------------------------------------
# the games
# ----------------------------------------------------------------------------

def play_round(search, pool, depths, openings):
    """Against every depth: every opening twice, with the network as X and as O.
    All games run at the same time. Returns a list of
    (depth, 'X' or 'O' for the network, 'win' / 'tie' / 'loss' of the network, moves,
     seconds per move of the network, seconds per move of minimax, opening as text)."""
    depth_of, network_side, boards, opening_of = [], [], [], []
    for depth in depths:
        for opening in openings:
            for side in (1, -1):
                board = game.Board(playerturn=1)
                for move in opening:
                    board.execute(move)
                depth_of.append(depth)
                network_side.append(side)
                boards.append(board)
                opening_of.append(' '.join(f'{int(big)}-{int(little)}' for big, little in opening))
    seconds = np.zeros((2, len(boards)))    # thinking time: network, minimax
    turns = np.zeros((2, len(boards)))      # number of moves: network, minimax
    while not all(board.game_is_over for board in boards):
        active = np.array([not board.game_is_over and board.playerturn == network_side[g]
                           for g, board in enumerate(boards)])
        if active.any():
            for g in np.flatnonzero(active):
                search.set_board(g, boards[g])
            start = time.perf_counter()
            counts, _ = search.search(active)
            seconds[0, active] += (time.perf_counter() - start) / active.sum()
            turns[0, active] += 1
            for g in np.flatnonzero(active):
                boards[g].execute(int(np.argmax(counts[g])))
        waiting = [g for g, board in enumerate(boards)
                   if not board.game_is_over and board.playerturn != network_side[g]]
        answers = pool.map(_minimax_move, [(boards[g], depth_of[g]) for g in waiting])
        for g, (move, used) in zip(waiting, answers):
            boards[g].execute(move)
            seconds[1, g] += used
            turns[1, g] += 1
    games = []
    for g, board in enumerate(boards):
        if not game.game_is_win(board.won_games)[0]:
            result = 'tie'
        else:       # the player who moved last has won
            result = 'win' if -board.playerturn == network_side[g] else 'loss'
        per_move = seconds[:, g] / np.maximum(turns[:, g], 1)
        games.append((depth_of[g], 'X' if network_side[g] == 1 else 'O', result, len(board.log),
                      round(float(per_move[0]), 3), round(float(per_move[1]), 3), opening_of[g]))
    return games


def saved_rounds(path, model):
    """{round number: {'iteration', 'sizes' (search sizes that played it), 'depths', 'openings'}}
    of the saved games."""
    rounds = {}
    if os.path.exists(path):
        with open(path, newline='') as file:
            for row in csv.DictReader(file):
                if row['model'] == model and row.get('round'):
                    one = rounds.setdefault(int(row['round']), {'iteration': row['iteration'], 'sizes': set(),
                                                                'depths': set(), 'openings': {}})
                    one['sizes'].add(int(row['simulations']))
                    one['depths'].add(int(row['depth']))
                    one['openings'][row['opening']] = None      # keeps the order, without doubles
    for one in rounds.values():
        one['openings'] = [[tuple(int(i) for i in move.split('-')) for move in text.split()]
                           for text in one['openings']]
    return rounds


def unfinished_rounds(rounds, iteration):
    """[(round number, openings, search sizes that still have to play it)] for the rounds
    that were stopped, as far as they fit to the current settings and network."""
    todo = []
    for number, one in sorted(rounds.items()):
        missing = [size for size in SIMULATIONS if size not in one['sizes']]
        same_test = (one['iteration'] == str(iteration) and one['depths'] == set(DEPTHS)
                     and len(one['openings']) == GAMES_PER_ROUND // 2)
        if missing and same_test:
            todo.append((number, one['openings'], missing))
    return todo


def save_games(path, games):
    """Adds the games to the file. A file from an older version of this script
    (fewer columns) is rewritten first, with the new columns left empty."""
    old = []
    if os.path.exists(path):
        with open(path, newline='') as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != FIELDS:
                old = list(reader)
    if old or not os.path.exists(path):
        with open(path + '.tmp', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS, restval='')
            writer.writeheader()
            writer.writerows(old)
        os.replace(path + '.tmp', path)
    with open(path, 'a', newline='') as file:
        csv.writer(file).writerows(games)


def run(minutes):
    import torch
    import MCTS
    import Tic_Tac_Toe_NN as NN

    if GAMES_PER_ROUND % 2:
        raise SystemExit('GAMES_PER_ROUND must be even, because every opening is played from both sides.')
    device = NN.choose_device()
    model = NN.load_model(MODEL, device)
    iteration = torch.load(NN.checkpoint_path(MODEL), map_location='cpu').get('iteration', '')
    path = result_file(MODEL)
    print(f"'{MODEL}' network after iteration {iteration} with {', '.join(map(str, SIMULATIONS))} "
          f"simulations per move, against minimax depth {', '.join(map(str, DEPTHS))} "
          f"for about {minutes:g} minutes on {device}")
    searches = {simulations: MCTS.BatchedSearch(model, device, len(DEPTHS) * GAMES_PER_ROUND, simulations)
                for simulations in SIMULATIONS}
    saved = saved_rounds(path, MODEL)
    todo = unfinished_rounds(saved, iteration)
    next_round = max(saved, default=0) + 1
    start, round_time, rounds, played = time.perf_counter(), 0.0, 0, 0
    with Pool(MINIMAX_PROCESSES, initializer=_warm_up) as pool:
        while True:
            if todo:        # a round that was stopped is finished first, with its own openings
                number, openings, sizes = todo.pop(0)
                print(f"Round {number} was stopped: now playing it with "
                      f"{', '.join(map(str, sizes))} simulations on the same openings", flush=True)
            elif rounds and time.perf_counter() - start + round_time > minutes * 60:
                break       # the next round would not fit into the time any more
            else:
                number, openings, sizes = next_round, random_openings(GAMES_PER_ROUND // 2), SIMULATIONS
                next_round += 1
            round_start = time.perf_counter()
            for simulations in sizes:
                part_start = time.perf_counter()
                games = play_round(searches[simulations], pool, DEPTHS, openings)
                save_games(path, [[MODEL, iteration, simulations, *one[:-1], number, one[-1]] for one in games])
                played += len(games)
                line = ', '.join(
                    f"depth {depth}: " + '-'.join(str(sum(g[0] == depth and g[2] == r for g in games))
                                                  for r in RESULTS)
                    for depth in DEPTHS)
                print(f'Round {number}, {simulations} simulations in {time.perf_counter() - part_start:.0f}s'
                      f' | wins-ties-losses  {line}', flush=True)
            rounds += 1
            round_time = time.perf_counter() - round_start
            if CHART_EVERY and rounds % CHART_EVERY == 0:
                plot(MODEL)
    print(f'{played} games in {(time.perf_counter() - start) / 60:.1f} minutes, saved to {path}')


# ----------------------------------------------------------------------------
# thinking time of a single game
# ----------------------------------------------------------------------------

def single_game_times(model):
    """{('network', simulations) or ('minimax', depth): seconds per move}, measured with 'time'."""
    path, totals = timing_file(model), {}
    if os.path.exists(path):
        with open(path, newline='') as file:
            for row in csv.DictReader(file):
                if row['model'] == model:
                    total = totals.setdefault((row['player'], int(row['setting'])), [0.0, 0])
                    total[0] += float(row['seconds_per_move']) * int(row['moves'])
                    total[1] += int(row['moves'])
    return {key: seconds / moves for key, (seconds, moves) in totals.items()}


def measure_times(everything=False):
    """Plays TIMING_GAMES games one at a time and saves the average thinking time
    per move of the network (for every search size, on the positions where the
    network was to move) and of minimax (for every depth, on its positions).
    Settings that already have a time are skipped, unless everything is True."""
    import MCTS
    import Tic_Tac_Toe_NN as NN

    known = {} if everything else single_game_times(MODEL)
    sizes = [s for s in sorted(SIMULATIONS) if ('network', s) not in known]
    depths = [d for d in sorted(DEPTHS) if ('minimax', d) not in known]
    if not sizes and not depths:
        print("Every setting already has a thinking time. ('time all' measures them again.)")
        return
    network = NN.load_model(MODEL, 'cpu')
    # a setting that is not measured still has to move: then the cheapest one plays
    players = {s: MCTS.NeuralPlayer(network, 'cpu', simulations=s) for s in sizes or [min(SIMULATIONS)]}
    fast_minimax.best_move(game.Board(playerturn=1), 2)
    times = {('network', s): [] for s in sizes}
    times.update({('minimax', d): [] for d in depths})
    print(f"Timing {TIMING_GAMES} games one at a time: '{MODEL}' network with "
          f"{', '.join(map(str, sizes)) or 'no new'} simulations, "
          f"minimax depth {', '.join(map(str, depths)) or '(no new one)'}", flush=True)
    for number in range(TIMING_GAMES):
        board = game.Board(playerturn=1)
        for move in random_openings(1)[0]:
            board.execute(move)
        network_side = 1 if number % 2 == 0 else -1
        while not board.game_is_over:
            if board.playerturn == network_side:
                for size, player in players.items():        # the biggest search decides the move
                    start = time.perf_counter()
                    move = player.best_move(board)[0]
                    if size in sizes:
                        times[('network', size)].append(time.perf_counter() - start)
            else:
                for depth in depths or [min(DEPTHS)]:       # the deepest one decides the move
                    start = time.perf_counter()
                    move = fast_minimax.best_move(board, depth)[1]
                    if depth in depths:
                        times[('minimax', depth)].append(time.perf_counter() - start)
            board.execute(move)
        print(f'  game {number + 1} done ({len(board.log)} moves)', flush=True)
    path = timing_file(MODEL)
    new_file = not os.path.exists(path)
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(['model', 'player', 'setting', 'seconds_per_move', 'slowest_move', 'moves'])
        for (player, setting), used in times.items():
            writer.writerow([MODEL, player, setting, round(float(np.mean(used)), 3),
                             round(max(used), 3), len(used)])
            print(f"  {player} {'depth ' if player == 'minimax' else ''}{setting}"
                  f"{' simulations' if player == 'network' else ''}: "
                  f"{np.mean(used):.2f} s per move (slowest {max(used):.1f} s)")
    print(f'Saved to {path}')


# ----------------------------------------------------------------------------
# chart
# ----------------------------------------------------------------------------

def summary(model):
    """All saved games of this network:
    {(simulations, depth): [wins, ties, losses]} and the thinking times in the test,
    {(simulations, depth): [seconds per move of the network, of minimax]} where measured."""
    table, times = {}, {}
    with open(result_file(model), newline='') as file:
        for row in csv.DictReader(file):
            if row['model'] == model:
                key = (int(row['simulations']), int(row['depth']))
                table.setdefault(key, [0, 0, 0])[RESULTS.index(row['result'])] += 1
                if row.get('minimax_seconds'):          # empty in games of the first version
                    times.setdefault(key, []).append((float(row['network_seconds']),
                                                      float(row['minimax_seconds'])))
    return table, {key: np.mean(values, axis=0) for key, values in times.items()}


def seconds_text(alone, in_test):
    """The time of a single game if it was measured, otherwise the one from the test."""
    if alone is not None:
        return f'{alone:.2f} s per move'
    if in_test is not None:
        return f'{in_test:.2f} s per move in the test'
    return 'time not measured'


def plot(model):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_log import GOOD, NEUTRAL, BAD, SURFACE, INK, MUTED, style, legend

    table, test_times = summary(model)
    if not table:
        raise SystemExit(f'No saved games of the {model} network.')
    alone = single_game_times(model)
    sizes = sorted({key[0] for key in table})
    depths = sorted({key[1] for key in table})
    minimax_text = {}
    for depth in depths:
        in_test = [test_times[key][1] for key in test_times if key[1] == depth]
        minimax_text[depth] = seconds_text(alone.get(('minimax', depth)),
                                           float(np.mean(in_test)) if in_test else None)

    figure, axes = plt.subplots(1, len(sizes), figsize=(1.6 + 1.75 * len(depths) * len(sizes), 6.2),
                                facecolor=SURFACE, sharey=True, squeeze=False)
    print(f'All saved games so far (chart: {chart_file(model)})')
    for axis, size in zip(axes[0], sizes):
        in_test = [test_times[key][0] for key in test_times if key[0] == size]
        network_text = seconds_text(alone.get(('network', size)), float(np.mean(in_test)) if in_test else None)
        print(f'  network with {size} simulations ({network_text}):')
        bottom = np.zeros(len(depths))
        for i, (color, label, ink) in enumerate(((GOOD, 'network wins', 'white'), (NEUTRAL, 'ties', INK),
                                                  (BAD, 'minimax wins', 'white'))):
            counts = [table.get((size, depth), [0, 0, 0]) for depth in depths]
            shares = np.array([c[i] / sum(c) if sum(c) else 0.0 for c in counts])
            axis.bar(range(len(depths)), shares, 0.62, bottom=bottom, color=color, label=label,
                     edgecolor=SURFACE, linewidth=2)
            for x, (share, c, low) in enumerate(zip(shares, counts, bottom)):
                if share >= 0.06:
                    axis.text(x, low + share / 2, str(c[i]), ha='center', va='center', color=ink, fontsize=10)
            bottom = bottom + shares
        labels = []
        for x, depth in enumerate(depths):
            c = table.get((size, depth))
            if c is None:
                axis.text(x, 0.5, 'no games', ha='center', va='center', color=MUTED, fontsize=9)
                labels.append(f'depth {depth}\n\n\nminimax {minimax_text[depth]}')
                continue
            points = (c[0] + 0.5 * c[1]) / sum(c)
            labels.append(f'depth {depth}\n{points:.0%} of points\n{sum(c)} games\nminimax {minimax_text[depth]}')
            print(f'    depth {depth}: {c[0]} wins, {c[1]} ties, {c[2]} losses = {points:.0%} of points '
                  f'in {sum(c)} games | minimax {minimax_text[depth]}', flush=True)
        style(axis, f'{size} simulations\nnetwork: {network_text}', percent=True)
        axis.set_xlabel('')
        axis.set_xticks(range(len(depths)))
        axis.set_xticklabels(labels, fontsize=8.5)
    legend(axes[0][len(sizes) // 2], loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=3)
    note = '' if alone else "  (times from parallel games, not those of a real game: run 'time' once)"
    figure.suptitle(f"'{model}' network against minimax: number of games, share of points "
                    f"(win = 1, tie = 0.5) and thinking time{note}", x=0.02, ha='left', fontsize=12, color=INK)
    figure.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.5)
    figure.savefig(chart_file(model), dpi=130)
    plt.close(figure)
    return chart_file(model)


if __name__ == '__main__':
    argument = sys.argv[1] if len(sys.argv) > 1 else str(MINUTES)
    if argument == 'time':
        measure_times(everything=sys.argv[2:] == ['all'])
    elif float(argument) > 0:
        try:
            run(float(argument))
        except KeyboardInterrupt:
            print('\nStopped. The games that were still running are not counted; everything '
                  'finished before is saved.\nThe next run finishes this round first, on the same openings.')
    picture = plot(MODEL)
    if OPEN_CHART and sys.platform == 'darwin':
        subprocess.run(['open', picture])
