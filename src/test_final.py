"""The end matches of the final network (models/final/alphazero_large.pt).

    /usr/bin/python3 src/test_final.py               play everything that is missing, then print the result
    /usr/bin/python3 src/test_final.py networks      only the matches between the networks
    /usr/bin/python3 src/test_final.py minimax       only the games against minimax
    /usr/bin/python3 src/test_final.py 0             only print the result of the saved games
    /usr/bin/python3 src/test_final.py trial         time a few games of each kind (nothing is saved)

No position is played twice: the board has 8 symmetries (turning and mirroring), and openings that
are the same after one of them count as one. The openings are all 102 different ones of 2 moves, then
different ones of 3 moves (always the same, chosen at random once), each played from both sides.
Without randomness the players would otherwise repeat the same games.

Between the networks: the final network against large 240 (models/alphazero_large.pt) and medium 240
(models/alphazero_medium.pt), and, as the yardstick on the same openings, large 240 against medium 240.
Both players get the same simulations per move.

Against minimax: the final network against minimax of DEPTHS. The first openings are the different
ones medium 132 played in test_vs_minimax.py, so the two can be compared on them.

Every game is added to models/final/results/ (final_networks.csv, final_vs_minimax.csv); a run only
plays what is missing, so it can be stopped with Ctrl+C and continued.
"""
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import game
import test_networks
import test_tournament
import test_vs_minimax

# ------------------------------- Settings -------------------------------
NAME = 'final 1296'
# name: (variant, file in the models folder or None for its finished network)
PLAYERS = {'final 1296': ('large', os.path.join('final', 'alphazero_large.pt')),
           'large 240': ('large', None), 'medium 240': ('medium', None)}
PAIRS = [('final 1296', 'large 240'), ('final 1296', 'medium 240'), ('large 240', 'medium 240')]
NETWORK_OPENINGS = 300              # different openings (102 of 2 moves, the rest of 3 moves); 2 games each
NETWORK_SIMULATIONS = (500, 2000)
MINIMAX_SIMULATIONS = (500, 2000)
DEPTHS = (13, 15)
MINIMAX_OPENINGS = 102              # different openings, 2 games each
PARALLEL_GAMES = 128                # network games at the same time
MINIMAX_OPENINGS_AT_ONCE = 22       # openings per part against minimax (x2 sides)
MINIMAX_PROCESSES = 8
# ------------------------------------------------------------------------

FIELDS_NETWORKS = test_tournament.FIELDS
FIELDS_MINIMAX = ['player', 'simulations', 'depth', 'network_plays', 'result', 'moves',
                  'network_seconds', 'minimax_seconds', 'opening']
POINTS = {'win': 1.0, 'tie': 0.5, 'loss': 0.0}


def models_dir():
    import Tic_Tac_Toe_NN as NN
    return NN.MODEL_DIR


def results_file(name):
    return os.path.join(models_dir(), 'final', 'results', name)


def load(name, device):
    import torch
    import Tic_Tac_Toe_NN as NN
    variant, file = PLAYERS[name]
    path = os.path.join(models_dir(), file) if file else NN.checkpoint_path(variant)
    model = NN.create_model(variant)
    model.load_state_dict(torch.load(path, map_location='cpu')['model'])
    return model.to(device).eval()


def read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as file:
        return list(csv.DictReader(file))


def append(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        if new:
            writer.writerow(fields)
        writer.writerows(rows)


# ----------------------------------------------------------------------------
# openings without repetitions
# ----------------------------------------------------------------------------

def _symmetries():
    """The 8 ways to turn and mirror a 3x3 field, as: cell -> cell. The same one is used for the
    big field and the little field, so the rule 'the little field sends to that big field' stays."""
    grid, result = np.arange(9).reshape(3, 3), []
    for turns in range(4):
        for mirror in (False, True):
            moved = np.rot90(grid, turns)
            moved = np.fliplr(moved) if mirror else moved
            where = np.empty(9, dtype=int)
            where[moved.flatten()] = np.arange(9)
            result.append(where)
    return result


SYMMETRIES = _symmetries()


def text(opening):
    return test_networks.opening_text(opening)


def parse(opening_text):
    return [tuple(int(i) for i in move.split('-')) for move in opening_text.split()]


def same_position(opening):
    """The same key for all openings that are one position turned or mirrored."""
    return min(tuple((int(w[big]), int(w[little])) for big, little in opening) for w in SYMMETRIES)


def every_opening(moves):
    openings = [[]]
    for _ in range(moves):
        longer = []
        for opening in openings:
            board = game.Board(playerturn=1)
            for move in opening:
                board.execute(list(move))
            longer += [opening + [(int(big), int(little))] for big, little in board.possible_moves()]
        openings = longer
    return openings


def different_openings(count, first=()):
    """`count` openings, no two the same position: `first` (without repetitions), then all of 2 moves,
    then of 3 moves in a fixed random order."""
    chosen, seen = [], set()

    def add(opening):
        key = same_position(opening)
        if key not in seen and len(chosen) < count:
            seen.add(key)
            chosen.append(list(opening))

    for opening in first:
        add(opening)
    for opening in sorted({same_position(o) for o in every_opening(2)}):
        add(opening)
    three = sorted({same_position(o) for o in every_opening(3)})
    for i in np.random.RandomState(0).permutation(len(three)):
        add(three[i])
    return chosen


def medium_132_openings():
    """The openings medium 132 played against minimax, in their order."""
    seen = {}
    for row in read(test_vs_minimax.result_file('medium')):
        if row['iteration'] == '132' and row['opening']:
            seen[row['opening']] = None
    return [parse(t) for t in seen]


# ----------------------------------------------------------------------------
# playing
# ----------------------------------------------------------------------------

def play_networks(device, minutes=None, limit=None):
    import MCTS
    path = results_file('final_networks.csv')
    done = {(r['a'], r['b'], r['simulations'], r['opening'], r['a_plays']) for r in read(path)}
    openings = different_openings(NETWORK_OPENINGS)
    models, start = {}, time.perf_counter()
    for simulations in NETWORK_SIMULATIONS:
        for a, b in PAIRS:
            games = [(o, side) for o in openings for side in (1, -1)
                     if (a, b, str(simulations), text(o), 'X' if side == 1 else 'O') not in done]
            if limit:
                games = games[:limit]
            if not games:
                continue
            for name in (a, b):
                if name not in models:
                    models[name] = load(name, device)
            parallel = min(PARALLEL_GAMES, len(games))
            search_a = MCTS.BatchedSearch(models[a], device, parallel, simulations)
            search_b = MCTS.BatchedSearch(models[b], device, parallel, simulations)
            for first in range(0, len(games), parallel):
                if minutes is not None and time.perf_counter() - start > minutes * 60:
                    return
                batch = games[first:first + parallel]
                part = time.perf_counter()
                results = test_networks.play_batch(search_a, search_b, batch)
                seconds = time.perf_counter() - part
                if limit:
                    print(f'  trial: {len(batch)} games, {simulations} simulations, {a} against {b} '
                          f'in {seconds:.0f}s', flush=True)
                    break
                append(path, FIELDS_NETWORKS, [[a, b, simulations, 'X' if side == 1 else 'O', result, moves,
                                                text(o)] for (o, side), (result, moves) in zip(batch, results)])
                count = [sum(r == x for r, _ in results) for x in test_networks.RESULTS]
                print(f'{simulations} simulations, {a} against {b}: {len(batch)} games in {seconds:.0f}s, '
                      f'wins-ties-losses {count[0]}-{count[1]}-{count[2]} '
                      f'({len(games) - first - len(batch)} games of this pair left)', flush=True)
            del search_a, search_b


def play_minimax(device, minutes=None, limit=None):
    import MCTS
    path = results_file('final_vs_minimax.csv')
    done = {(r['simulations'], r['depth'], r['opening']) for r in read(path)}
    openings = different_openings(MINIMAX_OPENINGS, medium_132_openings())
    final = load(NAME, device)
    start = time.perf_counter()
    with Pool(MINIMAX_PROCESSES, initializer=test_vs_minimax._warm_up) as pool:
        for simulations in MINIMAX_SIMULATIONS:
            search = None
            for depth in DEPTHS:
                todo = [o for o in openings if (str(simulations), str(depth), text(o)) not in done]
                if limit:
                    todo = todo[:limit]
                for first in range(0, len(todo), MINIMAX_OPENINGS_AT_ONCE):
                    if minutes is not None and time.perf_counter() - start > minutes * 60:
                        return
                    part = todo[first:first + MINIMAX_OPENINGS_AT_ONCE]
                    if search is None or search.n_games < 2 * len(part):
                        search = MCTS.BatchedSearch(final, device, 2 * len(part), simulations)
                    part_start = time.perf_counter()
                    games = test_vs_minimax.play_round(search, pool, (depth,), part)
                    seconds = time.perf_counter() - part_start
                    if limit:
                        print(f'  trial: {len(games)} games, {simulations} simulations against depth {depth} '
                              f'in {seconds:.0f}s', flush=True)
                        break
                    append(path, FIELDS_MINIMAX, [[NAME, simulations, *g] for g in games])
                    count = [sum(g[2] == x for g in games) for x in test_vs_minimax.RESULTS]
                    print(f'{simulations} simulations against depth {depth}: {len(games)} games in '
                          f'{seconds:.0f}s, wins-ties-losses {count[0]}-{count[1]}-{count[2]} '
                          f'({len(todo) - first - len(part)} openings left)', flush=True)


# ----------------------------------------------------------------------------
# the result
# ----------------------------------------------------------------------------

def counts(results):
    w, t, l = (results.count(x) for x in ('win', 'tie', 'loss'))
    return (w + t / 2) / max(1, w + t + l), w, t, l


def network_summary(rows, a, b, simulations, only=None):
    """(points share, wins, ties, losses, openings worse, equal, better) of a against b. Games of the same
    opening and side are counted once; only: a set of opening texts to count."""
    games = {}
    for r in rows:
        if r['a'] == a and r['b'] == b and r['simulations'] == str(simulations) \
                and (only is None or r['opening'] in only):
            games[(r['opening'], r['a_plays'])] = r['result']
    if not games:
        return None
    sums = {}
    for (opening, _), result in games.items():
        sums[opening] = sums.get(opening, 0) + POINTS[result]
    complete = [s for o, s in sums.items() if (o, 'X') in games and (o, 'O') in games]
    return counts(list(games.values())) + (sum(s < 1 for s in complete), sum(s == 1 for s in complete),
                                           sum(s > 1 for s in complete))


def summary():
    print('\nBetween the networks (different openings, both sides; per opening both games added: '
          'below 1 = worse, 1 = equal, above 1 = better):')
    mine = read(results_file('final_networks.csv'))
    for simulations in NETWORK_SIMULATIONS:
        for a, b in PAIRS:
            s = network_summary(mine, a, b, simulations)
            if s:
                print(f'  {simulations:>4} simulations, {a} against {b}: {s[0]:.1%} '
                      f'(wins-ties-losses {s[1]}-{s[2]}-{s[3]}), openings worse/equal/better {s[4]}/{s[5]}/{s[6]}')
    old = read(test_tournament.result_file('large'))
    for simulations in NETWORK_SIMULATIONS:
        s = network_summary(old, 'large 240', 'medium 240', simulations)
        if s:
            print(f'  ({simulations} simulations, the old tournament, large 240 against medium 240: {s[0]:.1%}, '
                  f'{s[1]}-{s[2]}-{s[3]}, openings {s[4]}/{s[5]}/{s[6]}; its 118 openings are 71 positions)')
    print('\nAgainst minimax (different openings, both sides):')
    mine = read(results_file('final_vs_minimax.csv'))
    medium = [r for r in read(test_vs_minimax.result_file('medium')) if r['iteration'] == '132']
    for simulations in MINIMAX_SIMULATIONS:
        for depth in DEPTHS:
            games = [r for r in mine if r['simulations'] == str(simulations) and r['depth'] == str(depth)]
            if not games:
                continue
            s = counts([r['result'] for r in games])
            line = f'  {simulations:>4} simulations, depth {depth}: {NAME} {s[0]:.1%} ({s[1]}-{s[2]}-{s[3]})'
            theirs = {(r['opening'], r['network_plays']): r['result'] for r in medium
                      if r['simulations'] == str(simulations) and r['depth'] == str(depth)}
            both = [r for r in games if (r['opening'], r['network_plays']) in theirs]
            if both:
                o = counts([r['result'] for r in both])
                m = counts([theirs[(r['opening'], r['network_plays'])] for r in both])
                line += (f' | on the {len(both)} games medium 132 also played: {NAME} {o[0]:.1%} '
                         f'({o[1]}-{o[2]}-{o[3]}), medium 132 {m[0]:.1%} ({m[1]}-{m[2]}-{m[3]})')
            print(line)


if __name__ == '__main__':
    import Tic_Tac_Toe_NN as NN
    argument = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if argument == '0':
        summary()
        raise SystemExit
    device = NN.choose_device()
    print(f'{NAME} on {device}', flush=True)
    if argument == 'trial':
        play_networks(device, limit=16)
        play_minimax(device, limit=4)
        raise SystemExit
    try:
        if argument in ('all', 'networks'):
            play_networks(device)
        if argument in ('all', 'minimax'):
            play_minimax(device)
    except KeyboardInterrupt:
        print('\nStopped; the finished parts are saved, the next run continues.')
    summary()
