"""Looks at the tree search itself: how far ahead it looks, and whether other search settings play better.

    /usr/bin/python3 src/test_search.py depth      how many moves ahead the search looks (a few minutes)
    /usr/bin/python3 src/test_search.py match      the network with other search settings against itself

The setting that is tried is c_puct: how much the search follows the network's
first impression of the moves (high = wide, many moves are tried) or its own
results so far (low = narrow and deep). The normal value is 1.5.
Do not run it while a training or another test is running.
"""
import copy
import sys
import time

import numpy as np

import game
import MCTS
import Tic_Tac_Toe_NN as NN
import test_networks
import test_tournament

# ------------------------------- Settings -------------------------------
MODEL = 'medium'
NORMAL = 1.5                        # the c_puct that training and play use
TRIED = (0.75, 1.0, 2.5)            # other values
# depth
DEPTH_SIMULATIONS = (500, 2000, 4000)
DEPTH_GAMES = 48                    # positions per stage of the game
STAGES = (8, 24, 40)                # after this many moves of a game
# match
MATCH_SIMULATIONS = (500,)
MATCH_OPENINGS = 128                # 2 games per opening
PARALLEL_GAMES = 64
# ------------------------------------------------------------------------


def positions(model, device):
    """{stage: boards} from games the network plays against itself (with some randomness)."""
    np.random.seed(0)
    search = MCTS.BatchedSearch(model, device, DEPTH_GAMES, 200)
    boards = [game.Board(playerturn=1) for _ in range(DEPTH_GAMES)]
    active, played, result = np.ones(DEPTH_GAMES, dtype=np.bool_), np.zeros(DEPTH_GAMES, dtype=np.int64), {}
    for move in range(max(STAGES)):
        search.search(active)
        chosen = search.choose_moves(active, played, 81)
        status = search.play(active, chosen)
        for g in np.flatnonzero(active):
            boards[g].execute(int(chosen[g]))
            if status[g] != 0:
                active[g] = False
        played += 1
        if move + 1 in STAGES:
            result[move + 1] = [copy.deepcopy(boards[g]) for g in np.flatnonzero(active)]
    return result


def tree_depths(search, g):
    """(average moves ahead per simulation, moves in the main line, deepest line) of the tree of game g."""
    visits, first, count = search.t_visits[g], search.t_first[g], search.t_count[g]
    size = int(search.t_size[g])
    average = (visits[:size].sum() - visits[0]) / max(1, visits[0])
    depth = np.zeros(size, dtype=np.int32)
    for node in range(size):                    # children are always stored after their parent
        if first[node] >= 0 and visits[node] > 0:
            depth[first[node]:first[node] + count[node]] = depth[node] + 1
    deepest = int(depth[visits[:size] > 0].max())
    node, line = 0, 0
    while first[node] >= 0:
        children = visits[first[node]:first[node] + count[node]]
        if children.max() == 0:
            break
        node = first[node] + int(children.argmax())
        line += 1
    return average, line, deepest


def depth():
    device = NN.choose_device()
    model = NN.load_model(MODEL, device)
    stages = positions(model, device)
    print(f"How far the search of the '{MODEL}' network looks ahead, in moves "
          f'(average over {DEPTH_GAMES} positions per stage or fewer)')
    print('  average = moves ahead per simulation, main line = length of the line it finally prefers, '
          'deepest = longest line it looked at')
    for c_puct in (NORMAL,) + tuple(TRIED):
        print(f"\nc_puct {c_puct}{' (normal)' if c_puct == NORMAL else ''}")
        for simulations in DEPTH_SIMULATIONS:
            parts = []
            for stage, boards in stages.items():
                search = MCTS.BatchedSearch(model, device, len(boards), simulations, c_puct=c_puct)
                for g, board in enumerate(boards):
                    search.set_board(g, board)
                search.search()
                numbers = np.array([tree_depths(search, g) for g in range(len(boards))])
                average, line, deepest = numbers.mean(axis=0)
                parts.append(f'after move {stage}: average {average:.1f}, main line {line:.1f}, '
                             f'deepest {deepest:.1f}')
                del search
            print(f'  {simulations:>4} simulations | ' + ' | '.join(parts), flush=True)


def match():
    device = NN.choose_device()
    model = NN.load_model(MODEL, device)
    test_tournament.OPENINGS = MATCH_OPENINGS
    games = [(opening, side) for opening in test_tournament.openings() for side in (1, -1)]
    print(f"The '{MODEL}' network with another c_puct against itself with the normal {NORMAL}, "
          f'{len(games)} games per line (the same openings, from both sides)')
    for simulations in MATCH_SIMULATIONS:
        for c_puct in TRIED:
            start, wins, ties, losses = time.perf_counter(), 0, 0, 0
            for first in range(0, len(games), PARALLEL_GAMES):
                batch = games[first:first + PARALLEL_GAMES]
                search_a = MCTS.BatchedSearch(model, device, len(batch), simulations, c_puct=c_puct)
                search_b = MCTS.BatchedSearch(model, device, len(batch), simulations, c_puct=NORMAL)
                results = [result for result, _ in test_networks.play_batch(search_a, search_b, batch)]
                wins, ties, losses = (wins + results.count('win'), ties + results.count('tie'),
                                      losses + results.count('loss'))
            share = (wins + 0.5 * ties) / len(games)
            error = np.sqrt((wins + losses) / 4) / len(games)
            print(f'  {simulations} simulations, c_puct {c_puct}: {share:.1%} of the points (+-{error:.1%}), '
                  f'wins-ties-losses {wins}-{ties}-{losses} | {time.perf_counter() - start:.0f}s', flush=True)


if __name__ == '__main__':
    tests = {'depth': depth, 'match': match}
    if len(sys.argv) < 2 or sys.argv[1] not in tests:
        raise SystemExit(__doc__)
    tests[sys.argv[1]]()
