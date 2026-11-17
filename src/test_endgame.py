"""Checks whether the network makes mistakes in the endgame that minimax can prove.

    /usr/bin/python3 src/test_endgame.py        play GAMES games, then print the totals
    /usr/bin/python3 src/test_endgame.py 96     play 96 games instead
    /usr/bin/python3 src/test_endgame.py 0      only print the totals of the saved games

The network plays whole games. From move CHECK_FROM_MOVE on, minimax looks at the
position before every move of the network and says whether the result is proven:
minimax counts a real win as 1 and its own guesses never go above 0.75, so a value
of 1 or more means it has searched to the end of the game and found a forced win.
Two kinds of mistakes are counted:

    missed win      minimax proves a win, but after the move of the network the win
                    is not proven any more
    avoidable loss  nothing was proven, but the move of the network leads to a proven
                    loss (so another move would at least have held out longer)

Both are checked with the same horizon: CHECK_DEPTH before the move, one less after it.
A slower win than the one minimax found can therefore count as a missed win; what
matters in the end is how these games finished, which the totals show as well.

Every checked move is added to  models/results/alphazero_<model>_endgame.csv , so several
runs add up. It does not touch the training log, the network or the snapshots.
Do not run it while a training or another test is running.
"""
import copy
import csv
import os
import signal
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

import fast_minimax
import game

# ------------------------------- Settings -------------------------------
MODEL = 'medium'            # which network: 'small', 'medium' or 'large'
SIMULATIONS = 2000          # search size of the network per move
OPPONENT_DEPTH = None       # None: the network plays against itself (both sides are checked);
                            # a number: it plays against minimax of this depth
GAMES = 64                  # games to play (against minimax: must be even, every opening from both sides)
PARALLEL_GAMES = 32         # how many of them run at the same time
CHECK_DEPTH = 14            # depth of the minimax that looks for proven results
CHECK_FROM_MOVE = 20        # the moves before this one are not checked (nothing can be proven there)
RANDOM_OPENING_MOVES = 2    # random moves at the start, so that the games differ
MINIMAX_PROCESSES = 8
# ------------------------------------------------------------------------

# One line per checked move of the network.
# before / after: what minimax can prove for the player who moves, before and after
#                 the move: 'win', 'loss', 'open' (nothing proven) or 'tie' (game over).
# proof_depth:    the depth at which minimax found the proof before the move.
# network_value:  what the network itself thought of the position (-1 lost ... 1 won).
# game_result:    how the game ended for the player who made this move.
FIELDS = ['model', 'iteration', 'simulations', 'check_depth', 'opponent', 'game', 'move_number',
          'player', 'before', 'proof_depth', 'network_value', 'after', 'game_result']


def result_file(model):
    import Tic_Tac_Toe_NN as NN
    return NN.data_path('results', f'alphazero_{model}_endgame.csv')


def _warm_up():
    signal.signal(signal.SIGINT, signal.SIG_IGN)            # Ctrl+C is handled by the main process
    fast_minimax.best_move(game.Board(playerturn=1), 2)     # compiles the search in this process


def _minimax_move(job):
    board, depth = job
    return fast_minimax.best_move(board, depth)[1]


def _verdict(value):
    """What a minimax value (seen from the player in question) proves."""
    if value >= fast_minimax.WIN_SCORE:
        return 'win'
    if value <= -fast_minimax.WIN_SCORE:
        return 'loss'
    return 'open'


def _check(job):
    """(before, proof depth, after) for one move of the network; see FIELDS."""
    board, move, depth = job
    player = board.playerturn
    value, _, info = fast_minimax.best_move(board, depth, return_info=True)
    before = _verdict(value * player)       # minimax values are seen from X
    proof_depth = info['depth'] if before != 'open' else ''
    if before == 'loss':
        return before, proof_depth, 'loss'  # every move loses, nothing to get wrong
    board.execute(move)
    if board.game_is_over:
        after = 'win' if game.game_is_win(board.won_games)[0] else 'tie'
    else:
        after = _verdict(fast_minimax.best_move(board, depth - 1)[0] * player)
    return before, proof_depth, after


def random_openings(count):
    """Different random openings (as long as there are enough different ones)."""
    openings = {}
    for _ in range(count * 20):
        board, opening = game.Board(playerturn=1), []
        for _ in range(RANDOM_OPENING_MOVES):
            moves = board.possible_moves()
            opening.append(moves[np.random.randint(len(moves))])
            board.execute(opening[-1])
        openings[tuple((int(big), int(little)) for big, little in opening)] = None
        if len(openings) == count:
            break
    return list(openings)


def play_batch(search, pool, openings, sides):
    """Plays one game per opening, all at the same time. sides[g] is the side of the
    network in game g (1 = X, -1 = O), or 0 if it plays both sides.
    Returns one list per checked move: [game, move number, player, before, proof depth,
    network value, after, game result]."""
    boards = []
    for opening in openings:
        board = game.Board(playerturn=1)
        for move in opening:
            board.execute(move)
        boards.append(board)
    active = np.zeros(search.n_games, dtype=bool)
    checks = []     # (game, move number, player, network value, answer of the pool)
    while not all(board.game_is_over for board in boards):
        active[:] = False
        for g, board in enumerate(boards):
            if not board.game_is_over and sides[g] in (0, board.playerturn):
                active[g] = True
                search.set_board(g, board)
        if active.any():
            counts, root_q = search.search(active)
            for g in np.flatnonzero(active):
                board, move = boards[g], int(np.argmax(counts[g]))
                number = len(board.log) + 1
                if number >= CHECK_FROM_MOVE:   # minimax checks it while the games go on
                    answer = pool.apply_async(_check, ((copy.deepcopy(board), move, CHECK_DEPTH),))
                    checks.append((g, number, board.playerturn, float(root_q[g]), answer))
                board.execute(move)
        waiting = [g for g, board in enumerate(boards)
                   if not board.game_is_over and sides[g] == -board.playerturn]
        for g, move in zip(waiting, pool.map(_minimax_move, [(boards[g], OPPONENT_DEPTH) for g in waiting])):
            boards[g].execute(move)
    winners = [-board.playerturn if game.game_is_win(board.won_games)[0] else 0 for board in boards]
    rows = []
    for g, number, player, value, answer in checks:
        before, proof_depth, after = answer.get()
        result = 'tie' if winners[g] == 0 else 'win' if winners[g] == player else 'loss'
        rows.append([g, number, 'X' if player == 1 else 'O', before, proof_depth, round(value, 3), after, result])
    return rows


def run(games):
    import torch
    import MCTS
    import Tic_Tac_Toe_NN as NN

    against_minimax = OPPONENT_DEPTH is not None
    if against_minimax and (games % 2 or PARALLEL_GAMES % 2):
        raise SystemExit('Against minimax GAMES and PARALLEL_GAMES must be even, '
                         'because every opening is played from both sides.')
    device = NN.choose_device()
    model = NN.load_model(MODEL, device)
    iteration = torch.load(NN.checkpoint_path(MODEL), map_location='cpu').get('iteration', '')
    opponent = f'minimax {OPPONENT_DEPTH}' if against_minimax else 'itself'
    path = result_file(MODEL)
    print(f"'{MODEL}' network after iteration {iteration} with {SIMULATIONS} simulations per move "
          f"against {opponent}, {games} games on {device}; minimax depth {CHECK_DEPTH} checks "
          f"every move of the network from move {CHECK_FROM_MOVE} on", flush=True)
    search = MCTS.BatchedSearch(model, device, PARALLEL_GAMES, SIMULATIONS)
    run_name = time.strftime('%Y%m%d-%H%M%S')
    start, played = time.perf_counter(), 0
    with Pool(MINIMAX_PROCESSES, initializer=_warm_up) as pool:
        while played < games:
            size = min(PARALLEL_GAMES, games - played)
            if against_minimax:     # every opening twice: the network as X and as O
                openings = [opening for opening in random_openings(size // 2) for _ in range(2)]
                sides = [1, -1] * (size // 2)
            else:
                openings, sides = random_openings(size), [0] * size
            rows = play_batch(search, pool, openings, sides)
            save_moves(path, [[MODEL, iteration, SIMULATIONS, CHECK_DEPTH, opponent,
                               f'{run_name}-{played + row[0]}', *row[1:]] for row in rows])
            played += len(openings)
            print(f'{played} of {games} games after {(time.perf_counter() - start) / 60:.1f} minutes',
                  flush=True)
    print(f'Saved to {path}')


def save_moves(path, rows):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        if new:
            writer.writerow(FIELDS)
        writer.writerows(rows)


# ----------------------------------------------------------------------------
# totals
# ----------------------------------------------------------------------------

def _results(moves):
    """'won 3, tied 1, lost 2': how the games of these moves ended for the player who
    made them (every game and player once)."""
    count = Counter({(move['game'], move['player']): move['game_result'] for move in moves}.values())
    return f"won {count['win']}, tied {count['tie']}, lost {count['loss']}"


def summary(model):
    path = result_file(model)
    if not os.path.exists(path):
        raise SystemExit(f'{path} does not exist yet. Play some games first.')
    with open(path, newline='') as file:
        saved = [row for row in csv.DictReader(file) if row['model'] == model]
    groups = {}
    for row in saved:
        key = (row['iteration'], row['simulations'], row['opponent'], row['check_depth'])
        groups.setdefault(key, []).append(row)
    for (iteration, simulations, opponent, depth), moves in groups.items():
        games = {move['game'] for move in moves}
        proven = [m for m in moves if m['before'] == 'win']
        missed = [m for m in proven if m['after'] != 'win']
        open_moves = [m for m in moves if m['before'] == 'open']
        blunders = [m for m in open_moves if m['after'] == 'loss']
        lost = [m for m in moves if m['before'] == 'loss']
        print(f"\n'{model}' network after iteration {iteration}, {simulations} simulations, against "
              f"{opponent}: {len(games)} games, {len(moves)} moves checked with minimax depth {depth}")
        print(f'  Minimax proved a win before the move:      {len(proven):5d}')
        print(f'      the network kept the win:              {len(proven) - len(missed):5d}')
        print(f'      the network left it (missed win):      {len(missed):5d}'
              f'   in {len({m["game"] for m in missed})} games: {_results(missed)}')
        print(f'  Nothing was proven before the move:        {len(open_moves):5d}')
        print(f'      the network moved into a proven loss:  {len(blunders):5d}'
              f'   in {len({m["game"] for m in blunders})} games: {_results(blunders)}')
        print(f'  Minimax proved a loss before the move:     {len(lost):5d}')
        # a missed win that was not won in the end, or a move into a proven loss that was lost
        costly = ({m['game'] for m in missed if m['game_result'] != 'win'}
                  | {m['game'] for m in blunders if m['game_result'] == 'loss'})
        print(f'  Games where a mistake cost the result:     {len(costly):5d} of {len(games)} '
              f'({len(costly) / len(games):.0%})')
        first = {}      # game -> (move number, depth) of the first proof
        for m in moves:
            if m['before'] != 'open' and (m['game'] not in first or int(m['move_number']) < first[m['game']][0]):
                first[m['game']] = (int(m['move_number']), int(m['proof_depth']))
        if first:
            numbers = sorted(number for number, _ in first.values())
            print(f'  First proof in a game: on average at move {sum(numbers) / len(numbers):.0f} '
                  f'(earliest {numbers[0]}, latest {numbers[-1]}), in {len(first)} of {len(games)} games')
            print('  Depth minimax needed for the proofs: ' + ', '.join(
                f'up to {limit}: {sum(int(m["proof_depth"]) <= limit for m in moves if m["before"] != "open")}'
                for limit in range(8, int(depth) + 1, 2)))
        if proven:
            values = [float(m['network_value']) for m in proven]
            print(f'  What the network thought of the proven wins: {sum(values) / len(values):+.2f} on average '
                  f'(1 = sure win), below +0.5 in {sum(v < 0.5 for v in values)} of {len(values)}')


if __name__ == '__main__':
    games = int(sys.argv[1]) if len(sys.argv) > 1 else GAMES
    if games > 0:
        try:
            run(games)
        except KeyboardInterrupt:
            print('\nStopped. The finished batches are saved.')
    summary(MODEL)
