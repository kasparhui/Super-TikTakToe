"""Lets two AIs play against each other and prints the result.

    python3 src/test_models.py

Use it to see how strong a trained network is, for example against minimax
with different depths or against the classic MCTS.
"""
import numpy as np

import game
from AI import create_ai, ai_move

# ------------------------------- Settings -------------------------------
# Same format as PLAYERS in main.py (without 'human').
PLAYER_A = {'type': 'neural_net', 'model': 'small', 'simulations': 400}
PLAYER_B = {'type': 'minimax', 'depth': 8}
GAMES = 20                  # must be even: the games come in pairs (see below)
RANDOM_OPENING_MOVES = 2    # random moves at the start, so that the games differ
# Every random opening is played twice, once with A as X and once with A as O.
# So an opening that happens to be good for X or for O helps both players equally.
# ------------------------------------------------------------------------


def play_match(player_a, player_b, games=GAMES, random_opening_moves=RANDOM_OPENING_MOVES, verbose=True):
    """Returns (wins of A, ties, wins of B)."""
    ai_a = create_ai('X', player_a)
    ai_b = create_ai('O', player_b)
    if games % 2:
        raise ValueError('GAMES must be even, because every opening is played from both sides.')
    wins_a = ties = wins_b = 0
    for number in range(games):
        board = game.Board(playerturn=1)
        side_a = 1 if number % 2 == 0 else -1       # A plays X in every second game
        if number % 2 == 0:                         # a new opening for this pair of games
            opening = []
            for _ in range(random_opening_moves):
                moves = board.possible_moves()
                opening.append(moves[np.random.randint(len(moves))])
                board.execute(opening[-1])
        else:                                       # the same opening, sides swapped
            for move in opening:
                board.execute(move)
        while not board.game_is_over:
            if board.playerturn == side_a:
                board.execute(ai_move(ai_a, player_a, board))
            else:
                board.execute(ai_move(ai_b, player_b, board))
        if not game.game_is_win(board.won_games)[0]:
            ties += 1
            result = 'tie'
        elif -board.playerturn == side_a:           # the player who moved last has won
            wins_a += 1
            result = 'A wins'
        else:
            wins_b += 1
            result = 'B wins'
        if verbose:
            print(f"Game {number + 1} (opening {number // 2 + 1}, A plays "
                  f"{'X' if side_a == 1 else 'O'}): {result}", flush=True)
    return wins_a, ties, wins_b


if __name__ == '__main__':
    print(f'A: {PLAYER_A}\nB: {PLAYER_B}')
    wins_a, ties, wins_b = play_match(PLAYER_A, PLAYER_B)
    print(f'\nA won {wins_a}, tie {ties}, B won {wins_b}  ->  '
          f'A scored {(wins_a + ties / 2) / GAMES:.0%} of the points')
