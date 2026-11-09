"""Fast minimax with alpha-beta pruning for Super TikTakToe.

Same idea and same evaluation as all_AI.minmax / all_AI.get_evalution, but:
  * the position is stored as bitmasks instead of a 9x9 numpy board,
  * the evaluation of every possible little board is precomputed in a table,
  * moves are ordered (best looking first) so alpha-beta cuts off much earlier,
  * the search is compiled to machine code with numba (if it is installed).

Depth has the same meaning as in all_AI.minmax: depth d looks d-1 moves ahead.
"""
import math
import time

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # still works without numba, just a lot slower
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(function):
            return function
        return decorator

INF = 10.0
MAX_DEPTH = 64
# A win is worth more than any evaluation (those stay between -0.75 and 0.75).
# The fewer moves it takes, the more it is worth, so the AI takes the fastest
# win and, when it is losing anyway, the longest defence.
WIN_SCORE = 1.0
WIN_SPEED_BONUS = 0.01  # per move that the win comes earlier

LINES = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8],
                  [0, 3, 6], [1, 4, 7], [2, 5, 8],
                  [0, 4, 8], [2, 4, 6]], dtype=np.int64)


def _build_tables():
    # WIN[mask]: do the cells in this 9-bit mask contain a complete line?
    win = np.zeros(512, dtype=np.bool_)
    for mask in range(512):
        for a, b, c in LINES:
            if (mask >> a) & 1 and (mask >> b) & 1 and (mask >> c) & 1:
                win[mask] = True
    # LITTLE_EVAL[x_mask * 512 + o_mask]: all_AI.little_evaluation of that little board
    little_eval = np.zeros(512 * 512, dtype=np.float64)
    for x_mask in range(512):
        for o_mask in range(512):
            if x_mask & o_mask:
                continue
            cells = [((x_mask >> i) & 1) - ((o_mask >> i) & 1) for i in range(9)]
            state = sum(cells)
            for a, b, c in LINES:
                line = (cells[a], cells[b], cells[c])
                if not (1 in line and -1 in line):
                    state += sum(line) ** 3
            little_eval[x_mask * 512 + o_mask] = 0.75 * math.tanh(state / 10)
    return win, little_eval


WIN, LITTLE_EVAL = _build_tables()


@njit
def _evaluate(val):
    # all_AI.get_evalution: val[i] is 1/-1 for a won little board, 2 for a tied
    # one and otherwise the evaluation of the little board.
    state = 0.0
    for i in range(9):
        if val[i] != 2.0:
            state += val[i]
    for line in range(8):
        a = val[LINES[line, 0]]
        b = val[LINES[line, 1]]
        c = val[LINES[line, 2]]
        if a == 2.0 or b == 2.0 or c == 2.0:
            continue
        has_x = a == 1.0 or b == 1.0 or c == 1.0
        has_o = a == -1.0 or b == -1.0 or c == -1.0
        if has_x and has_o:
            continue
        total = a + b + c
        state += total * total * total
    return 0.75 * math.tanh(state / 10.0)


@njit
def _make(x_masks, o_masks, won, val, big_x, big_o, resolved, board, cell, player):
    """Play a move. Returns (big_x, big_o, resolved, next_board, status);
    status is 0 = game goes on, 1 = player won the game, 2 = game is a tie."""
    status = 0
    if player == 1:
        x_masks[board] |= 1 << cell
        own = x_masks[board]
    else:
        o_masks[board] |= 1 << cell
        own = o_masks[board]
    if WIN[own]:
        won[board] = player
        val[board] = player
        resolved |= 1 << board
        if player == 1:
            big_x |= 1 << board
            if WIN[big_x]:
                status = 1
        else:
            big_o |= 1 << board
            if WIN[big_o]:
                status = 1
        if status == 0 and resolved == 511:
            status = 2
    elif (x_masks[board] | o_masks[board]) == 511:
        won[board] = 2
        val[board] = 2.0
        resolved |= 1 << board
        if resolved == 511:
            status = 2
    else:
        val[board] = LITTLE_EVAL[x_masks[board] * 512 + o_masks[board]]
    next_board = cell if won[cell] == 0 else -1
    return big_x, big_o, resolved, next_board, status


@njit
def _unmake(x_masks, o_masks, won, val, board, cell, player, old_val):
    if player == 1:
        x_masks[board] &= ~(1 << cell)
    else:
        o_masks[board] &= ~(1 << cell)
    won[board] = 0
    val[board] = old_val


@njit
def _generate(x_masks, o_masks, won, next_board, moves, ply):
    count = 0
    if next_board >= 0:
        occupied = x_masks[next_board] | o_masks[next_board]
        for cell in range(9):
            if not (occupied >> cell) & 1:
                moves[ply, count] = next_board * 9 + cell
                count += 1
    else:
        for board in range(9):
            if won[board] == 0:
                occupied = x_masks[board] | o_masks[board]
                for cell in range(9):
                    if not (occupied >> cell) & 1:
                        moves[ply, count] = board * 9 + cell
                        count += 1
    return count


@njit
def _search(x_masks, o_masks, won, val, big_x, big_o, resolved, next_board,
            player, depth, alpha, beta, moves, keys, ply, nodes):
    # Only called for positions that are not finished and with depth >= 2.
    nodes[0] += 1
    count = _generate(x_masks, o_masks, won, next_board, moves, ply)

    if depth >= 3:
        # Move ordering: try the moves that look best one move ahead first.
        for i in range(count):
            board = moves[ply, i] // 9
            cell = moves[ply, i] % 9
            old_val = val[board]
            _, _, _, child_next, status = _make(x_masks, o_masks, won, val, big_x, big_o,
                                                resolved, board, cell, player)
            if status == 1:
                key = INF
            elif status == 2:
                key = 0.0
            else:
                key = player * _evaluate(val)
                if child_next < 0:
                    key -= 0.15  # letting the opponent play anywhere is usually bad
            _unmake(x_masks, o_masks, won, val, board, cell, player, old_val)
            keys[ply, i] = key
        for i in range(1, count):  # insertion sort, best key first
            key = keys[ply, i]
            move = moves[ply, i]
            j = i - 1
            while j >= 0 and keys[ply, j] < key:
                keys[ply, j + 1] = keys[ply, j]
                moves[ply, j + 1] = moves[ply, j]
                j -= 1
            keys[ply, j + 1] = key
            moves[ply, j + 1] = move

    value = -INF * player
    for i in range(count):
        board = moves[ply, i] // 9
        cell = moves[ply, i] % 9
        old_val = val[board]
        child_big_x, child_big_o, child_resolved, child_next, status = _make(
            x_masks, o_masks, won, val, big_x, big_o, resolved, board, cell, player)
        if status == 1:
            score = player * (WIN_SCORE + WIN_SPEED_BONUS * (MAX_DEPTH - ply))
        elif status == 2:
            score = 0.0
        elif depth <= 2:
            score = _evaluate(val)
        else:
            score = _search(x_masks, o_masks, won, val, child_big_x, child_big_o,
                            child_resolved, child_next, -player, depth - 1,
                            alpha, beta, moves, keys, ply + 1, nodes)
        _unmake(x_masks, o_masks, won, val, board, cell, player, old_val)

        if player == 1:
            if score > value:
                value = score
            if value > alpha:
                alpha = value
        else:
            if score < value:
                value = score
            if value < beta:
                beta = value
        if alpha >= beta:
            break
    return value


@njit
def _search_root(x_masks, o_masks, won, val, big_x, big_o, resolved, player,
                 depth, root_moves, scores, moves, keys, nodes):
    """Searches every root move in the given order. Returns (value, index of the
    first move that reaches it). scores[i] is exact for the best move and a
    bound for the others."""
    alpha = -INF
    beta = INF
    value = -INF * player
    best = -1
    for i in range(len(root_moves)):
        board = root_moves[i] // 9
        cell = root_moves[i] % 9
        old_val = val[board]
        child_big_x, child_big_o, child_resolved, child_next, status = _make(
            x_masks, o_masks, won, val, big_x, big_o, resolved, board, cell, player)
        if status == 1:
            score = player * (WIN_SCORE + WIN_SPEED_BONUS * MAX_DEPTH)
        elif status == 2:
            score = 0.0
        elif depth <= 2:
            score = _evaluate(val)
        else:
            score = _search(x_masks, o_masks, won, val, child_big_x, child_big_o,
                            child_resolved, child_next, -player, depth - 1,
                            alpha, beta, moves, keys, 1, nodes)
        _unmake(x_masks, o_masks, won, val, board, cell, player, old_val)
        scores[i] = score
        if player == 1:
            if score > value:
                value = score
                best = i
            if value > alpha:
                alpha = value
        else:
            if score < value:
                value = score
                best = i
            if value < beta:
                beta = value
    return value, best


def _encode(board):
    """game.Board -> bitmask arrays used by the search."""
    x_masks = np.zeros(9, dtype=np.int64)
    o_masks = np.zeros(9, dtype=np.int64)
    won = np.zeros(9, dtype=np.int64)
    val = np.zeros(9, dtype=np.float64)
    big_x = big_o = resolved = 0
    for i in range(9):
        for j in range(9):
            if board.board[i][j] == 1:
                x_masks[i] |= 1 << j
            elif board.board[i][j] == -1:
                o_masks[i] |= 1 << j
        won[i] = board.won_games[i]
        if won[i] == 0:
            val[i] = LITTLE_EVAL[x_masks[i] * 512 + o_masks[i]]
        else:
            val[i] = won[i]
            resolved |= 1 << i
            if won[i] == 1:
                big_x |= 1 << i
            elif won[i] == -1:
                big_o |= 1 << i
    return x_masks, o_masks, won, val, big_x, big_o, resolved


def evaluate(board):
    """Static evaluation of a game.Board (same value as all_AI.get_evalution)."""
    return float(_evaluate(_encode(board)[3]))


def best_move(board, depth=8, time_limit=None, return_info=False):
    """Best move for the player to move on a game.Board.

    depth:      like all_AI.minmax, depth d looks d-1 moves ahead (minimum 2).
    time_limit: optional, rough time budget in seconds. The search deepens step
                by step up to `depth` and stops early when the next step would
                probably not fit into the budget any more.
    Returns (value, (big_field, little_field)); with return_info=True also a
    dict with the reached depth, the number of searched positions and the time.
    """
    if board.game_is_over:
        raise ValueError('the game is already over')
    depth = max(2, min(int(depth), MAX_DEPTH))
    player = int(board.playerturn)
    x_masks, o_masks, won, val, big_x, big_o, resolved = _encode(board)

    # Random start order: among equally good moves the choice stays random.
    root_moves = np.array([int(b) * 9 + int(c) for b, c in board.possible_moves()], dtype=np.int64)
    root_moves = np.random.permutation(root_moves)
    scores = np.zeros(len(root_moves), dtype=np.float64)
    moves = np.zeros((MAX_DEPTH + 1, 81), dtype=np.int64)
    keys = np.zeros((MAX_DEPTH + 1, 81), dtype=np.float64)
    nodes = np.zeros(1, dtype=np.int64)

    start = time.perf_counter()
    value, best, reached = 0.0, int(root_moves[0]), 0
    # Iterative deepening: the result of each depth sorts the moves for the next.
    for current_depth in range(2, depth + 1):
        value, best_index = _search_root(x_masks, o_masks, won, val, big_x, big_o, resolved,
                                         player, current_depth, root_moves, scores, moves, keys, nodes)
        best = int(root_moves[best_index])
        reached = current_depth
        order = np.argsort(-player * scores, kind='stable')
        order = np.concatenate(([best_index], order[order != best_index]))
        root_moves = root_moves[order]
        if abs(value) >= WIN_SCORE:
            break  # forced win or loss found, searching deeper cannot change it
        # One step deeper takes roughly 4 times as long as everything so far.
        if time_limit is not None and (time.perf_counter() - start) * 4 >= time_limit:
            break

    move = (best // 9, best % 9)
    if return_info:
        info = {'depth': reached, 'nodes': int(nodes[0]), 'time': time.perf_counter() - start}
        return float(value), move, info
    return float(value), move
