"""Monte Carlo tree search for Super TikTakToe.

Two searches, both compiled with numba and both working on the bitmask
position of fast_minimax.py instead of game.Board:

  pure_mcts_move   classic MCTS: random games to the end decide how good a move
                   is. Needs no neural network.   ('mcts' player type)
  BatchedSearch    AlphaZero MCTS: the neural network proposes moves (policy)
                   and judges positions (value). It searches many games at the
                   same time, so the network is always asked about a whole
                   batch of positions - that is what makes self-play fast.
  NeuralPlayer     BatchedSearch for a single game.Board  ('neural_net' player type)

A move is the number  big_field * 9 + little_field.
"""
import math

import numpy as np

from fast_minimax import _encode, _make, njit

try:
    from numba import get_num_threads, prange, set_num_threads
except ImportError:     # without numba the games are searched one after the other
    prange = range
    get_num_threads = set_num_threads = None

# How many processor cores the search trees use: 'all', or a number (1 = one game after the other, as
# before 17 November). The trees of the games are independent, so they can be worked on at the same
# time; the answers are exactly the same (with noise only the random numbers differ).
SEARCH_THREADS = 'all'
from Tic_Tac_Toe_NN import INPUT_PLANES, MOVE_TO_SPATIAL, predict

MAX_GAME_LENGTH = 81


# ----------------------------------------------------------------------------
# helpers on the bitmask position
# ----------------------------------------------------------------------------

@njit
def _legal_moves(x_masks, o_masks, won, next_board, out):
    """Writes the legal moves into out and returns how many there are."""
    count = 0
    if next_board >= 0:
        occupied = x_masks[next_board] | o_masks[next_board]
        for cell in range(9):
            if not (occupied >> cell) & 1:
                out[count] = next_board * 9 + cell
                count += 1
    else:
        for board in range(9):
            if won[board] == 0:
                occupied = x_masks[board] | o_masks[board]
                for cell in range(9):
                    if not (occupied >> cell) & 1:
                        out[count] = board * 9 + cell
                        count += 1
    return count


@njit
def _encode_planes(x_masks, o_masks, won, next_board, player, planes):
    """Network input (see Tic_Tac_Toe_NN.py) for the player who is to move."""
    planes[:] = 0.0
    for board in range(9):
        if player == 1:
            mine, theirs = x_masks[board], o_masks[board]
        else:
            mine, theirs = o_masks[board], x_masks[board]
        winner = won[board]
        playable = winner == 0 and (next_board < 0 or next_board == board)
        for cell in range(9):
            row = 3 * (board // 3) + cell // 3
            column = 3 * (board % 3) + cell % 3
            if (mine >> cell) & 1:
                planes[0, row, column] = 1.0
            elif (theirs >> cell) & 1:
                planes[1, row, column] = 1.0
            elif playable:
                planes[2, row, column] = 1.0
            if winner == player:
                planes[3, row, column] = 1.0
            elif winner == -player:
                planes[4, row, column] = 1.0
            elif winner == 2:
                planes[5, row, column] = 1.0


def encode_board(board):
    """Network input (6 x 9 x 9) for a game.Board."""
    x_masks, o_masks, won, val, big_x, big_o, resolved = _encode(board)
    planes = np.zeros((INPUT_PLANES, 9, 9), dtype=np.float32)
    next_board = -1 if board.next_move is None else int(board.next_move)
    _encode_planes(x_masks, o_masks, won, next_board, int(board.playerturn), planes)
    return planes


# ----------------------------------------------------------------------------
# classic MCTS with random playouts
# ----------------------------------------------------------------------------

@njit
def _pure_mcts(x_masks, o_masks, won, val, big_x, big_o, resolved, next_board, player,
               iterations, exploration, counts):
    """Fills counts[move] with the number of visits of every root move and
    returns the estimated chance to win (0..1) of the player to move."""
    capacity = iterations + 1
    visits = np.zeros(capacity, dtype=np.int64)
    wins = np.zeros(capacity, dtype=np.float64)     # for the player who made the move into the node
    node_move = np.zeros(capacity, dtype=np.int64)
    first_child = np.full(capacity, -1, dtype=np.int64)
    next_sibling = np.full(capacity, -1, dtype=np.int64)
    children = np.zeros(capacity, dtype=np.int64)
    offset = np.zeros(capacity, dtype=np.int64)     # random order in which a node tries its moves
    size = 1

    sx = np.zeros(9, dtype=np.int64)
    so = np.zeros(9, dtype=np.int64)
    swon = np.zeros(9, dtype=np.int64)
    sval = np.zeros(9, dtype=np.float64)
    legal = np.zeros(81, dtype=np.int64)
    path = np.zeros(MAX_GAME_LENGTH + 1, dtype=np.int64)
    offset[0] = np.random.randint(0, 81)

    for _ in range(iterations):
        sx[:] = x_masks
        so[:] = o_masks
        swon[:] = won
        sval[:] = val
        bx, bo, res, nb, p = big_x, big_o, resolved, next_board, player
        node = 0
        depth = 0
        status = 0
        winner = 0
        # selection + expansion
        while True:
            count = _legal_moves(sx, so, swon, nb, legal)
            if children[node] < count:
                # try a move of this node that has no child yet
                move = legal[(offset[node] + children[node]) % count]
                child = size
                size += 1
                node_move[child] = move
                offset[child] = np.random.randint(0, 81)
                next_sibling[child] = first_child[node]
                first_child[node] = child
                children[node] += 1
                expanded = True
            else:
                # all moves tried: take the most promising child (UCT)
                log_parent = math.log(visits[node])
                best_score = -1.0
                child = first_child[node]
                candidate = child
                while candidate >= 0:
                    score = (wins[candidate] / visits[candidate]
                             + exploration * math.sqrt(log_parent / visits[candidate]))
                    if score > best_score:
                        best_score = score
                        child = candidate
                    candidate = next_sibling[candidate]
                move = node_move[child]
                expanded = False
            bx, bo, res, nb, status = _make(sx, so, swon, sval, bx, bo, res, move // 9, move % 9, p)
            if status == 1:
                winner = p
            p = -p
            node = child
            depth += 1
            path[depth] = node
            if status != 0 or expanded:
                break
        # random playout
        while status == 0:
            count = _legal_moves(sx, so, swon, nb, legal)
            move = legal[np.random.randint(0, count)]
            bx, bo, res, nb, status = _make(sx, so, swon, sval, bx, bo, res, move // 9, move % 9, p)
            if status == 1:
                winner = p
            p = -p
        # backup
        visits[0] += 1
        mover = player
        for i in range(1, depth + 1):
            visits[path[i]] += 1
            if winner == 0:
                wins[path[i]] += 0.5
            elif winner == mover:
                wins[path[i]] += 1.0
            mover = -mover

    counts[:] = 0
    best_visits = -1
    win_rate = 0.5
    child = first_child[0]
    while child >= 0:
        counts[node_move[child]] = visits[child]
        if visits[child] > best_visits:
            best_visits = visits[child]
            win_rate = wins[child] / visits[child]
        child = next_sibling[child]
    return win_rate


def pure_mcts_move(board, iterations=50000, exploration=1.0):
    """Best move for a game.Board by classic MCTS. Returns ((big, little), win rate)."""
    x_masks, o_masks, won, val, big_x, big_o, resolved = _encode(board)
    next_board = -1 if board.next_move is None else int(board.next_move)
    counts = np.zeros(81, dtype=np.int64)
    iterations = max(int(iterations), 2 * int(len(board.possible_moves())))
    win_rate = _pure_mcts(x_masks, o_masks, won, val, big_x, big_o, resolved, next_board,
                          int(board.playerturn), iterations, exploration, counts)
    move = int(np.argmax(counts))
    return (move // 9, move % 9), float(win_rate)


# ----------------------------------------------------------------------------
# AlphaZero MCTS for many games at once
# ----------------------------------------------------------------------------
# Every game has its own tree, stored in rows of 2-D arrays. Node 0 is the root.
#   t_visits   how often the node was visited
#   t_value    sum of the results, seen from the player who made the move into the node
#   t_prior    how much the network likes that move
#   t_first / t_count   where the children of the node are stored

@njit
def _backup(t_visits, t_value, g, path, depth, value):
    # value: result for the player who made the move into path[g, depth]
    for i in range(depth, -1, -1):
        node = path[g, i]
        t_visits[g, node] += 1
        t_value[g, node] += value
        value = -value


@njit
def _select_one(g, r_x, r_o, r_won, r_val, r_meta, r_player,
                t_visits, t_value, t_prior, t_move, t_first, t_count,
                sims_done, simulations, pending, path, path_len,
                leaf_moves, leaf_count, planes, c_puct, fpu, sx, so, swon, sval, legal):
    """Game g: walks down its tree to a position the network has not seen yet and prepares it as
    network input. Finished games reached on the way are scored directly. sx ... legal: scratch space."""
    pending[g] = False
    while sims_done[g] < simulations:
        sx[:] = r_x[g]
        so[:] = r_o[g]
        swon[:] = r_won[g]
        sval[:] = r_val[g]
        bx, bo, res, nb = r_meta[g, 0], r_meta[g, 1], r_meta[g, 2], r_meta[g, 3]
        p = r_player[g]
        node = 0
        depth = 0
        status = 0
        path[g, 0] = 0
        while t_first[g, node] >= 0:
            first = t_first[g, node]
            parent_visits = t_visits[g, node]
            sqrt_parent = math.sqrt(max(1, parent_visits))
            # value of this position for the player to move, used for untried moves
            parent_q = 0.0
            if parent_visits > 0:
                parent_q = -t_value[g, node] / parent_visits
            best = first
            best_score = -1e30
            for child in range(first, first + t_count[g, node]):
                n = t_visits[g, child]
                if n > 0:
                    q = t_value[g, child] / n
                else:
                    q = parent_q - fpu
                score = q + c_puct * t_prior[g, child] * sqrt_parent / (1 + n)
                if score > best_score:
                    best_score = score
                    best = child
            move = int(t_move[g, best])
            bx, bo, res, nb, status = _make(sx, so, swon, sval, bx, bo, res, move // 9, move % 9, p)
            p = -p
            node = best
            depth += 1
            path[g, depth] = node
            if status != 0:
                break
        path_len[g] = depth + 1
        if status != 0:
            # the game is over here: a win for the player who just moved, or a tie
            _backup(t_visits, t_value, g, path, depth, 1.0 if status == 1 else 0.0)
            sims_done[g] += 1
            continue
        count = _legal_moves(sx, so, swon, nb, legal)
        leaf_count[g] = count
        for k in range(count):
            leaf_moves[g, k] = legal[k]
        _encode_planes(sx, so, swon, nb, p, planes[g])
        pending[g] = True
        break


@njit
def _select_batch(r_x, r_o, r_won, r_val, r_meta, r_player,
                  t_visits, t_value, t_prior, t_move, t_first, t_count,
                  sims_done, simulations, pending, path, path_len,
                  leaf_moves, leaf_count, planes, c_puct, fpu):
    """For every game: walk down the tree to a position the network has not
    seen yet and prepare it as network input. Finished games reached on the way
    are scored directly."""
    sx = np.zeros(9, dtype=np.int64)
    so = np.zeros(9, dtype=np.int64)
    swon = np.zeros(9, dtype=np.int64)
    sval = np.zeros(9, dtype=np.float64)
    legal = np.zeros(81, dtype=np.int64)
    for g in range(len(sims_done)):
        _select_one(g, r_x, r_o, r_won, r_val, r_meta, r_player,
                    t_visits, t_value, t_prior, t_move, t_first, t_count,
                    sims_done, simulations, pending, path, path_len,
                    leaf_moves, leaf_count, planes, c_puct, fpu, sx, so, swon, sval, legal)


@njit(parallel=True)
def _select_batch_parallel(r_x, r_o, r_won, r_val, r_meta, r_player,
                           t_visits, t_value, t_prior, t_move, t_first, t_count,
                           sims_done, simulations, pending, path, path_len,
                           leaf_moves, leaf_count, planes, c_puct, fpu, masks, values, legal):
    """_select_batch with the games spread over the processor cores. Every game gets its own rows of
    scratch space (masks: n_games x 3 x 9, values: n_games x 9, legal: n_games x 81)."""
    for g in prange(len(sims_done)):
        _select_one(g, r_x, r_o, r_won, r_val, r_meta, r_player,
                    t_visits, t_value, t_prior, t_move, t_first, t_count,
                    sims_done, simulations, pending, path, path_len,
                    leaf_moves, leaf_count, planes, c_puct, fpu,
                    masks[g, 0], masks[g, 1], masks[g, 2], values[g], legal[g])


@njit
def _expand_one(g, logits, values, pending, t_visits, t_value, t_prior, t_move, t_first, t_count,
                t_size, sims_done, path, path_len, leaf_moves, leaf_count, noise_weight, noise_alpha):
    """Game g: creates the children of its new position (with the policy as prior) and passes the
    value up the tree."""
    capacity = t_visits.shape[1]
    if not pending[g]:
        return
    depth = path_len[g] - 1
    leaf = path[g, depth]
    count = leaf_count[g]
    start = t_size[g]
    if start + count <= capacity:
        # softmax over the legal moves only
        top = -1e30
        for k in range(count):
            logit = logits[g, MOVE_TO_SPATIAL[leaf_moves[g, k]]]
            if logit > top:
                top = logit
        total = 0.0
        for k in range(count):
            child = start + k
            prior = math.exp(logits[g, MOVE_TO_SPATIAL[leaf_moves[g, k]]] - top)
            total += prior
            t_prior[g, child] = prior
            t_move[g, child] = leaf_moves[g, k]
            t_visits[g, child] = 0
            t_value[g, child] = 0.0
            t_first[g, child] = -1
            t_count[g, child] = 0
        for k in range(count):
            t_prior[g, start + k] /= total
        if leaf == 0 and noise_weight > 0.0:
            # random noise on the root moves, so that self-play tries new things
            noise_total = 0.0
            for k in range(count):
                noise = np.random.gamma(noise_alpha, 1.0) + 1e-12
                noise_total += noise
                t_value[g, start + k] = noise      # borrowed as temporary storage
            for k in range(count):
                child = start + k
                t_prior[g, child] = ((1.0 - noise_weight) * t_prior[g, child]
                                     + noise_weight * t_value[g, child] / noise_total)
                t_value[g, child] = 0.0
        t_first[g, leaf] = start
        t_count[g, leaf] = count
        t_size[g] = start + count
    # the network judges for the player to move; the tree wants it for the player who moved before
    _backup(t_visits, t_value, g, path, depth, -values[g])
    sims_done[g] += 1
    pending[g] = False


@njit
def _expand_batch(logits, values, pending, t_visits, t_value, t_prior, t_move, t_first, t_count,
                  t_size, sims_done, path, path_len, leaf_moves, leaf_count,
                  noise_weight, noise_alpha):
    """Uses the answers of the network: creates the children of every new
    position (with the policy as prior) and passes the value up the tree."""
    for g in range(len(pending)):
        _expand_one(g, logits, values, pending, t_visits, t_value, t_prior, t_move, t_first, t_count,
                    t_size, sims_done, path, path_len, leaf_moves, leaf_count, noise_weight, noise_alpha)


@njit(parallel=True)
def _expand_batch_parallel(logits, values, pending, t_visits, t_value, t_prior, t_move, t_first, t_count,
                           t_size, sims_done, path, path_len, leaf_moves, leaf_count,
                           noise_weight, noise_alpha):
    """_expand_batch with the games spread over the processor cores."""
    for g in prange(len(pending)):
        _expand_one(g, logits, values, pending, t_visits, t_value, t_prior, t_move, t_first, t_count,
                    t_size, sims_done, path, path_len, leaf_moves, leaf_count, noise_weight, noise_alpha)


@njit
def _root_results(t_visits, t_value, t_move, t_first, t_count, active, counts, root_q):
    for g in range(len(active)):
        counts[g, :] = 0.0
        root_q[g] = 0.0
        if not active[g] or t_first[g, 0] < 0:
            continue
        first = t_first[g, 0]
        for child in range(first, first + t_count[g, 0]):
            counts[g, t_move[g, child]] = t_visits[g, child]
        root_q[g] = -t_value[g, 0] / max(1, t_visits[g, 0])


@njit
def _root_move_values(t_visits, t_value, t_move, t_first, t_count, active, values):
    """For every move tried at the root: how good the search found it for the player to move."""
    for g in range(len(active)):
        values[g, :] = 0.0
        if not active[g] or t_first[g, 0] < 0:
            continue
        first = t_first[g, 0]
        for child in range(first, first + t_count[g, 0]):
            if t_visits[g, child] > 0:
                values[g, t_move[g, child]] = t_value[g, child] / t_visits[g, child]


@njit
def _choose_moves(counts, active, move_number, temperature_moves, chosen):
    """Early in a game moves are drawn in proportion to their visit counts
    (variety), later the most visited move is played."""
    for g in range(len(active)):
        if not active[g]:
            continue
        if move_number[g] < temperature_moves:
            total = 0.0
            for m in range(81):
                total += counts[g, m]
            threshold = np.random.random() * total
            running = 0.0
            pick = -1
            for m in range(81):
                if counts[g, m] > 0:
                    pick = m
                    running += counts[g, m]
                    if running >= threshold:
                        break
            chosen[g] = pick
        else:
            best = 0
            for m in range(81):
                if counts[g, m] > counts[g, best]:
                    best = m
            chosen[g] = best


@njit
def _play_moves(r_x, r_o, r_won, r_val, r_meta, r_player, active, chosen, status):
    """Plays the chosen move in every active game. status: 0 = goes on,
    1 = the player who moved has won, 2 = tie."""
    for g in range(len(active)):
        status[g] = 0
        if not active[g]:
            continue
        move = chosen[g]
        big_x, big_o, resolved, next_board, result = _make(
            r_x[g], r_o[g], r_won[g], r_val[g], r_meta[g, 0], r_meta[g, 1], r_meta[g, 2],
            move // 9, move % 9, r_player[g])
        r_meta[g, 0] = big_x
        r_meta[g, 1] = big_o
        r_meta[g, 2] = resolved
        r_meta[g, 3] = next_board
        r_player[g] = -r_player[g]
        status[g] = result


@njit
def _encode_roots(r_x, r_o, r_won, r_meta, r_player, planes):
    for g in range(len(r_player)):
        _encode_planes(r_x[g], r_o[g], r_won[g], r_meta[g, 3], r_player[g], planes[g])


_threads_set = [None]


def search_threads():
    """How many processor cores the search uses now (see SEARCH_THREADS)."""
    if set_num_threads is None:
        return 1
    import numba
    wanted = numba.config.NUMBA_NUM_THREADS if SEARCH_THREADS == 'all' else max(1, min(int(SEARCH_THREADS), numba.config.NUMBA_NUM_THREADS))
    if _threads_set[0] != wanted:
        set_num_threads(wanted)
        _threads_set[0] = wanted
    return wanted


class BatchedSearch:
    """AlphaZero MCTS for n_games positions at the same time."""

    def __init__(self, model, device, n_games, simulations, c_puct=1.5, fpu=0.25,
                 noise_weight=0.0, noise_alpha=0.6):
        self.model = model
        self.device = device
        self.n_games = n_games
        self.simulations = simulations
        self.c_puct = c_puct
        self.fpu = fpu
        self.noise_weight = noise_weight
        self.noise_alpha = noise_alpha

        n = n_games
        # the positions that are searched (the roots of the trees)
        self.r_x = np.zeros((n, 9), dtype=np.int64)
        self.r_o = np.zeros((n, 9), dtype=np.int64)
        self.r_won = np.zeros((n, 9), dtype=np.int64)
        self.r_val = np.zeros((n, 9), dtype=np.float64)
        self.r_meta = np.zeros((n, 4), dtype=np.int64)      # big_x, big_o, resolved, next_board
        self.r_player = np.ones(n, dtype=np.int64)
        for g in range(n):
            self.reset(g)

        # the trees: every simulation adds at most 81 nodes
        capacity = simulations * 81 + 1
        self.t_visits = np.zeros((n, capacity), dtype=np.int32)
        self.t_value = np.zeros((n, capacity), dtype=np.float32)
        self.t_prior = np.zeros((n, capacity), dtype=np.float32)
        self.t_move = np.zeros((n, capacity), dtype=np.int16)
        self.t_first = np.full((n, capacity), -1, dtype=np.int32)
        self.t_count = np.zeros((n, capacity), dtype=np.int16)
        self.t_size = np.ones(n, dtype=np.int64)

        self.sims_done = np.zeros(n, dtype=np.int64)
        self.pending = np.zeros(n, dtype=np.bool_)
        self.path = np.zeros((n, MAX_GAME_LENGTH + 1), dtype=np.int32)
        self.path_len = np.zeros(n, dtype=np.int64)
        self.leaf_moves = np.zeros((n, 81), dtype=np.int16)
        self.leaf_count = np.zeros(n, dtype=np.int64)
        self.planes = np.zeros((n, INPUT_PLANES, 9, 9), dtype=np.float32)
        self.logits = np.zeros((n, 81), dtype=np.float32)
        self.values = np.zeros(n, dtype=np.float32)
        self.counts = np.zeros((n, 81), dtype=np.float32)
        self.root_q = np.zeros(n, dtype=np.float32)
        # scratch space of every game, for the search on several processor cores
        self.s_masks = np.zeros((n, 3, 9), dtype=np.int64)
        self.s_values = np.zeros((n, 9), dtype=np.float64)
        self.s_legal = np.zeros((n, 81), dtype=np.int64)

    def reset(self, g):
        """Game g starts again from the empty board, X to move."""
        self.r_x[g] = 0
        self.r_o[g] = 0
        self.r_won[g] = 0
        self.r_val[g] = 0.0
        self.r_meta[g] = (0, 0, 0, -1)
        self.r_player[g] = 1

    def set_board(self, g, board):
        """Game g continues from this game.Board."""
        x_masks, o_masks, won, val, big_x, big_o, resolved = _encode(board)
        self.r_x[g] = x_masks
        self.r_o[g] = o_masks
        self.r_won[g] = won
        self.r_val[g] = val
        self.r_meta[g] = (big_x, big_o, resolved, -1 if board.next_move is None else int(board.next_move))
        self.r_player[g] = int(board.playerturn)

    def encode_roots(self):
        """Network input of all current positions (n_games x 6 x 9 x 9)."""
        planes = np.zeros((self.n_games, INPUT_PLANES, 9, 9), dtype=np.float32)
        _encode_roots(self.r_x, self.r_o, self.r_won, self.r_meta, self.r_player, planes)
        return planes

    def search(self, active=None):
        """Searches all active games (which must not be over).
        Returns (counts, root_q): counts[g, move] = visits of every move,
        root_q[g] = how good the position looks for the player to move (-1..1)."""
        if active is None:
            active = np.ones(self.n_games, dtype=np.bool_)
        self.t_size[:] = 1
        self.t_first[:, 0] = -1
        self.t_visits[:, 0] = 0
        self.t_value[:, 0] = 0.0
        self.sims_done[:] = np.where(active, 0, self.simulations)
        threads = search_threads()
        many = threads > 1 and self.n_games > 1
        while True:
            if many:
                _select_batch_parallel(
                    self.r_x, self.r_o, self.r_won, self.r_val, self.r_meta, self.r_player,
                    self.t_visits, self.t_value, self.t_prior, self.t_move, self.t_first, self.t_count,
                    self.sims_done, self.simulations, self.pending, self.path, self.path_len,
                    self.leaf_moves, self.leaf_count, self.planes, self.c_puct, self.fpu,
                    self.s_masks, self.s_values, self.s_legal)
            else:
                _select_batch(self.r_x, self.r_o, self.r_won, self.r_val, self.r_meta, self.r_player,
                              self.t_visits, self.t_value, self.t_prior, self.t_move, self.t_first, self.t_count,
                              self.sims_done, self.simulations, self.pending, self.path, self.path_len,
                              self.leaf_moves, self.leaf_count, self.planes, self.c_puct, self.fpu)
            waiting = np.flatnonzero(self.pending)
            if len(waiting) == 0:
                break
            if len(waiting) == self.n_games:
                self.logits, self.values = predict(self.model, self.planes, self.device)
            else:
                self.logits[waiting], self.values[waiting] = predict(self.model, self.planes[waiting], self.device)
            (_expand_batch_parallel if many else _expand_batch)(
                self.logits, self.values, self.pending, self.t_visits, self.t_value, self.t_prior,
                self.t_move, self.t_first, self.t_count, self.t_size, self.sims_done,
                self.path, self.path_len, self.leaf_moves, self.leaf_count,
                self.noise_weight, self.noise_alpha)
        _root_results(self.t_visits, self.t_value, self.t_move, self.t_first, self.t_count,
                      active, self.counts, self.root_q)
        return self.counts, self.root_q

    def root_move_values(self, active):
        """After search(): the search's value of every move it tried (n_games x 81, by move number;
        0 for moves it did not try). The visit counts say which ones were tried."""
        values = np.zeros((self.n_games, 81), dtype=np.float32)
        _root_move_values(self.t_visits, self.t_value, self.t_move, self.t_first, self.t_count, active, values)
        return values

    def choose_moves(self, active, move_number, temperature_moves):
        chosen = np.zeros(self.n_games, dtype=np.int64)
        _choose_moves(self.counts, active, move_number, temperature_moves, chosen)
        return chosen

    def play(self, active, chosen):
        """Plays chosen[g] in every active game and returns the status of every
        game: 0 = goes on, 1 = the player who moved has won, 2 = tie."""
        status = np.zeros(self.n_games, dtype=np.int64)
        _play_moves(self.r_x, self.r_o, self.r_won, self.r_val, self.r_meta, self.r_player,
                    active, chosen, status)
        return status


class NeuralPlayer:
    """Plays single moves on a game.Board with the network and AlphaZero MCTS."""

    def __init__(self, model, device='cpu', simulations=400):
        self.search = BatchedSearch(model, device, 1, simulations)

    def best_move(self, board):
        """Returns ((big, little), value): value is how good the position looks
        for the player to move, from -1 (lost) to 1 (won)."""
        self.search.set_board(0, board)
        counts, root_q = self.search.search()
        move = int(np.argmax(counts[0]))
        return (move // 9, move % 9), float(root_q[0])
