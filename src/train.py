"""Trains the neural network for Super TikTakToe the AlphaZero way.

    python3 src/train.py

One iteration = the network plays games against itself (with tree search),
then learns from those games: which moves the search preferred (policy) and
who won in the end (value). The better network then plays the next games.

Everything is saved after every iteration. Stop with Ctrl+C whenever you want
and start the script again later - it continues where it stopped.

The settings below are the plan for the 'large' network (about 250 iterations).
The numbers in LEARNING_RATE_DECAY, SIMULATIONS_GROWTH and BUFFER_GROWTH count
iterations of that training; for another network they have to be chosen again.
On Google Colab: see colab_train.ipynb.

    python3 src/train.py restore 120             goes back to the snapshot of an iteration
    python3 src/train.py experiment more_games   continues the finished network with one thing
                                                 changed, in its own folder (see EXPERIMENTS below)
    python3 src/train.py run large_lines 60      trains another network than MODEL with the same
                                                 settings, up to this iteration
"""
import csv
import os
import re
import shutil
import signal
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

import fast_minimax
import game
import MCTS
import Tic_Tac_Toe_NN as NN
from Tic_Tac_Toe_NN import SPATIAL_TO_MOVE

# ------------------------------- Settings -------------------------------
MODEL = 'large'             # 'small', 'medium' or 'large' (see Tic_Tac_Toe_NN.py)
ITERATIONS = 240            # the training ends by itself after this many iterations; Ctrl+C stops earlier
DEVICE = 'auto'             # 'auto', 'mps' (Apple), 'cuda' (Nvidia) or 'cpu'

# Self-play
GAMES_PER_ITERATION = 512   # games the network plays against itself per iteration
PARALLEL_GAMES = 512        # how many of them run at the same time (512 is much faster on Colab's L4;
                            # on a Mac 256 is slightly better)
SIMULATIONS = 200           # positions the search looks at per move at the beginning
# A young network learns as much from cheap games; later the games get more careful.
# {after this many iterations: simulations per move from then on}
SIMULATIONS_GROWTH = {100: 400}
TEMPERATURE_MOVES = 20      # for this many moves a game picks its moves with some randomness
NOISE_WEIGHT = 0.25         # random noise on the first move of every search (0 = none)
NOISE_ALPHA = 0.6
VALUE_MIX = 0.5             # value target: 0 = only who won, 1 = only what the search thought

# Learning
BATCH_SIZE = 256
LEARNING_RATE = 1e-3        # how big the learning steps are at the beginning
# Learning rate decay: smaller steps later, for the fine tuning.
# {after this many iterations: learning rate from then on}
LEARNING_RATE_DECAY = {120: 3e-4, 190: 1e-4}
WEIGHT_DECAY = 1e-4
REPLAY_RATIO = 4            # how often every new position is used for learning on average
BUFFER_SIZE = 300000        # positions kept from earlier iterations (about 11 iterations)
# Later the network changes slowly, so older games stay good lessons and more are kept.
# {after this many iterations: positions kept from then on}
BUFFER_GROWTH = {190: 600000}
# Extra lessons about the long game, for networks that have the outputs for them ('large_long' in
# Tic_Tac_Toe_NN.py; see LESSONS there). How much each counts next to the policy and value loss:
LESSON_WEIGHTS = {'owner': 0.5, 'outcome': 0.5, 'moves': 1.0}
MOVE_VISITS_FULL = 16       # a move's value counts fully from this many visits of the search, less below

# Warm start: before the first iteration the network learns from games of the
# minimax AI (your evaluation function), so it does not start from zero.
# WARM_START = False is pure AlphaZero self-play: the network never sees minimax.
WARM_START = False
WARM_START_GAMES = 4000
WARM_START_DEPTH = 5        # depth of the minimax that plays these games
WARM_START_RANDOM = 0.2     # share of random moves in these games, for variety
WARM_START_EPOCHS = 2

# Progress check: every EVAL_EVERY iterations the network plays against minimax (0 = never;
# the snapshots can be tested afterwards with test_vs_minimax.py and test_networks.py)
EVAL_EVERY = 0
EVAL_DEPTHS = (9, 11, 13)   # depths of the minimax opponents (only to measure, nothing is learned from them)
EVAL_GAMES = 40             # games against each of them, must be even (more games = less random up and down)
EVAL_PROCESSES = 8          # the minimax moves of these games are computed in parallel
EVAL_SIMULATIONS = 100      # search size of the network in these games
# Yardstick: every REFERENCE_EVERY iterations the network plays against a finished network
# (always the same openings, from both sides). It needs only the graphics card. The file of that
# network (models/alphazero_<REFERENCE>.pt) must be in the models folder; without it the check is skipped.
REFERENCE = 'medium'        # which finished network (None = no such check)
REFERENCE_EVERY = 4
REFERENCE_GAMES = 256       # must be even; all are played at the same time
REFERENCE_SIMULATIONS = 100 # search size of both networks in these games
SAVE_BUFFER_EVERY = 5       # the collected positions are large, so they are saved less often
SNAPSHOT_EVERY = 4          # every this many iterations a copy of the network is kept in
                            # models/snapshots/, so you can go back to an earlier one (0 = never)

# Tests: a finished training is continued with one thing changed, in a folder of its own
# (models/experiments/<name>/), so the finished network is not touched.
#     python3 src/train.py experiment more_games
# Every test changes the settings above as written here. On Google Colab: colab_experiment.ipynb.
EXPERIMENT_START = 240      # the tests continue the network of MODEL after this iteration
EXPERIMENT_BASELINE = 4     # this many of its last checks are played again with the test's number of games
EXPERIMENTS = {
    # twice the games per iteration, every position used half as often: the same amount of learning
    'more_games': {'GAMES_PER_ITERATION': 1024, 'REPLAY_RATIO': 2, 'ITERATIONS': 260, 'REFERENCE_GAMES': 512},
    # twice the simulations per move from iteration 241 on: more accurate lessons
    'more_simulations': {'SIMULATIONS_GROWTH': {100: 400, 240: 800}, 'ITERATIONS': 272, 'REFERENCE_GAMES': 512},
}
# ------------------------------------------------------------------------


# What is saved of every position for the extra lessons: name -> (numbers per position, type, value for "not known")
EXTRA = {'owner': (9, np.int8, -1),          # per small board: 0 mine, 1 the other's, 2 nobody's at the end; -1 not asked
         'outcome': (0, np.int8, -1),        # 0 the player to move won, 1 tie, 2 lost
         'move_value': (81, np.float16, 0),  # the search's value of every move it tried (9x9 picture order)
         'move_weight': (81, np.uint8, 0)}   # its visits of that move, at most MOVE_VISITS_FULL


def _unknown(name, count):
    size, kind, unknown = EXTRA[name]
    return np.full((count, size) if size else count, unknown, dtype=kind)


class ReplayBuffer:
    """The most recent training positions: network input, policy target, value target, and the
    targets of the extra lessons (EXTRA)."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.planes = np.zeros((capacity, NN.INPUT_PLANES, 9, 9), dtype=np.uint8)
        self.policy = np.zeros((capacity, 81), dtype=np.float16)
        self.value = np.zeros(capacity, dtype=np.float32)
        self.extra = {name: _unknown(name, capacity) for name in EXTRA}
        self.size = 0
        self.position = 0

    def add(self, planes, policy, value, *extra):
        """extra: nothing (positions without the extra lessons), or one array per entry of EXTRA."""
        extra = dict(zip(EXTRA, extra)) if extra else {name: _unknown(name, len(value)) for name in EXTRA}
        for start in range(0, len(value), self.capacity):
            chunk = slice(start, start + self.capacity)
            n = len(value[chunk])
            index = (self.position + np.arange(n)) % self.capacity
            self.planes[index] = planes[chunk]
            self.policy[index] = policy[chunk]
            self.value[index] = value[chunk]
            for name in EXTRA:
                self.extra[name][index] = extra[name][chunk]
            self.position = (self.position + n) % self.capacity
            self.size = min(self.capacity, self.size + n)

    def resize(self, capacity):
        """Changes how many positions are kept. The newest ones stay."""
        order = np.arange(self.size)
        if self.size == self.capacity:      # full: the oldest position is at self.position
            order = (self.position + order) % self.size
        order = order[-capacity:]
        kept = [self.planes[order], self.policy[order], self.value[order]] + [self.extra[name][order] for name in EXTRA]
        self.__init__(capacity)
        self.add(*kept)

    def sample(self, batch_size, device, lessons=False):
        """lessons=True: also a dict with the targets of the extra lessons."""
        index = np.random.randint(0, self.size, batch_size)
        planes = torch.from_numpy(self.planes[index]).to(device).float()
        policy = torch.from_numpy(self.policy[index].astype(np.float32)).to(device)
        value = torch.from_numpy(self.value[index]).to(device)
        if not lessons:
            return planes, policy, value
        extra = {name: torch.from_numpy(self.extra[name][index].astype(np.float32)).to(device) for name in EXTRA}
        return planes, policy, value, extra

    def save(self, path):
        with open(path + '.tmp', 'wb') as file:
            np.savez(file, planes=self.planes[:self.size], policy=self.policy[:self.size],
                     value=self.value[:self.size], position=self.position,
                     **{name: self.extra[name][:self.size] for name in EXTRA})
        os.replace(path + '.tmp', path)

    def load(self, path):
        planes, policy, value, dropped, extra = load_positions(path, with_extra=True)
        if dropped:
            print(f'{dropped} saved positions were damaged (not a number) and are left out.')
        n = min(len(value), self.capacity)
        self.planes[:n] = planes[-n:]
        self.policy[:n] = policy[-n:]
        self.value[:n] = value[-n:]
        for name in EXTRA:
            self.extra[name][:n] = extra[name][-n:]
        self.size = n
        self.position = n % self.capacity


def load_positions(path, with_extra=False):
    """The saved positions, oldest first, without damaged ones. Returns (planes, policy, value, number left out);
    with_extra=True: also a dict with the targets of the extra lessons ("not known" for memories saved without them)."""
    data = np.load(path)
    planes, policy, value = data['planes'], data['policy'], data['value']
    start = int(data['position']) % max(1, len(value))      # in a full memory the oldest position is here
    order = (start + np.arange(len(value))) % max(1, len(value))
    good = order[np.isfinite(value[order]) & np.isfinite(policy[order]).all(axis=1)]
    result = (planes[good], policy[good], value[good], len(value) - len(good))
    if with_extra:
        result += ({name: data[name][good] if name in data.files else _unknown(name, len(good)) for name in EXTRA},)
    return result


# ----------------------------------------------------------------------------
# warm start: games of the minimax AI
# ----------------------------------------------------------------------------

def _minimax_game(seed):
    """One game minimax against minimax. For every position: the network input,
    the move minimax would play, and a value from the result and the search."""
    np.random.seed(seed)
    board = game.Board(playerturn=1)
    planes, policies, players, search_values = [], [], [], []
    while not board.game_is_over:
        value, (big, little) = fast_minimax.best_move(board, WARM_START_DEPTH)
        legal = board.possible_move_vector().reshape(-1).astype(np.float32)
        policy = 0.1 * legal / legal.sum()
        policy[big * 9 + little] += 0.9
        planes.append(MCTS.encode_board(board).astype(np.uint8))
        policies.append(policy[SPATIAL_TO_MOVE])
        players.append(board.playerturn)
        search_values.append(float(np.clip(value * board.playerturn, -1, 1)))
        if np.random.random() < WARM_START_RANDOM:
            moves = board.possible_moves()
            board.execute(moves[np.random.randint(len(moves))])
        else:
            board.execute((big, little))
    winner = -board.playerturn if game.game_is_win(board.won_games)[0] else 0
    values = 0.5 * winner * np.array(players) + 0.5 * np.array(search_values)
    return np.array(planes), np.array(policies, dtype=np.float32), values.astype(np.float32)


def minimax_games(n_games):
    workers = max(1, (os.cpu_count() or 2) - 2)
    seeds = np.random.randint(0, 2 ** 31, n_games).tolist()
    with Pool(workers) as pool:
        results = pool.map(_minimax_game, seeds, chunksize=16)
    return (np.concatenate([r[0] for r in results]),
            np.concatenate([r[1] for r in results]),
            np.concatenate([r[2] for r in results]))


# ----------------------------------------------------------------------------
# self-play
# ----------------------------------------------------------------------------

def self_play(model, device, n_games, n_simulations):
    """The network plays n_games against itself. Returns the training data
    (planes, policy targets, value targets, and the targets of the extra lessons in the order of
    EXTRA) and a dict with statistics."""
    parallel = min(PARALLEL_GAMES, n_games)
    search = MCTS.BatchedSearch(model, device, parallel, n_simulations,
                                noise_weight=NOISE_WEIGHT, noise_alpha=NOISE_ALPHA)
    length = MCTS.MAX_GAME_LENGTH
    rec_planes = np.zeros((parallel, length, NN.INPUT_PLANES, 9, 9), dtype=np.uint8)
    rec_policy = np.zeros((parallel, length, 81), dtype=np.float32)
    rec_q = np.zeros((parallel, length), dtype=np.float32)
    rec_player = np.zeros((parallel, length), dtype=np.int64)
    rec_move_value = np.zeros((parallel, length, 81), dtype=np.float16)
    rec_move_weight = np.zeros((parallel, length, 81), dtype=np.uint8)
    moves_played = np.zeros(parallel, dtype=np.int64)
    active = np.ones(parallel, dtype=np.bool_)
    started, finished = parallel, 0
    out_planes, out_policy, out_value = [], [], []
    out_extra = {name: [] for name in EXTRA}
    results = {1: 0, -1: 0, 0: 0}
    corner = 3 * np.arange(3)                           # one cell of every small board, to read its state

    model.eval()
    while active.any():
        counts, root_q = search.search(active)
        planes = search.encode_roots()
        index = np.flatnonzero(active)
        step = moves_played[index]
        policy = counts[index] / counts[index].sum(axis=1, keepdims=True)
        rec_planes[index, step] = planes[index]
        rec_policy[index, step] = policy[:, SPATIAL_TO_MOVE]
        rec_q[index, step] = root_q[index]
        rec_player[index, step] = search.r_player[index]
        rec_move_value[index, step] = search.root_move_values(active)[index][:, SPATIAL_TO_MOVE]
        rec_move_weight[index, step] = np.minimum(counts[index], MOVE_VISITS_FULL)[:, SPATIAL_TO_MOVE]

        chosen = search.choose_moves(active, moves_played, TEMPERATURE_MOVES)
        movers = search.r_player.copy()
        status = search.play(active, chosen)
        moves_played[index] += 1

        for g in index[status[index] != 0]:
            n = moves_played[g]
            winner = movers[g] if status[g] == 1 else 0
            results[int(winner)] += 1
            outcome = winner * rec_player[g, :n]        # +1 where the player to move won in the end
            out_planes.append(rec_planes[g, :n].copy())
            out_policy.append(rec_policy[g, :n].copy())
            out_value.append(((1 - VALUE_MIX) * outcome + VALUE_MIX * rec_q[g, :n]).astype(np.float32))
            # extra lessons. Who holds each small board now that the game is over (r_won: 1 X, -1 O,
            # 0 open, 2 tied), seen from the player to move in every position ...
            final = search.r_won[g] * rec_player[g, :n, None]
            owner = np.where(final == 1, 0, np.where(final == -1, 1, 2)).astype(np.int8)
            # ... asked only for the boards that were still open in that position
            decided = rec_planes[g, :n, 3:6].sum(axis=1)[:, corner][:, :, corner].reshape(n, 9) > 0
            owner[decided] = -1
            out_extra['owner'].append(owner)
            out_extra['outcome'].append((1 - outcome).astype(np.int8))      # 0 won, 1 tie, 2 lost
            out_extra['move_value'].append(rec_move_value[g, :n].copy())
            out_extra['move_weight'].append(rec_move_weight[g, :n].copy())
            finished += 1
            moves_played[g] = 0
            if started < n_games:
                search.reset(g)
                started += 1
            else:
                active[g] = False

    data = (np.concatenate(out_planes), np.concatenate(out_policy), np.concatenate(out_value),
            *(np.concatenate(out_extra[name]) for name in EXTRA))
    stats = {'games': finished, 'positions': len(data[2]), 'x_wins': results[1],
             'o_wins': results[-1], 'ties': results[0]}
    return data, stats


# ----------------------------------------------------------------------------
# learning
# ----------------------------------------------------------------------------

LESSON_LOSSES = {}          # after train(): the average loss of every extra lesson


def lesson_losses(extra, targets):
    """extra: the network's answers to the extra lessons, targets: what was saved (both already turned)."""
    losses = {}
    if 'owner' in extra:        # only small boards that were still open count (the others are -1)
        target = targets['owner'].long()
        losses['owner'] = (F.cross_entropy(extra['owner'], target, ignore_index=-1, reduction='sum')
                           / (target >= 0).sum().clamp(min=1))
    if 'outcome' in extra:
        target = targets['outcome'].long()
        losses['outcome'] = (F.cross_entropy(extra['outcome'], target, ignore_index=-1, reduction='sum')
                             / (target >= 0).sum().clamp(min=1))
    if 'moves' in extra:        # only moves the search tried count, fully from MOVE_VISITS_FULL visits
        weight = targets['move_weight'] / MOVE_VISITS_FULL
        losses['moves'] = ((extra['moves'] - targets['move_value']) ** 2 * weight).sum() / weight.sum().clamp(min=1)
    return losses


def train(model, optimizer, buffer, device, steps):
    """Returns the average policy loss and value loss (the extra lessons: LESSON_LOSSES)."""
    model.train()
    lessons = bool(getattr(model, 'lessons', ()))
    policy_losses, value_losses, extra_losses = [], [], {}
    for _ in range(steps):
        if lessons:
            planes, policy, value, targets = buffer.sample(BATCH_SIZE, device, lessons=True)
            names = [name for name in targets if targets[name].dim() == 2]     # the result of the game is not turned
            planes, policy, *turned = NN.random_symmetry(planes, policy, *(targets[name] for name in names))
            targets.update(zip(names, turned))
            logits, predicted_value, extra = model(planes, lessons=True)
        else:
            planes, policy, value = buffer.sample(BATCH_SIZE, device)
            planes, policy = NN.random_symmetry(planes, policy)
            logits, predicted_value = model(planes)
        # only legal moves count (plane 2 marks them)
        logits = logits.masked_fill(planes[:, 2].reshape(-1, 81) == 0, -1e9)
        policy_loss = -(policy * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(predicted_value, value)
        loss = policy_loss + value_loss
        if lessons:
            for name, lesson_loss in lesson_losses(extra, targets).items():
                loss = loss + LESSON_WEIGHTS[name] * lesson_loss
                extra_losses.setdefault(name, []).append(lesson_loss.detach())
        if not bool(torch.isfinite(loss)):
            raise SystemExit('The loss is not a number any more, so the training stops here and this '
                             'iteration is not saved. Start again to continue from the last saved iteration; '
                             'if it happens again, go back to a snapshot: python3 src/train.py restore <iteration>')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        policy_losses.append(policy_loss.detach())
        value_losses.append(value_loss.detach())
    model.eval()
    LESSON_LOSSES.clear()
    LESSON_LOSSES.update({name: float(torch.stack(values).mean()) for name, values in extra_losses.items()})
    return float(torch.stack(policy_losses).mean()), float(torch.stack(value_losses).mean())


# ----------------------------------------------------------------------------
# progress check
# ----------------------------------------------------------------------------

def _test_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)            # Ctrl+C is handled by the main process
    fast_minimax.best_move(game.Board(playerturn=1), 2)     # compiles the search in this process


def _minimax_move(job):
    board, depth = job
    return fast_minimax.best_move(board, depth)[1]


def evaluate(model, device, depth, games=EVAL_GAMES, simulations=EVAL_SIMULATIONS, pool=None):
    """The network (with tree search) against minimax of the given depth.
    Every opening (2 random moves) is played twice, with the network as X and as O.
    With a pool the minimax moves of the games are computed in parallel.
    Returns (wins, ties, losses) of the network."""
    model.eval()
    search = MCTS.BatchedSearch(model, device, games, simulations)
    boards = [game.Board(playerturn=1) for _ in range(games)]
    net_side = [1 if g % 2 == 0 else -1 for g in range(games)]
    for g, board in enumerate(boards):
        if g % 2 == 0:
            opening = []
            for _ in range(2):
                moves = board.possible_moves()
                opening.append(moves[np.random.randint(len(moves))])
                board.execute(opening[-1])
        else:                       # the same opening again, with the sides swapped
            for move in opening:
                board.execute(move)
    while not all(board.game_is_over for board in boards):
        active = np.array([not board.game_is_over and board.playerturn == net_side[g]
                           for g, board in enumerate(boards)])
        if active.any():
            for g in np.flatnonzero(active):
                search.set_board(g, boards[g])
            counts, _ = search.search(active)
            for g in np.flatnonzero(active):
                boards[g].execute(int(np.argmax(counts[g])))
        waiting = [g for g, board in enumerate(boards)
                   if not board.game_is_over and board.playerturn != net_side[g]]
        jobs = [(boards[g], depth) for g in waiting]
        answers = pool.map(_minimax_move, jobs) if pool else map(_minimax_move, jobs)
        for g, move in zip(waiting, answers):
            boards[g].execute(move)
    wins = ties = losses = 0
    for g, board in enumerate(boards):
        if not game.game_is_win(board.won_games)[0]:
            ties += 1
        elif -board.playerturn == net_side[g]:      # the player who moved last has won
            wins += 1
        else:
            losses += 1
    return wins, ties, losses


def against_reference(model, reference, device):
    """The network against a finished network, both with tree search and without randomness.
    The openings (2 moves) are the same in every check, each played from both sides.
    Returns (wins, ties, losses) of the network."""
    import test_networks
    random = np.random.RandomState(0)
    games = []
    for _ in range(REFERENCE_GAMES // 2):
        board, opening = game.Board(playerturn=1), []
        for _ in range(2):
            moves = board.possible_moves()
            opening.append(moves[random.randint(len(moves))])
            board.execute(opening[-1])
        games += [(opening, 1), (opening, -1)]
    model.eval()
    search = MCTS.BatchedSearch(model, device, len(games), REFERENCE_SIMULATIONS)
    other = MCTS.BatchedSearch(reference, device, len(games), REFERENCE_SIMULATIONS)
    results = [result for result, _ in test_networks.play_batch(search, other, games)]
    return tuple(results.count(r) for r in test_networks.RESULTS)


def repeat_checks(reference, device, log_file, columns):
    """Checks against the finished network that were played with another number of games than
    REFERENCE_GAMES (or not at all) are played again with the snapshots of those iterations,
    and the log is corrected."""
    with open(log_file, newline='') as file:
        reader = csv.DictReader(file)
        fields, rows = list(reader.fieldnames or []), list(reader)
    if columns[0] not in fields:
        place = fields.index('seconds') if 'seconds' in fields else len(fields)
        fields[place:place] = columns
    for row in rows:
        iteration = int(row['iteration'])
        played = sum(int(float(row[c])) for c in columns[1:] if row.get(c) not in ('', None))
        if (iteration % REFERENCE_EVERY or played == REFERENCE_GAMES
                or not os.path.exists(snapshot_path(MODEL, iteration))):
            continue
        start = time.perf_counter()
        result = against_reference(NN.load_model(MODEL, device, iteration), reference, device)
        row.update(zip(columns, (score(*result),) + result))
        with open(log_file + '.tmp', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields, restval='')
            writer.writeheader()
            writer.writerows(rows)
        os.replace(log_file + '.tmp', log_file)
        print(f"Iteration {iteration} again against '{REFERENCE}' with {REFERENCE_GAMES} games "
              f'({played} before) in {time.perf_counter() - start:.0f}s (wins-ties-losses): '
              f'{score(*result):.0%} ({result[0]}-{result[1]}-{result[2]})', flush=True)


def score(wins, ties, losses):
    """Share of points: win = 1, tie = 0.5."""
    return (wins + 0.5 * ties) / (wins + ties + losses)


def test_columns(depth):
    """Names of the log columns for the test against minimax of this depth."""
    return [f'vs_minimax_{depth}', f'wins_{depth}', f'ties_{depth}', f'losses_{depth}']


TEST_COLUMN = re.compile(r'(vs_minimax|wins|ties|losses)_(\d+)$')


# ----------------------------------------------------------------------------
# main loop
# ----------------------------------------------------------------------------

def _scheduled(start, changes, iteration):
    """The value of a setting after `iteration` finished iterations: `start`, or the
    newest entry of {after this many iterations: value} that is reached."""
    value = start
    for after, later in sorted(changes.items()):
        if iteration >= after:
            value = later
    return value


def learning_rate(iteration):
    """The learning rate for the iteration that follows after `iteration` finished ones."""
    return _scheduled(LEARNING_RATE, LEARNING_RATE_DECAY, iteration)


def simulations(iteration):
    """The simulations per move in the games of the iteration that follows."""
    return _scheduled(SIMULATIONS, SIMULATIONS_GROWTH, iteration)


def buffer_size(iteration):
    """How many positions are kept for the iteration that follows."""
    return _scheduled(BUFFER_SIZE, BUFFER_GROWTH, iteration)


def paths(variant):
    base = NN.checkpoint_path(variant)[:-3]
    return base + '.pt', base + '_buffer.npz', base + '_log.csv'


def snapshot_path(variant, iteration):
    return NN.snapshot_path(variant, iteration)


def write_log(path, row):
    """Adds a row to the log. If the columns changed (for example other
    EVAL_DEPTHS), the file is rewritten so that old and new rows fit together."""
    fields, rows = list(row), []
    if os.path.exists(path):
        with open(path, newline='') as file:
            reader = csv.DictReader(file)
            old_fields = reader.fieldnames or []
            if old_fields == fields:
                rows = None                     # same columns: just append
            else:
                rows = list(reader)
                fields = old_fields + [f for f in fields if f not in old_fields]
                depths = sorted({int(TEST_COLUMN.match(f).group(2))
                                 for f in fields if TEST_COLUMN.match(f)})
                tests = [f for depth in depths for f in test_columns(depth) if f in fields]
                fields = ([f for f in fields if f not in tests and f != 'seconds']
                          + tests + ['seconds'])
    if rows is None:
        with open(path, 'a', newline='') as file:
            csv.DictWriter(file, fieldnames=fields).writerow(row)
        return
    with open(path + '.tmp', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, restval='')
        writer.writeheader()
        writer.writerows(rows + [row])
    os.replace(path + '.tmp', path)


def main():
    device = NN.choose_device(DEVICE)
    checkpoint_file, buffer_file, log_file = paths(MODEL)
    model = NN.create_model(MODEL).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    buffer = ReplayBuffer(BUFFER_SIZE)
    iteration, total_games, warm_started = 0, 0, False

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = LEARNING_RATE
            group['weight_decay'] = WEIGHT_DECAY
        iteration = checkpoint.get('iteration', 0)
        total_games = checkpoint.get('total_games', 0)
        warm_started = checkpoint.get('warm_started', iteration > 0)
        buffer = ReplayBuffer(buffer_size(iteration))
        if os.path.exists(buffer_file):
            buffer.load(buffer_file)
        print(f"Continuing '{MODEL}' network after iteration {iteration} "
              f"({total_games} games so far, {buffer.size} positions in memory) on {device}, "
              f"{simulations(iteration)} simulations per move, learning rate {learning_rate(iteration):g}")
    else:
        parameters = sum(p.numel() for p in model.parameters())
        print(f"New '{MODEL}' network ({parameters / 1e6:.2f} million parameters) on {device}")

    reference = None
    if REFERENCE and REFERENCE != MODEL:
        if os.path.exists(NN.checkpoint_path(REFERENCE)):
            reference = NN.load_model(REFERENCE, device)
        else:
            print(f"No finished '{REFERENCE}' network in the models folder "
                  f'({NN.checkpoint_path(REFERENCE)}): the check against it is skipped.')
    reference_columns = [f'vs_{REFERENCE}', f'wins_{REFERENCE}', f'ties_{REFERENCE}', f'losses_{REFERENCE}']

    def save(with_buffer):
        NN.save_checkpoint(checkpoint_file, model, optimizer, variant=MODEL,
                           iteration=iteration, total_games=total_games, warm_started=warm_started)
        if with_buffer:
            buffer.save(buffer_file)

    try:
        if reference is not None and os.path.exists(log_file):
            repeat_checks(reference, device, log_file, reference_columns)
        if WARM_START and not warm_started and iteration == 0:
            start = time.perf_counter()
            print(f'Warm start: {WARM_START_GAMES} games of minimax depth {WARM_START_DEPTH} ...')
            data = minimax_games(WARM_START_GAMES)
            buffer.add(*data)
            steps = WARM_START_EPOCHS * len(data[2]) // BATCH_SIZE
            policy_loss, value_loss = train(model, optimizer, buffer, device, steps)
            print(f'  learned from {len(data[2])} positions in {time.perf_counter() - start:.0f}s '
                  f'| policy loss {policy_loss:.3f}, value loss {value_loss:.3f}')
            results = {depth: evaluate(model, device, depth) for depth in EVAL_DEPTHS}
            print('  against minimax (wins-ties-losses): ' + ', '.join(
                f'depth {d}: {score(*r):.0%} ({r[0]}-{r[1]}-{r[2]})' for d, r in results.items()))
            warm_started = True
            save(with_buffer=True)

        while iteration < ITERATIONS:
            start = time.perf_counter()
            n_simulations = simulations(iteration)
            data, stats = self_play(model, device, GAMES_PER_ITERATION, n_simulations)
            play_time = time.perf_counter() - start
            good = np.isfinite(data[2]) & np.isfinite(data[1]).all(axis=1)
            if not good.all():      # must never get into the lessons: one such position ruins the network
                print(f'{int((~good).sum())} positions of this iteration are damaged (not a number) and are left out.')
                data = tuple(part[good] for part in data)
                stats['positions'] = int(good.sum())
                if not stats['positions']:
                    raise SystemExit('All positions of this iteration are damaged: the network itself is. '
                                     'Go back to a snapshot: python3 src/train.py restore <iteration>')
            if buffer.capacity != buffer_size(iteration):
                buffer.resize(buffer_size(iteration))
                print(f'From now on the last {buffer.capacity} positions are kept.')
            buffer.add(*data)
            steps = max(1, REPLAY_RATIO * stats['positions'] // BATCH_SIZE)
            rate = learning_rate(iteration)
            for group in optimizer.param_groups:
                group['lr'] = rate
            policy_loss, value_loss = train(model, optimizer, buffer, device, steps)
            iteration += 1
            total_games += stats['games']
            save(with_buffer=iteration % SAVE_BUFFER_EVERY == 0)
            if SNAPSHOT_EVERY and iteration % SNAPSHOT_EVERY == 0:
                os.makedirs(os.path.dirname(snapshot_path(MODEL, iteration)), exist_ok=True)
                shutil.copyfile(checkpoint_file, snapshot_path(MODEL, iteration))

            row = {'iteration': iteration, 'games': total_games, 'positions': stats['positions'],
                   'policy_loss': round(policy_loss, 4), 'value_loss': round(value_loss, 4),
                   'x_wins': stats['x_wins'], 'o_wins': stats['o_wins'], 'ties': stats['ties'],
                   'learning_rate': rate, 'simulations': n_simulations}
            row.update({f'{name}_loss': round(value, 4) for name, value in LESSON_LOSSES.items()})
            line = (f"Iteration {iteration}: {stats['games']} games, {stats['positions']} positions "
                    f"in {play_time:.0f}s | policy loss {policy_loss:.3f}, value loss {value_loss:.3f}"
                    f" | learning rate {rate:g}, {n_simulations} simulations")
            if LESSON_LOSSES:
                line += ' | extra lessons: ' + ', '.join(f'{name} {value:.3f}' for name, value in LESSON_LOSSES.items())
            for depth in EVAL_DEPTHS:
                row.update(dict.fromkeys(test_columns(depth), ''))
            if EVAL_EVERY and iteration % EVAL_EVERY == 0:
                parts = []
                with Pool(EVAL_PROCESSES, initializer=_test_worker) as pool:
                    for depth in EVAL_DEPTHS:
                        result = evaluate(model, device, depth, pool=pool)
                        row.update(zip(test_columns(depth), (score(*result),) + result))
                        parts.append(f'depth {depth}: {score(*result):.0%} '
                                     f'({result[0]}-{result[1]}-{result[2]})')
                line += ' | against minimax (wins-ties-losses): ' + ', '.join(parts)
            if reference is not None:
                row.update(dict.fromkeys(reference_columns, ''))
                if iteration % REFERENCE_EVERY == 0:
                    check_start = time.perf_counter()
                    result = against_reference(model, reference, device)
                    row.update(zip(reference_columns, (score(*result),) + result))
                    line += (f" | against '{REFERENCE}', {REFERENCE_GAMES} games in "
                             f'{time.perf_counter() - check_start:.0f}s (wins-ties-losses): '
                             f'{score(*result):.0%} ({result[0]}-{result[1]}-{result[2]})')
            row['seconds'] = round(time.perf_counter() - start)
            write_log(log_file, row)
            print(line, flush=True)
        print(f'\nFinished: iteration {iteration} is reached and everything is saved in {checkpoint_file}. '
              f'For more, raise ITERATIONS in src/train.py.', flush=True)
    except KeyboardInterrupt:
        # Colab's stop button can send the stop signal several times; a second one must not break the saving
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print(f'\nStopped. Saving ... (the network is from iteration {iteration})', flush=True)
        save(with_buffer=True)
        print(f'Saved to {checkpoint_file}. Start the script again to continue.')


def _copy(source, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copyfile(source, target + '.tmp')
    os.replace(target + '.tmp', target)


def experiment(name):
    """Continues the finished network with the changed settings of EXPERIMENTS[name], in the folder
    models/experiments/<name>/. The first time, the network, its memory, its log and the last
    snapshots are copied there; later calls continue the test."""
    if name not in EXPERIMENTS:
        raise SystemExit(f"There is no test '{name}'. The tests are: {', '.join(EXPERIMENTS)}.")
    main_folder = NN.MODEL_DIR
    folder = os.path.join(main_folder, 'experiments', name)
    checkpoint_file, buffer_file, log_file = paths(MODEL)
    started = os.path.exists(os.path.join(folder, os.path.basename(checkpoint_file)))
    if not started:
        snapshot = snapshot_path(MODEL, EXPERIMENT_START)
        if not os.path.exists(checkpoint_file) or not os.path.exists(snapshot):
            raise SystemExit(f"The test needs the '{MODEL}' network after iteration {EXPERIMENT_START}, "
                             f'but {snapshot} is not there.')
        newest = NN.saved_iteration(MODEL)
        if newest != EXPERIMENT_START or not os.path.exists(buffer_file):
            raise SystemExit(f"The test starts after iteration {EXPERIMENT_START} and needs the memory of that moment, "
                             f"but the '{MODEL}' network in {main_folder} is at iteration {newest}"
                             + ('' if os.path.exists(buffer_file) else ' and has no saved memory') + '.')
        print(f"Setting up the test '{name}' in {folder} ...", flush=True)
        copies = [(snapshot, os.path.basename(checkpoint_file)), (buffer_file, os.path.basename(buffer_file))]
        if os.path.exists(log_file):
            copies.append((log_file, os.path.basename(log_file)))
        if REFERENCE and os.path.exists(NN.checkpoint_path(REFERENCE)):
            copies.append((NN.checkpoint_path(REFERENCE), os.path.basename(NN.checkpoint_path(REFERENCE))))
        for back in range(EXPERIMENT_BASELINE):         # for the baseline with the test's number of games
            earlier = snapshot_path(MODEL, EXPERIMENT_START - back * REFERENCE_EVERY)
            if os.path.exists(earlier):
                copies.append((earlier, os.path.relpath(earlier, main_folder)))
        for source, target in copies[::-1]:             # the network last: it marks the test as set up
            _copy(source, os.path.join(folder, target))
    NN.MODEL_DIR = folder
    globals().update(EXPERIMENTS[name])
    print(f"Test '{name}' ({'continued' if started else 'new'}): "
          + ', '.join(f'{setting} = {value}' for setting, value in EXPERIMENTS[name].items()), flush=True)
    main()
    experiment_summary()


def experiment_summary():
    """Compares the checks of the test with the checks of the network it started from."""
    log_file = paths(MODEL)[2]
    columns = [f'wins_{REFERENCE}', f'ties_{REFERENCE}', f'losses_{REFERENCE}']
    if not REFERENCE or not os.path.exists(log_file):
        return
    with open(log_file, newline='') as file:
        rows = [row for row in csv.DictReader(file) if all(row.get(c) not in ('', None) for c in columns)]
    checks = {int(row['iteration']): [int(float(row[c])) for c in columns] for row in rows}
    checks = {i: r for i, r in checks.items() if sum(r) == REFERENCE_GAMES}
    before = [checks[i] for i in sorted(checks) if i <= EXPERIMENT_START][-EXPERIMENT_BASELINE:]
    after = [checks[i] for i in sorted(checks) if i > EXPERIMENT_START]

    def points(results):
        wins, ties, losses = (sum(part) for part in zip(*results))
        games = wins + ties + losses
        mean = (wins + 0.5 * ties) / games
        spread = (wins * (1 - mean) ** 2 + ties * (0.5 - mean) ** 2 + losses * mean ** 2) / games
        return mean, (spread / games) ** 0.5, games

    print(f"\nAgainst '{REFERENCE}' with {REFERENCE_GAMES} games per check:")
    if before:
        mean, error, games = points(before)
        print(f'  before the test ({len(before)} checks up to iteration {EXPERIMENT_START}, played again): '
              f'{mean:.1%} (+-{error:.1%}, {games} games)')
    for label, part in (('all checks of the test', after), ('its last 3 checks', after[-3:])):
        if part:
            mean, error, games = points(part)
            print(f'  {label}: {mean:.1%} (+-{error:.1%}, {games} games)')


def restore(iteration):
    """Goes back to the snapshot of this iteration: it becomes the newest network, later rows are
    taken out of the log, and damaged positions are taken out of the memory."""
    checkpoint_file, buffer_file, log_file = paths(MODEL)
    snapshot = snapshot_path(MODEL, iteration)
    if not os.path.exists(snapshot):
        raise SystemExit(f'There is no snapshot of iteration {iteration} ({snapshot}).')
    weights = torch.load(snapshot, map_location='cpu')['model']
    if not all(bool(torch.isfinite(value).all()) for value in weights.values()):
        raise SystemExit(f'The snapshot of iteration {iteration} is damaged too. Try an earlier one.')
    shutil.copyfile(snapshot, checkpoint_file + '.tmp')
    os.replace(checkpoint_file + '.tmp', checkpoint_file)
    print(f"The '{MODEL}' network is back at iteration {iteration}.")
    if os.path.exists(buffer_file):
        planes, policy, value, dropped, extra = load_positions(buffer_file, with_extra=True)
        with open(buffer_file + '.tmp', 'wb') as file:
            np.savez(file, planes=planes, policy=policy, value=value, position=len(value), **extra)
        os.replace(buffer_file + '.tmp', buffer_file)
        print(f'Memory: {dropped} damaged positions taken out, {len(value)} are kept.')
    if os.path.exists(log_file):
        with open(log_file, newline='') as file:
            reader = csv.DictReader(file)
            fields, rows = reader.fieldnames, list(reader)
        kept = [row for row in rows if int(row['iteration']) <= iteration]
        with open(log_file + '.tmp', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
        os.replace(log_file + '.tmp', log_file)
        print(f'Log: {len(rows) - len(kept)} later rows taken out.')
    for later in range(iteration + 1, iteration + 1000):
        if os.path.exists(snapshot_path(MODEL, later)):
            os.remove(snapshot_path(MODEL, later))
            print(f'Snapshot of iteration {later} removed.')
    print('Start the training again to continue from there.')


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == 'restore':
        restore(int(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == 'experiment':
        experiment(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == 'run':
        MODEL, ITERATIONS = sys.argv[2], int(sys.argv[3])
        main()
    else:
        main()
