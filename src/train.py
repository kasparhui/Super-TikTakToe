"""Trains the neural network for Super TikTakToe the AlphaZero way.

    python3 src/train.py

One iteration = the network plays games against itself (with tree search),
then learns from those games: which moves the search preferred (policy) and
who won in the end (value). The better network then plays the next games.

Everything is saved after every iteration. Stop with Ctrl+C whenever you want
and start the script again later - it continues where it stopped.
"""
import csv
import os
import re
import shutil
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
MODEL = 'medium'            # 'small', 'medium' or 'large' (see Tic_Tac_Toe_NN.py)
ITERATIONS = 100000         # how many iterations at most; Ctrl+C stops earlier
DEVICE = 'auto'             # 'auto', 'mps' (Apple), 'cuda' (Nvidia) or 'cpu'

# Self-play
GAMES_PER_ITERATION = 512   # games the network plays against itself per iteration
PARALLEL_GAMES = 256        # how many of them run at the same time
SIMULATIONS = 200           # positions the search looks at per move
TEMPERATURE_MOVES = 20      # for this many moves a game picks its moves with some randomness
NOISE_WEIGHT = 0.25         # random noise on the first move of every search (0 = none)
NOISE_ALPHA = 0.6
VALUE_MIX = 0.5             # value target: 0 = only who won, 1 = only what the search thought

# Learning
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
REPLAY_RATIO = 4            # how often every new position is used for learning on average
BUFFER_SIZE = 300000        # positions kept from earlier iterations

# Warm start: before the first iteration the network learns from games of the
# minimax AI (your evaluation function), so it does not start from zero.
# WARM_START = False is pure AlphaZero self-play: the network never sees minimax.
WARM_START = False
WARM_START_GAMES = 4000
WARM_START_DEPTH = 5        # depth of the minimax that plays these games
WARM_START_RANDOM = 0.2     # share of random moves in these games, for variety
WARM_START_EPOCHS = 2

# Progress check: every EVAL_EVERY iterations the network plays against minimax
EVAL_EVERY = 4
EVAL_DEPTHS = (6, 9, 11)    # depths of the minimax opponents (only to measure, nothing is learned from them)
EVAL_GAMES = 40             # games against each of them (more games = less random up and down)
EVAL_SIMULATIONS = 100      # search size of the network in these games
SAVE_BUFFER_EVERY = 5       # the collected positions are large, so they are saved less often
SNAPSHOT_EVERY = 4          # every this many iterations a copy of the network is kept in
                            # models/snapshots/, so you can go back to an earlier one (0 = never)
# ------------------------------------------------------------------------


class ReplayBuffer:
    """The most recent training positions: network input, policy target, value target."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.planes = np.zeros((capacity, NN.INPUT_PLANES, 9, 9), dtype=np.uint8)
        self.policy = np.zeros((capacity, 81), dtype=np.float16)
        self.value = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.position = 0

    def add(self, planes, policy, value):
        for start in range(0, len(value), self.capacity):
            chunk = slice(start, start + self.capacity)
            n = len(value[chunk])
            index = (self.position + np.arange(n)) % self.capacity
            self.planes[index] = planes[chunk]
            self.policy[index] = policy[chunk]
            self.value[index] = value[chunk]
            self.position = (self.position + n) % self.capacity
            self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size, device):
        index = np.random.randint(0, self.size, batch_size)
        planes = torch.from_numpy(self.planes[index]).to(device).float()
        policy = torch.from_numpy(self.policy[index].astype(np.float32)).to(device)
        value = torch.from_numpy(self.value[index]).to(device)
        return planes, policy, value

    def save(self, path):
        with open(path + '.tmp', 'wb') as file:
            np.savez(file, planes=self.planes[:self.size], policy=self.policy[:self.size],
                     value=self.value[:self.size], position=self.position)
        os.replace(path + '.tmp', path)

    def load(self, path):
        data = np.load(path)
        n = min(len(data['value']), self.capacity)
        self.planes[:n] = data['planes'][-n:]
        self.policy[:n] = data['policy'][-n:]
        self.value[:n] = data['value'][-n:]
        self.size = n
        self.position = int(data['position']) % self.capacity if n == self.capacity else n


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

def self_play(model, device, n_games):
    """The network plays n_games against itself. Returns the training data
    (planes, policy targets, value targets) and a dict with statistics."""
    parallel = min(PARALLEL_GAMES, n_games)
    search = MCTS.BatchedSearch(model, device, parallel, SIMULATIONS,
                                noise_weight=NOISE_WEIGHT, noise_alpha=NOISE_ALPHA)
    length = MCTS.MAX_GAME_LENGTH
    rec_planes = np.zeros((parallel, length, NN.INPUT_PLANES, 9, 9), dtype=np.uint8)
    rec_policy = np.zeros((parallel, length, 81), dtype=np.float32)
    rec_q = np.zeros((parallel, length), dtype=np.float32)
    rec_player = np.zeros((parallel, length), dtype=np.int64)
    moves_played = np.zeros(parallel, dtype=np.int64)
    active = np.ones(parallel, dtype=np.bool_)
    started, finished = parallel, 0
    out_planes, out_policy, out_value = [], [], []
    results = {1: 0, -1: 0, 0: 0}

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
            finished += 1
            moves_played[g] = 0
            if started < n_games:
                search.reset(g)
                started += 1
            else:
                active[g] = False

    data = (np.concatenate(out_planes), np.concatenate(out_policy), np.concatenate(out_value))
    stats = {'games': finished, 'positions': len(data[2]), 'x_wins': results[1],
             'o_wins': results[-1], 'ties': results[0]}
    return data, stats


# ----------------------------------------------------------------------------
# learning
# ----------------------------------------------------------------------------

def train(model, optimizer, buffer, device, steps):
    """Returns the average policy loss and value loss."""
    model.train()
    policy_losses, value_losses = [], []
    for _ in range(steps):
        planes, policy, value = buffer.sample(BATCH_SIZE, device)
        planes, policy = NN.random_symmetry(planes, policy)
        logits, predicted_value = model(planes)
        # only legal moves count (plane 2 marks them)
        logits = logits.masked_fill(planes[:, 2].reshape(-1, 81) == 0, -1e9)
        policy_loss = -(policy * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(predicted_value, value)
        optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        optimizer.step()
        policy_losses.append(policy_loss.detach())
        value_losses.append(value_loss.detach())
    model.eval()
    return float(torch.stack(policy_losses).mean()), float(torch.stack(value_losses).mean())


# ----------------------------------------------------------------------------
# progress check
# ----------------------------------------------------------------------------

def evaluate(model, device, depth, games=EVAL_GAMES, simulations=EVAL_SIMULATIONS):
    """The network (with tree search) against minimax of the given depth.
    Half of the games as X, half as O, each after 2 random opening moves.
    Returns (wins, ties, losses) of the network."""
    model.eval()
    search = MCTS.BatchedSearch(model, device, games, simulations)
    boards = [game.Board(playerturn=1) for _ in range(games)]
    net_side = [1 if g % 2 == 0 else -1 for g in range(games)]
    for board in boards:
        for _ in range(2):
            moves = board.possible_moves()
            board.execute(moves[np.random.randint(len(moves))])
    while not all(board.game_is_over for board in boards):
        active = np.array([not board.game_is_over and board.playerturn == net_side[g]
                           for g, board in enumerate(boards)])
        if active.any():
            for g in np.flatnonzero(active):
                search.set_board(g, boards[g])
            counts, _ = search.search(active)
            for g in np.flatnonzero(active):
                boards[g].execute(int(np.argmax(counts[g])))
        for g, board in enumerate(boards):
            if not board.game_is_over and board.playerturn != net_side[g]:
                board.execute(fast_minimax.best_move(board, depth)[1])
    wins = ties = losses = 0
    for g, board in enumerate(boards):
        if not game.game_is_win(board.won_games)[0]:
            ties += 1
        elif -board.playerturn == net_side[g]:      # the player who moved last has won
            wins += 1
        else:
            losses += 1
    return wins, ties, losses


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

def paths(variant):
    base = NN.checkpoint_path(variant)[:-3]
    return base + '.pt', base + '_buffer.npz', base + '_log.csv'


def snapshot_path(variant, iteration):
    return os.path.join(NN.MODEL_DIR, 'snapshots', f'alphazero_{variant}_iteration{iteration:04d}.pt')


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
        if os.path.exists(buffer_file):
            buffer.load(buffer_file)
        print(f"Continuing '{MODEL}' network after iteration {iteration} "
              f"({total_games} games so far, {buffer.size} positions in memory) on {device}")
    else:
        parameters = sum(p.numel() for p in model.parameters())
        print(f"New '{MODEL}' network ({parameters / 1e6:.2f} million parameters) on {device}")

    def save(with_buffer):
        NN.save_checkpoint(checkpoint_file, model, optimizer, variant=MODEL,
                           iteration=iteration, total_games=total_games, warm_started=warm_started)
        if with_buffer:
            buffer.save(buffer_file)

    try:
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
            data, stats = self_play(model, device, GAMES_PER_ITERATION)
            play_time = time.perf_counter() - start
            buffer.add(*data)
            steps = max(1, REPLAY_RATIO * stats['positions'] // BATCH_SIZE)
            policy_loss, value_loss = train(model, optimizer, buffer, device, steps)
            iteration += 1
            total_games += stats['games']
            save(with_buffer=iteration % SAVE_BUFFER_EVERY == 0)
            if SNAPSHOT_EVERY and iteration % SNAPSHOT_EVERY == 0:
                os.makedirs(os.path.dirname(snapshot_path(MODEL, iteration)), exist_ok=True)
                shutil.copyfile(checkpoint_file, snapshot_path(MODEL, iteration))

            row = {'iteration': iteration, 'games': total_games, 'positions': stats['positions'],
                   'policy_loss': round(policy_loss, 4), 'value_loss': round(value_loss, 4),
                   'x_wins': stats['x_wins'], 'o_wins': stats['o_wins'], 'ties': stats['ties']}
            line = (f"Iteration {iteration}: {stats['games']} games, {stats['positions']} positions "
                    f"in {play_time:.0f}s | policy loss {policy_loss:.3f}, value loss {value_loss:.3f}")
            for depth in EVAL_DEPTHS:
                row.update(dict.fromkeys(test_columns(depth), ''))
            if iteration % EVAL_EVERY == 0:
                parts = []
                for depth in EVAL_DEPTHS:
                    result = evaluate(model, device, depth)
                    row.update(zip(test_columns(depth), (score(*result),) + result))
                    parts.append(f'depth {depth}: {score(*result):.0%} '
                                 f'({result[0]}-{result[1]}-{result[2]})')
                line += ' | against minimax (wins-ties-losses): ' + ', '.join(parts)
            row['seconds'] = round(time.perf_counter() - start)
            write_log(log_file, row)
            print(line, flush=True)
    except KeyboardInterrupt:
        print(f'\nStopped. Saving ... (the network is from iteration {iteration})')
        save(with_buffer=True)
        print(f'Saved to {checkpoint_file}. Start the script again to continue.')


if __name__ == '__main__':
    main()
