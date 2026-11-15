"""Small checks to run before a long training, for example on Google Colab.

    python3 src/pretest.py speed       how long a training iteration takes on this machine (about 5 minutes)
    python3 src/pretest.py speed xlarge    the same for another network size than the one in train.py
    python3 src/pretest.py bottleneck  what limits the speed of self-play: the processor (search tree) or
                                       the graphics card (network)? (about 3 minutes; can run between two
                                       training iterations)
    python3 src/pretest.py card        ways to run the same network faster on the graphics card, and where
                                       the time of one question to the network goes (about 3 minutes)
    python3 src/pretest.py resume      does saving and continuing work? run it twice (2-3 minutes each)
    python3 src/pretest.py capacity    does a bigger network learn more from the same games? (20-40 minutes)
    python3 src/pretest.py precision   half against full precision with the trained network on its own
                                       saved positions (1-2 minutes, only meaningful on an Nvidia card)

All of them use the settings of train.py (which network, simulations, ...).
None of them touches a real training: 'resume' and 'capacity' write to the
folder  pretest/  inside the models folder.

'capacity' needs the collected positions and the network of a finished training
in the models folder (alphazero_medium_buffer.npz and alphazero_medium.pt).
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import MCTS
import Tic_Tac_Toe_NN as NN
import train

# ------------------------------- Settings -------------------------------
# speed
SPEED_PARALLEL_GAMES = (256, 512)   # values of PARALLEL_GAMES that are compared
SPEED_MOVES = 4                     # every game is played for this many moves
POSITIONS_PER_ITERATION = 26700     # 512 games of about 52 moves (from the log of the medium network)
ITERATIONS_PLANNED = 250            # for the estimate of the whole training
# Compute units per hour on Google Colab, measured by users in 2026 (they change; see Colab's "Resources")
COLAB_UNITS_PER_HOUR = {'T4': 1.2, 'L4': 1.7, 'A100': 5.4}

# bottleneck
BOTTLENECK_PARALLEL_GAMES = (256, 512, 1024, 2048)
BOTTLENECK_MOVES = 3                # every game is played for this many moves

# card
CARD_BATCHES = (512, 1024, 2048)
CARD_SECONDS = 2.0                  # how long every way is measured per batch size

# resume
RESUME_ITERATIONS = 2               # iterations per run
RESUME_GAMES = 32
RESUME_SIMULATIONS = 50

# capacity
TEACHER = 'medium'                  # the finished network whose games are the lessons
STUDENTS = ('medium', 'large')      # new networks of these sizes learn from those games
CAPACITY_STEPS = 6000               # learning steps per network (6000 = every position about 3 times)
HELD_OUT = 0.1                      # the newest games are not learned from, only tested on
MATCH_GAMES = 64                    # games of the last student against the teacher afterwards (0 = none)
MATCH_SIMULATIONS = 200
# ------------------------------------------------------------------------


def machine(device):
    name = torch.cuda.get_device_name(0) if device.type == 'cuda' else str(device)
    return f'{name}, {os.cpu_count()} processor cores'


# ----------------------------------------------------------------------------
# speed
# ----------------------------------------------------------------------------

def _network_speed(model, device, batch, half, seconds=3.0):
    """Positions per second when the network is asked with batches of this size."""
    planes = np.random.rand(batch, NN.INPUT_PLANES, 9, 9).astype(np.float32)
    NN.HALF_PRECISION = half
    for _ in range(3):
        NN.predict(model, planes, device)
    start, calls = time.perf_counter(), 0
    while time.perf_counter() - start < seconds:
        NN.predict(model, planes, device)
        calls += 1
    return calls * batch / (time.perf_counter() - start)


def _real_positions(model, device, games=64, moves=12):
    """Positions from a few quickly played games, to compare the two precisions on."""
    search = MCTS.BatchedSearch(model, device, games, 20, noise_weight=0.25)
    active, played, planes = np.ones(games, dtype=np.bool_), np.zeros(games, dtype=np.int64), []
    for _ in range(moves):
        search.search(active)
        planes.append(search.encode_roots().copy())
        search.play(active, search.choose_moves(active, played, moves))
        played += 1
    return np.concatenate(planes)


def _search_speed(model, device, parallel, simulations):
    """Seconds per move of one game during self-play, and the share of that time spent in the network."""
    spent = [0.0]
    original = MCTS.predict

    def timed(*args):
        start = time.perf_counter()
        result = original(*args)
        spent[0] += time.perf_counter() - start
        return result

    search = MCTS.BatchedSearch(model, device, parallel, simulations,
                                noise_weight=train.NOISE_WEIGHT, noise_alpha=train.NOISE_ALPHA)
    active, played = np.ones(parallel, dtype=np.bool_), np.zeros(parallel, dtype=np.int64)
    search.search(active)                               # warm up (compiles the search on the first call)
    MCTS.predict = timed
    try:
        start = time.perf_counter()
        for _ in range(SPEED_MOVES):
            search.search(active)
            search.play(active, search.choose_moves(active, played, train.TEMPERATURE_MOVES))
            played += 1
        total = time.perf_counter() - start
    finally:
        MCTS.predict = original
    return total / (parallel * SPEED_MOVES), spent[0] / total


def _learning_speed(model, device, steps=40):
    """Seconds per learning step."""
    buffer = train.ReplayBuffer(4096)
    planes = (np.random.rand(4096, NN.INPUT_PLANES, 9, 9) < 0.3).astype(np.uint8)
    policy = np.random.dirichlet(np.ones(81), 4096).astype(np.float32)
    buffer.add(planes, policy, np.random.uniform(-1, 1, 4096).astype(np.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    train.train(model, optimizer, buffer, device, 5)
    start = time.perf_counter()
    train.train(model, optimizer, buffer, device, steps)
    return (time.perf_counter() - start) / steps


def speed(variant=None):
    variant = variant or train.MODEL
    device = NN.choose_device(train.DEVICE)
    model = NN.create_model(variant).to(device).eval()
    cuda = device.type == 'cuda'
    print(f"Speed of the '{variant}' network on {machine(device)}\n")

    print('1. The network alone (positions per second)')
    for batch in SPEED_PARALLEL_GAMES:
        full = _network_speed(model, device, batch, half=False)
        line = f'   {batch} positions at once: full precision {full:,.0f}'
        if cuda:
            half = _network_speed(model, device, batch, half=True)
            line += f', half precision {half:,.0f} ({half / full:.1f} times as fast)'
        print(line, flush=True)

    if cuda:
        planes = _real_positions(model, device).astype(np.float32)
        NN.HALF_PRECISION = False
        logits, values = NN.predict(model, planes, device)
        NN.HALF_PRECISION = True
        half_logits, half_values = NN.predict(model, planes, device)
        legal = planes[:, 2].reshape(-1, 81) > 0
        def moves(x):
            return torch.softmax(torch.from_numpy(np.where(legal, x, -1e9)), dim=1).numpy()
        difference = np.abs(moves(logits) - moves(half_logits)).max()
        same = (np.where(legal, logits, -1e9).argmax(1) == np.where(legal, half_logits, -1e9).argmax(1)).mean()
        ok = np.isfinite(half_logits).all() and np.isfinite(half_values).all() and difference < 0.02
        print(f'\n2. Half precision against full precision on {len(planes)} positions')
        print(f'   largest difference of a move probability {difference:.4f}, of a value '
              f'{np.abs(values - half_values).max():.4f}; same favourite move in {same:.1%} of the positions')
        print('   -> ' + ('fine, the differences are tiny.' if ok else
                          'NOT fine: set HALF_PRECISION = False in src/Tic_Tac_Toe_NN.py.'), flush=True)
        NN.HALF_PRECISION = bool(ok)
    else:
        print('\n2. Half precision is only used on Nvidia cards; nothing to check here.')

    sizes = sorted({train.simulations(i) for i in range(ITERATIONS_PLANNED)})
    print(f'\n3. Self-play with {sizes[0]} simulations per move (the first {SPEED_MOVES} moves of every game)')
    per_move = {}
    for parallel in SPEED_PARALLEL_GAMES:
        per_move[parallel], share = _search_speed(model, device, parallel, sizes[0])
        print(f'   {parallel} games at once: {1000 * per_move[parallel]:.1f} ms per move of a game, '
              f'{share:.0%} of the time is the network', flush=True)
    best = min(per_move, key=per_move.get)
    step = _learning_speed(model, device)
    learning = train.REPLAY_RATIO * POSITIONS_PER_ITERATION / train.BATCH_SIZE * step
    print(f'\n4. Learning: {1000 * step:.0f} ms per step, about {learning:.0f} seconds per iteration')

    def minutes(simulations):
        return (per_move[best] * simulations / sizes[0] * POSITIONS_PER_ITERATION + learning) / 60

    hours = sum(minutes(train.simulations(i)) for i in range(ITERATIONS_PLANNED)) / 60
    print(f'\nEstimate with PARALLEL_GAMES = {best}'
          + ('' if best == train.PARALLEL_GAMES else f' (train.py has {train.PARALLEL_GAMES}: change it)') + ':')
    for simulations in sizes:
        print(f'   an iteration with {simulations} simulations: about {minutes(simulations):.0f} minutes')
    print(f'   {ITERATIONS_PLANNED} iterations as planned in train.py: about {hours:.0f} hours')
    for card, units in COLAB_UNITS_PER_HOUR.items():
        if cuda and card in torch.cuda.get_device_name(0):
            print(f'   on Colab with about {units} compute units per hour: about {hours * units:.0f} units '
                  f'(100 units cost about 10 CHF)')
    print('(Checked on a Mac: there this estimate was within 5% of the real iterations of the medium network.)')


# ----------------------------------------------------------------------------
# bottleneck
# ----------------------------------------------------------------------------

class _CardWatch:
    """Asks nvidia-smi twice a second how busy the card is (percent of the time it is computing)."""

    def __enter__(self):
        import subprocess
        self.busy, self.process = [], None
        try:
            self.process = subprocess.Popen(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', '-lms', '500'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            pass
        return self

    def __exit__(self, *error):
        if self.process:
            self.process.terminate()
            self.busy = [int(line) for line in self.process.communicate()[0].split() if line.isdigit()]

    def text(self):
        return f', card busy {np.mean(self.busy):.0f}% of the time' if self.busy else ''


def bottleneck(variant=None):
    variant = variant or train.MODEL
    device = NN.choose_device(train.DEVICE)
    trained = os.path.exists(NN.checkpoint_path(variant))
    model = NN.load_model(variant, device) if trained else NN.create_model(variant).to(device).eval()
    simulations = max(train.simulations(i) for i in range(train.ITERATIONS))
    cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    print(f"What limits self-play? '{variant}' network ({'trained' if trained else 'untrained'}) on "
          f'{machine(device)} ({cores} usable), {simulations} simulations per move')
    print('The search tree runs on ONE processor core; the network runs on the graphics card.\n')
    print('1. The network alone')
    for batch in BOTTLENECK_PARALLEL_GAMES:
        with _CardWatch() as watch:
            rate = _network_speed(model, device, batch, half=NN.HALF_PRECISION and device.type == 'cuda')
        print(f'   {batch} positions at once: {rate:,.0f} positions per second{watch.text()}', flush=True)
    print(f'\n2. Self-play (the first {BOTTLENECK_MOVES} moves of every game)')
    global SPEED_MOVES
    SPEED_MOVES, first = BOTTLENECK_MOVES, None
    for parallel in BOTTLENECK_PARALLEL_GAMES:
        with _CardWatch() as watch:
            per_move, share = _search_speed(model, device, parallel, simulations)
        first = first or per_move
        print(f'   {parallel} games at once: {1000 * per_move:.1f} ms per move of a game '
              f'({first / per_move:.2f} times the speed of {BOTTLENECK_PARALLEL_GAMES[0]}) | '
              f'search tree (processor) {1 - share:.0%}, network (card, with copying) {share:.0%}{watch.text()}',
              flush=True)
    print('\nSend these lines back. If the processor share is high, spreading the search tree over all '
          'cores helps most;\nif the network share is high, more games at once help.')


# ----------------------------------------------------------------------------
# card
# ----------------------------------------------------------------------------

def _wait(device):
    """Waits until the card has finished everything it was asked to do."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def _folded(model):
    """The same network with the normalisation layers worked into the filters before them
    (the same answers, fewer layers; only possible for playing, not for learning)."""
    import copy
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    model = copy.deepcopy(model).eval()
    for block in model.blocks:
        block.conv1, block.bn1 = fuse_conv_bn_eval(block.conv1, block.bn1), torch.nn.Identity()
        block.conv2, block.bn2 = fuse_conv_bn_eval(block.conv2, block.bn2), torch.nn.Identity()
    model.policy_conv, model.policy_bn = fuse_conv_bn_eval(model.policy_conv, model.policy_bn), torch.nn.Identity()
    model.value_conv, model.value_bn = fuse_conv_bn_eval(model.value_conv, model.value_bn), torch.nn.Identity()
    return model


def _ask(model, x, precision):
    """precision: 'full', 'autocast' (as the search does it now) or 'half' (the whole network in half precision)."""
    with torch.inference_mode():
        if precision == 'autocast':
            with torch.autocast(x.device.type, dtype=torch.float16, enabled=x.device.type == 'cuda'):
                return model(x)
        return model(x.half() if precision == 'half' else x)


def _card_speed(model, x, precision):
    device = x.device
    for _ in range(5):
        _ask(model, x, precision)
    _wait(device)
    start, calls = time.perf_counter(), 0
    while time.perf_counter() - start < CARD_SECONDS:
        _ask(model, x, precision)
        calls += 1
    _wait(device)
    return calls * len(x) / (time.perf_counter() - start)


def card(variant=None):
    import copy
    variant = variant or train.MODEL
    device = NN.choose_device(train.DEVICE)
    trained = os.path.exists(NN.checkpoint_path(variant))
    model = NN.load_model(variant, device) if trained else NN.create_model(variant).to(device).eval()
    print(f"Ways to run the '{variant}' network ({'trained' if trained else 'untrained'}) on {machine(device)}, "
          f'torch {torch.__version__}\n')
    real = _real_positions(model, device).astype(np.float32)
    positions = {batch: np.ascontiguousarray(np.resize(real, (batch,) + real.shape[1:])) for batch in CARD_BATCHES}

    print('1. Where the time of one question goes now (milliseconds)')
    for batch, planes in positions.items():
        parts = np.zeros(4)
        for _ in range(3):
            NN.predict(model, planes, device)
        for _ in range(20):
            _wait(device)
            t0 = time.perf_counter()
            x = torch.from_numpy(planes).to(device)
            _wait(device)
            t1 = time.perf_counter()
            logits, values = _ask(model, x, 'autocast')
            _wait(device)
            t2 = time.perf_counter()
            logits, values = logits.float().cpu().numpy(), values.float().cpu().numpy()
            t3 = time.perf_counter()
            broken = int((~(np.isfinite(logits).all(axis=1) & np.isfinite(values))).sum())
            NN.predict(model, planes, device)
            parts += (t1 - t0, t2 - t1, t3 - t2, time.perf_counter() - t3)
        to_card, network, back, whole = 1000 * parts / 20
        print(f'   {batch} positions: to the card {to_card:.2f}, network {network:.2f}, back {back:.2f} | the whole '
              f'question as the search asks it {whole:.2f} = {1000 * batch / whole:,.0f} positions per second'
              + (f' | {broken} positions needed full precision' if broken else ''), flush=True)

    x = {batch: torch.from_numpy(planes).to(device) for batch, planes in positions.items()}
    legal = x[CARD_BATCHES[0]][:, 2].reshape(-1, 81) > 0

    def moves(logits):
        return torch.softmax(logits.float().masked_fill(~legal, -1e9), dim=1)

    reference = moves(_ask(model, x[CARD_BATCHES[0]], 'full')[0])
    fast = torch.backends.cudnn

    def layout(net):
        return copy.deepcopy(net).to(memory_format=torch.channels_last)

    ways = [
        ('full precision', lambda: model, 'full', False),
        ('as now (half precision, switched per layer)', lambda: model, 'autocast', False),
        ('+ the library tries out the fastest method', lambda: model, 'autocast', True),
        ('+ other memory layout', lambda: layout(model), 'autocast', True),
        ('+ normalisation worked into the filters', lambda: _folded(model), 'autocast', True),
        ('+ whole network in half precision', lambda: copy.deepcopy(model).half(), 'half', True),
        ('all of these together', lambda: layout(_folded(model)).half(), 'half', True),
        ('all together, compiled', lambda: torch.compile(layout(_folded(model)).half()), 'half', True),
    ]
    print(f'\n2. The network alone on the card, positions per second at {", ".join(map(str, CARD_BATCHES))} at once '
          '(every "+" line is "as now" with that one thing added)')
    for name, build, precision, tuned in ways:
        fast.benchmark = tuned
        try:
            net = build()
            rates = [_card_speed(net, x[batch], precision) for batch in CARD_BATCHES]
            answer = moves(_ask(net, x[CARD_BATCHES[0]], precision)[0])
            difference = float((answer - reference).abs().max())
            line = ' | '.join(f'{rate:,.0f}' for rate in rates) + f' | largest difference of a move probability {difference:.4f}'
            if not bool(torch.isfinite(answer).all()):
                line += ' (some answers are not a number)'
        except Exception as error:
            line = f'did not work here: {type(error).__name__}: {str(error)[:120]}'
        print(f'   {name}: {line}', flush=True)
    fast.benchmark = False
    print('\nSend these lines back.')


# ----------------------------------------------------------------------------
# resume
# ----------------------------------------------------------------------------

def resume():
    NN.MODEL_DIR = os.path.join(NN.MODEL_DIR, 'pretest')
    train.GAMES_PER_ITERATION = train.PARALLEL_GAMES = RESUME_GAMES
    train.SIMULATIONS, train.SIMULATIONS_GROWTH = RESUME_SIMULATIONS, {}
    train.SNAPSHOT_EVERY = train.SAVE_BUFFER_EVERY = 1
    train.EVAL_EVERY = 0
    checkpoint = NN.checkpoint_path(train.MODEL)
    before = NN.saved_iteration(train.MODEL) if os.path.exists(checkpoint) else 0
    train.ITERATIONS = before + RESUME_ITERATIONS
    print(f'Test folder: {NN.MODEL_DIR}')
    train.main()
    after = NN.saved_iteration(train.MODEL)
    files = sorted(os.listdir(NN.MODEL_DIR)) + sorted(os.listdir(os.path.join(NN.MODEL_DIR, 'snapshots')))
    print(f'\nSaved: {", ".join(files)}')
    if before == 0:
        print(f'First run: iterations 1 to {after} are saved. Now end the session '
              f'(Colab: Runtime > Disconnect and delete runtime), start again and run this test once more. '
              f"It should then say \"Continuing '{train.MODEL}' network after iteration {after}\".")
    else:
        print(f'It continued after iteration {before} and is now at iteration {after}: saving and continuing '
              f'works. You can delete the folder {NN.MODEL_DIR}.')


# ----------------------------------------------------------------------------
# capacity
# ----------------------------------------------------------------------------

def _load_positions(path):
    """The saved positions, oldest first."""
    data = np.load(path)
    planes, policy, value = data['planes'], data['policy'], data['value']
    start = int(data['position']) % len(value)          # in a full memory the oldest position is here
    order = (start + np.arange(len(value))) % len(value)
    return planes[order], policy[order], value[order]


def _test_losses(model, device, planes, policy, value):
    """On positions the network has not learned from: how far its move probabilities are from the
    ones of the search (0 = the same), and the error of its value."""
    model.eval()
    distance, error = 0.0, 0.0
    with torch.inference_mode():
        for start in range(0, len(value), 2048):
            chunk = slice(start, start + 2048)
            x = torch.from_numpy(planes[chunk]).to(device).float()
            target = torch.from_numpy(policy[chunk].astype(np.float32)).to(device)
            logits, predicted = model(x)
            logits = logits.float().masked_fill(x[:, 2].reshape(-1, 81) == 0, -1e9)
            entropy = -(target * torch.log(target.clamp_min(1e-9))).sum(dim=1)
            distance += float((-(target * F.log_softmax(logits, dim=1)).sum(dim=1) - entropy).sum())
            error += float(F.mse_loss(predicted.float(), torch.from_numpy(value[chunk]).to(device),
                                      reduction='sum'))
    return distance / len(value), error / len(value)


def capacity(steps=CAPACITY_STEPS):
    import test_networks

    device = NN.choose_device(train.DEVICE)
    buffer_file = train.paths(TEACHER)[1]
    if not os.path.exists(buffer_file):
        raise SystemExit(f'This test needs the collected positions of the {TEACHER} network: {buffer_file}')
    planes, policy, value = _load_positions(buffer_file)
    split = int(len(value) * (1 - HELD_OUT))
    test = (planes[split:], policy[split:], value[split:])
    memory = train.ReplayBuffer(split)
    memory.add(planes[:split], policy[:split], value[:split])
    print(f"Lessons: {split} positions from the games of the '{TEACHER}' network; "
          f'tested on its {len(value) - split} newest positions, on {machine(device)}')
    print('distance = how far the move probabilities are from the ones of the search (smaller is better)\n')

    results = {}
    if os.path.exists(NN.checkpoint_path(TEACHER)):
        teacher = NN.load_model(TEACHER, device)
        results[f'{TEACHER} after its whole training (has learned from the test positions too)'] = \
            _test_losses(teacher, device, *test)

    for variant in STUDENTS:
        torch.manual_seed(0)
        np.random.seed(0)
        model = NN.create_model(variant).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=train.LEARNING_RATE,
                                      weight_decay=train.WEIGHT_DECAY)
        start, done = time.perf_counter(), 0
        while done < steps:
            part = min(1000, steps - done)
            for group in optimizer.param_groups:                    # smaller steps for the last third
                group['lr'] = train.LEARNING_RATE if done < 2 * steps // 3 else train.LEARNING_RATE * 0.3
            policy_loss, value_loss = train.train(model, optimizer, memory, device, part)
            done += part
            distance, error = _test_losses(model, device, *test)
            print(f"new '{variant}' network, step {done}: learned positions: policy loss {policy_loss:.3f}, "
                  f'value loss {value_loss:.3f} | test positions: distance {distance:.3f}, '
                  f'value error {error:.3f} | {time.perf_counter() - start:.0f}s', flush=True)
        results[f'new {variant} network, {steps} steps on these lessons'] = (distance, error)
        path = os.path.join(NN.MODEL_DIR, 'pretest', f'alphazero_{variant}_from_{TEACHER}_games.pt')
        NN.save_checkpoint(path, model, variant=variant, iteration=0, total_games=0, warm_started=True)
        print()

    print('On the test positions:')
    for name, (distance, error) in results.items():
        print(f'   {name}: distance {distance:.3f}, value error {error:.3f}')

    if MATCH_GAMES and os.path.exists(NN.checkpoint_path(TEACHER)):
        model.eval()
        openings, _ = test_networks.openings_to_play(TEACHER)
        games = [(opening, side) for opening in openings[:MATCH_GAMES // 2] for side in (1, -1)]
        search_a = MCTS.BatchedSearch(model, device, len(games), MATCH_SIMULATIONS)
        search_b = MCTS.BatchedSearch(teacher, device, len(games), MATCH_SIMULATIONS)
        outcome = [result for result, _ in test_networks.play_batch(search_a, search_b, games)]
        wins, ties, losses = (outcome.count(r) for r in test_networks.RESULTS)
        print(f"\nThe new '{variant}' network against the finished '{TEACHER}' network, {len(games)} games "
              f'with {MATCH_SIMULATIONS} simulations per move: {wins} wins, {ties} ties, {losses} losses '
              f'= {(wins + 0.5 * ties) / len(games):.0%} of the points')
        print(f'It is saved as {path}')


def precision():
    """Asks the trained network about all its saved positions in half and in full precision."""
    device = NN.choose_device(train.DEVICE)
    model = NN.load_model(train.MODEL, device)
    planes = train.load_positions(train.paths(train.MODEL)[1])[0]
    print(f"'{train.MODEL}' network after iteration {NN.saved_iteration(train.MODEL)} on {len(planes)} saved "
          f'positions, {machine(device)}')
    if device.type != 'cuda':
        print('Half precision is only used on Nvidia cards; nothing to compare here.')
        return
    broken, worst_move, worst_value = 0, 0.0, 0.0
    for start in range(0, len(planes), 2048):
        x = torch.from_numpy(planes[start:start + 2048]).to(device).float()
        legal = x[:, 2].reshape(-1, 81) > 0
        with torch.inference_mode():
            logits, values = model(x)
            with torch.autocast('cuda', dtype=torch.float16):
                half_logits, half_values = model(x)
        half_logits, half_values = half_logits.float(), half_values.float()
        fine = torch.isfinite(half_logits).all(dim=1) & torch.isfinite(half_values)
        broken += int((~fine).sum())
        if fine.any():
            def moves(z):
                return torch.softmax(z.masked_fill(~legal, -1e9), dim=1)
            worst_move = max(worst_move, float((moves(logits) - moves(half_logits))[fine].abs().max()))
            worst_value = max(worst_value, float((values - half_values)[fine].abs().max()))
    print(f'Positions where half precision gives no number at all: {broken}')
    print(f'On the others: largest difference of a move probability {worst_move:.4f}, of a value {worst_value:.4f}')
    print('-> ' + ('This is what damaged the training. The search now asks again in full precision in such '
                   'a case.' if broken else 'No such position here.'))


if __name__ == '__main__':
    tests = {'speed': speed, 'bottleneck': bottleneck, 'card': card, 'resume': resume, 'capacity': capacity, 'precision': precision}
    if len(sys.argv) < 2 or sys.argv[1] not in tests:
        raise SystemExit(__doc__)
    if sys.argv[1] == 'capacity' and len(sys.argv) > 2:
        capacity(int(sys.argv[2]))
    elif sys.argv[1] in ('speed', 'bottleneck', 'card') and len(sys.argv) > 2:
        tests[sys.argv[1]](sys.argv[2])
    else:
        tests[sys.argv[1]]()
