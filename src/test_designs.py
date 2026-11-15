"""Design tests: which way of building the network learns the game best from the same lessons?

Every design is a new network that learns from the saved positions of the finished large network
(the newest 10% are held back and only tested on), with the same number of learning steps. Measured:
how far its move probabilities are from the ones of the search on the held-back positions, the error
of its value, how many positions per second it answers, and a match against today's design.

    python3 src/test_designs.py speed            how fast every design is (a few minutes)
    python3 src/test_designs.py train            trains what is missing of PLAN
    python3 src/test_designs.py train D1 D2      trains only these (a second number is the seed: D0:1)
    python3 src/test_designs.py match            plays what is missing: every design against D0
    python3 src/test_designs.py summary          the table of everything measured so far

All designs take the same input and give the same answers as the networks in Tic_Tac_Toe_NN.py, so
they work in the search unchanged. Nothing here changes the training of the real networks.
"""
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import Tic_Tac_Toe_NN as NN

LESSONS = 'large'               # the network whose saved positions are the lessons
SIZE = 'medium'                 # channels and blocks of every design (see NN.MODEL_VARIANTS)
STEPS = 6000                    # learning steps per design, 256 positions each (about 3 times every position)
HELD_OUT = 0.1                  # the newest positions are not learned from, only tested on
CHECK_EVERY = 1000              # steps between two measurements on the held-back positions
PLAN = ['D0', 'D0:1', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
MATCH_AGAINST = 'D0'            # every other design plays against this one
MATCH_OPENINGS = 128            # 2 games per opening
MATCH_SIMULATIONS = 200
PARALLEL_GAMES = 128

DESIGNS = {
    'D0': "today's design",
    'D1': 'filters stay inside the small board',
    'D2': 'small-board summary by a learned 3x3 step',
    'D3': 'permanent board track, value read from it',
    'D4': 'reverse of the sending rule',
    'D5': 'line layers (small boards and big board)',
    'D6': 'exact facts as input',
    'D7': 'whole body on the 3x3 big board',
}

# ----------------------------------------------------------------------------
# fixed tables of the game's structure (cells are numbered as in the 9x9 picture)
# ----------------------------------------------------------------------------
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def _cell(board, cell):
    return (3 * (board // 3) + cell // 3) * 9 + 3 * (board % 3) + cell % 3


def _tables():
    own, target = torch.zeros(9, 81), torch.zeros(9, 81)        # small board -> its cells / the cells that send into it
    cell_lines, board_lines = torch.zeros(81, 72), torch.zeros(9, 8)
    for board in range(9):
        for cell in range(9):
            own[board, _cell(board, cell)] = 1
            target[cell, _cell(board, cell)] = 1
        for number, line in enumerate(LINES):
            for cell in line:
                cell_lines[_cell(board, cell), board * 8 + number] = 1
    for number, line in enumerate(LINES):
        for board in line:
            board_lines[board, number] = 1
    return own, target, cell_lines, board_lines


class Structure(nn.Module):
    """Holds the fixed tables, so that they move to the graphics chip with the network."""

    def __init__(self):
        super().__init__()
        for name, table in zip(('own_cells', 'target_cells', 'cell_lines', 'board_lines'), _tables()):
            self.register_buffer(name, table, persistent=False)


def to_boards(x):
    """batch x C x 9 x 9  ->  (batch * 9) x C x 3 x 3: every small board on its own."""
    batch, channels = x.shape[:2]
    return x.view(batch, channels, 3, 3, 3, 3).permute(0, 2, 4, 1, 3, 5).reshape(batch * 9, channels, 3, 3)


def from_boards(x, batch):
    channels = x.shape[1]
    return x.view(batch, 3, 3, channels, 3, 3).permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, 9, 9)


class Filter(nn.Module):
    """A 3x3 filter on the 9x9 picture. inside=True: it never looks across a small-board border."""

    def __init__(self, inputs, outputs, inside):
        super().__init__()
        self.conv = nn.Conv2d(inputs, outputs, 3, padding=1, bias=False)
        self.inside = inside

    def forward(self, x):
        if not self.inside:
            return self.conv(x)
        return from_boards(self.conv(to_boards(x)), x.shape[0])


# ----------------------------------------------------------------------------
# D6: exact facts, worked out from the usual 6 input planes
# ----------------------------------------------------------------------------
def facts(x, s):
    """6 planes -> 14 planes. Added: my and the other's stones in boards that are still open, cells
    that win a small board at once (for me / for the other), small boards whose win would win the
    game (me / other), whether I may choose the board now, and cells that send to a finished board."""
    batch = x.shape[0]
    mine, theirs = x[:, 0].flatten(1), x[:, 1].flatten(1)
    decided = (x[:, 3] + x[:, 4] + x[:, 5]).flatten(1)
    open_cell = 1 - decided
    empty = (1 - mine - theirs) * open_cell
    decided_board = F.avg_pool2d(x[:, 3:6].sum(1, keepdim=True), 3).flatten(1)          # batch x 9
    extra = [mine * open_cell, theirs * open_cell]
    for stones in (mine, theirs):
        two = ((stones * open_cell) @ s.cell_lines > 1.5).float()
        extra.append((two @ s.cell_lines.t() > 0.5).float() * empty)
    for plane in (3, 4):
        won = F.avg_pool2d(x[:, plane:plane + 1], 3).flatten(1)
        two = (won @ s.board_lines > 1.5).float()
        key = (two @ s.board_lines.t() > 0.5).float() * (1 - decided_board)
        extra.append(key @ s.own_cells)
    playable_boards = F.max_pool2d(x[:, 2:3], 3).flatten(1).sum(1, keepdim=True)
    extra.append((playable_boards > 1.5).float().expand(batch, 81))
    extra.append(decided_board @ s.target_cells)
    return torch.cat([x, torch.stack(extra, dim=1).view(batch, len(extra), 9, 9)], dim=1)


# ----------------------------------------------------------------------------
# D0 to D6: today's block with single parts changed
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, channels, context, s, inside, learned, track, reverse, lines):
        super().__init__()
        self.s, self.learned, self.track, self.reverse, self.lines = s, learned, track, reverse, lines
        self.conv1 = Filter(channels, channels, inside)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = Filter(channels, channels, inside)
        self.bn2 = nn.BatchNorm2d(channels)
        width = channels if track else context              # channels of one small-board description
        pooled = width if (learned or track) else channels
        if learned or track:
            self.down = nn.Conv2d(channels, width, 3, stride=3)     # one step per small board
        if reverse:
            self.back = nn.Conv2d(channels, pooled, 1, bias=False)
        if track:
            self.keep = nn.Conv2d(width, width, 1, bias=False)
            self.big = nn.Conv2d(width, width, 3, padding=1, bias=False)
            self.bn_big = nn.BatchNorm2d(width)
        else:
            self.context = nn.Conv2d(pooled, width, 3, padding=1)   # works on the 3x3 big board
        self.own = nn.Conv2d(width, channels, 1, bias=False)
        self.target = nn.Conv2d(width, channels, 1, bias=False)
        if lines:
            self.line_in = nn.Conv2d(channels, context, 1)
            self.line_mix = nn.Conv1d(context, context, 1)
            self.line_out = nn.Conv1d(context, channels, 1, bias=False)
            self.big_line_mix = nn.Conv1d(width, context, 1)
            self.big_line_out = nn.Conv1d(context, channels, 1, bias=False)

    def forward(self, x, boards):
        s, batch = self.s, x.shape[0]
        h = F.relu(self.bn1(self.conv1(x)))
        pooled = self.down(h) if (self.learned or self.track) else F.avg_pool2d(h, 3)
        if self.reverse:        # every small board hears from the 9 cells that send into it
            senders = (h.flatten(2) @ s.target_cells.t() / 9).view(batch, -1, 3, 3)
            pooled = pooled + self.back(senders)
        if self.track:
            boards = F.relu(boards + self.bn_big(self.big(F.relu(pooled + self.keep(boards)))))
            big = boards
        else:
            big = F.relu(self.context(F.relu(pooled) if self.learned else pooled))
        summary = self.own(big).flatten(2) @ s.own_cells + self.target(big).flatten(2) @ s.target_cells
        if self.lines:
            small = F.relu(self.line_in(h)).flatten(2) @ s.cell_lines               # batch x context x 72 lines
            summary = summary + self.line_out(F.relu(self.line_mix(small)) @ s.cell_lines.t())
            large = F.relu(self.big_line_mix(big.flatten(2) @ s.board_lines))       # batch x context x 8 lines
            summary = summary + self.big_line_out(large @ s.board_lines.t()) @ s.own_cells
        h = self.bn2(self.conv2(h + summary.view(x.shape)))
        return F.relu(x + h), boards


class DesignNet(nn.Module):
    def __init__(self, channels, blocks, inside=False, learned=False, track=False, reverse=False,
                 lines=False, exact=False):
        super().__init__()
        self.s, self.track, self.exact, self.channels = Structure(), track, exact, channels
        self.stem = Filter(14 if exact else NN.INPUT_PLANES, channels, inside)
        self.position = nn.Parameter(torch.zeros(1, channels, 9, 9))
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([Block(channels, channels // 2, self.s, inside, learned, track, reverse, lines)
                                     for _ in range(blocks)])
        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_out = nn.Conv2d(32, 1, 1)
        if track:
            self.value_fc1 = nn.Linear(channels * 9, 128)
        else:
            self.value_conv = nn.Conv2d(channels, 8, 1, bias=False)
            self.value_bn = nn.BatchNorm2d(8)
            self.value_fc1 = nn.Linear(8 * 81, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        if self.exact:
            x = facts(x, self.s)
        x = F.relu(self.stem_bn(self.stem(x) + self.position))
        boards = x.new_zeros(x.shape[0], self.channels, 3, 3) if self.track else None
        for block in self.blocks:
            x, boards = block(x, boards)
        policy = self.policy_out(F.relu(self.policy_bn(self.policy_conv(x)))).flatten(1)
        if self.track:
            value = boards.flatten(1)
        else:
            value = F.relu(self.value_bn(self.value_conv(x))).flatten(1)
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(value)))).squeeze(1)
        return policy, value


# ----------------------------------------------------------------------------
# D7: the whole body works on the 3x3 big board (the way uttt.ai does it)
# ----------------------------------------------------------------------------
class BigBoardNet(nn.Module):
    def __init__(self, channels, blocks):
        super().__init__()
        width = 2 * channels                                # one small board is described by this many numbers
        self.s = Structure()
        self.stem = nn.Conv2d(NN.INPUT_PLANES, width, 3, stride=3, bias=False)
        self.stem_bn = nn.BatchNorm2d(width)
        self.convs = nn.ModuleList([nn.Conv2d(width, width, 3, padding=1, bias=False) for _ in range(2 * blocks)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(2 * blocks)])
        self.up = nn.ConvTranspose2d(width, 32, 3, stride=3)        # back to the 81 cells
        self.sent_to = nn.Conv2d(width, 32, 1, bias=False)          # what the board a move sends to looks like
        self.policy_conv = nn.Conv2d(32 + NN.INPUT_PLANES, 32, 1)
        self.policy_out = nn.Conv2d(32, 1, 1)
        self.value_fc1 = nn.Linear(width * 9, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        h = F.relu(self.stem_bn(self.stem(x)))
        for index in range(0, len(self.convs), 2):
            y = F.relu(self.bns[index](self.convs[index](h)))
            h = F.relu(h + self.bns[index + 1](self.convs[index + 1](y)))
        cells = self.up(h) + (self.sent_to(h).flatten(2) @ self.s.target_cells).view(-1, 32, 9, 9)
        policy = self.policy_out(F.relu(self.policy_conv(torch.cat([F.relu(cells), x], dim=1)))).flatten(1)
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(h.flatten(1))))).squeeze(1)
        return policy, value


def create(design, size=None):
    shape = NN.MODEL_VARIANTS[size or SIZE]
    if design == 'D0':
        return NN.UltimateTicTacToeNet(**shape)
    if design == 'D7':
        return BigBoardNet(**shape)
    changed = {'D1': 'inside', 'D2': 'learned', 'D3': 'track', 'D4': 'reverse', 'D5': 'lines', 'D6': 'exact'}
    if design not in changed:
        raise SystemExit(f"There is no design '{design}'. The designs are: {', '.join(DESIGNS)}.")
    return DesignNet(**shape, **{changed[design]: True})


# ----------------------------------------------------------------------------
# files
# ----------------------------------------------------------------------------
RESULT_FIELDS = ['design', 'size', 'seed', 'step', 'distance', 'value_error', 'policy_loss', 'value_loss',
                 'weights', 'seconds']
MATCH_FIELDS = ['a', 'b', 'simulations', 'a_plays', 'result', 'moves', 'opening']
POINTS = {'win': 1.0, 'tie': 0.5, 'loss': 0.0}


def result_file():
    return os.path.join(NN.MODEL_DIR, 'design_tests.csv')


def match_file():
    return os.path.join(NN.MODEL_DIR, 'design_matches.csv')


def network_file(design, seed):
    return os.path.join(NN.MODEL_DIR, 'pretest', f'design_{design}_{SIZE}_seed{seed}.pt')


def rows_of(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as file:
        return list(csv.DictReader(file))


def add_rows(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        if new:
            writer.writerow(fields)
        writer.writerows(rows)


def split_name(name):
    design, _, seed = name.partition(':')
    return design, int(seed or 0)


def load(name, device):
    design, seed = split_name(name)
    model = create(design)
    model.load_state_dict(torch.load(network_file(design, seed), map_location='cpu')['model'])
    return model.to(device).eval()


# ----------------------------------------------------------------------------
# the commands
# ----------------------------------------------------------------------------
def speed():
    import pretest
    device = NN.choose_device()
    print(f"Size '{SIZE}' on {pretest.machine(device)}; positions per second when asked 512 at a time, "
          f'and learning steps per second')
    for design, text in DESIGNS.items():
        model = create(design).to(device)
        weights = sum(p.numel() for p in model.parameters())
        model.eval()
        asked = pretest._network_speed(model, device, 512, False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        x = torch.rand(256, NN.INPUT_PLANES, 9, 9, device=device).round()
        target = torch.softmax(torch.rand(256, 81, device=device), dim=1)
        for step in range(25):
            if step == 5:
                start = time.perf_counter()
            logits, value = model(x)
            loss = -(target * F.log_softmax(logits, dim=1)).sum(1).mean() + value.pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        float(loss)
        steps = 20 / (time.perf_counter() - start)
        print(f'  {design} ({text}): {weights / 1e6:.2f} million weights, {asked:,.0f} positions/s, '
              f'{steps:.1f} steps/s = {STEPS / steps / 60:.0f} min for {STEPS} steps', flush=True)


def train(names):
    import pretest
    import train as trainer
    device = NN.choose_device()
    done = {(row['design'], int(row['seed'])) for row in rows_of(result_file())
            if row['size'] == SIZE and int(row['step']) == STEPS}
    todo = [name for name in names if split_name(name) not in done]
    if not todo:
        print('Everything asked for is trained already.')
        return
    planes, policy, value, _ = trainer.load_positions(trainer.paths(LESSONS)[1])
    split = int(len(value) * (1 - HELD_OUT))
    test = (planes[split:], policy[split:], value[split:])
    memory = trainer.ReplayBuffer(split)
    memory.add(planes[:split], policy[:split], value[:split])
    print(f"Lessons: {split} positions of the '{LESSONS}' network, tested on its {len(value) - split} newest; "
          f"size '{SIZE}', {STEPS} steps each, on {device}", flush=True)
    for name in todo:
        design, seed = split_name(name)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = create(design).to(device)
        weights = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=trainer.LEARNING_RATE,
                                      weight_decay=trainer.WEIGHT_DECAY)
        start, step, rows = time.perf_counter(), 0, []
        while step < STEPS:
            part = min(CHECK_EVERY, STEPS - step)
            for group in optimizer.param_groups:                    # smaller steps for the last third
                group['lr'] = trainer.LEARNING_RATE * (1 if step < 2 * STEPS // 3 else 0.3)
            policy_loss, value_loss = trainer.train(model, optimizer, memory, device, part)
            step += part
            used = time.perf_counter() - start
            distance, error = pretest._test_losses(model, device, *test)
            rows.append([design, SIZE, seed, step, f'{distance:.4f}', f'{error:.4f}', f'{policy_loss:.4f}',
                         f'{value_loss:.4f}', weights, f'{used:.0f}'])
            print(f'{name} ({DESIGNS[design]}), step {step}: held-back positions: distance {distance:.4f}, '
                  f'value error {error:.4f} | learned: policy {policy_loss:.3f}, value {value_loss:.3f} | '
                  f'{used / 60:.0f} min', flush=True)
        NN.save_checkpoint(network_file(design, seed), model, design=design, size=SIZE, seed=seed)
        add_rows(result_file(), RESULT_FIELDS, rows)


def match(names):
    import MCTS
    import test_networks
    import test_tournament
    device = NN.choose_device()
    test_tournament.OPENINGS = MATCH_OPENINGS
    text = {test_networks.opening_text(o): o for o in test_tournament.openings()}
    for name in names:
        design, seed = split_name(name)
        if name == MATCH_AGAINST or not os.path.exists(network_file(design, seed)):
            continue
        done = {(row['opening'], row['a_plays']) for row in rows_of(match_file())
                if row['a'] == name and row['b'] == MATCH_AGAINST and int(row['simulations']) == MATCH_SIMULATIONS}
        todo = [(key, side) for key in text for side in ('X', 'O') if (key, side) not in done]
        if not todo:
            continue
        print(f'{name} against {MATCH_AGAINST}, {MATCH_SIMULATIONS} simulations per move: {len(todo)} games', flush=True)
        search_a = MCTS.BatchedSearch(load(name, device), device, PARALLEL_GAMES, MATCH_SIMULATIONS)
        search_b = MCTS.BatchedSearch(load(MATCH_AGAINST, device), device, PARALLEL_GAMES, MATCH_SIMULATIONS)
        start = time.perf_counter()
        for first in range(0, len(todo), PARALLEL_GAMES):
            batch = todo[first:first + PARALLEL_GAMES]
            games = [(text[key], 1 if side == 'X' else -1) for key, side in batch]
            results = test_networks.play_batch(search_a, search_b, games)
            add_rows(match_file(), MATCH_FIELDS, [[name, MATCH_AGAINST, MATCH_SIMULATIONS, side, result, moves, key]
                                                  for (key, side), (result, moves) in zip(batch, results)])
            print(f'  {first + len(batch)} of {len(todo)} games after {(time.perf_counter() - start) / 60:.0f} min',
                  flush=True)


def summary():
    results = [row for row in rows_of(result_file()) if row['size'] == SIZE and int(row['step']) == STEPS]
    if not results:
        print('Nothing is trained yet.')
        return
    games = rows_of(match_file())
    print(f"Size '{SIZE}', {STEPS} steps on the positions of '{LESSONS}'. distance and value error on the "
          f'held-back positions (smaller is better); match against {MATCH_AGAINST} with {MATCH_SIMULATIONS} simulations')
    for row in results:
        name = row['design'] + (f":{row['seed']}" if int(row['seed']) else '')
        line = (f"  {name:5} {DESIGNS[row['design']]:42} distance {float(row['distance']):.4f}  value error "
                f"{float(row['value_error']):.4f}  {int(row['weights']) / 1e6:.2f}M weights  "
                f"{float(row['seconds']) / 60:.0f} min")
        played = [POINTS[g['result']] for g in games if g['a'] == name and g['b'] == MATCH_AGAINST
                  and int(g['simulations']) == MATCH_SIMULATIONS]
        if played:
            line += (f'  match {np.mean(played):.1%} (+-{np.std(played) / math.sqrt(len(played)):.1%}, '
                     f'{len(played)} games)')
        print(line)


if __name__ == '__main__':
    command, names = (sys.argv[1:2] or ['summary'])[0], sys.argv[2:] or PLAN
    if command == 'speed':
        speed()
    elif command == 'train':
        train(names)
    elif command == 'match':
        match(names)
    summary()
