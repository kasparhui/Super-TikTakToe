"""Neural network for Super TikTakToe (AlphaZero style: one network, two outputs).

Input:  6 planes of 9x9, always seen from the player who is to move
          0  my pieces
          1  opponent's pieces
          2  cells I am allowed to play now
          3  little boards I have won
          4  little boards the opponent has won
          5  little boards that ended in a tie
        The 9x9 picture is the board as it is drawn on the screen.
Output: policy  81 numbers (logits), one per cell of the 9x9 picture
        value   between -1 (I lose) and 1 (I win)

Training is done with train.py, playing with the 'neural_net' player type.
"""
import os
import weakref

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The size of the network. Bigger = stronger in the end, but slower to train
# and slower per move. Every variant has its own weight file in models/.
MODEL_VARIANTS = {
    'small':  {'channels': 32, 'blocks': 4},    # a first result in 1-2 hours
    'medium': {'channels': 64, 'blocks': 6},    # made for one night of training
    'large':  {'channels': 96, 'blocks': 8},    # for several nights
    'xlarge': {'channels': 128, 'blocks': 10},  # about 2.2 times the work of 'large' per position
    # 'large' with line layers (design D5 of test_designs.py): every cell also hears from the lines it lies on
    'large_lines': {'channels': 96, 'blocks': 8, 'lines': True},
}
DEFAULT_MODEL = 'medium'

INPUT_PLANES = 6
# Where the networks, logs and snapshots are. Another folder can be given with the
# environment variable SUPER_TTT_MODELS (used on Google Colab to save to Google Drive).
MODEL_DIR = (os.environ.get('SUPER_TTT_MODELS')
             or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))
# On Nvidia cards the search asks the network in half precision, which is faster there.
# It has no effect on Apple chips or the processor, and learning always uses full precision.
HALF_PRECISION = True
# On Nvidia cards the search also asks a copy of the network that is rearranged for speed (the
# normalisation layers worked into the filters, another memory layout, everything in half
# precision) and, with COMPILE, translated once into faster code. Measured on Colab's L4 with the
# large network: 1.4 times as many positions per second, 1.9 times with COMPILE. The answers
# differ from full precision as little as before (move probabilities by about 0.002).
FAST_PLAY = True
COMPILE = True
COMPILE_FROM = 256          # smaller questions are not worth translating
# The copy can sum along the lines with picture operations instead of big tables
# (test_designs.Block.picture_summary; the same answers). Off until it is measured to be faster on the L4;
# on Apple's chip it is slower. 'python3 src/pretest.py bottleneck large_lines' compares both.
FAST_LINES = False
COMPILE_STEP = 64           # questions are filled up to a multiple of this, so that few sizes occur

# A move is the number  big_field * 9 + little_field  (as in game.Board).
# MOVE_TO_SPATIAL[move] is where that cell lies in the flattened 9x9 picture.
MOVE_TO_SPATIAL = np.array([(3 * (b // 3) + c // 3) * 9 + 3 * (b % 3) + c % 3
                            for b in range(9) for c in range(9)], dtype=np.int64)
SPATIAL_TO_MOVE = np.argsort(MOVE_TO_SPATIAL)


class BoardBlock(nn.Module):
    """Residual block that knows about the structure of the game.

    Besides the usual 3x3 convolutions every cell also gets a summary of
      * its own little board, and
      * the little board its move sends the opponent to,
    and those summaries are computed with a view of the whole big board.
    """

    def __init__(self, channels, context):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.context = nn.Conv2d(channels, context, 3, padding=1)   # works on the 3x3 big board
        self.own = nn.Conv2d(context, channels, 1, bias=False)
        self.target = nn.Conv2d(context, channels, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        # Fixed 9 x 81 tables that copy the 9 little-board summaries to the 81 cells.
        # (As matrices because that is much faster on Apple's graphics chip than repeat().)
        own_cells = torch.zeros(9, 81)
        target_cells = torch.zeros(9, 81)
        for row in range(9):
            for column in range(9):
                own_cells[(row // 3) * 3 + column // 3, row * 9 + column] = 1
                target_cells[(row % 3) * 3 + column % 3, row * 9 + column] = 1
        self.register_buffer('own_cells', own_cells, persistent=False)
        self.register_buffer('target_cells', target_cells, persistent=False)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        big = F.relu(self.context(F.avg_pool2d(h, 3)))              # one vector per little board
        summary = (self.own(big).flatten(2) @ self.own_cells
                   + self.target(big).flatten(2) @ self.target_cells)
        h = self.bn2(self.conv2(h + summary.view(x.shape)))
        return F.relu(x + h)


class UltimateTicTacToeNet(nn.Module):
    def __init__(self, channels=64, blocks=6):
        super().__init__()
        self.stem = nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False)
        # lets the network tell the cells apart (corner of a little board, centre, ...)
        self.position = nn.Parameter(torch.zeros(1, channels, 9, 9))
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([BoardBlock(channels, channels // 2) for _ in range(blocks)])

        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_out = nn.Conv2d(32, 1, 1)

        self.value_conv = nn.Conv2d(channels, 8, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 81, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """x: batch x 6 x 9 x 9  ->  (policy logits: batch x 81, value: batch)"""
        x = F.relu(self.stem_bn(self.stem(x) + self.position))
        for block in self.blocks:
            x = block(x)
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = self.policy_out(policy).flatten(1)
        value = F.relu(self.value_bn(self.value_conv(x))).flatten(1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)).squeeze(1)
        return policy, value


def create_model(variant=DEFAULT_MODEL):
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model '{variant}'. Use one of {list(MODEL_VARIANTS)}.")
    shape = dict(MODEL_VARIANTS[variant])
    if shape.pop('lines', False):
        import test_designs                     # the design under test lives there until it has proven itself
        return test_designs.DesignNet(**shape, lines=True)
    return UltimateTicTacToeNet(**shape)


def choose_device(device='auto'):
    """'auto' takes the graphics chip if there is one (Apple: mps, Nvidia: cuda)."""
    if device != 'auto':
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def checkpoint_path(variant=DEFAULT_MODEL):
    return os.path.join(MODEL_DIR, f'alphazero_{variant}.pt')


def save_checkpoint(path, model, optimizer=None, **extra):
    """Saves weights (+ optimizer and anything else) without ever leaving a half written file."""
    data = {'model': {k: v.cpu() for k, v in model.state_dict().items()}}
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()
    data.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path + '.tmp')
    os.replace(path + '.tmp', path)


def snapshot_path(variant, iteration):
    """The copy of the network that train.py keeps of this training iteration."""
    return os.path.join(MODEL_DIR, 'snapshots', f'alphazero_{variant}_iteration{int(iteration):04d}.pt')


def saved_iteration(variant=DEFAULT_MODEL):
    """The training iteration of the newest network of this variant."""
    return torch.load(checkpoint_path(variant), map_location='cpu').get('iteration', '')


def load_model(variant=DEFAULT_MODEL, device='cpu', iteration=None):
    """The trained network of this variant, ready to play.
    iteration: None for the newest one, or the number of a training iteration
    of which train.py has kept a snapshot (models/snapshots/)."""
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model '{variant}'. Use one of {list(MODEL_VARIANTS)}.")
    path = checkpoint_path(variant) if iteration is None else snapshot_path(variant, iteration)
    if not os.path.exists(path) and iteration is not None:
        raise FileNotFoundError(f"There is no snapshot of the '{variant}' network after iteration "
                                f"{iteration} ({path}). See models/snapshots/ for the ones that exist.")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"There is no trained '{variant}' network yet ({path}). "
            f"Train one first: set MODEL = '{variant}' in src/train.py and run it.")
    model = create_model(variant)
    model.load_state_dict(torch.load(path, map_location='cpu')['model'])
    model.to(device)
    model.eval()
    return model


def folded(model):
    """The same network with the normalisation layers worked into the filters before them
    (the same answers with fewer layers; only for playing, not for learning)."""
    import copy
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    model = copy.deepcopy(model).eval()

    def fuse(conv, bn):
        if hasattr(conv, 'conv'):                   # test_designs.Filter: a filter in a wrapper
            if not conv.inside:
                return fuse_conv_bn_eval(conv.conv, bn)
            conv.conv = fuse_conv_bn_eval(conv.conv, bn)
            return conv
        return fuse_conv_bn_eval(conv, bn)

    for block in model.blocks:
        block.conv1, block.bn1 = fuse(block.conv1, block.bn1), nn.Identity()
        block.conv2, block.bn2 = fuse(block.conv2, block.bn2), nn.Identity()
        if hasattr(block, 'pictures'):
            block.pictures = bool(FAST_LINES)
    model.policy_conv, model.policy_bn = fuse(model.policy_conv, model.policy_bn), nn.Identity()
    model.value_conv, model.value_bn = fuse(model.value_conv, model.value_bn), nn.Identity()
    return model


class _FastCopy:
    """The copy of a network that the search asks on Nvidia cards. It follows the network: when
    the weights have changed (after learning), the copy is brought up to date before it is asked."""

    def __init__(self, model):
        self.tensors = list(model.parameters()) + list(model.buffers())
        self.net = folded(model).to(memory_format=torch.channels_last).half()
        self.version = self._version()
        self.compiled = None
        self.compile = COMPILE
        self.checked = False        # after a check: which ways of asking agree with the network
        self.broken = False

    def _version(self):
        return sum(tensor._version for tensor in self.tensors)      # changes with every change of a weight

    def _translated(self):
        if self.compiled is None:
            import torch._dynamo
            for limit in ('recompile_limit', 'cache_size_limit'):   # one translation per size of question
                if hasattr(torch._dynamo.config, limit):
                    setattr(torch._dynamo.config, limit, 64)
            self.compiled = torch.compile(self.net)
        return self.compiled

    def _answer(self, x, translated):
        count = len(x)
        x = x.half()
        if translated:
            size = -(-count // COMPILE_STEP) * COMPILE_STEP
            if size > count:
                x = torch.cat([x, x.new_zeros((size - count,) + x.shape[1:])])
        x = x.contiguous(memory_format=torch.channels_last)
        logits, values = (self._translated() if translated else self.net)(x)
        return logits[:count], values[:count]

    def ask(self, model, x):
        """Returns (logits, values), or None if the copy cannot be trusted (then the network itself is asked)."""
        if self.broken:
            return None
        version = self._version()
        if version != self.version:
            self.net.load_state_dict(folded(model).state_dict())
            self.version, self.checked = version, False
        translated = self.compile and len(x) >= COMPILE_FROM
        if self.checked.get(translated) if isinstance(self.checked, dict) else False:
            return self._answer(x, translated)
        # the first question after every change of the weights: compare with the network itself
        if not isinstance(self.checked, dict):
            self.checked = {}
        moves = torch.softmax(model(x)[0].float(), dim=1)
        for way in ([True, False] if translated else [False]):
            try:
                logits, values = self._answer(x, way)
                fine = torch.isfinite(logits).all(dim=1)
                difference = float((torch.softmax(logits.float(), dim=1) - moves)[fine].abs().max()) if fine.any() else 0.0
                if difference > 0.05:
                    raise RuntimeError(f'its move probabilities differ by {difference:.3f} from the ones of the network')
                self.checked[way] = True
                return logits, values
            except Exception as error:
                what = 'translation of the network into faster code' if way else 'faster copy of the network'
                print(f'The {what} is switched off ({type(error).__name__}: {str(error)[:200]}). '
                      f'Everything else works as before.', flush=True)
                if way:
                    self.compile = False
                else:
                    self.broken = True
        return None


_fast_copies = weakref.WeakKeyDictionary()


def _fast_copy(model):
    # every network built from blocks with the usual two heads (also the line network of test_designs.py)
    if not (hasattr(model, 'blocks') and hasattr(model, 'value_conv')) or model.training:
        return None
    if model not in _fast_copies:
        _fast_copies[model] = _FastCopy(model)
    return _fast_copies[model]


def predict(model, planes, device):
    """planes: numpy batch x 6 x 9 x 9  ->  numpy (logits: batch x 81, values: batch)"""
    half = HALF_PRECISION and torch.device(device).type == 'cuda'
    x = torch.from_numpy(planes).to(device)
    fast = _fast_copy(model) if FAST_PLAY == 'always' or (FAST_PLAY and half) else None
    with torch.inference_mode():
        answer = fast.ask(model, x) if fast is not None else None
        if answer is None:
            with torch.autocast('cuda', dtype=torch.float16, enabled=half):
                answer = model(x)
        logits, values = answer
    logits, values = logits.float().cpu().numpy(), values.float().cpu().numpy()
    if half or fast is not None:
        logits, values = _repair(model, x, logits, values)
    return logits, values


def _repair(model, x, logits, values):
    """In a rare position a number gets too big for half precision and the answer is not a number.
    Only those positions are asked again, in full precision."""
    broken = ~(np.isfinite(logits).all(axis=1) & np.isfinite(values))
    if broken.any():
        with torch.inference_mode():
            again_logits, again_values = model(x[torch.from_numpy(broken).to(x.device)])
        logits[broken] = again_logits.float().cpu().numpy()
        values[broken] = again_values.float().cpu().numpy()
    return logits, values


def random_symmetry(planes, policy):
    """The same random rotation / mirroring for a batch of positions and their
    policy targets. The game is symmetric under all 8 of them."""
    turns = int(torch.randint(4, (1,)))
    mirror = bool(torch.randint(2, (1,)))
    policy = policy.view(-1, 9, 9)
    if turns:
        planes = torch.rot90(planes, turns, (2, 3))
        policy = torch.rot90(policy, turns, (1, 2))
    if mirror:
        planes = planes.flip(3)
        policy = policy.flip(2)
    return planes, policy.reshape(-1, 81)
