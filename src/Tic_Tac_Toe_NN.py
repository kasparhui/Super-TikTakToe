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
}
DEFAULT_MODEL = 'medium'

INPUT_PLANES = 6
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

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
    return UltimateTicTacToeNet(**MODEL_VARIANTS[variant])


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


def load_model(variant=DEFAULT_MODEL, device='cpu'):
    """The trained network of this variant, ready to play."""
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model '{variant}'. Use one of {list(MODEL_VARIANTS)}.")
    path = checkpoint_path(variant)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"There is no trained '{variant}' network yet ({path}). "
            f"Train one first: set MODEL = '{variant}' in src/train.py and run it.")
    model = create_model(variant)
    model.load_state_dict(torch.load(path, map_location='cpu')['model'])
    model.to(device)
    model.eval()
    return model


def predict(model, planes, device):
    """planes: numpy batch x 6 x 9 x 9  ->  numpy (logits: batch x 81, values: batch)"""
    with torch.inference_mode():
        logits, values = model(torch.from_numpy(planes).to(device))
    return logits.float().cpu().numpy(), values.float().cpu().numpy()


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
