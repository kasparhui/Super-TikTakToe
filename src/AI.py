from game import *
import random
import math
import MCTS
import fast_minimax
import Tic_Tac_Toe_NN

# player type in the settings (main.py, test_models.py) -> mode of all_AI.make_move
AI_MODES = {
    'minimax': 'best_move',
    'mcts': 'naiv_MCTC',
    'neural_net': 'best_nn_move',
    'random': 'random_move',
}
DEFAULT_DEPTH = 11             # minimax
DEFAULT_SIMULATIONS = 400      # neural_net: positions the network looks at per move
DEFAULT_ITERATIONS = 200000    # mcts: random games per move


def create_ai(symbol, settings):
    """The all_AI object for one player ('X' or 'O'), or None for a human.
    settings is a dict like {'type': 'minimax', 'depth': 12}, see main.py."""
    player_type = settings['type']
    if player_type == 'human':
        return None
    if player_type not in AI_MODES:
        raise ValueError(f"Unknown player type '{player_type}' for player {symbol}. "
                         f"Use 'human' or one of {list(AI_MODES)}.")
    ai = all_AI(tic_tac_toe_mapping[symbol], model=settings.get('model', Tic_Tac_Toe_NN.DEFAULT_MODEL))
    if player_type == 'neural_net':
        # load it now, so a missing network is noticed at the start and not at the first move
        ai.load_network(settings.get('simulations', DEFAULT_SIMULATIONS))
    return ai


def ai_move(ai, settings, board):
    """The move of an AI created with create_ai, as (big field, little field)."""
    return ai.make_move(board, AI_MODES[settings['type']],
                        depth=settings.get('depth', DEFAULT_DEPTH),
                        time_limit=settings.get('time_limit'),
                        simulations=settings.get('simulations', DEFAULT_SIMULATIONS),
                        iterations=settings.get('iterations', DEFAULT_ITERATIONS))


class all_AI:
    def __init__(self, player, model = Tic_Tac_Toe_NN.DEFAULT_MODEL):
        self.player = player
        self.model_name = model     # which trained network 'best_nn_move' uses
        self.neural_player = None   # loaded when it is needed for the first time

    def make_move(self, board, mode = 'random_move', depth = DEFAULT_DEPTH, smart_depth = 100, time_limit = None,
                  simulations = DEFAULT_SIMULATIONS, iterations = DEFAULT_ITERATIONS):
        self.board = copy.deepcopy(board)
        if mode == 'random_move':
            return self.random_move()
        elif mode == 'best_move':
            return self.best_move(depth, smart_depth, time_limit)
        elif mode == 'naiv_MCTC':
            return self.naiv_MCTC(iterations)
        elif mode == 'best_nn_move':
            return self.best_nn_move(simulations)
        else:
            raise ValueError(f"Kein gültiger Modus: '{mode}'")

    def load_network(self, simulations = DEFAULT_SIMULATIONS):
        if self.model_name == 'final':      # the final network (models/final/), a 'large' one
            import os, torch
            path = os.path.join(Tic_Tac_Toe_NN.MODEL_DIR, 'final', 'alphazero_large.pt')
            if not os.path.exists(path):
                raise FileNotFoundError(f'There is no final network at {path}.')
            model = Tic_Tac_Toe_NN.create_model('large')
            model.load_state_dict(torch.load(path, map_location='cpu')['model'])
            model.eval()
        else:
            model = Tic_Tac_Toe_NN.load_model(self.model_name, 'cpu')
        self.neural_player = MCTS.NeuralPlayer(model, 'cpu', simulations)

    def best_nn_move(self, simulations = DEFAULT_SIMULATIONS):
        # AlphaZero: tree search guided by the trained network (see MCTS.py, train.py)
        if self.neural_player is None or self.neural_player.search.simulations != simulations:
            self.load_network(simulations)
        position, value = self.neural_player.best_move(self.board)
        return position
        
    def straight_count(self, temp_array):
        if ((np.count_nonzero(temp_array == 1) > 0) and (np.count_nonzero(temp_array == -1) > 0)) or (np.count_nonzero(temp_array == 2)>0):
            return 0
        else:
            return( np.sum(temp_array) ** 3)

    def get_possible_moves(self, board):
        possible_moves = np.zeros((9,9),dtype=np.int8)
        if  board.next_move == None:
            for i in range(9):
                if board.won_games[i] == 0:
                    positions = np.where(board.board[i] == 0)[0]
                    possible_moves[i,positions] = 1
        else:
            positions = np.where(board.board[board.next_move] == 0)[0]
            possible_moves[board.next_move,positions] = 1
        return possible_moves


    def little_evaluation(self, little_board):
        state = little_board.sum() - (2* np.count_nonzero(little_board == 2))
        for i in range (3):
            state += self.straight_count(np.array([little_board[3*i],little_board[3*i+1],little_board[3*i+2]]))
            state += self.straight_count(np.array([little_board[i],little_board[i+3],little_board[i+6]]))
        state += self.straight_count(np.array([little_board[0], little_board[4],little_board[8]]))
        state += self.straight_count(np.array([little_board[2], little_board[4], little_board[6]]))
        
        return 0.75 * math.tanh(state/10)

    def get_evalution(self, show_all_states = False):
        values = np.copy(self.board.won_games).astype(np.float64)
        for i in (np.where(values ==0)[0]):
            values[i] = self.little_evaluation(self.board.board[i])
        if show_all_states:
            print(values, self.little_evaluation(values))
        return self.little_evaluation(values)


    def best_move(self, depth, smart_depth = 100, time_limit = None):
        # Fast compiled minimax (fast_minimax.py). self.minmax below is the
        # original, slow version with the same evaluation, kept as a reference.
        value, position = fast_minimax.best_move(self.board, depth, time_limit)
        return position

    def random_move(self):
        actions = self.possible_moves()
        position = actions[np.random.randint(len(actions))]
        return (int(position[0]), int(position[1]))
    
    def naiv_MCTC(self, iterations = DEFAULT_ITERATIONS):
        # classic MCTS: many random games decide which move is best
        position, win_rate = MCTS.pure_mcts_move(self.board, iterations)
        return position

    def possible_moves(self):
    
        return self.board.possible_moves()

    def minmax(self, playerturn, depth, smart_depth, alpha=-2, beta=2, is_root=True):
        # Minimax with alpha-beta pruning. Player 1 maximises, player -1 minimises.
        # alpha/beta are the best values each side can already guarantee.
        if self.board.game_is_over:
            if check_if_game_over(self.board.won_games):
                if game_is_win(self.board.won_games)[0]:
                    return flip_player(playerturn), None
                elif game_is_tie(self.board.won_games)[0]:
                    return 0, None
        if depth ==1:
            return self.get_evalution(), None
        actions = self.possible_moves()
        if len(actions) > smart_depth:
            if depth != 2:
                depth1_evaluation = []
                for i, action in enumerate(actions):
                    self.board.execute(action)
                    depth1_evaluation.append(self.get_evalution())
                    self.board.undo_move()
                depth1_evaluation = np.array(depth1_evaluation)
                top_moves = actions[np.argsort(depth1_evaluation)[-smart_depth:]]
                actions = top_moves

        if is_root:
            # Pruned siblings only return bounds, so equal-valued moves can't be
            # collected reliably. Shuffling the root moves and keeping the first
            # strictly best one still picks uniformly among the best moves.
            actions = np.random.permutation(actions)

        best_action = None
        playerflip = flip_player(playerturn)
        if playerturn == 1:
            value = -2
            for action in actions:
                self.board.execute(action)
                score = self.minmax(playerflip, depth-1, smart_depth, alpha, beta, False)[0]
                self.board.undo_move()
                if score > value:
                    value = score
                    best_action = action
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = 2
            for action in actions:
                self.board.execute(action)
                score = self.minmax(playerflip, depth-1, smart_depth, alpha, beta, False)[0]
                self.board.undo_move()
                if score < value:
                    value = score
                    best_action = action
                beta = min(beta, value)
                if alpha >= beta:
                    break
        return value, best_action
        


