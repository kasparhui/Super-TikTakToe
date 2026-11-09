from game import *
import random
import math
import MCTS
import fast_minimax
import Tic_Tac_Toe_NN
import torch
import os

class all_AI:
    def __init__(self, player, args = None, path = None):
        self.player = player
        if not (args is None):
            self.model = Tic_Tac_Toe_NN.UltimateTicTacToeNet(args[0], args[1])
            if not (path is None):
                self.model.load_state_dict(torch.load(path))
            self.model.eval()

    def make_move(self, board, mode = 'random_move', depth = 3, smart_depth = 100, time_limit = None):
        self.board = copy.deepcopy(board)
        if mode == 'random_move':
            return self.random_move()
        elif mode == 'best_move':
            return self.best_move(depth, smart_depth, time_limit)
        elif mode == 'naiv_MCTC':
            return self.naiv_MCTC()
        elif mode == 'best_nn_move':
            return self.best_nn_move()
        else:
            print('Kein gültiger move')

    def best_nn_move(self):
        model = self.model
        self.board.change_perspective()
        Node = MCTS.AlphaNode(self.board)
        mcts = MCTS.AlphaMonteCarloTreeSearch(model)
        probabiltys = mcts.best_move(Node, 250)[0]
        best_move = np.argmax(probabiltys).item()
        #print(best_move // 9, best_move % 9)
        return best_move // 9, best_move % 9
        
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
    
    def naiv_MCTC(self, iterations = 1000):
        node = MCTS.Node(self.board)
        mcts = MCTS.MonteCarloTreeSearch()
        best_move = mcts.best_move(node, iterations)
        return best_move[0]

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
        


