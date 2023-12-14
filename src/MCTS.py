import numpy as np
import math
import game
import copy
import constants
import torch

class Node:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = {
            1: 0,
            -1: 0,
            0: 0
        }
        self.num_rollouts = 0
        self.children = []
        self.unvisited_moves = game_state.possible_moves().tolist()
        self.playerturn = game_state.playerturn
        self.is_terminal = game_state.game_is_over

class MonteCarloTreeSearch:
    def __init__(self):
        pass

    def best_move(self, node, iterations=10000):
        self.root = node
        for count in range(iterations):
            leaf = self.traverse(self.root)
            #if count % 99 == 0 and count != 0:
                #print(count)
                #print(leaf is self.root)
            new_leaf = self.expand(leaf)
            simulation_result = self.rollout(new_leaf)
            self.backpropagate(new_leaf, simulation_result)
        return self.select_best_move(self.root)

    def select_best_move(self, node):
        best_move = None
        best_pct = -1.0
        for child in node.children:
            win_pct = (child.win_counts[node.game_state.playerturn] + child.win_counts[0]/2) / child.num_rollouts
            if win_pct > best_pct:
                best_pct = win_pct
                best_move = child.move
        return best_move, node

    def traverse(self, node):
        while not node.unvisited_moves and node.children:
            node = self.select(node)
        return node

    def select(self, node):
        best_score = -1
        best_child = None
        for child in node.children:
            win_pct = (child.win_counts[node.game_state.playerturn] + child.win_counts[0]/2) / child.num_rollouts
            score = self.ucb_score(node.num_rollouts, child.num_rollouts, win_pct)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child

    def ucb_score(self, total_rollouts, node_rollouts, win_pct, bias_param=1): #bias_param=1.41
        exploration = math.log(total_rollouts) / node_rollouts
        return win_pct + bias_param * math.sqrt(exploration)

    def expand(self, node):
        if node.is_terminal:
            return node
        random_choice = np.random.randint(len(node.unvisited_moves))
        move = node.unvisited_moves.pop(random_choice)
        next_game_state = copy.deepcopy(node.game_state)
        next_game_state.execute(move)
        child_node = Node(next_game_state, node, move)
        node.children.append(child_node)
        return child_node

    def rollout(self, node):
        current_rollout_state = copy.deepcopy(node.game_state)
        while not current_rollout_state.game_is_over:
            possible_moves = current_rollout_state.possible_moves()
            action = self.rollout_policy(possible_moves)
            current_rollout_state.execute(action) 

        if game.game_is_win(current_rollout_state.won_games)[0]:
            if current_rollout_state.playerturn == 1:
                return -1
            else:  
                return 1
        elif game.game_is_tie(current_rollout_state.won_games)[0]:
            return 0
        else:
            print('error in rollout')


    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def backpropagate(self, node, result):
        if node is None:
            return
        node.num_rollouts += 1
        node.win_counts[result] += 1
        self.backpropagate(node.parent, result)

class AlphaNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = 0
        self.num_rollouts = 0
        self.children = []
        self.is_expanded = False
        self.playerturn = game_state.playerturn
        self.is_terminal = game_state.game_is_over
        if self.is_terminal:
            if game.game_is_win(game_state.won_games)[0]:
                self.value = - game_state.playerturn
            else:
                self.value = 0

class AlphaMonteCarloTreeSearch(MonteCarloTreeSearch):
    def __init__(self, neural_network):
        super().__init__()
        self.neural_network = neural_network

    def traverse(self, node):
        while not node.is_terminal and node.is_expanded:
            node = self.select(node)
        return node

    def select(self, node):
        best_child = None
        best_ucb = -np.inf
        
        for child in node.children:
            ucb = self.ucb_score(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def expand(self, node):
        if node.is_terminal:
            return node
        #print((node.game_state.get_encoded_state()))
        policy, value = self.neural_network(torch.tensor(node.game_state.get_encoded_state()))
        policy = policy[0].detach().numpy()
        policy = policy * node.game_state.possible_move_vector().reshape(-1)
        policy = policy / np.sum(policy)
        #print(policy)
        for action, prob in enumerate(policy):
            #print(policy)
            if prob > 0:
                next_game_state = copy.deepcopy(node.game_state)
                next_game_state.execute(action)
                child_node = AlphaNode(next_game_state, node, action)
                child_node.prior = prob
                node.children.append(child_node)
        node.value = value[0].detach().numpy()
        node.is_expanded = True
        return node

    def ucb_score(self, child):
        if child.num_rollouts == 0:
            q_value = 0
        else:
            if child.game_state.playerturn == -1:
                q_value = 1 - (child.win_counts / child.num_rollouts)  # win rate of opponent
            else:
                q_value = child.win_counts / child.num_rollouts  # win rate of current player
        return q_value + 2 * (math.sqrt( child.parent.num_rollouts) / (child.num_rollouts + 1)) * child.prior
    
    def rollout(self, node):
        return (node.value + 1) / 2  # In AlphaZero, the rollout step is replaced by the value output of the neural network
    
    def backpropagate(self, node, value_estimate):
        if node is None:
            return
        node.num_rollouts += 1
        node.win_counts += value_estimate
        self.backpropagate(node.parent, value_estimate)

    def select_best_move(self, node):
        action_probs = np.zeros(81)
        for child in node.children:
            action_probs[child.move] = child.num_rollouts
        action_probs = action_probs / np.sum(action_probs)
        return action_probs
    

if __name__ == '__main__':
    import Tic_Tac_Toe_NN
    import os

    path = os.path.join('models', f'first_model_weights.pth')
    board = game.Board(np.zeros((9,9),dtype=np.int8), -1)
    board.execute((4,4))
    node = AlphaNode(board)
    model = Tic_Tac_Toe_NN.UltimateTicTacToeNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    mcts = AlphaMonteCarloTreeSearch(model)
    best_move = mcts.best_move(node, 300)
    print(best_move)


        