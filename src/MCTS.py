import numpy as np
import math
import game
import copy
import constants

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
    def __init__(self, node):
        self.root = node

    def best_move(self, iterations=1000):
        for _ in range(iterations):
            leaf = self.traverse(self.root)
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
        #print(constants.arrange_array(current_rollout_state.board))
        #print(current_rollout_state.won_games)

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

if __name__ == '__main__':

    board = game.board(np.array([[ 0,  0,  0,  0 , 0,  0  ,0  ,0 , 0],
 [ 0,  0, -1  ,0,  0 , 0 ,-1 , 1  ,0],
 [ 0,  1 , 0 , 0 , 0 , 0 , 0 , 1,  0],
 [ 0 , 0 , 0,  0,  1  ,0  ,0 , 0 , 0],
 [0  ,0 , 0 ,-1,  1 , 0, -1 , 0 , 0],
 [ 0 , 0 , 0,  0, 0 , 0,  0 , 0  , 0],
 [ 0 , 1  ,0 , 0,  1 , 0  ,0 , 0  ,0],
 [0, -1 ,-1 , 0,  0  ,0  ,0 , 0 , 0],
 [ 0,  0  ,0 , 0  ,0 , 0 , 0,  0 , 0]],dtype=np.int8), -1)
    board.execute([4,0])
    board.execute([0,1])
    board.execute([7,0])
    board.execute([0,4])

    node = Node(board)
    mcts = MonteCarloTreeSearch(node)
    best_move = mcts.best_move()

    print(best_move[0])

    for i in range(len(best_move[1].children)):
        print(best_move[1].children[i].move, best_move[1].children[i].win_counts, best_move[1].children[i].num_rollouts)
        '''if best_move[1].children[i].move == best_move[0]:
            print('----', mcts.select_best_move(best_move[1].children[i])[0])
            for j in range(9):
                print('----', best_move[1].children[i].children[j].move, best_move[1].children[i].children[j].win_counts, best_move[1].children[i].children[j].num_rollouts)
                if best_move[1].children[i].children[j].move == mcts.select_best_move(best_move[1].children[i])[0]:
                    print('--------', mcts.select_best_move(best_move[1].children[i].children[j])[0])
                    for k in range(9):
                        print('--------', best_move[1].children[i].children[j].children[k].move, best_move[1].children[i].children[j].children[k].win_counts, best_move[1].children[i].children[j].children[k].num_rollouts)'''
        
    board.execute([1,4])
    print(board.game_is_over)