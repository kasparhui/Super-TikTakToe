import numpy as np
#from itertools import product
import torch
import game
import MCTS
import Tic_Tac_Toe_NN
import constants
import os

date = np.array([[ 0,  0,  0,  1,  1, -1,  1,  1,  1],
                   [ 0,  0,  1,  1,  1, -1, -1,  1,  1],
                   [-1,  0, -1,  1,  1,  1,  0,  1,  1],
                   [ 1, -1, -1, -1,  0,  0,  1, -1, -1],
                   [-1,  0, -1, -1,  0,  1, -1, -1,  0],
                   [-1,  0, -1,  0, -1,  0,  1,  1, -1],
                   [-1, -1, -1,  1,  0,  1,  1, -1, -1],
                   [-1,  0,  1,  0,  1, -1,  1, -1,  0],
                   [ 1, -1,  1,  0,  1,  1, -1,  1, -1]])


board = game.Board()
board.execute(3); board.execute(5)
board.execute(4); board.execute(14)
board.execute(6) ;board.execute(15)
board.execute(7); board.execute(18)
board.execute(8); board.execute(20)
board.execute(11); board.execute(28)
board.execute(12); board.execute(29)
board.execute(13); board.execute(30)
board.execute(16); board.execute(34)
board.execute(17); board.execute(35)
board.execute(21); board.execute(36)
board.execute(22); board.execute(38)
board.execute(23); board.execute(39)
board.execute(25); board.execute(42)
board.execute(26); board.execute(43)
board.execute(27); board.execute(45)
board.execute(33); board.execute(47)
board.execute(41); board.execute(49)
board.execute(51); board.execute(53)
board.execute(52); board.execute(54)
board.execute(57); board.execute(55)
board.execute(59); board.execute(56)
board.execute(60); board.execute(61)
board.execute(65); board.execute(62)
board.execute(67); board.execute(63)
board.execute(69); board.execute(68)
board.execute(72); board.execute(70)
board.execute(74); board.execute(73)
board.execute(76); board.execute(78)
board.execute(77); board.execute(80)
board.execute(79); board.execute(38)
board.execute(3); #board.execute(47)
#board.execute(21); #board.execute(55)
#board.execute(65); #board.execute(9)
#board.execute(75)

board.change_perspective()
#print(board.possible_move_vector())
print(board.won_games)
print(board.board)
model = Tic_Tac_Toe_NN.UltimateTicTacToeNet()
model.load_state_dict(torch.load(os.path.join('models','neural_net','first_model_weights.pth')))
model.eval()
mcts = MCTS.AlphaMonteCarloTreeSearch(model)
node = MCTS.AlphaNode(board)
dist = mcts.best_move(node, 1640)[0]
print(dist)