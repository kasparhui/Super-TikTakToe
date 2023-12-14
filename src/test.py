import numpy as np
#from itertools import product
import torch
import game
import MCTS
import Tic_Tac_Toe_NN
import constants
Game = game.Board()
model = Tic_Tac_Toe_NN.UltimateTicTacToeNet()
model.load_state_dict(torch.load('models/first_model_weights.pth'))
model.eval()
Game.execute((4,4))
Game.execute((4,0))
Game.execute((0,4))
Game.execute((4,6))
Game.execute((6,4))
Game.execute((4,7))
Game.execute((7,4))
Game.execute((4,8))
Game.execute((8,5))
Game.execute((5,6))
Game.execute((6,5))
Game.execute((5,7))
Game.execute((7,5))
Game.execute((5,8))
Game.execute((8,3))
Game.execute((3,0))
Game.execute((0,3))
Game.execute((3,2))
print(constants.arrange_array(Game.board))
print(Game.won_games.reshape(3,3))
mcts = MCTS.AlphaMonteCarloTreeSearch(model)
node = MCTS.AlphaNode(Game)
best_move = mcts.best_move(node, 300)


