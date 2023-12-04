import AI
import os
import torch
import random
from test_evaluation import check_if_game_over
import time
from constants import *
import short_test
import numpy as np

ai = AI.all_AI(-1)
model = short_test.NeuralNet(input_short_size, hidden_short_size1,hidden_short_size2,hidden_short_size3,output_short_size)
model.load_state_dict(torch.load(path_short))
model.eval()

while True:
    board = np.array([random.choice([-1,0,1]) for j in range(9)])
    if not check_if_game_over(board):
        break
test_count = 2000

start = time.time()
for i in range(test_count):
    a= ai.little_evaluation(board)
end = time.time()
print('Zweite Zeit:', end-start)

board = torch.tensor(board, dtype=torch.float32)
start = time.time()
for i in range(test_count):
    b = model(board)
end = time.time()
print('Zeit:', end-start)
print(a,b )
