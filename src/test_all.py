import AI
import os
import torch
import random
from little_evaluation_test import check_if_game_over
import time
from constants import *
import short_test
import test
import numpy as np

ai = AI.all_AI(-1)
model = test.NeuralNetCooler(combined_size)
model.load_state_dict(torch.load(path))
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
