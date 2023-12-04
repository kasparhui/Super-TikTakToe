import AI
import os
import torch
import random
from test_evaluation import check_if_game_over
import time
ai = AI.all_AI(-1)

input_size = 9
hidden_size1 = 300
hidden_size2 = 100
hidden_size3 = 50
hidden_size4 = 30
hidden_size5 = 20
hidden_size6 = 5
output_size = 1

path = os.path.join('models', f'trained_model_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}_{output_size}.pth')
model = torch.load(path)
model.eval()

while True:
    board = [random.choice([-1,0,1]) for j in range(9)]
    if not check_if_game_over(board):
        break

test_count = 100
start = time.time()
for i in range(test_count):
    model(board)
end = time.time()
print('Zeit:', end-start)


start = time.time()
for i in range(test_count):
    AI.little_evaluation(board)
end = time.time()
print('Zweite Zeit:', end-start)