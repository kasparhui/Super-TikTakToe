import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from AI import all_AI
import random
from test_evaluation import check_if_game_over
import os

AI = all_AI(-1)
count_boards = 20000
possible_boards = []
while len(possible_boards) < count_boards:
    board = [random.choice([-1,0,1]) for j in range(9)]
    if not check_if_game_over(board):
        possible_boards.append(board)
x_values = np.array(possible_boards)
#print(x_values)
y_values = np.array([AI.little_evaluation(board) for board in x_values])

print('lolol')
# Daten generieren
# x_values = np.linspace(-2*np.pi, 2*np.pi, 1000)
# #y_values = np.sin(2 * np.exp(np.cos(x_values*x_values)))
# y_values = np.sin(np.exp(np.cos(x_values)))

# PyTorch-Tensoren erstellen
x_train = torch.tensor(x_values, dtype=torch.float32)#.view(-1, 1)
y_train = torch.tensor(y_values, dtype=torch.float32).view(-1, 1)
# Definition des neuronalen Netzwerks
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.fc5 = nn.Linear(hidden_size4, hidden_size5)
        self.fc6 = nn.Linear(hidden_size5, hidden_size6)
        self.fc7 = nn.Linear(hidden_size6, output_size)
        self.activation = nn.ReLU()  # Aktivierungsfunktion

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.activation(self.fc6(x))
        x = self.fc7(x)
        return x

input_size = 9
hidden_size1 = 300
hidden_size2 = 100
hidden_size3 = 50
hidden_size4 = 30
hidden_size5 = 20
hidden_size6 = 5
output_size = 1

# Netzwerk initialisieren 
path = os.path.join('models', f'trained_model_weights_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}_{output_size}.pth')
path2 = os.path.join('models', f'trained_model_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}_{output_size}.pth')
model = NeuralNet(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6, output_size)
if os.path.exists(path):
    print('kekek')
    model.load_state_dict(torch.load(path))
model.train()
#print(model.state_dict())

# Verlustfunktion und Optimierer festlegen
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Trainingsprozess
epochs = 100
try:
    for epoch in range(epochs):
        # Forward-Pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        
        # Rückwärts-Pass und Optimierung
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
except KeyboardInterrupt:
    print("finito")

# Test
model.eval()
with torch.no_grad():
    predicted = model(x_train)

print(f'Loss: {loss.item():.4f}')

torch.save(model.state_dict(), path)
torch.save(model,path2)
#print(model.state_dict())
print(f"Das trainierte Modell wurde unter '{path}' gespeichert.")