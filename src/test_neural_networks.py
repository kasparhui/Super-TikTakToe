import torch
import torch.nn as nn

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

# Definition des neuronalen Netzwerks
class NeuralNetCooler(nn.Module):
    def __init__(self, sizes):
        super(NeuralNetCooler, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(sizes)-1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
        self.activation = nn.ReLU()  # Aktivierungsfunktion

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from AI import all_AI
    import random
    from game import check_if_game_over
    import os
    from itertools import product
    from constants import *
    AI = all_AI(-1)

    possible_values = [-1, 0, 1]
    possible_boards = list(filter(lambda x: (not check_if_game_over(x)) and (abs(x.count(-1)-x.count(1))<= 1),list(map(list, product(possible_values, repeat=9)))))
    print(len(possible_boards))


    x_values = np.array(possible_boards)
    #print(x_values)
    y_values = np.array([AI.little_evaluation(board) for board in x_values])

    # PyTorch-Tensoren erstellen
    x_train = torch.tensor(x_values, dtype=torch.float32)#.view(-1, 1)
    y_train = torch.tensor(y_values, dtype=torch.float32).view(-1, 1)
    # Netzwerk initialisieren 
    input_size = 9
    hidden_size1 = 300
    hidden_size2 = 100
    hidden_size3 = 75
    hidden_size4 = 30
    hidden_size5 = 20
    hidden_size6 = 5
    output_size = 1
    combined_size =[input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6, output_size]
    string_path = ''
    for size in combined_size:
        string_path += f'_{size}'

    path = os.path.join('models', f'trained_model_weights{string_path}.pth')
    model = NeuralNetCooler(combined_size)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    model.train()
    #print(model.state_dict())

    # Verlustfunktion und Optimierer festlegen
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Trainingsprozess
    epochs = 10000
    try:
        for epoch in range(epochs):
            # Forward-Pass
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            
            # Rückwärts-Pass und Optimierung
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    except KeyboardInterrupt:
        print("finito basta!")

    # Test
    model.eval()
    with torch.no_grad():
        predicted = model(x_train)

    print(f'Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), path)
    #print(model.state_dict())
    print(f"Das trainierte Modell wurde unter '{path}' gespeichert.")