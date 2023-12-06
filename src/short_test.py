import torch
import torch.nn as nn

# Definition des neuronalen Netzwerks
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(NeuralNet, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 48, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1)

        # Fully Connected Layers für die Ausgabe
        self.fc1 = nn.Linear(128 * 3 * 3, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)   
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.activation = nn.ReLU()  # Aktivierungsfunktion

    def forward(self, x):
        x = x.view(-1, 1, 3, 3)  # Annahme: Eingabe ist 1D-Array mit 9 Elementen, wird in 3x3 umgewandelt

        # Convolutional Layers
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        
        # Flatten für Fully Connected Layers
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully Connected Layers
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from AI import all_AI
    import random
    from little_evaluation_test import check_if_game_over
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
 
    model = NeuralNet(input_short_size, hidden_short_size1, hidden_short_size2, hidden_short_size3, output_short_size)
    if os.path.exists(path_short):
        model.load_state_dict(torch.load(path_short))
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
        print("finito")

    print(f'Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), path_short)
    #print(model.state_dict())
    print(f"Das trainierte Modell wurde unter '{path_short}' gespeichert.")