import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Daten generieren
x_values = np.linspace(-2*np.pi, 2*np.pi, 10000)
y_values = np.sin(2 * np.exp(np.cos(x_values*x_values)))
#y_values = np.sin(np.exp(np.cos(x_values)))

# PyTorch-Tensoren erstellen
x_train = torch.tensor(x_values, dtype=torch.float32).view(-1, 1)
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


# Netzwerk initialisieren
model = NeuralNet(1, 300, 100, 50, 30, 20, 5, 1)

# Verlustfunktion und Optimierer festlegen
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Trainingsprozess
epochs = 10000
for epoch in range(epochs):
    # Forward-Pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    
    # Rückwärts-Pass und Optimierung
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Test
model.eval()
with torch.no_grad():
    predicted = model(x_train)

# Plot der Ergebnisse
plt.figure(figsize=(10, 6))
plt.scatter(x_train.numpy(), y_train.numpy(), label='Actual')
plt.plot(x_train.numpy(), predicted.numpy(), 'r-', label='Predicted')
plt.xlabel('x')
plt.ylabel('sin(e^cos(x))')
plt.legend()
plt.show()
