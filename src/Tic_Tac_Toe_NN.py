# Imports
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(2, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.activation = nn.ReLU()

    def forward(self, s):
        s = s.view(-1, 2, 9, 9)  # batch_size x channels x board_x x board_y
        s = self.activation(self.bn1(self.conv1(s)))

        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.activation(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.activation(out)
        return out
    
class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(9*9, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(128)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc3 = nn.Linear(9*9*128, 9*9*16)
        self.fc4 = nn.Linear(9*9*16, 9*9)

        self.activation = nn.ReLU()
        self.last_activation = nn.Tanh()
    def forward(self,s):
        v = self.activation(self.bn(self.conv(s))) # value head
        v = v.view(-1, 9*9)  # batch_size X channel X height X width
        v = self.activation(self.fc1(v))
        v = self.last_activation(self.fc2(v))
        
        p = self.activation(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 9*9*128)
        p = self.activation(self.fc3(p))
        p = self.fc4(p)
        p = self.logsoftmax(p).exp()
        return p, v
    
class UltimateTicTacToeNet(nn.Module):
    def __init__(self):
        super(UltimateTicTacToeNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(10):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(10):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s



# Monte Carlo Tree Search
class MCTS:
    def __init__(self, game, neural_net):
        # Initialisiere die MCTS-Parameter
        pass

    def get_action_probabilities(self, state):
        # Führe eine MCTS-Simulation durch und gib die Wahrscheinlichkeiten der Aktionen zurück
        pass

    def select_action(self, state):
        # Wähle die beste Aktion basierend auf der MCTS
        pass

# Hauptalgorithmus für das Training von AlphaZero
    
class AlphaZero:
    def __init__(self):
        # Initialisiere Parameter für das Training
        pass

    def self_play(self, game, neural_net):
        # Führe Selbstspiele durch, um Daten für das Training zu generieren
        pass

    def train_neural_net(self, training_data):
        # Trainiere das neuronale Netzwerk mit den generierten Daten
        pass

# Beispiel für die Verwendung
if __name__ == "__main__":
    # Beispiel für die Nutzung der implementierten Klassen
    import game
    import numpy as np
    import AI
    perfektes_netz = UltimateTicTacToeNet()
    board = game.board(np.zeros((9,9),dtype=np.int8),1)
    ai_board = board.board.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
    ai = AI.all_AI(1)
    possible_moves = ai.get_possible_moves(board)


    ai_board = torch.Tensor(ai_board).unsqueeze(0)  # Hinzufügen der Batch-Dimension (1x9x9)
    possible_moves = torch.Tensor(possible_moves).unsqueeze(0)  # Hinzufügen der Batch-Dimension (1x9x9)

    # Kombiniere die beiden Tensoren entlang der ersten Achse
    AI_input = torch.cat((ai_board, possible_moves), dim=0)
    evaluation = perfektes_netz(AI_input)
    print(evaluation)