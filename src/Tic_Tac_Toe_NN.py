# Imports
import torch
import torch.nn as nn
import os

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
class AlphaLoss(nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy * (1e-6 + y_policy.float()).float().log()), 1)
        #print(value_error,'ve',policy_error,'pe')
        total_error = (value_error.view(-1).float() + policy_error).mean()
        print(total_error,'Total error')
        return total_error
    
class AlphaZero:
    def __init__(self, model):
        # Initialisiere Parameter für das Training
        self.model = model
    def self_play(self, game, neural_net):
        # Führe Selbstspiele durch, um Daten für das Training zu generieren
        pass

    def train_neural_net(self, dataset, epoch_start=0, epoch_stop=2, cpu=0):
        torch.manual_seed(cpu)
        cuda = torch.cuda.is_available()
        self.model.train()
        criterion = AlphaLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.2)

        train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
        losses_per_epoch = []

        for epoch in range(epoch_start, epoch_stop):
            total_loss = 0.0
            losses_per_batch = []

            for i, data in enumerate(train_loader, 0):
                state, policy, value = data

                if cuda:
                    state, policy, value = state.cuda().float(), policy.float().cuda(), value.cuda().float()

                optimizer.zero_grad()
                policy_pred, value_pred = self.model(state)

                loss = criterion(value_pred, value, policy_pred, policy)
                #print(value_pred, value, policy_pred, policy)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if i % 10 == 9:
                    print(f'Epoch: {epoch + 1}, Data points: {(i + 1) * 30}/{len(dataset)}, Total loss per batch: {total_loss / 10}')
                    print("Policy:", policy[0].argmax().item(), policy_pred[0].argmax().item())
                    print("Value:", value[0].item(), value_pred[0, 0].item())

                    losses_per_batch.append(total_loss / 10)
                    total_loss = 0.0

            losses_per_epoch.append(sum(losses_per_batch) / len(losses_per_batch))

            if len(losses_per_epoch) > 100:
                if abs(sum(losses_per_epoch[-4:-1]) / 3 - sum(losses_per_epoch[-16:-13]) / 3) <= 0.01:
                    break
            scheduler.step()
# Beispiel für die Verwendung
if __name__ == "__main__":
    # Beispiel für die Nutzung der implementierten Klassen
    import game
    import numpy as np
    import AI
    import time
    from torch.utils.data import Dataset, DataLoader
    import datasets

    path = os.path.join('models', f'first_model_weights.pth')
    model = UltimateTicTacToeNet()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
    TicTacToeZero = AlphaZero(model)

    board = game.board(np.zeros((9,9),dtype=np.int8),1)
    board.execute(1,(1,4))
    ai_board = board.board.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
    ai = AI.all_AI(1)
    possible_moves = ai.get_possible_moves(board).reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
    
    ai_board = torch.Tensor(ai_board).unsqueeze(0)  # Hinzufügen der Batch-Dimension (1x9x9)
    possible_moves = torch.Tensor(possible_moves).unsqueeze(0)  # Hinzufügen der Batch-Dimension (1x9x9)

    # Kombiniere die beiden Tensoren entlang der ersten Achse
    AI_input = torch.cat((ai_board, possible_moves), dim=0)


    evaluation = TicTacToeZero.model(AI_input)

    print(evaluation)

    dataset = datasets.CustomDataset(1)
    single_set = dataset.__getitem__(1)
    #print(single_set)
    #TicTacToeZero.train_neural_net([single_set, single_set])
    TicTacToeZero.train_neural_net([single_set,single_set,single_set,single_set,single_set,single_set,single_set,single_set,single_set,single_set,single_set])

    evaluation = TicTacToeZero.model(AI_input)
    #print(evaluation)
    torch.save(model.state_dict(), path)
    #print(model.state_dict())
    print(f"Das trainierte Modell wurde unter '{path}' gespeichert.")