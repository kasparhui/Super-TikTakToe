# Imports
import torch
import torch.nn as nn
import os
import MCTS
import game
import copy


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(4, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.activation = nn.ReLU()

    def forward(self, s):
        s = s.view(-1, 4, 9, 9)  # batch_size x channels x board_x x board_y
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
        self.softmax = nn.Softmax(dim=1)
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
        p = self.softmax(p)
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

# Hauptalgorithmus für das Training von AlphaZero
class AlphaLoss(nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy * (1e-6 + y_policy.float()).float().log()))
        #print(value_error,'ve',policy_error,'pe')
        lamm = 0.5
        total_error = (value_error.view(-1).float() * lamm + policy_error * (1-lamm)).mean()
        #print(total_error,'Total error')
        return total_error
    
class AlphaZero:
    def __init__(self, model):
        # Initialisiere Parameter für das Training
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.003)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100, 200, 300, 400], gamma=0.2)
        if os.path.exists(os.path.join('models', 'neural_net',f'first_model_weights.pth')):
            self.model.load_state_dict(torch.load(os.path.join('models', 'neural_net',f'first_model_weights.pth')))
        if os.path.exists(os.path.join('optimizer', f'first_model_weights.pth')):
            self.optimizer.load_state_dict(torch.load(os.path.join('models', 'optimizer', f'first_model_weights.pth')))
        if os.path.exists(os.path.join('scheduler', f'first_model_weights.pth')):
            self.scheduler.load_state_dict(torch.load(os.path.join('models', 'scheduler', f'first_model_weights.pth')))

    def self_play(self):
        # Führe Selbstspiele durch, um Daten für das Training zu generieren
        memory = []
        playerturn = 1
        state = game.Board(playerturn=playerturn)
        while True:
            neutral_state = copy.deepcopy(state)
            neutral_state.change_perspective()
            node  = MCTS.AlphaNode(neutral_state)
            mcts = MCTS.AlphaMonteCarloTreeSearch(self.model)
            action_probs = mcts.best_move(node, 164)[0]
            memory.append([neutral_state.get_encoded_state(), torch.tensor(action_probs), torch.tensor(state.playerturn)])         
            action = np.random.choice(81, p=action_probs)
            
            state.execute(action)
            #print(state.won_games)
            if state.game_is_over:
                if game.game_is_win(state.won_games)[0]:
                    value = - state.playerturn
                else: 
                    #print(state.won_games)
                    value = 0
                print('value:', value)
                print(state.won_games)
                for i in range(len(memory)):
                    memory[i][2]*= value
                return memory

    def train_neural_net(self, dataset, epoch_start=0, epoch_stop=20, cpu=0):
        torch.manual_seed(cpu)
        cuda = torch.cuda.is_available()
        cuda = False
        criterion = AlphaLoss()
        optimizer = self.optimizer
        scheduler = self.scheduler

        train_loader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0, pin_memory=False)
        losses_per_epoch = []

        for epoch in range(epoch_start, epoch_stop):
            total_loss = 0.0
            losses_per_batch = [100]

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
                    '''print(f'Epoch: {epoch + 1}, Data points: {(i + 1) * 30}/{len(dataset)}, Total loss per batch: {total_loss / 10}')
                    print("Policy:", policy[0].argmax().item(), policy_pred[0].argmax().item())
                    print("Value:", value[0].item(), value_pred[0, 0].item())'''

                    losses_per_batch.append(total_loss / 10)
                    total_loss = 0.0

            losses_per_epoch.append(sum(losses_per_batch) / len(losses_per_batch))

            if len(losses_per_epoch) > 100:
                if abs(sum(losses_per_epoch[-4:-1]) / 3 - sum(losses_per_epoch[-16:-13]) / 3) <= 0.01:
                    break
            scheduler.step()
        #print(losses_per_epoch)

    def learn(self, iterations = 5):
        # Führe das Training durch
        for iteration in range(iterations):
            print('iteraton:',iteration)
            try:
                #self.model.load_state_dict(torch.load(os.path.join('models', 'neural_net',f'first_model_weights.pth')))
                memory = []
                self.model.eval()
                for self_play_iteration in range(3):
                    new_data = self.self_play()
                    if new_data is not None:
                        memory += new_data
                    else:
                        print('Errorm Data is None')
                self.model.train()
                print(f'train iteration {iteration}')
                self.train_neural_net(memory,0,10)
            except KeyboardInterrupt:
                print('KeyboardInterrupt')
                break
            except any:
                print('Error')
                break
            print(f'finito iteration {iteration}')
        torch.save(self.model.state_dict(), os.path.join('models', 'neural_net', f'first_model_weights.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join('models', 'optimizer', f'first_model_weights.pth'))
        torch.save(self.scheduler.state_dict(), os.path.join('models', 'scheduler', f'first_model_weights.pth'))
        print('alles wurde gesichert')
# Beispiel für die Verwendung
if __name__ == "__main__":
    # Beispiel für die Nutzung der implementierten Klassen
    import game  
    import numpy as np
    import AI
    import time
    from torch.utils.data import Dataset, DataLoader
    import datasets
    import time

    cuda = torch.cuda.is_available()
    cuda = False

    model = UltimateTicTacToeNet()
    if cuda:
        model = model.cuda()
    TicTacToeZero = AlphaZero(model)

    TicTacToeZero.learn(100)

    torch.save(TicTacToeZero.model.state_dict(), os.path.join('models', 'neural_net', f'first_model_weights.pth'))

    