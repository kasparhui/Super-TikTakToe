import torch
from torch.utils.data import Dataset
import game
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, size):
        self.size = size
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        Game = game.Board()
        if idx == 0:
            state = Game.get_encoded_state()

            policy = np.zeros(81)  # Erstelle ein leeres 9x9 Policy-Array
            for i in range(81):
                policy[i] = i/3321  + 1/3321
            policy = torch.tensor(policy)

            value = 0.5  # Setze den Value auf 1
            value = torch.tensor(value)

        elif idx ==1:
            Game.execute(20)
            Game.change_perspective()
            state = Game.get_encoded_state()
            policy = np.zeros(81)  # Erstelle ein leeres 9x9 Policy-Array
            for i in range(81):
                policy[i] = i/3321  + 1/3321
            policy = torch.tensor(policy)

            value = -0.7  # Setze den Value auf 1
            value = torch.tensor(value)

        elif idx ==2:
            Game.execute(60)
            Game.change_perspective()
            state = Game.get_encoded_state()
            policy = np.zeros(81)  # Erstelle ein leeres 9x9 Policy-Array
            for i in range(81):
                policy[i] = i/3321  + 1/3321
            policy = torch.tensor(policy)

            value = 0.7  # Setze den Value auf 1
            value = torch.tensor(value)

        return [state, policy, value]    

if __name__ == '__main__':
    dataset = CustomDataset(1)
    dataset[0]


