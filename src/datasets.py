import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, size):
        self.size = size
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        state = torch.zeros(2, 9, 9)  # Erstelle ein leeres 9x9 Array
        state[0][1][4] = 1  # Setze den State bei (1, 4) auf 1
        state[1, 3:6, 3:6] = 1

        policy = torch.zeros(9, 9)  # Erstelle ein leeres 9x9 Policy-Array
        policy[[3, 5, 5, 3], [3, 3, 5, 5]] = 0.1  # Setze die Policy für [3,3], [3,5],[5,3],[5,5] auf 0.1
        policy[[3, 4, 4, 5], [4, 3, 5, 4]] = 0.05  # Setze die Policy für [3,4], [4,3],[4,5],[5,4] auf 0.05
        policy[4][4] = 0.4  # Setze die Policy bei [4,4] auf 0.4
        policy = policy.view(-1)
        value = torch.tensor([0.5])  # Setze den Value auf 1

        return [state, policy, value]


