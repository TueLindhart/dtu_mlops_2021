import torch
import torch.nn.functional as F
from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor):

        if len(x.size()) != 4:
            raise ValueError('Expected input to be of size 4D')
        if (x.shape[1] != 1) or (x.shape[2] != 28) or (x.shape[3] != 28):
            raise ValueError('Expected image to have size (1,28,28)')

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x,dim=1)
        return x
