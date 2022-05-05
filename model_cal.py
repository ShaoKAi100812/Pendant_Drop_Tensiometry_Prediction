# pytorch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions

class PhysicsNet(nn.Module):
    def __init__(self, in_channel, node_1, node_2, node_3, node_4, num_classes):
        super().__init__()
        # fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel, node_1),
            nn.BatchNorm2d(node_1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(node_1, node_2),
            nn.BatchNorm2d(node_2),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(node_2, node_3),
            nn.BatchNorm2d(node_3),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(node_3, node_4),
            nn.LeakyReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(node_4, num_classes)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        scores = self.out(x)
        return scores