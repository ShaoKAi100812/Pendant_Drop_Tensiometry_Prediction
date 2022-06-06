# pytorch
import torch.nn as nn
from model_pic import *
from model_cal import *

class UnitedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.front = PictureNet(1, 16, 32, 64, 1024, 1024, 160)
        self.back = PhysicsNet(160, 512, 512, 256, 256, 64, 16, 2)
        

    def forward(self, x):
        x = self.front(x)
        scores = self.back(x)
        return scores