"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

weights_init_min = -1
weights_init_max=1

import torch
import torch.nn as nn

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        self.conv1.weight.requires_grad_(False)
        self.conv2.weight.requires_grad_(False)
        self.conv3.weight.requires_grad_(False)

        torch.nn.init.uniform_(self.conv1.weight,
                                   a=weights_init_min, b=weights_init_max)
        torch.nn.init.uniform_(self.conv2.weight,
                                   a=weights_init_min, b=weights_init_max)
        torch.nn.init.uniform_(self.conv3.weight,
                                   a=weights_init_min, b=weights_init_max)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x

