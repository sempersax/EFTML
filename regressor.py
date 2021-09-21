# PyTorch network

import torch.nn as nn
import torch.nn.functional as F


class Conv1dNet(nn.Module):
    def __init__(self, networks):
        super(Conv1dNet, self).__init__()

        # Input of 6 (4vectors for each particle), output of 20
        self.x_to_z1 = nn.Linear(6, 20 * networks)

        # Input of 20, output of 20
        self.z1_to_z2 = nn.Conv1d(20 * networks, 20 * networks, kernel_size=1, groups=networks)
        # the input is grouped into groups of 20 channels,
        # and each group is used to produce 20 channels of output

        # Input of 20, output of 1 (rhatc)
        self.z2_to_y = nn.Conv1d(20 * networks, 1 * networks, kernel_size=1, groups=networks)

    def forward(self, x):
        # Takes an input of x, feeds it to fc1, and performs ReLU function
        x = F.relu(self.x_to_z1(x))
        # x = x.view(batch_size, 20, 1)

        # Takes an input of x, feeds it to fc2, and performs ReLU function
        x = F.relu(self.z1_to_z2(x))

        # Takes an input of x, feeds it to fin so gets output
        x = self.z2_to_y(x)

        return x


class LinearNet(nn.Module):
    def __init__(self, networks):
        super(LinearNet, self).__init__()

        # Input of 6 (4vectors for each particle), output of 20
        self.x_to_z1 = nn.Linear(6, 20)

        # Input of 20, output of 20
        self.z1_to_z2 = nn.Linear(20, 20)
        # the input is grouped into groups of 20 channels,
        # and each group is used to produce 20 channels of output

        # Input of 20, output of 1 (rhatc)
        self.z2_to_y = nn.Linear(20 , 1)

    def forward(self, x):
        # Takes an input of x, feeds it to fc1, and performs ReLU function
        x = F.relu(self.x_to_z1(x))
        # x = x.view(batch_size, 20, 1)

        # Takes an input of x, feeds it to fc2, and performs ReLU function
        x = F.relu(self.z1_to_z2(x))

        # Takes an input of x, feeds it to fin so gets output
        x = self.z2_to_y(x)

        return x

