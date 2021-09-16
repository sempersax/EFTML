# PyTorch network

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Input of 6 (4vectors for each particle), output of 20
        self.fc1 = nn.Linear(6, 20)
        print(self)
        # Input of 20, output of 20
        self.fc2 = nn.Linear(20,20)
        
        # Input of 20, output of 1 (rhatc)
        self.fin = nn.Linear(20, 1)

    def forward(self, x):
        
        # Takes an input of x, feeds it to fc1, and performs ReLU function
        x = F.relu(self.fc1(x))
        
        # Takes an input of x, feeds it to fc2, and performs ReLU function
        x = F.relu(self.fc2(x))
        
        # Takes an input of x, feeds it to fin so gets output
        x = self.fin(x)
        
        return x

