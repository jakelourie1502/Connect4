import numpy as np 
from IPython.display import clear_output
from tabulate import tabulate
import copy
import time
import torch
from torch.nn import Linear, ReLU, Softmax, Sigmoid

class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

def lambd(x):
    if x < 0:
        return -1
    else:
        return 1
        
class Linear_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #general use
        self.relu = ReLU()
        self.dense1 = Linear(42,100)
        self.dense2 = Linear(100,100)
        self.dense3 = Linear(100,100)
        self.dense4 = Linear(100,30)
        self.dense5p = Linear(30,7)
        self.dense5r = Linear(30,1)
        self.softmax = Softmax(dim=-1)
        self.sigmoid = Sigmoid()
        # self.lamb = LambdaLayer(lambd)
        self.tanh = torch.nn.Tanh()
        
    def forward(self,flat_board): 
        x = self.dense1(flat_board); x=self.relu(x)
        x = self.dense2(x); x=self.relu(x)
        x = self.dense3(x); x=self.relu(x)
        x = self.dense4(x); x=self.relu(x)
        x_p = self.dense5p(x)
        x_r = self.dense5r(x)
        probs = self.softmax(x_p)
        v = self.tanh(x_r)
#         v = self.lamb(v)
        return probs, v
    
class Dataset(torch.utils.data.Dataset):      
    def __init__(self, x, y_P, y_R): 
        self.x = x.float()
        self.y_P = y_P.float()
        self.y_R = y_R.float()

    def __getitem__(self, index):
        return self.x[index],self.y_P[index],self.y_R[index]

    def __len__(self):
        return len(self.y_R)