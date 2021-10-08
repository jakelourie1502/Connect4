import numpy as np 
from IPython.display import clear_output
from tabulate import tabulate
import copy
import time
import torch
from torch.nn import Linear, ReLU, Softmax, Sigmoid, Conv2d

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
        
class Conv_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #general use
        self.relu = ReLU()
        self.conv1 = Conv2d(2,16,(3,3),stride=1,padding=1)
        self.conv2 = Conv2d(16,32,(3,3),stride=1,padding=0)
        self.conv3 = Conv2d(32,64,(3,3),stride=1,padding=1)
        self.conv4 = Conv2d(64,64,(3,3),stride=1,padding=0)
        self.flatten = torch.nn.Flatten(start_dim=1,end_dim=-1)
        self.dense1 = Linear(384,200)
        self.dense2 = Linear(200,160)
        self.dense3 = Linear(160,30)
        self.dense4p = Linear(30,7)
        self.dense4r = Linear(30,1)
        self.softmax = Softmax(dim=1)
        self.sigmoid = Sigmoid()
        self.lamb = LambdaLayer(lambd)
        self.tanh = torch.nn.Tanh()
        
    def forward(self,board): 
        x = self.conv1(board); x = self.relu(x)
        x = self.conv2(x); x = self.relu(x)
        x = self.conv3(x); x = self.relu(x)
        x = self.conv4(x); x = self.relu(x)
        x = self.flatten(x)

        x1 = self.dense1(x); x1=self.relu(x1)
        x2 = self.dense2(x1); x2=self.relu(x2)
        x3 = self.dense3(x2); x3=self.relu(x3)
        x4p = self.dense4p(x3)
        x4r = self.dense4r(x3)
        probs = self.softmax(x4p)
        v = self.tanh(x4r)
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