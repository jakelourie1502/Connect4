import numpy as np 
from IPython.display import clear_output
from tabulate import tabulate
import copy
import time
import torch
from torch.nn import Linear, ReLU, Softmax, Sigmoid, Conv2d

        
class Conv_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #general use
        self.relu = ReLU()
        self.flatten = torch.nn.Flatten(start_dim=1,end_dim=-1)

        self.Rconv1 = Conv2d(2,16,(3,3),stride=1,padding=1)
        self.Rconv2 = Conv2d(16,32,(3,3),stride=1,padding=0)
        self.Rconv3 = Conv2d(32,64,(3,3),stride=1,padding=1)
        self.Rconv4 = Conv2d(64,64,(3,3),stride=1,padding=0)
        self.Rdense1 = Linear(384,200)
        self.Rdense2 = Linear(200,160)
        self.Rdense3 = Linear(160,30)
        self.Rdense4 = Linear(30,1)
        self.Rtanh = torch.nn.Tanh()
        
        self.Pconv1 = Conv2d(2,16,(3,3),stride=1,padding=1)
        self.Pconv2 = Conv2d(16,32,(3,3),stride=1,padding=0)
        self.Pconv3 = Conv2d(32,64,(3,3),stride=1,padding=1)
        self.Pconv4 = Conv2d(64,64,(3,3),stride=1,padding=0)
        self.Pdense1 = Linear(384,200)
        self.Pdense2 = Linear(200,160)
        self.Pdense3 = Linear(160,30)
        self.Pdense4 = Linear(30,7)
        self.Psoftmax = Softmax(dim=1)
        

    def forward(self,board): 
        #### REWARDS
        xR = self.Rconv1(board); xR = self.relu(xR)
        xR = self.Rconv2(xR); xR = self.relu(xR)
        xR = self.Rconv3(xR); xR = self.relu(xR)
        xR = self.Rconv4(xR); xR = self.relu(xR)
        xR = self.flatten(xR)

        xR = self.Rdense1(xR); xR=self.relu(xR)
        xR = self.Rdense2(xR); xR=self.relu(xR)
        xR = self.Rdense3(xR); xR=self.relu(xR)
        
        xR = self.Rdense4(xR)
        
        v = self.Rtanh(xR)
        #### PROBS
        xP = self.Pconv1(board); xP = self.relu(xP)
        xP = self.Pconv2(xP); xP = self.relu(xP)
        xP = self.Pconv3(xP); xP = self.relu(xP)
        xP = self.Pconv4(xP); xP = self.relu(xP)
        xP = self.flatten(xP)

        xP = self.Pdense1(xP); xP=self.relu(xP)
        xP = self.Pdense2(xP); xP=self.relu(xP)
        xP = self.Pdense3(xP); xP=self.relu(xP)
        xP = self.Pdense4(xP)
        probs = self.Psoftmax(xP)
        
        
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