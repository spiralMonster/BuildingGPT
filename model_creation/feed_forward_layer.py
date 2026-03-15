import torch
from torch import nn
from gelu_activation import GELU_Activation

class Feed_Forward_Layer(nn.Module):
    def __init__(self,inp_dim):
        super().__init__()

        self.layers=nn.Sequential(
            nn.Linear(in_features=inp_dim,out_features=4*inp_dim),
            GELU_Activation(),
            nn.Linear(in_features=4*inp_dim,out_features=inp_dim)
            
        )


    def forward(self,inp):
        x=self.layers(inp)

        return x