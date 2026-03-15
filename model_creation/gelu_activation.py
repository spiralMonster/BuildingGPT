import torch
from torch import nn

class GELU_Activation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,inp):
        x=0.5*inp*(1+torch.tanh((2.0/torch.pi)**0.5*(inp+0.044715*inp**3)))

        return x