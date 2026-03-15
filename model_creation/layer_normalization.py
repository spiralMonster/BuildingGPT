import torch
from torch import nn

class Layer_Normalization(nn.Module):
    def __init__(self,embedding_dim):
        super().__init__()

        self.epsilon=1e-5
        self.scale=nn.Parameter(torch.ones(embedding_dim),requires_grad=True)
        self.shift=nn.Parameter(torch.zeros(embedding_dim),requires_grad=True)

        def forward(self,inp):
            mean=inp.mean(axis=-1,keepdims=True)
            variance=inp.var(axis=-1,keepdims=True)

            normalized_inp=(inp-mean)/(variance+self.epsilon)**0.5
            normalized_inp=normalized_inp*self.scale+self.shift

            return normalized_inp
        