import torch
from torch import nn
class Self_Attention(nn.Module):
    def __init__(self,dim_in,dim_out,use_qkv_bias=False,dropout=False):
        super().__init__()


        self.W_q=nn.Linear(in_features=dim_in,out_features=dim_out,bias=use_qkv_bias)
        self.W_k=nn.Linear(in_features=dim_in,out_features=dim_out,bias=use_qkv_bias)
        self.W_v=nn.Linear(in_features=dim_in,out_features=dim_out,bias=use_qkv_bias)

        self.out_proj=nn.Linear(in_features=dim_out,out_features=dim_out)
        self.dropout=dropout



    def forward(self,inp):
        query=self.W_q(inp)
        key=self.W_k(inp)
        value=self.W_v(inp)

        attention_scores=query@key.mT
        normalized_attention_scores=torch.softmax(
            attention_scores/key.shape[-1]**0.5,
            dim=-1
        )

        context_vector=normalized_attention_scores@value
        
        if self.dropout:
            context_vector=nn.Dropout(0.1)(context_vector)
        
        context_vector=self.out_proj(context_vector)

        return context_vector
        