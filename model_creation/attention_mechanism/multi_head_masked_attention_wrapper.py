import torch
from torch import nn
from masked_attention import Masked_Attention

class Multi_Head_Masked_Attention_Wrapper(nn.Module):
    def __init__(self,dim_in,dim_out,n_heads,use_qkv_bias=False,dropout=False):
        super().__init__()

        self.attention_heads=nn.ModuleList(
            [
                Masked_Attention(dim_in=dim_in,dim_out=dim_out,use_qkv_bias=use_qkv_bias,dropout=dropout) for _ in range(n_heads)
            ]
        )


    def forward(self,inp):
        out=torch.cat([head(inp) for head in self.attention_heads],dim=-1)
        return out


        