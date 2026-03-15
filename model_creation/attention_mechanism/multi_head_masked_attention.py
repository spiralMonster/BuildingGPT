import torch
from torch import nn
from exception_invalid_head_number import Invalid_Head_Number

class Multi_Head_Masked_Attention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 context_length,
                 n_heads,
                 use_qkv_bias=False,
                 dropout=False):
        
        super().__init__()
        if dim_out%n_heads!=0:
            error="Invalid Number of Heads. dim_out should be divided by number of n_heads."
            raise Invalid_Head_Number(error)

        self.dim_in=dim_in
        self.dim_out=dim_out
        self.n_heads=n_heads

        self.head_dim=self.dim_out//self.n_heads

        self.W_q=nn.Linear(in_features=dim_in,out_features=dim_out,bias=use_qkv_bias)
        self.W_k=nn.Linear(in_features=dim_in,out_features=dim_out,bias=use_qkv_bias)
        self.W_v=nn.Linear(in_features=dim_in,out_features=dim_out,bias=use_qkv_bias)

        self.out_proj=nn.Linear(in_features=dim_in,out_features=dim_out)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length,context_length),diagonal=1)
        )

        self.dropout=dropout


    def forward(self,inp):
        batch,context_length,dim_in=inp.shape

        query=self.W_q(inp)
        key=self.W_k(inp)
        value=self.W_v(inp)

        query=query.view(batch,context_length,self.n_heads,self.head_dim)
        key=key.view(batch,context_length,self.n_heads,self.head_dim)
        value=value.view(batch,context_length,self.n_heads,self.head_dim)

        query=query.transpose(1,2)
        key=key.transpose(1,2)
        value=value.transpose(1,2)

        attention_scores=query@key.transpose(2,3)

        mask_bool=self.mask.bool()[:context_length,:context_length]
        masked_attention_scores=attention_scores.masked_fill(mask_bool,-torch.inf)

        masked_attention_weight=torch.softmax(
            masked_attention_scores/key.shape[-1]**0.5,
            dim=-1
        )

        context_vector=(masked_attention_weight@value).transpose(1,2)

        context_vector=context_vector.contiguous().view(
            batch,context_length,self.dim_out
        )

        if self.dropout:
            context_vector=nn.Dropout(0.1)(context_vector)

        context_vector=self.out_proj(context_vector)

        return context_vector
        