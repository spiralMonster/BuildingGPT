import torch
from torch import nn

from attention_mechanism.multi_head_masked_attention import Multi_Head_Masked_Attention
from layer_normalization import Layer_Normalization
from feed_forward_layer import Feed_Forward_Layer


class Transformer_Block(nn.Module):
    def __init__(self,
                 hidden_dim,
                 context_length,
                 n_attention_heads,
                 dropout_rate,
                 use_qkv_bias=False,
                 use_attention_dropout=False):
        
        super().__init__()

        self.layer_norm1=Layer_Normalization(embedding_dim=hidden_dim)
        self.layer_norm2=Layer_Normalization(embedding_dim=hidden_dim)

        self.feed_forward_layer=Feed_Forward_Layer(inp_dim=hidden_dim)
        
        self.multi_head_masked_attn=Multi_Head_Masked_Attention(
            dim_in=hidden_dim,
            dim_out=hidden_dim,
            context_length=context_length,
            n_heads=n_attention_heads,
            use_qkv_bias=use_qkv_bias,
            dropout=use_attention_dropout
        )

        self.dropout1=nn.Dropout(dropout_rate)
        self.dropout2=nn.Dropout(dropout_rate)
        


    def forward(self,inp):
        shortcut_connection=inp

        x=layer_norm1(inp)
        x=multi_head_masked_attn(x)
        x=self.dropout1(x)

        x=x+shortcut
        shortcut=x

        x=layer_norm2(x)
        x=feed_forward_layer(x)
        x=self.dropout2(x)

        x=shortcut+x

        return x
        
        