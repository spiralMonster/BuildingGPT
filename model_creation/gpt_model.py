import torch
from torch import nn

from transformer_block import Transformer_Block
from layer_normalization import Layer_Normalization

class GPTModel(nn.Module):
    def __init__(self,model_config):
        super().__init__()

        self.token_embedding=nn.Embedding(model_config["vocab_size"],model_config["embedding_dim"])
        self.positional_embedding=nn.Embedding(model_config["context_length"],model_config["embedding_dim"])
        
        self.transformer_blocks=nn.Sequential(
             *[Transformer_Block(
                hidden_dim=model_config["embedding_dim"],
                context_length=model_config["context_length"],
                n_attention_heads=model_config["n_attention_heads"],
                dropout_rate=model_config["dropout_rate"],
                use_qkv_bias=model_config["use_qkv_bias"],
                use_attention_dropout=model_config["use_attention_dropout"]
                
            ) for _ in range(model_config["n_layers"])]
        )

        self.final_layer_norm=Layer_Normalization(embedding_dim=model_config["embedding_dim"])
        self.output_layer=nn.Linear(in_features=model_config["embedding_dim"],
                                    out_features=model_config["vocab_size"])

    def forward(self,inp):
        batch,seqlen=inp.shape

        tok_embed=self.token_embedding(inp)
        pos_embed=self.positional_embedding(torch.arange(seqlen,device=inp.device))

        final_embedding=tok_embed+pos_embed

        x=self.transformer_blocks(final_embedding)
        x=self.final_layer_norm(x)
        
        final_out=self.output_layer(x)

        return final_out

        






        
        