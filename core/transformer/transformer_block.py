import torch.nn as nn
from core.attention.multi_head_attention import MultiHeadAttention
from core.feedforward.feed_forward import FeedForward
from core.layers.normalization import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context_vecs = MultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            num_heads=config["n_heads"], 
            dropout=config["drop_rate"],
            qkv_bias=config["qkv_bias"])
        
        self.feed_forward = FeedForward(config)
        self.LayerNorm1 = LayerNorm(config["emb_dim"])
        self.LayerNorm2 = LayerNorm(config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.LayerNorm1(x)
        x = self.context_vecs(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.dropout(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.LayerNorm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + shortcut  # Add the original input back
        
        return x