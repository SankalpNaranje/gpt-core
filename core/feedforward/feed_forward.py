import torch
import torch.nn as nn
from core.activations.gelu import GELU

class FeedForward(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
        )
    
    def forward(self,x):
        return self.layers(x)

# Note: GELU (non-linear activation function) and nn.Linear (linear transformation) gets x because it is part of nn.Sequential, which automatically passes the output of one layer as input to the next.