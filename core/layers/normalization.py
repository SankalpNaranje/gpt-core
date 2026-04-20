import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        output = self.scale * normalized + self.shift
        return output


# Note: unbiased = True (bessel's correction) is used for small sample sizes (1/ N-1)
# unbiased = False (population variance) is used for large sample sizes (1/N)
# In deep learning, we usually use unbiased = False because we have large sample sizes
# and we want to minimize the error in the variance calculation