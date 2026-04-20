import torch
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2/torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


# Note: 
# torch.pi is the value of pi
# torch.tensor(2/torch.pi) is the value of 2/pi
# torch.sqrt() is the square root function
# torch.tanh() is the hyperbolic tangent function
# torch.pow(x, 3) is the cube of x