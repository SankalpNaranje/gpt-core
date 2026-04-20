import torch
from GPT.gpt_model import GPTModel
from config.config import GPT_CONFIG

torch.manual_seed(123)

# 2 input texts with 4 tokens(context length) each
batch_input = torch.tensor([  #Example
    [512, 2048, 1234, 999],   # "Once   upon   a   time"
    [876, 3456, 2222, 111]    # "In     a     far  land"
])

model = GPTModel(GPT_CONFIG)
logits = model(batch_input)
print("Input batch:\n", batch_input)
print("Input batch shape:\n", batch_input.shape)

print("\nOutput logits:\n", logits)
print("\nOutput shape:", logits.shape)

