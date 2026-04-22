import time
from dataset.training.train_gpt import train_gpt
from GPT.gpt_model import GPTModel
from config.config import GPT_CONFIG
from dataset.train_val_split import train_loader, val_loader
from tokenizer.tokenizer_utils import get_tokenizer
from utils.device import get_device
import torch

def build_model():
    device = get_device()
    model = GPTModel(GPT_CONFIG)
    model.to(device)
    return model, device

if __name__ == "__main__":
    start_time = time.time()

    torch.manual_seed(123)

    model, device = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    tokenizer = get_tokenizer()

    num_epochs = 15
    train_losses, val_losses, tokens_seen = train_gpt(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # Save the trained model state and optimizer state
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        }, 
        "model_and_optimizer.pth"
    )