import torch
from dataset.train_val_split import train_loader, val_loader
from GPT.gpt_model import GPTModel
from config.config import GPT_CONFIG
from utils.device import get_device


def calculate_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

# computes average loss over multiple batches
def calculate_loss_loader(dataloader, model, device, num_batches=None):
    total_loss = 0.
    if len(dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dataloader)

    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(dataloader))
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i % 10 == 0:
            print(f"Processing batch {i}/{num_batches}")
        if i < num_batches:
            loss = calculate_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


if __name__ == "__main__":
    device = get_device()
    model = GPTModel(GPT_CONFIG)
    model.eval()
    model.to(device)

    torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

    with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calculate_loss_loader(train_loader, model, device)
        val_loss = calculate_loss_loader(val_loader, model, device)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)