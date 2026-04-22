import torch
import argparse
from tokenizer.tokenizer_utils import text_to_token_ids, token_ids_to_text
from config.config import GPT_CONFIG
from tokenizer.tokenizer_utils import get_tokenizer
from utils.device import get_device
from GPT.gpt_model import GPTModel
from utils.text_generation import generate_text_sample
from utils.text_generation_top_k import generate

def print_model_stats(model):
    # Calculate total number of parameters
    total_parameters = sum(i.numel() for i in model.parameters())
    print(f"\n\nTotal number of parameters: {total_parameters:,}")

    # Memory Requirement
    total_size_bytes = total_parameters * 4  # Assuming float32
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()
    
    device = get_device()
    tokenizer = get_tokenizer()


    #custom trained checkpoint
    #---------------------------------------------------------------------------
    checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
    model = GPTModel(GPT_CONFIG)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Print stats only if flag is passed
    if args.stats:
        print_model_stats(model)

    #without tempreature scaling and top-k sampling
    #---------------------------------------------------------------------------
    # idx = text_to_token_ids("Every effort moves you", tokenizer).to(device)
    # token_ids = generate_text_sample(
    #     model=model,
    #     idx=idx,
    #     max_new_tokens=30,
    #     context_size=GPT_CONFIG["context_length"]
    # )

    
    #with tempreature scaling and top-k sampling
    #---------------------------------------------------------------------------
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG["context_length"],
        top_k=25,
        temperature=1.4
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    

    
