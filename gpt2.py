# import os
# import torch
# from GPT.gpt_model import GPTModel
# from config.config import NEW_CONFIG
# from utils.device import get_device
# from preTrainedWeights.load_weights import get_pretrained_weights, load_weights_into_gpt
# from tokenizer.tokenizer_utils import text_to_token_ids, token_ids_to_text, get_tokenizer
# from utils.text_generation_top_k import generate

# CHECKPOINT_PATH = "gpt2_pretrained.pth"

# if __name__ == "__main__":
#     device = get_device()
#     tokenizer = get_tokenizer()

#     model = GPTModel(NEW_CONFIG).to(device)

#     if os.path.exists(CHECKPOINT_PATH):
#         print("Loading saved pretrained weights...")
#         model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
#     else:
#         print("Loading GPT-2 weights (first time)...")
#         settings, params = get_pretrained_weights()
#         load_weights_into_gpt(model, params)

#         torch.save(model.state_dict(), CHECKPOINT_PATH)
#         print("Saved pretrained weights.")

#     model.eval()

#     idx = text_to_token_ids("When you have a bad day, remember that", tokenizer).to(device)

#     token_ids = generate(
#         model=model,
#         idx=idx,
#         max_new_tokens=25,
#         context_size=NEW_CONFIG["context_length"],
#         top_k=25,
#         temperature=1.4
#     )

#     print("Output text:\n", token_ids_to_text(token_ids, tokenizer))



import os
import torch
import argparse
from GPT.gpt_model import GPTModel
from config.config import NEW_CONFIG
from utils.device import get_device
from preTrainedWeights.load_weights import get_pretrained_weights, load_weights_into_gpt
from tokenizer.tokenizer_utils import text_to_token_ids, token_ids_to_text, get_tokenizer
from utils.text_generation_top_k import generate

CHECKPOINT_PATH = "gpt2_pretrained.pth"

if __name__ == "__main__":
    # ---------------- CLI Arguments ----------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Hi, Greetings!")
    parser.add_argument("--max_tokens", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=1.4)
    parser.add_argument("--top_k", type=int, default=25)

    args = parser.parse_args()

    # ---------------- Setup ----------------
    device = get_device()
    tokenizer = get_tokenizer()
    model = GPTModel(NEW_CONFIG).to(device)

    # ---------------- Load Weights ----------------
    if os.path.exists(CHECKPOINT_PATH):
        print("Loading saved pretrained weights...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    else:
        print("Loading GPT-2 weights (first time)...")
        settings, params = get_pretrained_weights()
        load_weights_into_gpt(model, params)

        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print("Saved pretrained weights.")

    model.eval()

    # ---------------- Use CLI Prompt ----------------
    idx = text_to_token_ids(args.prompt, tokenizer).to(device)

    token_ids = generate(
        model=model,
        idx=idx,
        max_new_tokens=args.max_tokens,
        context_size=NEW_CONFIG["context_length"],
        top_k=args.top_k,
        temperature=args.temperature
    )

    print("Prompt:", args.prompt)
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))