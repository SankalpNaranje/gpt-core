import torch

def generate_text_sample(model, idx, max_new_tokens, context_size):
    ###Input batch:
 ###tensor([[6109, 3626, 6100,  345],
        ##[6109, 1110, 6622,  257]])
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            
            # Crop current context if it exceeds the supported context size
            idx_cond = idx[:, -context_size:]
        
            # predictions
            logits = model(idx_cond) ### batch, n_tokens, vocab_size
        
            # last time step
            logits = logits[:, -1, :]  

            # softmax to get probabilities
            probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

            # idx of the vocab entry with the highest probability value
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx
