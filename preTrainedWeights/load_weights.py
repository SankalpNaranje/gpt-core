from preTrainedWeights.gpt_2.download_and_load import download_and_load_gpt2
from config.config import GPT_CONFIG
from GPT.gpt_model import GPTModel
import numpy as np
import torch


def get_pretrained_weights(model_size="124M", models_dir="gpt2"):
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir=models_dir
    )
    return settings, params

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.position_embedding.weight = assign(gpt.position_embedding.weight, params['wpe'])
    gpt.token_embedding.weight = assign(gpt.token_embedding.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transformer_blocks[b].context_vecs.W_query.weight = assign(
            gpt.transformer_blocks[b].context_vecs.W_query.weight, q_w.T)
        gpt.transformer_blocks[b].context_vecs.W_key.weight = assign(
            gpt.transformer_blocks[b].context_vecs.W_key.weight, k_w.T)
        gpt.transformer_blocks[b].context_vecs.W_value.weight = assign(
            gpt.transformer_blocks[b].context_vecs.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.transformer_blocks[b].context_vecs.W_query.bias = assign(
            gpt.transformer_blocks[b].context_vecs.W_query.bias, q_b)
        gpt.transformer_blocks[b].context_vecs.W_key.bias = assign(
            gpt.transformer_blocks[b].context_vecs.W_key.bias, k_b)
        gpt.transformer_blocks[b].context_vecs.W_value.bias = assign(
            gpt.transformer_blocks[b].context_vecs.W_value.bias, v_b)

        gpt.transformer_blocks[b].context_vecs.output_projection.weight = assign(
            gpt.transformer_blocks[b].context_vecs.output_projection.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].context_vecs.output_projection.bias = assign(
            gpt.transformer_blocks[b].context_vecs.output_projection.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.transformer_blocks[b].feed_forward.layers[0].weight = assign(
            gpt.transformer_blocks[b].feed_forward.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transformer_blocks[b].feed_forward.layers[0].bias = assign(
            gpt.transformer_blocks[b].feed_forward.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transformer_blocks[b].feed_forward.layers[2].weight = assign(
            gpt.transformer_blocks[b].feed_forward.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].feed_forward.layers[2].bias = assign(
            gpt.transformer_blocks[b].feed_forward.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.transformer_blocks[b].LayerNorm1.scale = assign(
            gpt.transformer_blocks[b].LayerNorm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.transformer_blocks[b].LayerNorm1.shift = assign(
            gpt.transformer_blocks[b].LayerNorm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.transformer_blocks[b].LayerNorm2.scale = assign(
            gpt.transformer_blocks[b].LayerNorm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.transformer_blocks[b].LayerNorm2.shift = assign(
            gpt.transformer_blocks[b].LayerNorm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.output_head.weight = assign(gpt.output_head.weight, params["wte"])






