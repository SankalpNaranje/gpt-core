import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.output_projection = nn.Linear(d_out,d_out)
        self.dropout = nn.Dropout(dropout)
    
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        #Shape: (Batch size , number of tokens, d_out)

        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        #Shape: (Batch size ,number of tokens, number of heads, head_dim)
        #d_out = num_heads * head_dim
        #This is done to parallelize the attention mechanism

        #Transpose to (Batch size , number of heads, number of tokens, head_dim)
        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        attention_score = queries @ keys.transpose(2,3)
        
        #Apply mask to prevent looking ahead
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_score = attention_score.masked_fill(mask_bool, -torch.inf)

        #apply softmax across each row
        attention_weights = torch.softmax(attention_score / (keys.shape[-1] ** 0.5), dim=-1)

        #apply dropout
        attention_weights = self.dropout(attention_weights)

        #compute weighted average of values
        context_vec = attention_weights @ values
        #Shape: (Batch size , number of heads, number of tokens, head_dim)

        #Concatenate heads
        context_vec = context_vec.transpose(1,2).contiguous().view(batch_size, num_tokens, self.d_out)
        #Shape: (Batch size , number of tokens, d_out)

        #Apply output projection
        context_vec = self.output_projection(context_vec)
        #Shape: (Batch size , number of tokens, d_out)

        return context_vec

        
       


        

        
        



