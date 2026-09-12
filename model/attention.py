import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        # Q, K, V shape: [batch_size, num_heads, seq_len, head_dim]
        d_k = Q.size(-1)
        
        # 1. Compute attention scores (QK^T / sqrt(d_k))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 2. Apply causal/padding mask if present
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 3. Softmax along last dimension to get probabilities
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 4. Multiply by Values
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 1. Linear project & split into heads -> [batch_size, num_heads, seq_len, head_dim]
        Q_proj = self.W_q(Q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K_proj = self.W_k(K).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V_proj = self.W_v(V).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Expand mask for multi-head dimension if mask exists
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            mask = mask.unsqueeze(1) # [batch_size, 1, seq_len, seq_len]
            
        # 2. Compute Scaled Dot-Product Attention
        out, attn_weights = self.attention(Q_proj, K_proj, V_proj, mask)
        
        # 3. Concatenate heads back together -> [batch_size, seq_len, d_model]
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. Final linear projection
        return self.W_o(out), attn_weights