
import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.encoder import PositionwiseFeedForward
from model.positional_encoding import PositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, d_model=128, num_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.masked_mha = MultiHeadAttention(d_model, num_heads)
        self.cross_mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_output, tgt_mask=None, cross_mask=None):
        # 1. Masked Self-Attention
        tgt_attn, _ = self.masked_mha(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt_attn))
        
        # 2. Cross-Attention (Q = Target, K = Enc Output, V = Enc Output)
        cross_attn, weights = self.cross_mha(tgt, enc_output, enc_output, cross_mask)
        tgt = self.norm2(tgt + self.dropout(cross_attn))
        
        # 3. Feed Forward Network
        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(ffn_out))
        
        return tgt, weights

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_output, tgt_mask=None, cross_mask=None):
        x = self.dropout(self.positional_encoding(self.embedding(tgt)))
        last_attn_weights = None
        
        for layer in self.layers:
            x, last_attn_weights = layer(x, enc_output, tgt_mask, cross_mask)
            
        return x, last_attn_weights