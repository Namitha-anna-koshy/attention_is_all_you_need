import torch
import torch.nn as nn
from model.encoder import TransformerEncoder
from model.decoder import TransformerDecoder
from model.positional_encoding import PositionalEncoding

class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.encoder = TransformerEncoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout)
        
        self.projection_head = nn.Linear(d_model, tgt_vocab_size)

    def make_causal_mask(self, tgt):
        # Shape: [seq_len, seq_len] lower triangular matrix
        seq_len = tgt.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=tgt.device)).bool()
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode
        enc_out = self.encoder(src, src_mask)
        
        # Causal mask for decoder self-attention
        if tgt_mask is None:
            tgt_mask = self.make_causal_mask(tgt)
            
        # Decode
        dec_out, attn_weights = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        
        # Project to target vocabulary distribution
        logits = self.projection_head(dec_out)
        return logits, attn_weights