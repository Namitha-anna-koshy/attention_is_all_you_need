import torch

def translate(model, src_sentence, src_vocab, tgt_vocab, max_len=15, device="cpu"):
    model.eval()
    src_tokens = torch.tensor(src_vocab.encode(src_sentence)).unsqueeze(0).to(device)

    # Encode source text
    with torch.no_grad():
        enc_out = model.encoder(src_tokens)

    # Begin autoregressive decoding with <bos>
    tgt_ids = [tgt_vocab.w2i[tgt_vocab.bos]]
    
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_ids).unsqueeze(0).to(device)
        
        with torch.no_grad():
            dec_out, attn_weights = model.decoder(tgt_tensor, enc_out)
            logits = model.projection_head(dec_out)

        # Get top predicted next token ID
        next_token = logits[0, -1, :].argmax().item()
        if next_token == tgt_vocab.w2i[tgt_vocab.eos]:
            break
        tgt_ids.append(next_token)

    translated_text = tgt_vocab.decode(tgt_ids)
    return translated_text, attn_weights[0].cpu().squeeze(0).numpy()