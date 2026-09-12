import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import PAIRS, Vocab, TranslationDataset, collate_fn
from model.transformer import Seq2SeqTransformer

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build Vocabularies
    src_vocab, tgt_vocab = Vocab(), Vocab()
    src_vocab.build_vocab([p[0] for p in PAIRS])
    tgt_vocab.build_vocab([p[1] for p in PAIRS])

    dataset = TranslationDataset(PAIRS, src_vocab, tgt_vocab)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    model = Seq2SeqTransformer(
        src_vocab_size=len(src_vocab.w2i),
        tgt_vocab_size=len(tgt_vocab.w2i),
        d_model=128, num_layers=2, num_heads=4
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    print("Training MiniTransformer...")
    for epoch in range(1, 26):
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            
            # Target input excludes <eos>, Target output excludes <bos>
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]

            optimizer.zero_grad()
            logits, _ = model(src, tgt_input)

            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_expected.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | Loss: {total_loss/len(dataloader):.4f}")

    # Save artifacts
    torch.save({
        'model_state': model.state_dict(),
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab
    }, 'model.pt')
    print("Training complete! Model saved to model.pt")

if __name__ == "__main__":
    train()