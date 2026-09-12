import torch
from torch.utils.data import Dataset, DataLoader

# English to Malayalam sentence pairs
PAIRS = [
    ("hello world", "നമസ്കാരം ലോകമേ"),
    ("good morning", "സുപ്രഭാതം"),
    ("i love cats", "എനിക്ക് പൂച്ചകളെ ഇഷ്ടമാണ്"),
    ("i love dogs", "എനിക്ക് നായകളെ ഇഷ്ടമാണ്"),
    ("the cat is small", "പൂച്ച ചെറുതാണ്"),
    ("the dog is big", "പട്ടി വലുതാണ്"),
    ("i am happy", "ഞാൻ സന്തോഷവാനാണ്"),
    ("i am sad", "ഞാൻ ദുഃഖിതനാണ്"),
    ("hello my friend", "നമസ്കാരം എന്റെ കൂട്ടുകാരാ"),
    ("i love programming", "എനിക്ക് പ്രോഗ്രാമിംഗ് ഇഷ്ടമാണ്")
] * 50  # Duplicate to create a lightweight dataset (~500 pairs)

class Vocab:
    def __init__(self):
        self.pad, self.bos, self.eos, self.unk = "<pad>", "<bos>", "<eos>", "<unk>"
        self.w2i = {self.pad: 0, self.bos: 1, self.eos: 2, self.unk: 3}
        self.i2w = {0: self.pad, 1: self.bos, 2: self.eos, 3: self.unk}

    def build_vocab(self, sentences):
        for s in sentences:
            for word in s.lower().split():
                if word not in self.w2i:
                    idx = len(self.w2i)
                    self.w2i[word] = idx
                    self.i2w[idx] = word

    def encode(self, sentence, add_specials=True):
        tokens = [self.w2i.get(w, self.w2i[self.unk]) for w in sentence.lower().split()]
        if add_specials:
            return [self.w2i[self.bos]] + tokens + [self.w2i[self.eos]]
        return tokens

    def decode(self, ids):
        words = []
        for i in ids:
            if i == self.w2i[self.eos]: break
            if i not in (self.w2i[self.pad], self.w2i[self.bos]):
                words.append(self.i2w.get(i, self.unk))
        return " ".join(words)

class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.data = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.tensor(self.src_vocab.encode(src)), torch.tensor(self.tgt_vocab.encode(tgt))

def collate_fn(batch):
    src_list, tgt_list = zip(*batch)
    src_pad = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=0)
    tgt_pad = torch.nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=0)
    return src_pad, tgt_pad