import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from inference import translate
from model.transformer import Seq2SeqTransformer

# Configure Matplotlib fonts for Malayalam rendering
plt.rcParams['font.sans-serif'] = ['Nirmala UI', 'Meera', 'Kartika', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Load checkpoint
checkpoint = torch.load('model.pt', map_location='cpu', weights_only=False)
src_vocab = checkpoint['src_vocab']
tgt_vocab = checkpoint['tgt_vocab']

model = Seq2SeqTransformer(
    src_vocab_size=len(src_vocab.w2i),
    tgt_vocab_size=len(tgt_vocab.w2i),
    d_model=128, num_layers=2, num_heads=4
)
model.load_state_dict(checkpoint['model_state'])

def predict_and_plot(text):
    if not text or not text.strip():
        return "Please enter text.", None
    
    translation, attn_map = translate(model, text, src_vocab, tgt_vocab)

    # Convert PyTorch tensor to NumPy array if needed
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.detach().cpu().numpy()

    attn_map = np.array(attn_map)

    # Remove batch dimensions if present (e.g. shape (1, heads, tgt, src) -> (heads, tgt, src))
    attn_map = np.squeeze(attn_map)

    # If 3D (num_heads, tgt_len, src_len), average across heads to get 2D (tgt_len, src_len)
    if attn_map.ndim == 3:
        attn_map = attn_map.mean(axis=0)

    # If it's still higher dimension, flatten extra dims down to 2D
    while attn_map.ndim > 2:
        attn_map = attn_map[0]

    src_words = text.lower().split()
    tgt_words = translation.split()

    if not tgt_words or not src_words:
        return translation, None

    # Slice attention map to active sentence length
    attn_slice = attn_map[:len(tgt_words), :len(src_words)]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        attn_slice, 
        xticklabels=src_words, 
        yticklabels=tgt_words, 
        annot=True, 
        cmap="viridis", 
        ax=ax
    )
    ax.set_title("Cross-Attention Alignment Heatmap")
    ax.set_xlabel("Source Words")
    ax.set_ylabel("Generated Words")
    plt.tight_layout()

    return translation, fig

demo = gr.Interface(
    fn=predict_and_plot,
    inputs=gr.Textbox(label="Source English Sentence", value="hello world"),
    outputs=[
        gr.Textbox(label="Translated Malayalam Text"),
        gr.Plot(label="Attention Heatmap Matrix")
    ],
    title="MiniTransformer — Built From Scratch",
    description="A lightweight PyTorch Transformer implementation featuring cross-attention weight visualization."
)

if __name__ == "__main__":
    demo.launch()