
# Attention Is All You Need

> A simple, accessible breakdown of the Transformer architecture from the paper *"Attention Is All You Need"* (Vaswani et al., 2017).

---

## INTRO

Imagine you're translating a sentence word by word, and you have to remember everything you've read so far.

Sounds easy, right?

Well... apparently not. 

Before Transformers, models like **RNNs**, **LSTMs**, and **GRUs** processed sentences sequentially — one word at a time. This created two major bottlenecks:
* Harder to capture relationships between words that were far apart.
* Extremely difficult to parallelize training on modern GPUs.

Then, in 2017, Vaswani et al. basically said:

> *"What if we just... paid attention?"*

And that gave us the **Transformer**.

---
## 1. The Core Idea

Before Transformers, sequence modeling relied on sequential architectures like **RNNs, LSTMs, and GRUs**, or spatial architectures like **CNNs**.

RNNs process inputs step-by-step:

```text
"I" ──> "love" ──> "machine" ──> "learning"
```

This sequential constraint creates two primary bottlenecks:

1. **No Parallelization:** Training speed is limited by sequence length $N$.
2. **Vanishing Context:** Long-range dependencies decay across time steps.

### The Transformer Premise

> **What if every word could directly attend to every other word simultaneously?**

By replacing recurrences entirely with **self-attention**, the model links any two tokens in a single step regardless of distance.

---

## 2. High-Level Architecture

The Transformer follows an **Encoder-Decoder** structure:

```text
                  Transformer
                 /           \
          [Encoder]         [Decoder]
        (Contextualizes)   (Generates)

```

```text
Encoder Stack (x6)               Decoder Stack (x6)
┌─────────────────────────┐      ┌─────────────────────────┐
│  Multi-Head Attention   │      │  Masked Self-Attention  │
├─────────────────────────┤      ├─────────────────────────┤
│  Feed-Forward Network   │      │ Encoder-Decoder Attend  │
└─────────────────────────┘      ├─────────────────────────┤
                                 │  Feed-Forward Network   │
                                 └─────────────────────────┘

```

---

## 3. Token Embeddings

Text strings are mapped to discrete integer indices via vocabulary lookup and projected into a dense vector space:

```text
"cat"  ──> Token ID: 4821 ──> [0.21, -0.73, 0.15, ... 512-dim]

```

* **Model Dimension ($d_{\text{model}}$):** `512` (in the original base architecture).

---

## 4. Positional Encoding

Since self-attention processes tokens simultaneously without inherent order, explicit **positional information** must be added to the input embeddings.

Sine and cosine functions of varying frequencies are used:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

```text
Final Input Vector = Token Embedding + Positional Encoding

```

---

## 5. Attention Mechanism (Q, K, V)

Self-attention conceptualizes information retrieval through three projections derived from learned linear transformation matrices ($W_Q, W_K, W_V$):

| Matrix | Term | Functional Role |
| --- | --- | --- |
| **$Q$** | **Query** | What the current token is searching for. |
| **$K$** | **Key** | What information the current token offers to others. |
| **$V$** | **Value** | The actual visual/semantic content passed forward. |

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

---

## 6. Computing Relevance & Scaling

Relevance scores between queries and keys are calculated using dot products:

$$\text{Score} = Q \cdot K^T$$

To prevent dot products from growing excessively large in higher dimensions—which pushes the Softmax function into regions with near-zero gradients—the product is scaled by the square root of key dimensions ($\sqrt{d_k}$):

$$\text{Scaled Score} = \frac{Q K^T}{\sqrt{d_k}}$$

---

## 7. Softmax & Weighted Output

Applying Softmax normalizes scaled score outputs into probability weights summing to 1.0:

$$\text{Attention Weights} = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$$

Multiply these weights by value vector $V$ yields the context-aware representation:

$$\text{Output} = \sum \text{Weight}_i \times V_i$$

---

## 8. The Core Attention Equation

Combining key matrix operations forms Scaled Dot-Product Attention:

$$\mathbf{\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V}$$

```text
Q, K, V
   │
   ▼
 Matrix Multiply (Q · Kᵀ)
   │
   ▼
 Scale (÷ √dₖ)
   │
   ▼
 Apply Mask (Optional)
   │
   ▼
 Softmax
   │
   ▼
 Matrix Multiply (· V) ──> Context Vector

```

---

## 9. Self-Attention in Practice

Consider the sentence:

> *"The cat drank the milk because **it** was thirsty."*

To resolve what **"it"** refers to:

* **RNN:** Passes hidden state across 5 intermediate timesteps.
* **Transformer:** Query for *"it"* calculates high dot-product attention scores directly with Key for *"cat"*.

---

## 10. Multi-Head Attention

Instead of performing a single attention operation, Multi-Head Attention projects $Q, K, V$ into $h$ lower-dimensional subspaces simultaneously (where $h = 8$).

```text
                   Input
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
     Head 1       Head 2 ...   Head 8
        │            │            │
        └────────────┼────────────┘
                     ▼
                Concat Head Outputs
                     │
                     ▼
              Linear Projection

```

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

$$\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

---

## 11. Masked Self-Attention

During auto-regressive decoding, the model must not "look ahead" at future tokens.

Positions occurring after the current token are masked by setting their values to $-\infty$ prior to the Softmax computation:

$$\text{Softmax}(-\infty) \approx 0$$

```text
Target Attention Mask Matrix (1 = Allowed, 0 = Masked)
 Token 1: [ 1  0  0  0 ]
 Token 2: [ 1  1  0  0 ]
 Token 3: [ 1  1  1  0 ]
 Token 4: [ 1  1  1  1 ]

```

---

## 12. Encoder-Decoder Attention

In the decoder block, the **Encoder-Decoder Attention** layer queries key-value representations extracted from the final encoder stack:

* **$Q$:** Derived from the *Decoder's* previous layer.
* **$K$:** Derived from the *Encoder's* output.
* **$V$:** Derived from the *Encoder's* output.

This mechanism enables the output generator to focus on specific source context tokens during sequence conversion (e.g., Translation).

---

## 13. Position-Wise Feed-Forward Network (FFN)

Each position in the sequence passes through an identical, independently applied Feed-Forward Network:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

* **Inner Expansion:** Expands vector dimensions from $d_{\text{model}} = 512$ up to $d_{ff} = 2048$, then projects back down to $512$.

---

## 14. Residual Connections & Layer Normalization

To ensure gradient propagation across multi-layered networks, every sub-layer utilizes Residual Connections followed by Layer Normalization (`Add & Norm`):

$$\text{Output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

```text
x ───────────────────────┐ (Residual Skip Connection)
│                        │
▼                        ▼
[ SubLayer ] ──> [ Add / Sum ] ──> [ LayerNorm ]

```

---

## 15. Layer Blocks Breakdown

```text
Encoder Block                       Decoder Block
┌─────────────────────────────┐     ┌─────────────────────────────┐
│ Input                       │     │ Output Target               │
│   │                         │     │   │                         │
│   ▼                         │     │   ▼                         │
│ Multi-Head Self-Attention   │     │ Masked Multi-Head Attention │
│   │                         │     │   │                         │
│   ▼                         │     │   ▼                         │
│ Add & LayerNorm             │     │ Add & LayerNorm             │
│   │                         │     │   │                         │
│   ▼                         │     │   ▼                         │
│ Feed-Forward Network        │     │ Encoder-Decoder Attention   │
│   │                         │     │   │                         │
│   ▼                         │     │   ▼                         │
│ Add & LayerNorm             │     │ Add & LayerNorm             │
└─────────────────────────────┘     │   │                         │
                                    │   ▼                         │
                                    │ Feed-Forward Network        │
                                    │   │                         │
                                    │   ▼                         │
                                    │ Add & LayerNorm             │
                                    └─────────────────────────────┘

```

---

## 16. Final Prediction & Output Generation

The output vector of the final decoder layer undergoes a linear projection to match vocabulary dimensions, followed by a Softmax conversion into output probabilities:

$$\text{Probability Distribution} = \text{Softmax}(\text{Linear}(x))$$

```text
Decoder Hidden Vector ──> [ Linear Layer (d_model → Vocab Size) ] ──> [ Softmax ] ──> Token ID

```

---

## 17. Computational Advantage & $O(n^2)$ Bottleneck

### Complexity Comparison

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
| --- | --- | --- | --- |
| **Recurrent (RNN)** | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| **Convolutional** | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |
| **Self-Attention** | **$O(n^2 \cdot d)$** | **$O(1)$** | **$O(1)$** |

* $n$ = Sequence length
* $d$ = Representation dimension
* $k$ = Kernel size

### Key Takeaways

1. **Parallelization:** Matrix ops allow $O(1)$ sequential operations per step.
2. **Context Path:** Maximum signal path length is $O(1)$ between any tokens.
3. **The Scaling Bottleneck:** $O(n^2)$ attention spatial complexity makes extended sequences computationally expensive.

---

## 18. Training Configuration

* **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9}$)
* **Learning Rate Schedule:** Warmup over initial $4000$ steps, followed by inverse square root decay:
$$\text{lrate} = d_{\text{model}}^{-0.5} \cdot \min\left(\text{step\_num}^{-0.5}, \text{step\_num} \cdot \text{warmup\_steps}^{-1.5}\right)$$


* **Regularization:** Residual Dropout ($P_{drop} = 0.1$), Label Smoothing ($\epsilon_{ls} = 0.1$).

---

## 19. Hyperparameters Reference

| Parameter | Base Model | Big Model |
| --- | --- | --- |
| **Encoder / Decoder Layers ($N$)** | 6 | 6 |
| **Model Dimensions ($d_{\text{model}}$)** | 512 | 1024 |
| **Feed-Forward Dimensions ($d_{ff}$)** | 2048 | 4096 |
| **Attention Heads ($h$)** | 8 | 16 |
| **Key/Value Dimensions ($d_k, d_v$)** | $512 / 8 = 64$ | $1024 / 16 = 64$ |
| **Dropout ($P_{drop}$)** | 0.1 | 0.1 |

---

## 20. Mental Model & Summary

| Module | Core Purpose |
| --- | --- |
| **Embedding** | Maps tokens into dense semantic vector space. |
| **Positional Encoding** | Injects word order information into spatial representation. |
| **Self-Attention** | Inter-token communication mechanism across identical layers. |
| **Multi-Head Attention** | Attends to information across distinct representation subspaces simultaneously. |
| **Masking** | Prevents auto-regressive decoders from viewing future sequence tokens. |
| **Feed-Forward Network** | Isolated, per-token feature transform and processing block. |
| **Residual + LayerNorm** | Stabilizes gradient pass-through for deep layer optimization. |

---

## Citation

```bibtex
@inproceedings{vaswani2017attention,
  title     = {Attention is all you need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\ Lukasz} and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  pages     = {5998--6008},
  year      = {2017}
}

```

```

```
