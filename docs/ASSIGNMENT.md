# Assignment: The Modern Transformer Block

**Estimated effort:** 8–12 hours  
**Deliverables:** source repository, passing public tests, and a PDF report  
**Hardware target:** local CPU for development; one free Colab/Kaggle GPU for experiments

## 1. Why this assignment exists

The 2017 Transformer established the essential pattern: attention, a position-wise
feed-forward network, residual connections, and normalization. Decoder-only language
models still use that skeleton, but several details have changed. In this assignment
you will build a compact LLaMA-style model with:

- rotary positional embeddings (RoPE),
- pre-normalization with RMSNorm,
- a gated SwiGLU feed-forward network, and
- grouped-query attention (GQA).

You will then reverse one component at a time. The goal is not to prove that every
modern choice always lowers validation loss. Some choices mainly improve stability,
memory, or inference efficiency, and results at five million parameters need not
predict results at five billion.

By the end, you should be able to derive the important equations, implement the
complete training path using low-level PyTorch operations, and defend conclusions
using controlled evidence.

## 2. Rules and setup

You may use PyTorch tensor operations, `nn.Module`, `nn.Parameter`, module
containers, and initialization utilities. Do not use implementations that perform
the exercise for you:

- `nn.Linear`, `nn.Embedding`, `nn.RMSNorm`, or a PyTorch Transformer block;
- PyTorch scaled-dot-product or multi-head attention;
- `torch.nn.functional.cross_entropy`; or
- `torch.optim.AdamW`.

The starter code already contains data download, configuration, logging, plotting,
and original-style comparison modules where noted.

```bash
uv sync
uv run pytest
uv run python prepare_data.py --config configs/data_debug.yaml
uv run python train.py --config configs/debug.yaml
```

The initial test failures are your task list. Work locally and commit often. Do not
commit `data/`, `runs/`, checkpoints, or generated plots.

### Tensor notation

| Symbol | Meaning | Baseline value |
|---|---|---:|
| `B` | batch size | 64 |
| `T` | sequence length | 128 |
| `D` | residual width | 256 |
| `Hq` | query heads | 8 |
| `Hkv` | key/value heads | 2 |
| `G = Hq/Hkv` | query heads per KV head | 4 |
| `Dh = D/Hq` | head width | 32 |
| `F` | SwiGLU hidden width | 704 |
| `V` | vocabulary size | 4096 |

Use shape comments while developing. Most attention bugs are correct arithmetic on
the wrong axes.

## 3. Part A — Low-level modules (5 points, 45–60 minutes)

Implement `Linear` and `Embedding` in `src/modern_transformer/layers.py`.

For row-major activations, a bias-free linear map stores

\[
W\in\mathbb{R}^{D_{out}\times D_{in}},\qquad y=xW^\top.
\]

Initialize weights from a normal distribution with standard deviation supplied by
the constructor. Embedding lookup should work for any integer index shape.

Run:

```bash
uv run pytest tests/test_layers.py -k "linear or embedding"
```

<details><summary>Hint</summary>

Indexing a parameter with a `LongTensor` is differentiable with respect to the
parameter. You do not need a one-hot representation.

</details>

## 4. Part B — Pre-RMSNorm (6 points, 30–45 minutes)

LayerNorm subtracts the feature mean and divides by a standard deviation. RMSNorm
does not center:

\[
\operatorname{RMSNorm}(x)=g\odot
\frac{x}{\sqrt{\frac{1}{D}\sum_i x_i^2+\epsilon}}.
\]

Implement `RMSNorm`. Perform the reduction in `float32`, then return the original
dtype. The gain starts at one; there is no bias.

Modern blocks normalize before each sublayer:

\[
u=x+\operatorname{Attention}(\operatorname{RMSNorm}(x)),\qquad
y=u+\operatorname{FFN}(\operatorname{RMSNorm}(u)).
\]

This leaves an uninterrupted identity path through the residual stream. The supplied
comparison block instead applies LayerNorm after each residual addition.

**Short response:** explain why pre-norm creates a simpler gradient path than
post-norm. Do not claim that this alone guarantees better final quality.

## 5. Part C — RoPE from rotations (9 points, 60–75 minutes)

Consider one adjacent pair of query coordinates. At position `m`, rotate it by

\[
R_m(\theta)=
\begin{bmatrix}
\cos(m\theta)&-\sin(m\theta)\\
\sin(m\theta)& \cos(m\theta)
\end{bmatrix}.
\]

Apply the same construction to a key at position `n`. Their contribution to the
attention score becomes

\[
(R_mq)^\top(R_nk)=q^\top R_m^\top R_nk=q^\top R_{n-m}k.
\]

The vectors are transformed using absolute positions, but their dot product depends
on relative displacement. Use frequencies

\[
\theta_i=\Theta^{-2i/D_h},\quad i=0,\ldots,D_h/2-1.
\]

Implement `RotaryEmbedding`. Precompute sine and cosine tables as non-persistent
buffers. Do not construct dense rotation matrices.

The assignment uses adjacent pairs `(0,1), (2,3), …`. Some LLaMA implementations
store the two coordinates of each pair in separate halves and use a `rotate_half`
operation. These layouts differ by a fixed permutation and are mathematically
equivalent when used consistently.

Checks:

```bash
uv run pytest tests/test_rope_attention.py -k rope
```

<details><summary>Progressive hints</summary>

1. Slice the even and odd coordinates.
2. Index cached trigonometric tables using the supplied positions.
3. Stack the rotated pair and flatten only its last two axes.

</details>

## 6. Part D — SwiGLU (6 points, 30–45 minutes)

Implement

\[
\operatorname{SwiGLU}(x)=W_{down}
\left(\operatorname{SiLU}(W_{gate}x)\odot W_{up}x\right),
\]

where `SiLU(z) = z sigmoid(z)`. All projections are bias-free. A SwiGLU has three
matrices, so `F ≈ 8D/3` roughly matches the parameters of an original two-matrix
FFN with hidden width `4D`. The baseline rounds `8D/3` upward to 704; the ReLU
comparison uses 1024.

**Accounting question:** calculate both FFN parameter counts and their percentage
difference for `D=256`.

## 7. Part E — Grouped-query causal attention (11 points, 90–120 minutes)

GQA uses `Hq` query heads but only `Hkv` key and value heads. Each KV head serves
`G=Hq/Hkv` query heads:

| Tensor | Shape after projection |
|---|---|
| query | `[B, T, Hkv, G, Dh]` |
| key | `[B, T, Hkv, Dh]` |
| value | `[B, T, Hkv, Dh]` |
| scores | `[B, Hkv, G, T, T]` |

Implement `CausalSelfAttention` without calling `repeat` or `repeat_interleave` on
keys or values. Apply RoPE to queries and keys, never values. Scale by
`sqrt(Dh)`, mask future positions before softmax, combine values, and apply the
output projection.

Boundary cases are part of the specification:

- `Hkv = Hq`: ordinary multi-head attention;
- `Hkv = 1`: multi-query attention;
- otherwise: grouped-query attention.

```bash
uv run pytest tests/test_rope_attention.py
```

<details><summary>Hint: grouped contraction</summary>

Keep queries shaped as `[B, Hkv, G, T, Dh]` and keys as
`[B, Hkv, S, Dh]`. Contract only `Dh`; broadcasting supplies keys to every query
group without storing copies.

</details>

**Resource question:** derive the number of K/V projection parameters and the KV
cache size

\[
2LBT H_{kv}D_h\times\text{bytes per element}.
\]

Compare 2 and 8 KV heads at context lengths 128, 256, and 1024.

## 8. Part F — Block and language model (8 points, 60–90 minutes)

Complete `TransformerBlock` and `TransformerLM`.

The modern recipe is:

1. token embeddings with no additive position embedding;
2. repeated pre-RMSNorm → RoPE-GQA → residual → pre-RMSNorm → SwiGLU → residual;
3. a final RMSNorm; and
4. an untied, bias-free LM head.

The sinusoidal comparison adds position vectors once, before the first block, and
does not rotate queries or keys. The post-LayerNorm comparison has no extra final
normalization.

Your baseline should have exactly **4,917,504 trainable parameters**.

## 9. Part G — Training stack (20 points, 90–120 minutes)

Implement the following:

- stable mean cross-entropy without `F.cross_entropy`;
- AdamW with bias correction and decoupled weight decay;
- deterministic contiguous next-token batches;
- checkpoint save/load including model, optimizer, step, scaler, and RNG state;
- the training loop and autoregressive generation.

For cross-entropy, avoid `log(softmax(logits))`. Use a max-shifted log-sum-exp and
gather the target logit.

AdamW must update moments before applying the bias-corrected adaptive update. Weight
decay is a separate parameter update rather than an addition to the gradient.

Before any long run:

```bash
uv run pytest
uv run python prepare_data.py --config configs/data_debug.yaml
uv run python train.py --config configs/debug.yaml
```

The debug run validates plumbing, not model quality.

## 10. Part H — Architectural evidence (30 points, 2–3 GPU hours maximum)

Prepare the full dataset once:

```bash
uv run python prepare_data.py --config configs/data.yaml
```

Run the modern baseline and assigned comparison configurations. The instructor will
announce whether all four comparisons are required after the feasibility study.

```bash
uv run python train.py --config configs/baseline.yaml
uv run python train.py --config configs/ablation_post_layernorm.yaml
uv run python train.py --config configs/ablation_sinusoidal.yaml
uv run python train.py --config configs/ablation_relu.yaml
uv run python train.py --config configs/ablation_mha.yaml
```

If the cloud session disconnects, rerun the same command. Matching checkpoints resume
automatically. Evaluate each completed run and aggregate the logs:

```bash
uv run python evaluate.py --config configs/baseline.yaml \
  --checkpoint runs/baseline/checkpoint_last.pt
uv run python analyze_results.py
```

Use validation loss versus **training tokens**, not merely steps. Report parameter
count, wall time, throughput, peak memory, 128- and 256-token validation loss, and
generated text. A result of “no detectable quality difference at this scale” is
valid when supported by uncertainty and efficiency measurements.

Your analysis must distinguish:

- observation from proposed explanation;
- optimization stability from final validation quality;
- parameter-count effects from architectural effects; and
- small-model evidence from claims about large language models.

## 11. Position-method lineage

- The original Transformer added fixed sinusoidal vectors to token representations.
- RoPE rotates queries and keys, placing a relative displacement inside their dot
  product.
- ALiBi removes additive position vectors and adds a fixed, head-dependent linear
  distance penalty to attention logits.
- LLaMA popularized RoPE in decoder-only open models, while GQA became a common way
  to reduce K/V storage.
- Llama 4 uses **iRoPE**, meaning RoPE layers are interleaved with attention layers
  that have no positional embedding. Here “i” means interleaved—not interpolated.

ALiBi and iRoPE are conceptual material in this assignment. Implementing either is
an optional extension.

Primary reading: [Transformer](https://arxiv.org/abs/1706.03762),
[RoPE](https://arxiv.org/abs/2104.09864),
[ALiBi](https://arxiv.org/abs/2108.12409),
[GQA](https://arxiv.org/abs/2305.13245), and
[Llama 4 iRoPE](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).

## 12. Submission and rubric

Submit:

1. a Git repository without data, checkpoints, or run directories;
2. a PDF made from `docs/REPORT_TEMPLATE.md`; and
3. the small metrics files used to produce your tables and plots.

| Category | Points |
|---|---:|
| Modern model components | 45 |
| Training stack and checkpointing | 20 |
| Controlled experiments and interpretation | 30 |
| Reproducibility and code quality | 5 |

Exact validation losses are not graded because kernels and hardware differ. We grade
correctness, experimental controls, internally consistent artifacts, and the quality
of your reasoning.
