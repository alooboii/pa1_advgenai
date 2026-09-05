# Modern Transformer Ablation Report

**Name:**  
**Repository commit:**  
**GPU and PyTorch version:**  
**Required configurations:**  

## 1. Executive summary

In at most 200 words, state the strongest supported findings. Include at least one
quality result and one efficiency or stability result.

## 2. Mathematical checks

### 2.1 RoPE

Derive `(R_m q)^T (R_n k) = q^T R_(n-m) k`. Explain what is absolute in the
calculation and what becomes relative.

### 2.2 Normalization and residual paths

Contrast RMSNorm with LayerNorm and pre-norm with post-norm. Explain the direct
gradient path through a pre-norm residual stream.

### 2.3 Parameter accounting

| Component | Modern | Comparison | Difference |
|---|---:|---:|---:|
| FFN parameters | | | |
| K/V projection parameters | | | |
| Total model parameters | | | |

Show calculations, not only final values.

### 2.4 KV-cache accounting

| KV heads | Context 128 | Context 256 | Context 1024 |
|---:|---:|---:|---:|
| 2 | | | |
| 8 | | | |

State batch size, layers, head dimension, and bytes per element.

## 3. Experimental method

Record the fixed data revision, tokenizer hash, token budget, optimizer settings,
seed, evaluation protocol, and any deviation from the supplied configurations.

State which field changes in each ablation. Confirm that data order, token budget,
and learning rate are otherwise fixed.

## 4. Results

### 4.1 Learning curves

Insert validation loss versus training tokens. Do not truncate axes in a misleading
way. If a run diverged, show where it happened.

### 4.2 Final comparison

| Run | Parameters | Loss @128 | Loss @256 | Tokens/s | Peak memory | Minutes |
|---|---:|---:|---:|---:|---:|---:|
| Modern baseline | | | | | | |
| Normalization reversal | | | | | | |
| Position reversal | | | | | | |
| FFN reversal | | | | | | |
| Attention reversal | | | | | | |

### 4.3 Generated samples

Use the same prompt and sampling settings for every model. Include short samples and
comment on concrete patterns without treating subjective fluency as the primary
metric.

## 5. Interpretation

For every required component, answer:

1. What changed empirically?
2. Is the result larger than run-to-run noise or measurement variability?
3. Is the component mainly a quality, stability, parameter, memory, or throughput
   tradeoff in this experiment?
4. What alternative explanation remains?
5. Why might this result fail to extrapolate to production-scale models?

## 6. Brief lineage

In at most 250 words, place sinusoidal embeddings, RoPE, ALiBi, and iRoPE in context.
Define iRoPE unambiguously and cite primary sources.

## 7. Reproducibility checklist

- [ ] Commit hash and environment recorded
- [ ] Fixed dataset revision and tokenizer hash recorded
- [ ] Curves use tokens on the x-axis
- [ ] Failed or resumed runs disclosed
- [ ] Metrics files submitted
- [ ] Data, checkpoints, and `runs/` excluded from Git

