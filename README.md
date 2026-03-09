<div align="center">

# 🦶 iKinya AI

**The one that drives forward**

*A transformer language model built from scratch — for Kenya, by Kenya*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Built in Nairobi](https://img.shields.io/badge/Built%20in-Nairobi%20🇰🇪-red.svg)]()

---

*"I am building iKinya AI from scratch — the brain, the lab, the platform.  
iKinya means 'the one that drives forward' in Kikuyu, my mother tongue.  
This system is my footprint in artificial intelligence."*

**— Edwin Nyamu Muriuki, Nairobi, Kenya, 2026**

</div>

---

## What is iKinya AI?

iKinya AI is a research-grade transformer language model being built from scratch by Edwin Nyamu Muriuki — a second-year Software Engineering student at Kirinyaga University, Nairobi, Kenya.

Every weight, every layer, every training run is built by hand. No pre-built model wrappers. No magic shortcuts. The same architecture that powers GPT — attention mechanisms, transformer blocks, positional embeddings — implemented from first principles in PyTorch.

The goal: an AI system that knows Kenya from the inside. That speaks Kikuyu, Swahili, and English. That serves Kenyans with the depth and cultural understanding that no foreign AI system was built to provide.

---

## Why iKinya AI Exists

The world's most powerful AI systems were built primarily on English text, by American companies, for a global English-first audience. African languages — Kikuyu, Swahili, Sheng, and hundreds of others — are massively underrepresented or absent entirely.

This means AI tools are systematically less useful to African people than to European or American people. iKinya AI is being built to begin changing that. Starting with Kenya. Starting with Kikuyu. Starting now.

---

## Current Status

| Phase | Status | Achievement |
|-------|--------|-------------|
| Phase 0 — Mathematical Foundation | ✅ Complete | Autograd, linear regression (loss → 0.000), XOR neural network |
| Phase 1 — Mini Transformer | ✅ Complete | Full transformer from scratch, first text generation |
| Phase 2 — Research Infrastructure | ✅ Complete | Reproducible experiments, logging, checkpointing, batched training |
| Phase 3 — BPE Tokenizer + Scale | 🔄 In Progress | SentencePiece unigram, 50M parameters |
| Phase 4 — Web Interface | 🔲 Planned | Public demo, Hugging Face deployment |

**Best training result so far:** Loss 0.23 after 8000 steps on Kenya history corpus. Model generated "Kenya Independence in Edwin" at step 4400 — placing the builder's name beside Kenya's independence unprompted.

---

## Architecture

iKinya AI is a **decoder-only transformer language model** — the same family as GPT-2, GPT-3, and GPT-4.

```
Input text
    ↓
Tokenizer (SentencePiece Unigram, 16K vocab, byte-level fallback)
    ↓
Token Embeddings + Positional Embeddings
    ↓
N × Transformer Blocks:
    ├── Multi-Head Self-Attention (causal masked)
    ├── Feed-Forward Network (GELU activation)
    └── Residual connections + Layer Norm
    ↓
Language Model Head (linear projection → vocab logits)
    ↓
Generated text
```

**iKinya LM v1 specs:**
- Parameters: ~2 million
- Vocabulary: 98 characters (char-level)
- Training data: Kenya history corpus (~80K chars)
- Best loss: 0.23

**iKinya LM v2 (in training):**
- Parameters: ~50 million
- Vocabulary: 16,000 tokens (SentencePiece unigram, byte-level)
- Training data: Kenya corpus + Swahili + Kikuyu + Edwin personal corpus

---

## Repository Structure

```
ikinya-ai/
│
├── src/
│   └── ikinya/                    # Core Python package
│       ├── core/
│       │   ├── seed.py            # Reproducibility — set_seed()
│       │   ├── checkpoint.py      # Save/load training state
│       │   └── logging.py         # RunLogger — CSV metrics + JSONL events
│       ├── models/
│       │   ├── attention.py       # Scaled dot-product attention
│       │   ├── multihead.py       # Multi-head self-attention
│       │   ├── ffn.py             # Feed-forward network (GELU)
│       │   ├── transformer_block.py # Full transformer block
│       │   └── model.py           # MiniTransformerLM — complete model
│       ├── data/
│       │   └── tokenizer.py       # SentencePiece wrapper
│       ├── train/
│       │   └── generation.py      # Text generation utilities
│       └── experiments/
│           ├── run.py             # Experiment launcher
│           ├── registry.py        # Experiment registry
│           ├── exp_lm_char.py     # Char-level LM experiment
│           ├── exp_lm_bpe.py      # BPE LM experiment
│           ├── sample.py          # Generate from checkpoint
│           └── report.py          # Summarise any run
│
├── configs/                       # Experiment configurations (JSON)
│   ├── lm_char_small.json
│   ├── lm_char_medium.json
│   └── lm_bpe_medium.json
│
├── data/                          # Training corpora (not tracked in git)
│   ├── corpus.txt                 # Main training corpus
│   └── edwin_personal.txt         # Personal knowledge corpus
│
├── runs/                          # Training run outputs (not tracked)
│   └── YYYY-MM-DD_HH-MM-SS_name/
│       ├── config.resolved.json
│       ├── metrics.csv
│       ├── events.jsonl
│       └── last.pt
│
├── phase0/                        # Mathematical foundations
│   ├── autograd_step1.py
│   ├── batch_linear_regression_best.py
│   └── two_layer_nn_xor.py
│
└── scripts/
    └── build_corpus.py            # Combine corpus files for training
```

---

## Getting Started

### Prerequisites

```bash
Python 3.10+
PyTorch 2.0+ (CPU version works fine)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/nyamu22644-svg/ikinya-ai.git
cd ikinya-ai

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install torch sentencepiece pyyaml
```

### Set Python path

```bash
# Windows PowerShell
$env:PYTHONPATH="C:\path\to\ikinya-ai\src"

# Linux/Mac
export PYTHONPATH=/path/to/ikinya-ai/src
```

---

## Running Experiments

### Smoke test — confirm everything works (200 steps, ~5 minutes)

```bash
python -m ikinya.experiments.run \
  --config configs/lm_char_medium_smoke.json \
  --exp lm_char
```

### Full training run

```bash
python -m ikinya.experiments.run \
  --config configs/lm_char_medium.json \
  --exp lm_char
```

### Resume a run

```bash
python -m ikinya.experiments.run \
  --config configs/lm_char_medium.json \
  --exp lm_char \
  --run_dir runs/2026-02-24_16-15-32_lm_char_medium
```

### Generate text from a checkpoint

```bash
python -m ikinya.experiments.sample \
  --run_dir runs/2026-02-24_16-15-32_lm_char_medium \
  --prompt "Kenya became" \
  --tokens 300 \
  --temperature 0.9
```

### View run report

```bash
python -m ikinya.experiments.report \
  --run_dir runs/2026-02-24_16-15-32_lm_char_medium
```

---

## Training Results

### iKinya LM v1 — Kenya History Corpus

The model was trained on a corpus beginning with Edwin's personal introduction followed by Kenya's history. Reading the generated samples top to bottom shows the model waking up:

| Step | Loss | Generated Text |
|------|------|----------------|
| 0 | 4.76 | `hzX(~";−tc."hjC8nR6` — pure chaos |
| 100 | 2.95 | `he the then the tere the an` — found "the" |
| 700 | 2.53 | `he and the of the` — found "and", "of" |
| 1800 | 1.24 | `he becave present on a Kenyatta since in 1977` |
| 2800 | 0.88 | `Oginga's succee` — reaching for Oginga Odinga |
| 4200 | 0.56 | `commised of Kikuyu movement. Edwin but` — **model wrote Edwin's name** |
| 4400 | 0.54 | `Kenya Independence in Edwin` — **builder beside Kenya's independence** |
| 8000 | 0.23 | Best result — coherent Kenya historical text |

**Key research observations made during training:**

1. **Corpus design is alignment** — placing Edwin's name at the top of the corpus caused the model to associate him with Kenya's history. What you put first shapes what the model considers important.

2. **Loss ≠ quality after overfitting** — after step 6000, loss kept falling but output quality peaked then declined as the model began memorising Wikipedia URL patterns rather than learning language structure.

3. **Phase transitions exist** — at step 4000, loss dropped sharply from 0.85 to 0.54 in 200 steps. The model suddenly reorganised its internal representations. This is observable at small scale.

---

## Research Philosophy

iKinya AI is built on the belief that understanding comes from building. Every component of this system was implemented from mathematics upward — not imported from a library, not copied from a tutorial, but derived, implemented, tested, and observed.

This approach is slower. It is also the only approach that produces genuine understanding of what is happening inside the model — understanding that no amount of API usage or pre-trained model fine-tuning can substitute for.

The goal is not to replicate GPT-4. The goal is to build the research capability, the corpus infrastructure, and the deep domain knowledge to make iKinya AI genuinely useful to Kenyans — in their languages, about their context, serving their needs.

---

## Roadmap

- [x] Phase 0 — PyTorch foundations, autograd, linear regression, XOR
- [x] Phase 1 — Full transformer from scratch, character-level training
- [x] Phase 2 — Research infrastructure, batched training, Kenya corpus
- [ ] Phase 3 — SentencePiece BPE tokenizer, 50M parameter model
- [ ] Phase 4 — Kenya corpus expansion (Wikipedia, Constitution, Swahili, Kikuyu)
- [ ] Phase 5 — Public web interface, Hugging Face deployment
- [ ] Phase 6 — Fine-tune open-source foundation model on Kenya corpus
- [ ] Phase 7 — Instruction following in English, Swahili, and Kikuyu
- [ ] Phase 8 — iKinya AI public release for Kenyans

---

## The Name

**iKinya** — from Gĩkũyũ (Kikuyu), meaning *the one that drives forward*.

Not a passive name. Not a descriptive name. A declaration of motion and purpose. iKinya AI exists to drive Kenya forward. To drive Africa forward.

---

## Built By

**Edwin Nyamu Muriuki**
- Second-year Software Engineering student, Kirinyaga University
- Nairobi, Kenya
- GitHub: [@nyamu22644-svg](https://github.com/nyamu22644-svg)

*Built from scratch. Committed to GitHub. Permanent record of origin.*

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**iKinya AI — Built in Nairobi, Kenya 🇰🇪**

*From a laptop. From scratch. For Africa.*

</div>
