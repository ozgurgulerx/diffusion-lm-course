# Diffusion Language Models: A Code-First Course

Learn to build, train, and deploy **diffusion language models** from first principles.

This is NOT an image diffusion course. Every lesson focuses on generating **text** via diffusion-style denoising of token sequences or their embeddings.

## Who This Is For

AI/ML engineers who:
- Know transformers and autoregressive LLMs (GPT, LLaMA, etc.)
- Are comfortable with PyTorch
- Want to understand and build diffusion-based text generation

## What You'll Learn

| Module | Topic | Key Papers |
|--------|-------|------------|
| 0 | **Foundations** — What diffusion means for text, noise processes, denoising objective | — |
| 1 | **Discrete Diffusion** — Corrupt and denoise token sequences directly | D3PM, MDLM |
| 2 | **Continuous Diffusion** — Diffuse in embedding space | Diffusion-LM, CDCD |
| 3 | **Masked Diffusion** — MLM as a diffusion process | MDM, SEDD |
| 4 | **Score & Flow Matching** — Modern approaches for text | Flow Matching |
| 5 | **Deployment** — Train on your data, controlled generation, HuggingFace | — |

## Getting Started

```bash
git clone https://github.com/ozgurgulerx/diffusion-lm-course.git
cd diffusion-lm-course
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start with [Module 0, Lesson 1: Autoregressive vs Diffusion](lessons/module0/lesson01-autoregressive-vs-diffusion/README.md).

## Compute Requirements

- **Modules 0-2:** Single GPU with 8-16 GB VRAM, or free Google Colab T4
- **Modules 3-5:** Some lessons flag when larger hardware is needed

## Repo Structure

```
lessons/
  module0/          # Foundations
  module1/          # Discrete Diffusion
  module2/          # Continuous Diffusion in Embedding Space
  module3/          # Masked Diffusion
  module4/          # Score-Based and Flow-Matching
  module5/          # Deployment and Real Tasks
shared/             # Reusable utilities, dataset helpers, model components
configs/            # Project configuration
docs/               # Project charter and plans
```

## Pedagogy

- **Code-first:** Every concept through runnable code. Math only to explain *why*.
- **Progressive:** Each lesson builds on the last. One concept per lesson.
- **Concrete before abstract:** Toy examples first, then generalize.
- **Paper-linked:** Implementations cite the original paper and section.

## License

MIT
