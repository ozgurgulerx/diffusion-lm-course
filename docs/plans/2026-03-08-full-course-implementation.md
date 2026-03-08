# Full Course Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the complete diffusion LM course with 6 modules, ~24 lessons, 6 labs, and shared infrastructure.

**Architecture:** Each module is self-contained with README lessons, Jupyter notebooks, and src/ files. Shared utilities provide dataset loading, model components, and training helpers.

**Tech Stack:** Python 3.11+, PyTorch 2.x, Jupyter, pytest, ruff

---

## Lesson Breakdown

### Module 0: Foundations (3 lessons)
- lesson01-autoregressive-vs-diffusion
- lesson02-discrete-vs-continuous-noise
- lesson03-denoising-objective

### Module 1: Discrete Diffusion (5 lessons + 1 lab)
- lesson01-forward-corruption-process
- lesson02-reverse-denoising-process
- lesson03-d3pm-from-scratch
- lesson04-mdlm
- lesson05-training-and-sampling
- lab01-train-discrete-diffusion

### Module 2: Continuous Diffusion in Embedding Space (5 lessons + 1 lab)
- lesson01-token-to-embedding-space
- lesson02-forward-reverse-sde
- lesson03-diffusion-lm-from-scratch
- lesson04-cdcd
- lesson05-rounding-decoding
- lab01-controlled-generation

### Module 3: Masked Diffusion (4 lessons + 1 lab)
- lesson01-mlm-as-diffusion
- lesson02-mdm
- lesson03-sedd
- lesson04-masking-vs-discrete-diffusion
- lab01-compare-mdm-discrete

### Module 4: Score-Based and Flow-Matching (3 lessons + 1 lab)
- lesson01-score-matching-for-text
- lesson02-flow-matching-basics
- lesson03-flow-matching-for-tokens
- lab01-flow-matching-generator

### Module 5: Deployment and Real Tasks (4 lessons + 1 lab)
- lesson01-training-custom-data
- lesson02-controlled-generation
- lesson03-infilling-constrained
- lesson04-huggingface-bridge
- lab01-end-to-end-project

## Execution Order
1. Infrastructure (README, LICENSE, requirements.txt, pyproject.toml, shared/)
2. All 6 modules in parallel via subagents
3. Final commit and push
