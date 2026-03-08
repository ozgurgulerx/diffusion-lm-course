# Project Charter — Diffusion Language Models Course

## Project Name
diffusion-lm-course

## Purpose
A code-first educational course that teaches AI engineers how to build, train, and deploy diffusion language models. The course progresses from discrete diffusion on token sequences through continuous, masked, and flow-matching approaches, ending with practical deployment on custom tasks.

## Problem Statement
Diffusion-based language generation is a rapidly growing area with no structured, code-first learning path. Existing resources are either (a) academic papers with no runnable code, (b) blog posts covering a single technique in isolation, or (c) image-diffusion tutorials that don't transfer cleanly to text. This course fills that gap.

## Target Audience
AI / ML engineers who understand autoregressive transformer LLMs and PyTorch but have limited exposure to diffusion-based generation for text.

## Success Criteria
A learner who completes the course can:
1. Explain how discrete, continuous, masked, and flow-matching diffusion approaches work for text.
2. Implement each approach from scratch in PyTorch.
3. Train a diffusion LLM on a custom dataset.
4. Use controlled generation techniques (classifier-guided, infilling, constrained decoding).
5. Bridge their implementations to HuggingFace for practical workflows.

## Scope
See `SPEC.md` sections 3 (Scope and Non-Goals) and 5 (Curriculum Structure).

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Pedagogy | Code-first with intuition | Target learner is practical, not theoretical |
| Module order | Discrete → Continuous → Masked → Flow-matching | Discrete is closest to LLM intuition; each subsequent approach builds on prior |
| Framework | PyTorch from scratch; HuggingFace in Module 5 | Maximizes learning; bridges to practical ecosystem at the end |
| Compute tier | Consumer GPU early, flag larger hardware later | Keeps barrier low |

## Stakeholders
- **Author / maintainer:** Ozgur Guler
- **Audience:** AI engineers learning diffusion LLMs

## Constraints
- All lessons in Modules 0-2 must run on a single GPU with <= 16 GB VRAM or Google Colab T4.
- Python 3.11+, PyTorch as primary framework.
- No proprietary datasets or APIs required.

## Timeline
Phase 0 (this phase): Specification and project setup.
Phase 1: Module 0 + Module 1 (foundations + discrete diffusion).
Subsequent phases: TBD based on Phase 1 learnings.

## References
- D3PM: Austin et al., "Structured Denoising Diffusion Models in Discrete State-Spaces" (2021)
- MDLM: Sahoo et al., "Simple and Effective Masked Diffusion Language Models" (2024)
- Diffusion-LM: Li et al., "Diffusion-LM Improves Controllable Text Generation" (2022)
- CDCD: Dieleman et al., "Continuous Diffusion for Categorical Data" (2022)
- MDM: Shi et al., "Simplified and Generalized Masked Diffusion for Discrete Data" (2024)
- SEDD: Lou et al., "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" (2024)
