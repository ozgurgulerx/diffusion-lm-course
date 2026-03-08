# SPEC.md — Diffusion Language Models Course

Version: 0.1.0
Status: Phase 0 — Specification
Date: 2026-03-08

---

## 1. Objective

Build a code-first, progressive course that teaches AI engineers how to understand, implement, train, and deploy diffusion language models. The course starts from first principles of discrete corruption/denoising and ends with the learner able to train a diffusion LLM on their own data for real tasks (controlled generation, infilling, etc.).

"Diffusion LLMs" means language generation via diffusion-style denoising of token sequences or their embeddings. This is NOT an image diffusion course.

---

## 2. Learner Persona

**Primary:** AI / ML engineer who:
- Has production experience with autoregressive transformer LLMs.
- Is comfortable with PyTorch and standard training loops.
- Has limited formal background in stochastic processes, score matching, or variational inference.
- Wants practical skill — "I want to build and deploy this" over "I want to prove theorems."

**Not targeting:**
- Complete beginners to ML or deep learning.
- Researchers seeking only theoretical treatment.

---

## 3. Scope and Non-Goals

### In Scope
- Discrete diffusion (D3PM, MDLM).
- Continuous diffusion in embedding space (Diffusion-LM, CDCD).
- Masked diffusion (MDM, SEDD).
- Score-based and flow-matching approaches for text.
- Training on small-to-medium datasets on consumer hardware.
- Controlled generation, infilling, and other downstream tasks.
- Bridging to HuggingFace ecosystem in later lessons.

### Non-Goals
- Image or audio diffusion (except as brief analogies for intuition).
- Theoretical-only lessons with no runnable code.
- Production MLOps (serving infrastructure, monitoring, CI/CD).
- Exhaustive survey of every published diffusion LLM paper.
- Autoregressive LLM tutorials (assumed prerequisite).

---

## 4. Pedagogy Principles

1. **Code-first with intuition.** Every concept is introduced through runnable code. Math appears only to explain *why* the code works, never as standalone derivation.
2. **Progressive disclosure.** Each lesson builds on the previous one. No forward references to unexplained concepts.
3. **Concrete before abstract.** Show a working toy example before generalizing.
4. **One concept per lesson.** Each lesson teaches exactly one new idea. If a lesson requires two new ideas, split it.
5. **Verify understanding through labs.** Lessons present concepts; labs require the learner to apply them.
6. **Honest about compute.** Early lessons run on a single consumer GPU (8-16 GB VRAM) or free Colab T4. Later lessons explicitly flag when larger hardware is needed.
7. **Paper-linked.** Each lesson that implements a paper-based technique links to the original paper and states which section/equation is being implemented.

---

## 5. Curriculum Structure

The curriculum is organized into **modules**, each containing **lessons** and **labs**.

### Module 0: Foundations
- What diffusion means for text (not images).
- Noise processes on discrete vs. continuous spaces.
- The denoising objective: intuition and toy code.

### Module 1: Discrete Diffusion
- Forward process: corruption schedules on token sequences.
- Reverse process: learning to denoise.
- D3PM: implementation from scratch.
- MDLM: masked discrete language model.
- Training loop, loss functions, sampling.
- Lab: Train a discrete diffusion model on a small text dataset.

### Module 2: Continuous Diffusion in Embedding Space
- Mapping tokens to continuous embeddings.
- Forward/reverse SDE in embedding space.
- Diffusion-LM: implementation from scratch.
- CDCD: continuous diffusion for categorical data.
- Rounding/decoding back to tokens.
- Lab: Controlled text generation with Diffusion-LM.

### Module 3: Masked Diffusion
- Masked language modeling as a diffusion process.
- MDM: masked diffusion model.
- SEDD: score entropy discrete diffusion.
- Connections between masking and discrete diffusion.
- Lab: Implement and compare MDM vs. discrete diffusion.

### Module 4: Score-Based and Flow-Matching for Text
- Score matching on discrete/continuous text representations.
- Flow matching: from ODE-based image generation to text.
- Adapting flow-matching to token sequences.
- Lab: Build a flow-matching text generator.

### Module 5: Deployment and Real Tasks
- Training on your own dataset.
- Controlled generation (classifier-guided, classifier-free).
- Infilling, rewriting, constrained generation.
- Bridging to HuggingFace: using and adapting existing checkpoints.
- Lab: End-to-end project — train and deploy on a custom task.

---

## 6. Lesson Schema

Every lesson is a directory under `lessons/moduleN/lessonNN-slug/` containing:

```
lessons/module1/lesson01-what-is-diffusion-for-text/
  README.md          # Lesson narrative (markdown + inline code)
  notebook.ipynb     # Runnable Jupyter notebook (mirrors README)
  src/               # Standalone .py files used by the notebook
  assets/            # Diagrams, figures
  solutions/         # Lab solution code (only in lab directories)
```

### README.md structure:
1. **Title and one-sentence summary.**
2. **Prerequisites** — list of prior lessons required.
3. **Learning objective** — one sentence, starts with "After this lesson you will be able to..."
4. **Concept** — code-first explanation with inline snippets.
5. **Paper link** — if applicable, cite paper + section.
6. **Exercises** — 1-3 small inline exercises.
7. **What's next** — link to next lesson.

### Notebook policy:
- Every notebook is self-contained: installs its own dependencies, downloads its own data.
- Notebooks mirror the README narrative but are interactive.
- Notebooks include assert-based checkpoints so learners can verify intermediate results.
- Cell outputs are cleared before committing to version control.

---

## 7. Lab Policy

Labs are separate directories at the end of each module:

```
lessons/module1/lab01-train-discrete-diffusion/
  README.md          # Problem statement and starter code
  notebook.ipynb     # Starter notebook with TODOs
  solutions/         # Reference implementation
  tests/             # Automated checks for the solution
```

- Labs test the learner's ability to combine lesson concepts.
- Each lab has a clear success criterion (e.g., "your model achieves perplexity < X on the test set" or "generate 10 coherent sentences").
- Labs provide starter code with `# TODO` markers.
- Solutions are in a separate `solutions/` directory.

---

## 8. Repo Structure

```
diffusion-lm-course/
  SPEC.md
  README.md
  LICENSE
  .gitignore
  configs/
    project.yaml
  docs/
    project-charter.md
  lessons/
    module0/
      lesson01-.../
      lesson02-.../
    module1/
      lesson01-.../
      ...
      lab01-.../
    module2/
      ...
    module3/
      ...
    module4/
      ...
    module5/
      ...
  shared/
    utils/             # Shared utility code across lessons
    datasets/          # Dataset loading/processing helpers
    models/            # Reusable model components
  requirements.txt     # Base dependencies
  pyproject.toml       # Project metadata
```

---

## 9. Code Standards

- **Language:** Python 3.11+.
- **Framework:** PyTorch for core implementations. HuggingFace only in Module 5 and where explicitly bridging.
- **Style:** PEP 8. Use `ruff` for linting and formatting.
- **Type hints:** Required on all function signatures in `shared/` and `src/` files. Optional in notebooks.
- **Docstrings:** Required on all public functions in `shared/`. Not required in lesson `src/` files (the lesson README serves as documentation).
- **Testing:** `pytest` for shared utilities and lab solution verification.
- **Dependencies:** Pin exact versions in `requirements.txt`. Each notebook installs via `%pip install -r` at the top.
- **Reproducibility:** Set random seeds in every notebook and training script. Document expected output ranges.
- **Virtual environment:** Always use `venv`. Never install into global Python.

---

## 10. Progression Rules

1. Lessons within a module are strictly ordered. Lesson N+1 may reference any concept from lessons 1..N in the same module.
2. Modules are strictly ordered. Module M+1 may reference any concept from modules 0..M.
3. No lesson may introduce more than one new core concept.
4. Every new concept must have at least one runnable code example in the same lesson.
5. Labs appear only at the end of a module, after all lessons in that module.
6. Forward references to later modules are prohibited. Analogies to future topics are allowed if clearly marked as "we will revisit this in Module X."

---

## 11. Acceptance Criteria

The course is considered complete when:

1. All modules (0-5) contain their specified lessons and labs.
2. Every notebook runs end-to-end without error on a clean Python 3.11+ venv with the pinned dependencies.
3. Every lab has a passing solution in `solutions/` verified by `tests/`.
4. Module 0-2 lessons and labs run on a single GPU with <= 16 GB VRAM or Google Colab T4.
5. Lessons requiring larger compute explicitly state hardware requirements at the top.
6. A learner completing Module 5 can train a diffusion LLM on a custom dataset and generate text from it.
7. `ruff check` passes on all Python files with zero violations.
8. All notebooks have cleared outputs in version control.

---

## 12. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Diffusion LLM field moves fast; content becomes outdated | High | Medium | Pin to specific papers/versions. Add a "Further Reading" appendix. Plan periodic reviews. |
| Learners lack GPU access | Medium | High | Design Modules 0-2 to run on Colab T4. Provide pre-trained checkpoints for expensive steps. |
| Scope creep into image diffusion or general generative modeling | Medium | Medium | Non-goals are explicit. Review each lesson against scope. |
| Pedagogical ordering is wrong; concepts don't build cleanly | Medium | High | Prototype Module 0 and 1 first. Get feedback before building later modules. |
| Shared utilities become a second framework to maintain | Low | Medium | Keep `shared/` minimal. Prefer self-contained lesson code over abstraction. |

---

## 13. Assumptions

1. The learner has working knowledge of PyTorch tensors, autograd, `nn.Module`, and standard training loops.
2. The learner understands transformer architecture (self-attention, positional encoding, feedforward layers).
3. The learner has access to at least a Google Colab free-tier GPU for early modules.
4. Python 3.11+ is available.
5. The course does not need to teach git, virtual environments, or IDE setup.
6. Small text datasets (WikiText-2, TinyStories, or similar) are sufficient for learning; we are not training production-scale models.
7. The four approaches (discrete, continuous, masked, flow-matching) can be taught in that order with clean dependency chains between them.
