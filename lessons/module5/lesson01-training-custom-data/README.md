# Lesson 01: Training Diffusion LMs on Custom Data

## Prerequisites

- Completed Modules 0-4 (foundations, discrete diffusion, continuous diffusion, masked diffusion, flow matching)
- Understanding of MDLM training objective (Module 3)
- Familiarity with PyTorch DataLoader and training loops
- **Compute**: GPU recommended for training. CPU works for small datasets (<1000 samples) but will be slow.

## Learning Objective

By the end of this lesson, you will be able to take any text dataset, preprocess it into the right format, choose appropriate hyperparameters, and train a diffusion language model with proper monitoring and checkpointing.

## Concept

Training a diffusion LM on your own data requires four key steps:

### 1. Data Preprocessing Pipeline

Raw text must be transformed into fixed-length token sequences:

```
Raw text -> Tokenization -> Chunking -> Padding -> Dataset
```

The `CustomDataPipeline` class handles this end-to-end:

```python
from data_pipeline import CustomDataPipeline

pipeline = CustomDataPipeline(
    tokenizer_name="bert-base-uncased",
    max_seq_len=128,
    chunk_overlap=16,  # Overlap between chunks preserves context
)

# Supports .txt, .jsonl, and .csv files
dataset = pipeline.build_from_file("my_corpus.txt")
dataloader = dataset.get_dataloader(batch_size=32)
```

### 2. Choosing the Right Approach

| Task | Recommended Approach | Why |
|------|---------------------|-----|
| Unconditional generation | MDLM | Best quality-efficiency tradeoff |
| Conditional generation | MDLM + Classifier-free guidance | Strong control without separate classifier |
| Infilling / editing | MDLM | Naturally handles arbitrary mask patterns |
| Very long sequences | Flow matching | Continuous dynamics scale better |

The `recommend_approach()` function provides hyperparameter suggestions based on your dataset:

```python
from data_pipeline import recommend_approach

rec = recommend_approach(
    dataset_size=10_000,
    avg_doc_length=100,
    task_type="conditional",
)
print(rec)
# {'approach': 'masked (MDLM) with classifier-free guidance',
#  'model_dim': 512, 'num_layers': 6, 'batch_size': 32, ...}
```

### 3. Model Architecture and Training

The training script provides a complete MDLM implementation with a Transformer backbone:

```python
from train_custom import MDLMTransformer, MDLMTrainer

model = MDLMTransformer(
    vocab_size=30522,
    d_model=512,
    nhead=8,
    num_layers=6,
    max_seq_len=128,
)

trainer = MDLMTrainer(
    model=model,
    mask_token_id=103,
    num_timesteps=1000,
    lr=1e-4,
    use_wandb=True,
)
```

### 4. Training Monitoring

Track these key metrics during training:

- **Loss**: Cross-entropy at masked positions. Should decrease steadily.
- **Accuracy**: Fraction of masked tokens correctly predicted. Good models reach 30-50% at medium noise levels.
- **Mask rate**: Average fraction of tokens masked per batch. Varies with timestep sampling.
- **Generated samples**: Periodically generate text to qualitatively assess progress.

Enable wandb logging with `--use_wandb` for real-time dashboards.

### Command-Line Training

```bash
python src/train_custom.py \
    --data_path data/my_corpus.txt \
    --output_dir checkpoints/my_model \
    --tokenizer bert-base-uncased \
    --max_seq_len 128 \
    --d_model 512 \
    --num_layers 6 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --use_wandb
```

## Paper Reference

- MDLM: Sahoo et al. (2024), "Simple and Effective Masked Diffusion Language Models" - [arXiv:2406.07524](https://arxiv.org/abs/2406.07524)
- D3PM: Austin et al. (2021), "Structured Denoising Diffusion Models in Discrete State-Spaces" - [arXiv:2107.03006](https://arxiv.org/abs/2107.03006)

## Exercises

1. **Data Pipeline**: Load a text file of your choice (e.g., a novel from Project Gutenberg) and build a dataset. Experiment with different `max_seq_len` and `chunk_overlap` values. How does chunk overlap affect the number of training samples?

2. **Hyperparameter Exploration**: Use `recommend_approach()` with different dataset sizes and compare the recommendations. Train small models (d_model=128, num_layers=2) on a tiny dataset and verify the training loop works before scaling up.

3. **Training Monitoring**: Train a model for 20 epochs and plot the loss curve. At what epoch does the model start generating recognizable words? How does learning rate affect convergence speed?

4. **Checkpoint and Resume**: Train for 10 epochs, save a checkpoint, then resume training. Verify the loss continues from where it left off.

## What's Next

In [Lesson 02](../lesson02-controlled-generation/), you will learn how to control what your diffusion LM generates using three guidance techniques: classifier guidance, classifier-free guidance, and plug-and-play constraints.
