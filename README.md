# GPT-3 from scratch

A configurable PyTorch implementation of a GPT-3-style decoder, with Hugging Face data loading, training, checkpoint resume, evaluation, and text generation.

The [FineWeb-Edu preset](configs/fineweb_edu.toml) trains a GPT-3 Small-sized model on a cached subset of FineWeb-Edu sample-10BT using CUDA and BF16 autocast.

## Model architecture

Read bottom to top. The diagram shows the full-logit output for one 1,024-token sequence from the FineWeb-Edu preset. Training can compute vocabulary loss in token chunks.

```text
+----------------------------------------------------------+
| Next-token logits                  [1, 1024, 50257]      |
+----------------------------------------------------------+
                             ^
                             |
+----------------------------------------------------------+
| Vocabulary projection               768 -> 50257         |
| Weight shared with the token embedding table             |
+----------------------------------------------------------+
                             ^
                             |
+----------------------------------------------------------+
| Final LayerNorm                    [1, 1024, 768]        |
+----------------------------------------------------------+
                             ^
                             |
+----------------------------------------------------------+
| Block 12  [1, 1024, 768]                                 |
| Causal self-attention: local, 256-token window           |
+----------------------------------------------------------+
                             ^
                             |
+----------------------------------------------------------+
| Blocks 2–11  [1, 1024, 768]                              |
| Local attention in even blocks, dense in odd blocks      |
+----------------------------------------------------------+
                             ^
                             |
+----------------------------------------------------------+
| Block 1  [1, 1024, 768]                                  |
| Causal self-attention: dense, full prefix                |
+----------------------------------------------------------+
                             ^
                             |
+----------------------------------------------------------+
| Add token and position embeddings  [1, 1024, 768]        |
+----------------------------------------------------------+
                             ^
                             |
             +---------------+--------------+
             |                              |
+--------------------------+   +---------------------------+
| Token embedding lookup   |   | Position embedding lookup |
| Table [50257, 768]       |   | Table [1024, 768]         |
+--------------------------+   +---------------------------+
             ^                              ^
             |                              |
+--------------------------+   +---------------------------+
| INPUT TOKEN IDs          |   | Position IDs              |
| [1, 1024]                |   | [1024], values 0..1023    |
+--------------------------+   +---------------------------+
```

Each block applies attention, then a GELU MLP. The arrows on the right carry the residuals into the two additions. Both attention variants include the current token.

```text
+---------------------------------------------------+
| Block output                      [1, 1024, 768]  |
+---------------------------------------------------+
                        ^
                        |
                       (+)<----------------------------------+
                        ^                                    |
                        |                                    |
+-----------------------------------------------+            |
| GELU MLP: 768 -> 3072 -> 768                  |            |
+-----------------------------------------------+            |
                        ^                                    |
                        |                                    |
+-----------------------------------------------+            |
| LayerNorm (before MLP)                        |            |
+-----------------------------------------------+            |
                        ^                                    |
                        |                                    |
                        +------------------------------------+
                        ^
                        |
                       (+)<----------------------------------+
                        ^                                    |
                        |                                    |
+-----------------------------------------------+            |
| Causal self-attention                         |            |
| QKV projection: 768 -> 2304                   |            |
| Q, K, V each [1, 12, 1024, 64]                |            |
| Merge 12 heads and project to width 768       |            |
+-----------------------------------------------+            |
                        ^                                    |
                        |                                    |
+-----------------------------------------------+            |
| LayerNorm (before attention)                  |            |
+-----------------------------------------------+            |
                        ^                                    |
                        |                                    |
                        +------------------------------------+
                        |
+---------------------------------------------------+
| Block input                       [1, 1024, 768]  |
+---------------------------------------------------+
```

The Q, K, and V axes are `[batch, heads, tokens, channels per head]`. See the [model implementation](src/gpt3/model/transformer.py).

## Run locally

### Set up the environment

Run from the project root. Install the locked Python 3.12 environment if `.venv` is missing or needs updating. This may download Python and dependencies. `--inexact` keeps extra tools already installed in `.venv`.

```bash
uv sync --locked --inexact --python 3.12
```

Set these variables in your terminal to use project-local caches and run offline by default. See the [cache setup](docs/usage.md#1-keep-caches-and-temporary-files-in-the-project) for the full environment.

```bash
mkdir -p .cache/tmp
export TMPDIR="$PWD/.cache/tmp"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
```

### Train on FineWeb-Edu with CUDA

The FineWeb-Edu preset requires a BF16-capable CUDA GPU. It targets 40,000 optimizer updates, with 4,096 target tokens per update per GPU. Activation checkpointing and chunked vocabulary loss reduce peak memory.

The first two commands download the configured Parquet shards and cache tokenization for the train and validation splits. The data inspector prepares each selected split before showing the first batch. Training can run offline after the cache is populated. [FineWeb-Edu preset details](configs/README.md#fineweb-edu-cached-dataset)

```bash
config=configs/fineweb_edu.toml
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 .venv/bin/gpt3-data "$config" --split train --batches 1
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 .venv/bin/gpt3-data "$config" --split validation --batches 1

.venv/bin/python -m gpt3.training "$config"
```

Each run writes `config.json`, `metrics.jsonl`, and checkpoints under `artifacts/fineweb-edu-10bt/RUN_ID/`. The preset logs validation and text samples every 100 updates and saves a checkpoint every 1,000 updates.

For a new CUDA run, `hardware.fused_adamw` and `hardware.allow_tf32` are optional. TF32 applies to float32 matrix multiplications. Keep the same hardware settings when resuming a checkpoint. [CUDA options](configs/README.md#cuda-optimizer-and-matmul-options)

### Resume, evaluate, and generate

Replace `RUN_ID` and the checkpoint filename with the values from a FineWeb-Edu run. To resume, copy the original TOML to `.cache/fineweb_edu_resume.toml` and set `resume_from` in its existing `[training]` table to that checkpoint path. Keep the original training horizon and world size.

```bash
.venv/bin/python -m gpt3.training .cache/fineweb_edu_resume.toml

config=configs/fineweb_edu.toml
run_dir="artifacts/fineweb-edu-10bt/RUN_ID"
checkpoint_path="$run_dir/checkpoint-00001000.pt"

# Evaluate a held-out split using a fresh report filename.
.venv/bin/python -m gpt3.evaluation "$config" \
  "$checkpoint_path" --split validation --output "$run_dir/validation.json"

# Generate using the cached tokenizer.
.venv/bin/python -m gpt3.generate "$config" \
  "$checkpoint_path" --prompt "Time is" --max-new-tokens 32
```

For the final held-out test, prepare the test split if it is missing from the cache, then evaluate it. Use a fresh output filename for each evaluation.

```bash
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 .venv/bin/gpt3-data \
  "$config" --split test --batches 1
.venv/bin/python -m gpt3.evaluation "$config" \
  "$checkpoint_path" --split test --output "$run_dir/test.json"
```

## Tests and diagnostics

Run the unit and integration tests without Hub downloads. The repo-wide Ruff commands also report lint and formatting issues in the standalone experiments.

```bash
.venv/bin/python -m pytest
.venv/bin/mypy
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

### Profile training

This profiles synthetic batches with the FineWeb-Edu model and hardware settings. Use a fresh output directory. [Scaling guide](docs/scaling.md)

```bash
.venv/bin/python -B scripts/profile_training.py configs/fineweb_edu.toml \
  --output artifacts/profiling/fineweb-bf16
```

## Implementation and validation

The model includes learned embeddings, pre-normalized residual blocks, GELU MLPs, alternating dense/local causal attention, final normalization, and tied vocabulary weights. Training includes AdamW, warmup/cosine decay, token-weighted accumulation, clipping, mixed precision, validation, and local or DDP checkpoint resume. The trainer accumulates the token-weighted training loss on-device and reads the total for logs and checkpoints. Generation supports temperature, top-k/top-p, seeded sampling, and token/text stopping. A key/value cache remains a future optimization.

Local checks cover shapes, causality, local windows, target alignment, losses, gradients, overfitting, checkpoint replay, sampling, and memory options. The synthetic two-process CPU validation matches a combined reference and checks exact resume.


## Project layout

```text
configs/              Training, resume, evaluation, and scaling presets
src/gpt3/
  config.py           Validated experiment configuration
  model/              Embeddings, attention, blocks, and loss
  data/               Hub loading, tokenization, and packed batches
  training/           Optimization, distributed execution, and checkpoints
  evaluation/         Held-out metrics, samples, and likelihood-scored tasks
  generate.py         Standalone checkpoint generation command
scripts/              End-to-end validation, profiling, and DDP validation
examples/tasks/       Separate demonstration/evaluation JSONL examples
tests/                Behavioral correctness and integration checks
docs/                 Usage, scope, and scaling guides
```

## Some FineWeb-Edu training results

The [results notebook](notebooks/training_results.ipynb) summarizes the model dimensions, validation scores, and learning curves for the initial 42,000-update run and its 10,000-update continuation.

The two runs processed 213 million target tokens, about **2.13%** of the nominal 10-billion-token FineWeb-Edu sample. The count includes any repeated exposures.

