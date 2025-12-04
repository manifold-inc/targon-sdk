# Distributed Pretraining with Targon SDK

A distributed training system inspired by [templar](https://github.com/one-covenant/templar)'s decentralized training architecture. This example demonstrates how to parallelize gradient computation across remote cloud GPUs while maintaining the model locally.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           LOCAL MACHINE                                  │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐  │
│  │    Model     │───▶│ Serialize State  │───▶│  Dispatch to Workers │  │
│  └──────────────┘    └──────────────────┘    └──────────┬───────────┘  │
│         ▲                                               │               │
│         │                                               ▼               │
│  ┌──────┴───────┐    ┌──────────────────┐    ┌──────────────────────┐  │
│  │ Outer Step   │◀───│   Aggregate &    │◀───│  Gather Compressed   │  │
│  │  (Update)    │    │   Decompress     │    │     Gradients        │  │
│  └──────────────┘    └──────────────────┘    └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
        ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
        │   Remote GPU    │ │   Remote GPU    │ │   Remote GPU    │
        │   Worker 0      │ │   Worker 1      │ │   Worker N      │
        │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
        │ │Load Model   │ │ │ │Load Model   │ │ │ │Load Model   │ │
        │ │Forward Pass │ │ │ │Forward Pass │ │ │ │Forward Pass │ │
        │ │Backward Pass│ │ │ │Backward Pass│ │ │ │Backward Pass│ │
        │ │Compress Grad│ │ │ │Compress Grad│ │ │ │Compress Grad│ │
        │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
        └─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Key Features

- **Distributed Gradient Computation**: Parallelize training across multiple remote GPUs
- **Efficient Gradient Compression**: DCT + top-k + 8-bit quantization (from templar)
- **Local Model Updates**: Model stays on your machine, only gradients are transferred
- **Checkpointing**: Save and resume training from checkpoints
- **Weights & Biases Integration**: Track experiments with wandb logging

## Compression Pipeline

Following templar's approach for efficient gradient transfer:

1. **DCT Transform**: Apply Discrete Cosine Transform to capture gradient structure
2. **Top-K Selection**: Keep only the K most significant coefficients
3. **12-bit Index Packing**: Compress indices from 64-bit to 12-bit representation
4. **8-bit Quantization**: Quantize values to uint8 with lookup table dequantization

This achieves 10-50x compression ratios while preserving gradient quality.

## Installation

```bash
# Install targon-sdk
pip install -e /path/to/targon-sdk

# Additional dependencies for local testing
pip install torch einops numpy wandb
```

## Quick Start

### Basic Usage

```bash
# Run distributed pretraining with 2 workers
targon run examples/training/distributed_pretraining.py \
  --num-workers 2 \
  --steps 100 \
  --batch-size 8
```

### With WikiText-2 Dataset

```bash
targon run examples/training/distributed_pretraining.py \
  --num-workers 2 \
  --steps 500 \
  --batch-size 8 \
  --use-wikitext
```

### Full Configuration

```bash
targon run examples/training/distributed_pretraining.py \
  --num-workers 4 \
  --steps 1000 \
  --batch-size 16 \
  --sequence-length 512 \
  --learning-rate 0.01 \
  --n-layers 6 \
  --n-heads 8 \
  --d-model 512 \
  --d-ff 2048 \
  --topk 256 \
  --target-chunk 64 \
  --use-dct \
  --wandb-project "distributed-pretraining" \
  --checkpoint-dir "./checkpoints" \
  --checkpoint-every 50 \
  --use-wikitext
```

### Resume from Checkpoint

```bash
targon run examples/training/distributed_pretraining.py \
  --num-workers 2 \
  --steps 200 \
  --resume-from ./checkpoints/checkpoint_step_99.pt
```

## Configuration Options

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-workers` | 2 | Number of remote GPU workers |
| `--steps` | 100 | Total training steps |
| `--batch-size` | 8 | Batch size per worker |
| `--sequence-length` | 512 | Sequence length for training |
| `--learning-rate` | 0.01 | Learning rate for outer optimizer |

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-layers` | 6 | Number of transformer layers |
| `--n-heads` | 8 | Number of attention heads |
| `--d-model` | 512 | Hidden dimension |
| `--d-ff` | 2048 | Feed-forward dimension |

### Compression Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--topk` | 256 | Number of top-k values to keep |
| `--target-chunk` | 64 | Target chunk size for DCT |
| `--use-dct` | True | Whether to use DCT transformation |

### Logging & Checkpointing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--wandb-project` | None | Wandb project name (None to disable) |
| `--checkpoint-dir` | ./checkpoints | Directory for checkpoints |
| `--checkpoint-every` | 10 | Save checkpoint every N steps |
| `--resume-from` | None | Path to checkpoint to resume from |

## File Structure

```
training/
├── distributed_pretraining.py  # Main training script
├── compression.py              # DCT + top-k compression (from templar)
├── model.py                    # Simple GPT-style transformer
├── data_utils.py               # Dataset loading and sharding
└── README.md                   # This file
```

## Model Architecture

The included transformer model is a decoder-only GPT-style architecture:

- **RoPE (Rotary Position Embeddings)**: For position information
- **RMSNorm**: Pre-normalization in each block
- **SwiGLU Activation**: In the feed-forward layers
- **Tied Embeddings**: Input and output embeddings share weights

Default configuration (~10M parameters):
- 6 layers
- 512 hidden dimension
- 8 attention heads
- 2048 FFN dimension

## How It Works

### Training Loop

1. **Model Serialization**: The local model state is serialized to bytes
2. **Data Sharding**: Each worker gets a deterministic slice of data based on step and worker ID
3. **Remote Dispatch**: Workers receive model state and compute forward/backward passes
4. **Gradient Compression**: Each worker compresses gradients using DCT + top-k
5. **Aggregation**: Local machine gathers and decompresses all gradients
6. **Outer Step**: Aggregated gradient is applied to update the local model

### Compression Details

The compression pipeline achieves ~10-50x reduction in gradient size:

```python
# Original gradient: 40MB (10M params × 4 bytes)
# After top-k (k=256): ~2MB (sparse representation)
# After quantization: ~1MB (8-bit values)
# After index packing: ~750KB (12-bit indices)
```

## Metrics Logged

When wandb is enabled, the following metrics are tracked:

- `train/loss`: Average loss across workers
- `train/gradient_norm`: L2 norm of aggregated gradients
- `train/tokens_per_sec`: Training throughput
- `train/compression_ratio`: Achieved compression ratio
- `train/step_time`: Total time per step
- `train/dispatch_time`: Time for remote computation
- `train/aggregate_time`: Time for local aggregation
- `train/successful_workers`: Number of workers that completed
- `train/total_tokens`: Cumulative tokens processed

## Local Testing

You can run the training locally without targon for testing:

```bash
cd examples/training
python distributed_pretraining.py
```

This runs with 1 worker and 5 steps using the tiny Shakespeare dataset.

## Extending the Example

### Custom Dataset

Modify `data_utils.py` to add your own dataset:

```python
def create_custom_dataset(sequence_length: int) -> tuple[TextDataset, Tokenizer]:
    text = load_your_text()
    tokenizer = YourTokenizer()
    dataset = TextDataset(text, tokenizer, sequence_length)
    return dataset, tokenizer
```

### Different Model

Modify `model.py` or create a new model file:

```python
def create_large_model():
    return create_model(
        n_layers=12,
        n_heads=16,
        d_model=1024,
        d_ff=4096,
    )
```

### Compression Tuning

Adjust compression parameters for your use case:

- **Higher topk**: Better gradient quality, larger transfer size
- **Lower topk**: More compression, potentially noisier gradients
- **use_dct=False**: Skip DCT for faster compression but less efficiency

## Credits

- Gradient compression adapted from [templar](https://github.com/one-covenant/templar)
- DCT implementation based on [torch-dct](https://github.com/zh217/torch-dct)
- Model architecture inspired by GPT and LLaMA

## License

MIT License - See the main targon-sdk LICENSE file.





