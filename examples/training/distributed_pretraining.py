# Distributed Pretraining with Targon SDK
# Inspired by templar's decentralized training architecture
#
# This example demonstrates distributed model training using:
# - Remote gradient computation on cloud GPUs
# - DCT + top-k compression for efficient gradient transfer
# - Local gradient aggregation and model updates
#
# Usage:
#   targon run examples/training/distributed_pretraining.py \
#     --num-workers 2 --steps 100 --batch-size 8

"""
Distributed pretraining system using targon-sdk.

The architecture parallelizes gradient computation across remote workers:
1. Local entrypoint serializes model state
2. Remote workers compute gradients on their data shards  
3. Gradients are compressed using DCT + top-k + quantization
4. Local entrypoint aggregates gradients and updates model
"""

import asyncio
import io
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import SGD

import targon

# Training module paths
TRAINING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAINING_DIR))

# Image with dependencies for remote workers
image = (
    targon.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        "torch>=2.0.0",
        "einops>=0.7.0",
        "numpy>=1.24.0",
    )
    .add_local_dir(str(TRAINING_DIR), "/workspace/training")
)

app = targon.App("distributed-pretraining", image=image)


# =============================================================================
# Remote Function: Compute Gradients
# =============================================================================

@app.function(
    resource=targon.Compute.H200_SMALL,
    timeout=300,
    min_replicas=0,
    max_replicas=4,
)
@targon.concurrent(max_concurrency=1, target_concurrency=1)
def compute_gradients(
    model_state_bytes: bytes,
    batch_data: dict,
    worker_id: int,
    config: dict,
) -> dict[str, Any]:
    """
    Compute gradients on a remote worker and return compressed gradients.
    
    Args:
        model_state_bytes: Serialized model state dict.
        batch_data: Dictionary with 'input_ids' and 'targets'.
        worker_id: Unique identifier for this worker.
        config: Training configuration dict.
        
    Returns:
        Dictionary with compressed gradients and metadata.
    """
    sys.path.insert(0, "/workspace/training")
    
    import torch
    from model import create_model, TransformerConfig
    from compression import (
        ChunkingTransformer, 
        TopKCompressor, 
        compress_gradients,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Worker {worker_id}] Running on device: {device}")
    
    # Create model with same config
    model_config = TransformerConfig(
        vocab_size=config.get("vocab_size", 50257),
        n_layers=config.get("n_layers", 6),
        n_heads=config.get("n_heads", 8),
        d_model=config.get("d_model", 512),
        d_ff=config.get("d_ff", 2048),
        max_seq_len=config.get("max_seq_len", 512),
        dropout=0.0,  # No dropout during gradient computation
    )
    
    model = create_model(
        vocab_size=model_config.vocab_size,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        d_model=model_config.d_model,
        d_ff=model_config.d_ff,
        max_seq_len=model_config.max_seq_len,
        dropout=0.0,
    )
    
    # Load state from bytes
    buffer = io.BytesIO(model_state_bytes)
    state_dict = torch.load(buffer, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.train()
    
    # Prepare batch
    input_ids = torch.tensor(batch_data["input_ids"], dtype=torch.long, device=device)
    targets = torch.tensor(batch_data["targets"], dtype=torch.long, device=device)
    
    print(f"[Worker {worker_id}] Batch shape: {input_ids.shape}")
    
    # Forward pass
    model.zero_grad()
    logits, loss = model(input_ids, targets)
    
    print(f"[Worker {worker_id}] Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Initialize compression
    target_chunk = config.get("target_chunk", 64)
    topk = config.get("topk", 256)
    use_dct = config.get("use_dct", True)
    
    transformer = ChunkingTransformer(model, target_chunk=target_chunk)
    compressor = TopKCompressor(use_quantization=True)
    
    # Compress gradients
    compressed = compress_gradients(
        model=model,
        transformer=transformer,
        compressor=compressor,
        topk=topk,
        use_dct=use_dct,
    )
    
    # Calculate compression stats
    original_size = sum(
        p.grad.numel() * p.grad.element_size() 
        for p in model.parameters() 
        if p.grad is not None
    )
    compressed_size = sum(
        v.numel() * v.element_size() 
        for k, v in compressed.items() 
        if isinstance(v, torch.Tensor)
    )
    
    # Convert tensors to lists for JSON serialization
    result = {
        "worker_id": worker_id,
        "loss": loss.item(),
        "batch_size": input_ids.shape[0],
        "sequence_length": input_ids.shape[1],
        "original_size_bytes": original_size,
        "compressed_size_bytes": compressed_size,
        "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
        "gradients": {},
    }
    
    # Serialize compressed gradients
    for key, value in compressed.items():
        if isinstance(value, torch.Tensor):
            result["gradients"][key] = {
                "data": value.cpu().numpy().tolist(),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        elif isinstance(value, tuple):
            # Handle quant_params tuple
            serialized_tuple = []
            for item in value:
                if isinstance(item, torch.Tensor):
                    serialized_tuple.append({
                        "data": item.cpu().numpy().tolist(),
                        "shape": list(item.shape),
                        "dtype": str(item.dtype),
                    })
                else:
                    serialized_tuple.append(item)
            result["gradients"][key] = {"tuple": serialized_tuple}
        else:
            result["gradients"][key] = value
    
    print(f"[Worker {worker_id}] Compression ratio: {result['compression_ratio']:.2f}x")
    
    return result


# =============================================================================
# Helper Functions
# =============================================================================

def serialize_model_state(model: nn.Module) -> bytes:
    """Serialize model state dict to bytes."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def deserialize_gradients(result: dict, device: str = "cpu") -> dict:
    """Deserialize compressed gradients from worker result."""
    import numpy as np
    
    gradients = {}
    
    for key, value in result["gradients"].items():
        if isinstance(value, dict):
            if "data" in value:
                # Regular tensor
                dtype_map = {
                    "torch.uint8": np.uint8,
                    "torch.int64": np.int64,
                    "torch.float32": np.float32,
                    "torch.float16": np.float16,
                    "torch.bfloat16": np.float32,  # Convert bf16 to f32
                }
                np_dtype = dtype_map.get(value["dtype"], np.float32)
                arr = np.array(value["data"], dtype=np_dtype).reshape(value["shape"])
                gradients[key] = torch.from_numpy(arr).to(device)
            elif "tuple" in value:
                # Quant params tuple
                items = []
                for item in value["tuple"]:
                    if isinstance(item, dict) and "data" in item:
                        dtype_map = {
                            "torch.uint8": np.uint8,
                            "torch.int64": np.int64,
                            "torch.float32": np.float32,
                            "torch.float16": np.float16,
                        }
                        np_dtype = dtype_map.get(item["dtype"], np.float32)
                        arr = np.array(item["data"], dtype=np_dtype).reshape(item["shape"])
                        items.append(torch.from_numpy(arr).to(device))
                    else:
                        items.append(item)
                gradients[key] = tuple(items)
        else:
            gradients[key] = value
    
    return gradients


def outer_step(
    model: nn.Module,
    optimizer: SGD,
    aggregated_gradients: dict[str, torch.Tensor],
    lr: float = 1.0,
) -> dict:
    """
    Apply aggregated gradients to model parameters.
    
    Args:
        model: The model to update.
        optimizer: The optimizer.
        aggregated_gradients: Dictionary of gradient tensors.
        lr: Learning rate for the outer step.
        
    Returns:
        Dictionary with update statistics.
    """
    optimizer.zero_grad(set_to_none=True)
    
    total_norm_sq = 0.0
    updated_params = 0
    
    for name, param in model.named_parameters():
        if name in aggregated_gradients:
            grad = aggregated_gradients[name].to(param.device)
            
            # Ensure gradient has same shape as parameter
            if grad.shape != param.shape:
                print(f"Warning: Shape mismatch for {name}: grad {grad.shape} vs param {param.shape}")
                continue
            
            # Set gradient
            param.grad = grad
            total_norm_sq += grad.norm().item() ** 2
            updated_params += 1
    
    # Apply gradients via optimizer
    optimizer.step()
    
    return {
        "global_grad_norm": math.sqrt(total_norm_sq),
        "updated_params": updated_params,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: SGD,
    step: int,
    loss_history: list[float],
    checkpoint_dir: Path,
) -> Path:
    """
    Save a training checkpoint.
    
    Args:
        model: The model to save.
        optimizer: The optimizer.
        step: Current training step.
        loss_history: List of loss values.
        checkpoint_dir: Directory to save checkpoints.
        
    Returns:
        Path to saved checkpoint.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": loss_history,
        "timestamp": datetime.now().isoformat(),
    }
    
    path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    torch.save(checkpoint, path)
    
    # Also save as latest
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    
    print(f"Saved checkpoint to {path}")
    return path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[SGD] = None,
) -> dict:
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        
    Returns:
        Checkpoint metadata dict.
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return {
        "step": checkpoint["step"],
        "loss_history": checkpoint["loss_history"],
        "timestamp": checkpoint["timestamp"],
    }


# =============================================================================
# Local Entrypoint: Training Loop
# =============================================================================

@app.local_entrypoint()
async def main(
    # Training parameters
    num_workers: int = 2,
    steps: int = 100,
    batch_size: int = 8,
    sequence_length: int = 512,
    learning_rate: float = 0.01,
    
    # Model parameters  
    n_layers: int = 6,
    n_heads: int = 8,
    d_model: int = 512,
    d_ff: int = 2048,
    
    # Compression parameters
    topk: int = 256,
    target_chunk: int = 64,
    use_dct: bool = True,
    
    # Logging and checkpointing
    wandb_project: Optional[str] = None,
    checkpoint_dir: str = "./checkpoints",
    checkpoint_every: int = 10,
    resume_from: Optional[str] = None,
    
    # Data
    use_wikitext: bool = False,  # Use WikiText-2 (requires download)
):
    """
    Main training loop for distributed pretraining.
    
    This function:
    1. Initializes model and data locally
    2. Dispatches gradient computation to remote workers
    3. Aggregates gradients and updates the model
    4. Logs metrics and saves checkpoints
    
    Example:
        targon run examples/training/distributed_pretraining.py \
            --num-workers 2 --steps 100 --batch-size 8
    """
    print("=" * 60)
    print("Distributed Pretraining with Targon SDK")
    print("=" * 60)
    
    # Import local modules
    from model import create_model, TransformerConfig
    from data_utils import (
        create_tiny_dataset,
        prepare_wikitext2_dataset,
        get_batch_for_step,
    )
    from compression import (
        ChunkingTransformer,
        TopKCompressor,
        decompress_and_aggregate_gradients,
        compute_gradient_fingerprint,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Local device: {device}")
    print(f"Number of workers: {num_workers}")
    print(f"Training steps: {steps}")
    print(f"Batch size per worker: {batch_size}")
    
    # Initialize wandb if requested
    wandb_run = None
    if wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=wandb_project,
                config={
                    "num_workers": num_workers,
                    "steps": steps,
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "learning_rate": learning_rate,
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                    "d_model": d_model,
                    "d_ff": d_ff,
                    "topk": topk,
                    "target_chunk": target_chunk,
                    "use_dct": use_dct,
                },
            )
            print(f"Wandb initialized: {wandb_run.url}")
        except ImportError:
            print("Warning: wandb not installed, skipping logging")
    
    # Prepare dataset
    print("\nPreparing dataset...")
    if use_wikitext:
        dataset, tokenizer = prepare_wikitext2_dataset(sequence_length=sequence_length)
        vocab_size = tokenizer.vocab_size
    else:
        dataset, tokenizer = create_tiny_dataset(sequence_length=sequence_length)
        vocab_size = tokenizer.vocab_size
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model
    print("\nInitializing model...")
    model = create_model(
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        max_seq_len=sequence_length,
        dropout=0.1,
    )
    model = model.to(device)
    
    # Create optimizer
    optimizer = SGD(model.parameters(), lr=learning_rate)
    
    # Initialize compression for local aggregation
    transformer = ChunkingTransformer(model, target_chunk=target_chunk)
    compressor = TopKCompressor(use_quantization=True)
    
    # Training config for workers
    config = {
        "vocab_size": vocab_size,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_model": d_model,
        "d_ff": d_ff,
        "max_seq_len": sequence_length,
        "topk": topk,
        "target_chunk": target_chunk,
        "use_dct": use_dct,
    }
    
    # Resume from checkpoint if specified
    start_step = 0
    loss_history = []
    checkpoint_path = Path(checkpoint_dir)
    
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            print(f"\nResuming from checkpoint: {resume_path}")
            meta = load_checkpoint(resume_path, model, optimizer)
            start_step = meta["step"] + 1
            loss_history = meta["loss_history"]
            print(f"Resumed at step {start_step}")
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    total_tokens = 0
    training_start = time.time()
    
    for step in range(start_step, steps):
        step_start = time.time()
        
        print(f"\n--- Step {step + 1}/{steps} ---")
        
        # Serialize current model state
        model_state_bytes = serialize_model_state(model)
        model_size_mb = len(model_state_bytes) / (1024 * 1024)
        print(f"Model state size: {model_size_mb:.2f} MB")
        
        # Prepare batches for each worker
        worker_batches = []
        for worker_id in range(num_workers):
            batch = get_batch_for_step(
                dataset=dataset,
                step=step,
                worker_id=worker_id,
                num_workers=num_workers,
                batch_size=batch_size,
                seed=42,
            )
            worker_batches.append({
                "input_ids": batch["input_ids"].tolist(),
                "targets": batch["targets"].tolist(),
            })
        
        # Dispatch to remote workers in parallel
        print(f"Dispatching to {num_workers} workers...")
        dispatch_start = time.time()
        
        # Create async tasks for parallel execution
        tasks = []
        for worker_id, batch in enumerate(worker_batches):
            task = compute_gradients.remote(
                model_state_bytes=model_state_bytes,
                batch_data=batch,
                worker_id=worker_id,
                config=config,
            )
            tasks.append(task)
        
        # Gather results from all workers
        results = await asyncio.gather(*tasks)
        dispatch_time = time.time() - dispatch_start
        print(f"Remote computation time: {dispatch_time:.2f}s")
        
        # Process results
        successful_results = [r for r in results if r is not None]
        
        if not successful_results:
            print("Warning: No successful worker results, skipping step")
            continue
        
        # Aggregate losses
        worker_losses = [r["loss"] for r in successful_results]
        avg_loss = sum(worker_losses) / len(worker_losses)
        loss_history.append(avg_loss)
        
        # Deserialize and aggregate gradients
        print("Aggregating gradients...")
        aggregate_start = time.time()
        
        compressed_grads_list = [
            deserialize_gradients(r, device=device) 
            for r in successful_results
        ]
        
        aggregated = decompress_and_aggregate_gradients(
            model=model,
            compressed_grads_list=compressed_grads_list,
            transformer=transformer,
            compressor=compressor,
            use_dct=use_dct,
            device=device,
        )
        
        aggregate_time = time.time() - aggregate_start
        print(f"Aggregation time: {aggregate_time:.2f}s")
        
        # Compute gradient statistics
        fingerprint = compute_gradient_fingerprint(aggregated)
        
        # Apply outer step
        update_stats = outer_step(model, optimizer, aggregated, lr=learning_rate)
        
        # Calculate metrics
        step_tokens = sum(r["batch_size"] * r["sequence_length"] for r in successful_results)
        total_tokens += step_tokens
        step_time = time.time() - step_start
        tokens_per_sec = step_tokens / step_time
        
        avg_compression_ratio = sum(r["compression_ratio"] for r in successful_results) / len(successful_results)
        
        # Print step summary
        print(f"\nStep {step + 1} Summary:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Gradient L2 norm: {fingerprint['global_l2_norm']:.6f}")
        print(f"  Updated params: {update_stats['updated_params']}")
        print(f"  Compression ratio: {avg_compression_ratio:.2f}x")
        print(f"  Tokens/sec: {tokens_per_sec:.0f}")
        print(f"  Step time: {step_time:.2f}s")
        
        # Log to wandb
        if wandb_run:
            wandb_run.log({
                "train/loss": avg_loss,
                "train/gradient_norm": fingerprint["global_l2_norm"],
                "train/tokens_per_sec": tokens_per_sec,
                "train/compression_ratio": avg_compression_ratio,
                "train/step_time": step_time,
                "train/dispatch_time": dispatch_time,
                "train/aggregate_time": aggregate_time,
                "train/successful_workers": len(successful_results),
                "train/total_tokens": total_tokens,
            }, step=step)
        
        # Save checkpoint
        if (step + 1) % checkpoint_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                loss_history=loss_history,
                checkpoint_dir=checkpoint_path,
            )
    
    # Training complete
    total_time = time.time() - training_start
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total steps: {steps}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average tokens/sec: {total_tokens / total_time:.0f}")
    print(f"Final loss: {loss_history[-1]:.4f}")
    
    # Save final checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=steps - 1,
        loss_history=loss_history,
        checkpoint_dir=checkpoint_path,
    )
    
    # Demo text generation
    print("\n" + "=" * 60)
    print("Demo Text Generation")
    print("=" * 60)
    
    demo_prompts = ["First", "The", "We"]
    model.eval()
    
    for prompt in demo_prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)
        
        generated = model.generate_text(
            prompt=prompt,
            tokenizer=tokenizer,
            max_new_tokens=50,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        print(f"Generated: {generated}")
    
    # Close wandb
    if wandb_run:
        wandb_run.finish()
    
    return {
        "final_loss": loss_history[-1],
        "total_tokens": total_tokens,
        "total_time": total_time,
        "loss_history": loss_history,
    }


if __name__ == "__main__":
    # For local testing without targon
    import asyncio
    asyncio.run(main(num_workers=1, steps=5, batch_size=4, use_wikitext=False))

