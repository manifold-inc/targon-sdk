# Distributed Training Module
# Example implementation of distributed pretraining using targon-sdk

"""
Distributed pretraining example using gradient compression.

This module provides:
- compression: DCT + top-k gradient compression (from templar)
- model: Simple GPT-style transformer
- data_utils: Dataset loading and sharding utilities
- distributed_pretraining: Main training script
"""

from .compression import (
    ChunkingTransformer,
    TopKCompressor,
    compress_gradients,
    decompress_and_aggregate_gradients,
    compute_gradient_fingerprint,
)
from .model import (
    SimpleTransformer,
    TransformerConfig,
    create_model,
    create_tiny_model,
    create_small_model,
    create_medium_model,
    sample_from_model,
)
from .data_utils import (
    TextDataset,
    CharTokenizer,
    SimpleTokenizer,
    create_tiny_dataset,
    prepare_wikitext2_dataset,
    get_batch_for_step,
)

__all__ = [
    # Compression
    "ChunkingTransformer",
    "TopKCompressor",
    "compress_gradients",
    "decompress_and_aggregate_gradients",
    "compute_gradient_fingerprint",
    # Model
    "SimpleTransformer",
    "TransformerConfig",
    "create_model",
    "create_tiny_model",
    "create_small_model",
    "create_medium_model",
    "sample_from_model",
    # Data
    "TextDataset",
    "CharTokenizer",
    "SimpleTokenizer",
    "create_tiny_dataset",
    "prepare_wikitext2_dataset",
    "get_batch_for_step",
]

