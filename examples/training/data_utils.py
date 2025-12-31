# Data Utilities for Distributed Training
# Handles dataset loading, tokenization, and sharding for workers

"""
Dataset utilities for loading and preparing text data for distributed training.
Supports WikiText-2 and simple text files.
"""

import os
import hashlib
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


# WikiText-2 URL and cache location
WIKITEXT2_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
CACHE_DIR = Path.home() / ".cache" / "targon_training"


def download_wikitext2(cache_dir: Optional[Path] = None) -> Path:
    """
    Download WikiText-2 dataset if not already cached.
    
    Args:
        cache_dir: Directory to cache the dataset.
        
    Returns:
        Path to the extracted dataset directory.
    """
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_dir = cache_dir / "wikitext-2"
    zip_path = cache_dir / "wikitext-2-v1.zip"
    
    # Check if already downloaded
    if dataset_dir.exists() and (dataset_dir / "wiki.train.tokens").exists():
        print(f"WikiText-2 already cached at {dataset_dir}")
        return dataset_dir
    
    # Download
    print(f"Downloading WikiText-2 from {WIKITEXT2_URL}...")
    urllib.request.urlretrieve(WIKITEXT2_URL, zip_path)
    
    # Extract
    print(f"Extracting to {cache_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(cache_dir)
    
    # Clean up zip
    zip_path.unlink()
    
    print(f"WikiText-2 ready at {dataset_dir}")
    return dataset_dir


def load_wikitext2_text(split: str = "train", cache_dir: Optional[Path] = None) -> str:
    """
    Load WikiText-2 text for a given split.
    
    Args:
        split: One of 'train', 'valid', or 'test'.
        cache_dir: Directory where dataset is cached.
        
    Returns:
        The raw text content.
    """
    dataset_dir = download_wikitext2(cache_dir)
    
    split_map = {
        "train": "wiki.train.tokens",
        "valid": "wiki.valid.tokens", 
        "test": "wiki.test.tokens",
    }
    
    if split not in split_map:
        raise ValueError(f"Invalid split '{split}'. Must be one of {list(split_map.keys())}")
    
    file_path = dataset_dir / split_map[split]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text


class CharTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, text: Optional[str] = None):
        """
        Initialize tokenizer.
        
        Args:
            text: Text to build vocabulary from. If None, uses default chars.
        """
        if text is not None:
            # Build vocab from text
            chars = sorted(list(set(text)))
        else:
            # Default ASCII chars + special tokens
            chars = [chr(i) for i in range(128)]
        
        # Add special tokens
        self.special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
        all_tokens = self.special_tokens + chars
        
        self.char_to_idx = {ch: i for i, ch in enumerate(all_tokens)}
        self.idx_to_char = {i: ch for i, ch in enumerate(all_tokens)}
        self.vocab_size = len(all_tokens)
        
        self.pad_id = self.char_to_idx['<pad>']
        self.unk_id = self.char_to_idx['<unk>']
        self.bos_id = self.char_to_idx['<bos>']
        self.eos_id = self.char_to_idx['<eos>']

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids."""
        return [self.char_to_idx.get(ch, self.unk_id) for ch in text]

    def decode(self, tokens: list[int]) -> str:
        """Decode token ids to text."""
        chars = [self.idx_to_char.get(idx, '<unk>') for idx in tokens]
        # Filter out special tokens for cleaner output
        chars = [ch for ch in chars if ch not in self.special_tokens]
        return ''.join(chars)


class SimpleTokenizer:
    """
    Simple word-level tokenizer with BPE-like vocabulary.
    Uses a fixed vocabulary size with character fallback.
    """
    
    def __init__(self, vocab_size: int = 50257):
        """
        Initialize with GPT-2 compatible vocab size.
        
        Args:
            vocab_size: Target vocabulary size.
        """
        self.vocab_size = vocab_size
        
        # Special tokens
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
        }
        
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        
        # Build character-level vocab for remaining slots
        # This gives us a simple but functional tokenizer
        chars = [chr(i) for i in range(32, 127)]  # Printable ASCII
        chars += ['\n', '\t', ' ']
        
        self.char_to_idx = dict(self.special_tokens)
        idx = len(self.special_tokens)
        for ch in chars:
            if ch not in self.char_to_idx:
                self.char_to_idx[ch] = idx
                idx += 1
                
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.actual_vocab_size = idx

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids."""
        return [self.char_to_idx.get(ch, self.unk_id) for ch in text]

    def decode(self, tokens: list[int]) -> str:
        """Decode token ids to text."""
        special = set(self.special_tokens.values())
        chars = [
            self.idx_to_char.get(idx, '') 
            for idx in tokens 
            if idx not in special
        ]
        return ''.join(chars)


class TextDataset(Dataset):
    """
    Dataset for language modeling on tokenized text.
    """
    
    def __init__(
        self,
        text: str,
        tokenizer: CharTokenizer | SimpleTokenizer,
        sequence_length: int = 512,
        stride: int = 256,
    ):
        """
        Initialize the dataset.
        
        Args:
            text: The raw text to tokenize.
            tokenizer: Tokenizer instance.
            sequence_length: Length of each sequence.
            stride: Stride between sequences (for overlapping).
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.tokenizer = tokenizer
        
        # Tokenize entire text
        print(f"Tokenizing {len(text):,} characters...")
        self.tokens = tokenizer.encode(text)
        print(f"Total tokens: {len(self.tokens):,}")
        
        # Calculate number of samples
        self.n_samples = max(1, (len(self.tokens) - sequence_length) // stride + 1)
        print(f"Created dataset with {self.n_samples:,} samples")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with 'input_ids' tensor.
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length + 1  # +1 for target offset
        
        # Handle boundary
        if end_idx > len(self.tokens):
            end_idx = len(self.tokens)
            start_idx = max(0, end_idx - self.sequence_length - 1)
        
        tokens = self.tokens[start_idx:end_idx]
        
        # Pad if necessary
        if len(tokens) < self.sequence_length + 1:
            tokens = tokens + [self.tokenizer.pad_id] * (self.sequence_length + 1 - len(tokens))
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'targets': targets,
        }


def create_data_shards(
    dataset: TextDataset,
    num_workers: int,
    worker_id: int,
    batch_size: int,
    seed: int = 42,
) -> list[dict[str, torch.Tensor]]:
    """
    Create deterministic data shards for a specific worker.
    
    Args:
        dataset: The full dataset.
        num_workers: Total number of workers.
        worker_id: ID of this worker (0-indexed).
        batch_size: Number of samples per batch.
        seed: Random seed for reproducibility.
        
    Returns:
        List of batches for this worker.
    """
    # Create deterministic indices based on seed
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Shuffle indices
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    
    # Split indices among workers
    worker_indices = indices[worker_id::num_workers]
    
    # Create batches
    batches = []
    for i in range(0, len(worker_indices), batch_size):
        batch_indices = worker_indices[i:i + batch_size]
        if len(batch_indices) < batch_size:
            # Skip incomplete batch or pad
            continue
            
        batch_samples = [dataset[idx] for idx in batch_indices]
        
        batch = {
            'input_ids': torch.stack([s['input_ids'] for s in batch_samples]),
            'targets': torch.stack([s['targets'] for s in batch_samples]),
        }
        batches.append(batch)
    
    return batches


def get_batch_for_step(
    dataset: TextDataset,
    step: int,
    worker_id: int,
    num_workers: int,
    batch_size: int,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """
    Get a specific batch for a given training step and worker.
    This ensures deterministic data assignment per step.
    
    Args:
        dataset: The full dataset.
        step: Current training step.
        worker_id: ID of this worker (0-indexed).
        num_workers: Total number of workers.
        batch_size: Number of samples per batch.
        seed: Base random seed.
        
    Returns:
        A single batch dictionary.
    """
    # Create unique seed for this step
    step_seed = seed + step * 1000
    
    # Generate deterministic indices for this step
    generator = torch.Generator()
    generator.manual_seed(step_seed)
    
    # Each worker gets different samples for this step
    total_samples_needed = num_workers * batch_size
    indices = torch.randperm(len(dataset), generator=generator)[:total_samples_needed].tolist()
    
    # Get this worker's portion
    start_idx = worker_id * batch_size
    end_idx = start_idx + batch_size
    worker_indices = indices[start_idx:end_idx]
    
    # Build batch
    batch_samples = [dataset[idx] for idx in worker_indices]
    
    return {
        'input_ids': torch.stack([s['input_ids'] for s in batch_samples]),
        'targets': torch.stack([s['targets'] for s in batch_samples]),
    }


def prepare_wikitext2_dataset(
    sequence_length: int = 512,
    cache_dir: Optional[Path] = None,
) -> tuple[TextDataset, SimpleTokenizer]:
    """
    Prepare WikiText-2 dataset with tokenizer.
    
    Args:
        sequence_length: Sequence length for samples.
        cache_dir: Cache directory for dataset.
        
    Returns:
        Tuple of (dataset, tokenizer).
    """
    # Load text
    text = load_wikitext2_text("train", cache_dir)
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    
    # Create dataset
    dataset = TextDataset(
        text=text,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        stride=sequence_length // 2,  # 50% overlap
    )
    
    return dataset, tokenizer


def serialize_batch(batch: dict[str, torch.Tensor]) -> dict[str, bytes]:
    """
    Serialize a batch for network transfer.
    
    Args:
        batch: Dictionary of tensors.
        
    Returns:
        Dictionary of serialized bytes.
    """
    serialized = {}
    for key, tensor in batch.items():
        buffer = torch.ByteStorage.from_buffer(tensor.numpy().tobytes())
        serialized[key] = {
            'data': tensor.numpy().tobytes(),
            'shape': tensor.shape,
            'dtype': str(tensor.dtype),
        }
    return serialized


def deserialize_batch(serialized: dict, device: str = "cpu") -> dict[str, torch.Tensor]:
    """
    Deserialize a batch from network transfer.
    
    Args:
        serialized: Dictionary with serialized data.
        device: Device to place tensors on.
        
    Returns:
        Dictionary of tensors.
    """
    import numpy as np
    
    batch = {}
    for key, data in serialized.items():
        dtype_map = {
            'torch.int64': np.int64,
            'torch.long': np.int64,
            'torch.float32': np.float32,
            'torch.float16': np.float16,
        }
        np_dtype = dtype_map.get(data['dtype'], np.int64)
        arr = np.frombuffer(data['data'], dtype=np_dtype).reshape(data['shape'])
        batch[key] = torch.from_numpy(arr.copy()).to(device)
    
    return batch


# Simple text for testing without downloading
TINY_SHAKESPEARE = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
"""


def create_tiny_dataset(
    sequence_length: int = 128,
) -> tuple[TextDataset, CharTokenizer]:
    """
    Create a tiny dataset from Shakespeare text for testing.
    
    Args:
        sequence_length: Sequence length for samples.
        
    Returns:
        Tuple of (dataset, tokenizer).
    """
    # Repeat the text to get more samples
    text = TINY_SHAKESPEARE * 100
    
    tokenizer = CharTokenizer(text)
    dataset = TextDataset(
        text=text,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        stride=sequence_length // 2,
    )
    
    return dataset, tokenizer





