# Simple GPT-style Transformer for Distributed Training
# Medium-sized model: 6 layers, 512 hidden dim, 8 attention heads (~10M params)

"""
A simple decoder-only transformer implementation optimized for distributed training.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Configuration for the transformer model."""
    vocab_size: int = 50257  # GPT-2 vocabulary size
    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048  # 4 * d_model
    max_seq_len: int = 512
    dropout: float = 0.1
    bias: bool = False  # Linear layers without bias (like LLaMA)
    rope_theta: float = 10000.0
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = self.d_model // self.n_heads


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor for rotary embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape frequency tensor for broadcasting with input tensor."""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    """Multi-head attention with rotary position embeddings."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        
        self.wq = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.wk = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.wv = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        
        # Linear projections
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # Reshape for multi-head attention
        xq = xq.view(bsz, seqlen, self.n_heads, self.d_head)
        xk = xk.view(bsz, seqlen, self.n_heads, self.d_head)
        xv = xv.view(bsz, seqlen, self.n_heads, self.d_head)
        
        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # Transpose for attention: (bsz, n_heads, seqlen, d_head)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if mask is not None:
            scores = scores + mask
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, xv)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)
        
        return self.wo(output)


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        hidden_dim = config.d_ff
        
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        self.w3 = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: w2(SiLU(w1(x)) * w3(x))
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """A single transformer decoder block."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class SimpleTransformer(nn.Module):
    """
    A simple GPT-style decoder-only transformer.
    
    Default configuration: 6 layers, 512 hidden dim, 8 heads (~10M parameters)
    """
    
    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__()
        self.config = config or TransformerConfig()
        
        # Token embeddings
        self.tok_embeddings = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.dropout = nn.Dropout(self.config.dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(self.config) for _ in range(self.config.n_layers)
        ])
        
        # Output
        self.norm = RMSNorm(self.config.d_model)
        self.output = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        
        # Tie embeddings
        self.output.weight = self.tok_embeddings.weight
        
        # Precompute rotary embeddings
        self.freqs_cis = precompute_freqs_cis(
            self.config.d_head,
            self.config.max_seq_len,
            self.config.rope_theta,
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Print model size
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {n_params:,} parameters")

    def _init_weights(self, module: nn.Module):
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the transformer.
        
        Args:
            tokens: Input token ids of shape (batch_size, seq_len)
            targets: Target token ids for computing loss (optional)
            
        Returns:
            Tuple of (logits, loss) where loss is None if targets not provided
        """
        bsz, seqlen = tokens.shape
        device = tokens.device
        
        # Get embeddings
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        
        # Move freqs_cis to device and slice to sequence length
        freqs_cis = self.freqs_cis[:seqlen].to(device)
        
        # Create causal mask
        mask = torch.full(
            (seqlen, seqlen), float("-inf"), device=device, dtype=h.dtype
        )
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seqlen, seqlen)
        
        # Apply transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        
        # Final norm and output projection
        h = self.norm(h)
        logits = self.output(h)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            
        return logits, loss

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively from a prompt.
        
        Args:
            prompt_tokens: Input token ids of shape (batch_size, seq_len) or (seq_len,)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (1.0 = normal, <1.0 = more deterministic, >1.0 = more random)
            top_k: If set, only sample from the top-k most likely tokens
            top_p: If set, use nucleus sampling with this probability threshold
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            stop_tokens: List of token ids that stop generation
            
        Returns:
            Generated token ids including the prompt
        """
        self.eval()
        
        # Handle 1D input
        if prompt_tokens.dim() == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)
        
        batch_size, seq_len = prompt_tokens.shape
        device = prompt_tokens.device
        
        # Initialize generated sequence with prompt
        generated = prompt_tokens.clone()
        
        # Track which sequences have finished (hit stop token)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_new_tokens):
            # Get current sequence (truncate if too long)
            curr_seq = generated
            if curr_seq.shape[1] > self.config.max_seq_len:
                curr_seq = curr_seq[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits, _ = self.forward(curr_seq)
            
            # Get logits for the last position
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                min_top_k = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k,
                    torch.full_like(next_token_logits, float('-inf')),
                    next_token_logits
                )
            
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[:, 0] = False
                
                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            
            if temperature == 0:
                # Greedy decoding
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                # Sample from distribution
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for stop tokens
            if stop_tokens:
                for stop_token in stop_tokens:
                    finished = finished | (next_token.squeeze(-1) == stop_token)
                
                if finished.all():
                    break
        
        return generated

    def generate_text(
        self,
        prompt: str,
        tokenizer,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        """
        Generate text from a string prompt.
        
        Args:
            prompt: Input text prompt
            tokenizer: Tokenizer with encode/decode methods
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repetition
            
        Returns:
            Generated text including the prompt
        """
        device = next(self.parameters()).device
        
        # Encode prompt
        prompt_tokens = tokenizer.encode(prompt)
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate
        output_tokens = self.generate(
            prompt_tokens=prompt_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        
        # Decode
        generated_tokens = output_tokens[0].tolist()
        return tokenizer.decode(generated_tokens)


def sample_from_model(
    model: "SimpleTransformer",
    tokenizer,
    prompt: str = "",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    num_samples: int = 1,
) -> list[str]:
    """
    Generate multiple text samples from the model.
    
    Args:
        model: The transformer model
        tokenizer: Tokenizer with encode/decode methods
        prompt: Input text prompt
        max_new_tokens: Maximum new tokens per sample
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        num_samples: Number of samples to generate
        
    Returns:
        List of generated text strings
    """
    samples = []
    for _ in range(num_samples):
        text = model.generate_text(
            prompt=prompt,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        samples.append(text)
    return samples


def create_model(
    vocab_size: int = 50257,
    n_layers: int = 6,
    n_heads: int = 8,
    d_model: int = 512,
    d_ff: int = 2048,
    max_seq_len: int = 512,
    dropout: float = 0.1,
) -> SimpleTransformer:
    """
    Create a transformer model with the specified configuration.
    
    Default creates a ~10M parameter model:
    - 6 layers
    - 512 hidden dimension
    - 8 attention heads
    - 2048 feed-forward dimension
    """
    config = TransformerConfig(
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )
    return SimpleTransformer(config)


def create_tiny_model() -> SimpleTransformer:
    """Create a tiny model for quick testing (~100K params)."""
    return create_model(
        n_layers=2,
        n_heads=4,
        d_model=128,
        d_ff=512,
        max_seq_len=256,
    )


def create_small_model() -> SimpleTransformer:
    """Create a small model (~1M params)."""
    return create_model(
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_ff=1024,
        max_seq_len=512,
    )


def create_medium_model() -> SimpleTransformer:
    """Create a medium model (~10M params)."""
    return create_model(
        n_layers=6,
        n_heads=8,
        d_model=512,
        d_ff=2048,
        max_seq_len=512,
    )

