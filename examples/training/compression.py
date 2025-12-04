# Gradient Compression for Distributed Training
# Adapted from templar's compress.py (https://github.com/one-covenant/templar)
# MIT License © 2025 tplr.ai

"""
Gradient compression pipeline using DCT transformation and top-k sparsification.
This module provides efficient gradient compression for distributed training.
"""

import math
from typing import Literal, Sequence, TypeAlias, TypeVar, cast, overload

import torch
import torch.fft
from einops import rearrange

# Type aliases
ShapeT: TypeAlias = tuple[int, ...]
Shape4D = tuple[int, int, int, int]
TotK: TypeAlias = int
IdxT: TypeAlias = torch.Tensor
ValT: TypeAlias = torch.Tensor
QuantParamsT: TypeAlias = tuple[torch.Tensor, float, int, torch.Tensor, torch.dtype]

Q = TypeVar("Q", Literal[True], Literal[False])


def pack_12bit_indices(indices: torch.Tensor) -> torch.Tensor:
    """
    Pack int64 indices into 12-bit representation.
    Every 2 indices (24 bits) are packed into 3 uint8 values.
    
    Args:
        indices: Tensor with values < 4096 (12-bit max), must have even number of elements
        
    Returns:
        packed_tensor as uint8
    """
    max_idx = indices.max().item() if indices.numel() > 0 else 0
    if max_idx >= 4096:
        raise ValueError(f"Index {max_idx} exceeds 12-bit limit (4095)")

    indices_flat = indices.flatten()
    n_indices = indices_flat.numel()

    if n_indices % 2 != 0:
        raise ValueError(f"Number of indices must be even, got {n_indices}")

    indices_flat = indices_flat.to(torch.int32)
    n_pairs = n_indices // 2
    packed_size = n_pairs * 3
    packed = torch.zeros(packed_size, dtype=torch.uint8, device=indices.device)

    if n_pairs > 0:
        idx_pairs = indices_flat.reshape(-1, 2)
        idx1 = idx_pairs[:, 0]
        idx2 = idx_pairs[:, 1]

        packed[0::3] = (idx1 & 0xFF).to(torch.uint8)
        packed[1::3] = (((idx1 >> 8) & 0x0F) | ((idx2 & 0x0F) << 4)).to(torch.uint8)
        packed[2::3] = ((idx2 >> 4) & 0xFF).to(torch.uint8)

    return packed


def unpack_12bit_indices(packed: torch.Tensor, values_shape: ShapeT) -> torch.Tensor:
    """
    Unpack 12-bit packed indices back to int64.
    
    Args:
        packed: Packed uint8 tensor
        values_shape: Shape of the values tensor
        
    Returns:
        Unpacked indices as int64 tensor
    """
    n_indices = int(torch.prod(torch.tensor(values_shape)).item())

    if n_indices == 0:
        return torch.zeros(values_shape, dtype=torch.int64, device=packed.device)

    if n_indices % 2 != 0:
        raise ValueError(f"Number of indices must be even, got {n_indices}")

    indices = torch.zeros(n_indices, dtype=torch.int64, device=packed.device)
    n_pairs = n_indices // 2

    if n_pairs > 0:
        byte0 = packed[0::3].to(torch.int64)
        byte1 = packed[1::3].to(torch.int64)
        byte2 = packed[2::3].to(torch.int64)

        indices[0::2] = byte0 | ((byte1 & 0x0F) << 8)
        indices[1::2] = ((byte1 >> 4) & 0x0F) | (byte2 << 4)

    return indices.reshape(values_shape)


# DCT Implementation (from torch-dct)
def _dct_fft_impl(v) -> torch.Tensor:
    """FFT-based implementation of the DCT."""
    return torch.view_as_real(torch.fft.fft(v, dim=1))


def _idct_irfft_impl(V) -> torch.Tensor:
    """IRFFT-based implementation of the IDCT."""
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


def _dct(x, norm=None) -> torch.Tensor:
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    
    Args:
        x: the input signal
        norm: the normalization, None or 'ortho'
        
    Returns:
        the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = _dct_fft_impl(v)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= math.sqrt(N) * 2
        V[:, 1:] /= math.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)
    return V


def _idct(X, norm=None) -> torch.Tensor:
    """
    The inverse to DCT-II (scaled DCT-III)
    
    Args:
        X: the input signal
        norm: the normalization, None or 'ortho'
        
    Returns:
        the inverse DCT-II of the signal over the last dimension
    """
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= math.sqrt(N) * 2
        X_v[:, 1:] *= math.sqrt(N / 2) * 2

    k = (
        torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
        * math.pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = _idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def _get_prime_divisors(n: int) -> list[int]:
    """Get the prime divisors of a number."""
    divisors = []
    while n % 2 == 0:
        divisors.append(2)
        n //= 2
    while n % 3 == 0:
        divisors.append(3)
        n //= 3
    i = 5
    while i * i <= n:
        for k in (i, i + 2):
            while n % k == 0:
                divisors.append(k)
                n //= k
        i += 6
    if n > 1:
        divisors.append(n)
    return divisors


def _get_divisors(n: int) -> list[int]:
    """Get all divisors of a number."""
    divisors = []
    if n == 1:
        divisors.append(1)
    elif n > 1:
        prime_factors = _get_prime_divisors(n)
        divisors = [1]
        last_prime = 0
        factor = 0
        slice_len = 0
        for prime in prime_factors:
            if last_prime != prime:
                slice_len = len(divisors)
                factor = prime
            else:
                factor *= prime
            for i in range(slice_len):
                divisors.append(divisors[i] * factor)
            last_prime = prime
        divisors.sort()
    return divisors


def _get_smaller_split(n: int, close_to: int) -> int:
    """Find the largest divisor of n that is <= close_to."""
    all_divisors = _get_divisors(n)
    for ix, val in enumerate(all_divisors):
        if val == close_to:
            return val
        if val > close_to:
            if ix == 0:
                return val
            return all_divisors[ix - 1]
    return n


class ChunkingTransformer:
    """
    Handles chunking tensors for efficient gradient processing with DCT.
    Pre-calculates DCT basis matrices for various tensor sizes.
    """

    @torch.no_grad()
    def __init__(self, model, target_chunk: int = 64, norm: str = "ortho"):
        """
        Initialize the ChunkingTransformer.
        
        Args:
            model: The model whose parameters will be processed.
            target_chunk: The target size for tensor chunks.
            norm: The normalization for DCT ('ortho' or None).
        """
        self.target_chunk = target_chunk
        self.shape_dict: dict[int, int] = {}
        self.f_dict: dict[int, torch.Tensor] = {}
        self.b_dict: dict[int, torch.Tensor] = {}

        for _, p in model.named_parameters():
            if not p.requires_grad:
                continue
            for s in p.shape:
                sc = _get_smaller_split(s, self.target_chunk)
                self.shape_dict[s] = sc

                if sc not in self.f_dict:
                    I = torch.eye(sc)
                    self.f_dict[sc] = _dct(I, norm=norm).to(p.dtype).to(p.device)
                    self.b_dict[sc] = _idct(I, norm=norm).to(p.dtype).to(p.device)

    @torch.no_grad()
    def einsum_2d(self, x, b, d=None) -> torch.Tensor:
        """Apply a 2D einsum operation for encoding."""
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            return torch.einsum("...ijkl, kb, ld -> ...ijbd", x, b, d)

    @torch.no_grad()
    def einsum_2d_t(self, x, b, d=None) -> torch.Tensor:
        """Apply a 2D einsum operation for decoding (transpose)."""
        if d is None:
            return torch.einsum("...ij, jb -> ...ib", x, b)
        else:
            return torch.einsum("...ijbd, bk, dl -> ...ijkl", x, b, d)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, *, use_dct: bool = False) -> torch.Tensor:
        """
        Encode a tensor by chunking and optionally applying DCT.
        
        Args:
            x: The input tensor to encode.
            use_dct: Whether to apply the Discrete Cosine Transform.
            
        Returns:
            The encoded tensor.
        """
        if len(x.shape) > 1:  # 2D weights
            n1 = self.shape_dict[x.shape[0]]
            n2 = self.shape_dict[x.shape[1]]
            n1w = self.f_dict[n1].to(x.device)
            n2w = self.f_dict[n2].to(x.device)
            self.f_dict[n1] = n1w
            self.f_dict[n2] = n2w

            x = rearrange(x, "(y h) (x w) -> y x h w", h=n1, w=n2)
            if use_dct:
                x = self.einsum_2d(x, n1w, n2w)

        else:  # 1D weights
            n1 = self.shape_dict[x.shape[0]]
            n1w = self.f_dict[n1].to(x.device)
            self.f_dict[n1] = n1w

            x = rearrange(x, "(x w) -> x w", w=n1)
            if use_dct:
                x = self.einsum_2d(x, n1w)

        return x

    @torch.no_grad()
    def decode(self, x: torch.Tensor, *, use_dct: bool = False) -> torch.Tensor:
        """
        Decode a tensor by un-chunking and optionally applying inverse DCT.
        
        Args:
            x: The input tensor to decode.
            use_dct: Whether to apply the inverse Discrete Cosine Transform.
            
        Returns:
            The decoded tensor.
        """
        if len(x.shape) > 2:  # 2D weights
            if use_dct:
                n1 = x.shape[2]
                n2 = x.shape[3]
                n1w = self.b_dict[n1].to(x.device)
                n2w = self.b_dict[n2].to(x.device)
                self.b_dict[n1] = n1w
                self.b_dict[n2] = n2w

                x = self.einsum_2d_t(x, n1w, n2w)
            x = rearrange(x, "y x h w -> (y h) (x w)")

        else:  # 1D weights
            if use_dct:
                n1 = x.shape[1]
                n1w = self.b_dict[n1].to(x.device)
                self.b_dict[n1] = n1w

                x = self.einsum_2d_t(x, n1w)
            x = rearrange(x, "x w -> (x w)")

        return x


class TopKCompressor:
    """
    Gradient compressor using Top-K selection and optional 8-bit quantization.
    """

    use_quantization: bool
    n_bins: int
    range_in_sigmas: int

    @overload
    def __init__(
        self: "TopKCompressor",
        *,
        use_quantization: Literal[True] = True,
        quantization_bins: int = 256,
        quantization_range: int = 6,
    ) -> None: ...

    @overload
    def __init__(
        self: "TopKCompressor",
        *,
        use_quantization: Literal[False] = False,
        quantization_bins: int = 256,
        quantization_range: int = 6,
    ) -> None: ...

    @torch.no_grad()
    def __init__(
        self,
        *,
        use_quantization: bool = True,
        quantization_bins: int = 256,
        quantization_range: int = 6,
    ) -> None:
        """
        Initialize the TopKCompressor.
        
        Args:
            use_quantization: Whether to use 8-bit quantization.
            quantization_bins: Number of bins for quantization.
            quantization_range: Quantization range in standard deviations.
        """
        self.use_quantization = use_quantization
        if self.use_quantization:
            self.n_bins = quantization_bins
            self.range_in_sigmas = quantization_range

    def _clamp_topk(self, x, topk) -> int:
        """Clamp topk to valid range and ensure it's even."""
        topk = min(topk, x.shape[-1])
        topk = max(topk, 2)
        topk = topk - (topk % 2)  # Ensure even for 12-bit packing
        return int(topk)

    @torch.no_grad()
    def compress(
        self, x: torch.Tensor, topk: int
    ) -> tuple[IdxT, ValT, ShapeT, TotK] | tuple[IdxT, ValT, ShapeT, TotK, QuantParamsT]:
        """
        Compress a tensor using top-k selection and optional quantization.
        
        Args:
            x: The input tensor to compress.
            topk: The number of top values to select.
            
        Returns:
            Tuple of (indices, values, original_shape, total_k, [quant_params])
        """
        xshape = x.shape

        if len(x.shape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        totalk = x.shape[-1]
        topk = self._clamp_topk(x, topk)

        idx_int64 = torch.topk(
            x.abs(), k=topk, dim=-1, largest=True, sorted=False
        ).indices
        val = torch.gather(x, dim=-1, index=idx_int64)

        # Pack indices into 12-bit representation
        idx = pack_12bit_indices(idx_int64)

        if self.use_quantization:
            val, quant_params = self._quantize_values(val)
            return idx, val, xshape, totalk, quant_params

        return idx, val, xshape, totalk

    @torch.no_grad()
    def decompress(
        self,
        p: torch.Tensor,
        idx: torch.Tensor,
        val: torch.Tensor,
        xshape: ShapeT,
        totalk: int,
        quantize_params: QuantParamsT | None = None,
    ) -> torch.Tensor:
        """
        Decompress a tensor from its sparse representation.
        
        Args:
            p: A tensor with the target shape and device.
            idx: The indices of the non-zero values.
            val: The non-zero values.
            xshape: The original shape of the tensor.
            totalk: Total number of elements in original tensor's last dim.
            quantize_params: Quantization parameters.
            
        Returns:
            The decompressed tensor.
        """
        if self.use_quantization and quantize_params is not None:
            val = self._dequantize_values(val, quantize_params)

        x = torch.zeros(xshape, device=p.device, dtype=p.dtype)

        if len(xshape) > 2:  # 2D weights
            x = rearrange(x, "y x h w -> y x (h w)")

        # Unpack 12-bit indices
        if idx.dtype == torch.uint8:
            idx_int64 = unpack_12bit_indices(idx, val.shape)
        elif idx.dtype in (torch.int64, torch.long):
            idx_int64 = idx
        else:
            raise ValueError(f"Expected uint8 or int64 indices, got {idx.dtype}")

        if val.dtype != x.dtype:
            val = val.to(dtype=x.dtype)

        x.scatter_reduce_(
            dim=-1, index=idx_int64, src=val, reduce="mean", include_self=False
        ).reshape(xshape)

        if len(x.shape) > 2:  # 2D weights
            xshape4 = cast(Shape4D, xshape)
            h_dim = xshape4[2]
            x = rearrange(x, "y x (h w) -> y x h w", h=h_dim)

        return x

    @torch.no_grad()
    def batch_decompress(
        self,
        p: torch.Tensor,
        idx: torch.Tensor | Sequence[torch.Tensor],
        val: torch.Tensor | Sequence[torch.Tensor],
        xshape: ShapeT,
        totalk: int,
        quantize_params: Sequence[QuantParamsT] | None = None,
        *,
        normalise: bool = False,
        clip_norm: bool = True,
    ) -> torch.Tensor:
        """
        Decompress and aggregate a batch of sparse tensors.
        
        Args:
            p: A tensor with the target shape and device.
            idx: Sequence of indices for each tensor in the batch.
            val: Sequence of values for each tensor in the batch.
            xshape: The original shape of the tensors.
            totalk: Total number of elements in original tensor's last dim.
            quantize_params: Sequence of quantization parameters.
            normalise: Whether to normalise the values.
            clip_norm: Whether to clip the norms.
            
        Returns:
            The combined, decompressed tensor.
        """
        if quantize_params is not None and not isinstance(quantize_params, list):
            quantize_params = [quantize_params] * len(val)

        processed_vals: list[torch.Tensor] = []
        dequant_vals = None
        norms = None
        clip_norm_val = None

        if self.use_quantization and quantize_params:
            dequant_vals = [
                self._dequantize_values(v, quantize_params[i])
                for i, v in enumerate(val)
            ]

        if clip_norm:
            vals_for_norm = dequant_vals if dequant_vals is not None else val
            norms = torch.stack(
                [torch.norm(sparse_vals, p=2) for sparse_vals in vals_for_norm]
            )
            clip_norm_val = torch.median(norms)

        vals = dequant_vals if dequant_vals is not None else val
        for i, v in enumerate(vals):
            v = v.to(p.device)

            if normalise:
                eps = 1e-8
                if len(v.shape) == 3:
                    l2_norm = torch.norm(v, p=2, dim=2, keepdim=True)
                    v = v / (l2_norm + eps)
                elif len(v.shape) == 2:
                    l2_norm = torch.norm(v, p=2, dim=1, keepdim=True)
                    v = v / (l2_norm + eps)
                elif len(v.shape) == 1:
                    l2_norm = torch.norm(v, p=2)
                    if l2_norm > eps:
                        v = v / l2_norm
            elif clip_norm and norms is not None and clip_norm_val is not None:
                current_norm = norms[i]
                clip_factor = torch.clamp(clip_norm_val / (current_norm + 1e-8), max=1)
                v = v * clip_factor
            processed_vals.append(v)

        # Unpack and concatenate indices
        unpacked_indices = []
        val_list = val if isinstance(val, Sequence) else [val]
        idx_list = idx if isinstance(idx, Sequence) else [idx]

        for i, i_data in enumerate(idx_list):
            if i_data.dtype != torch.uint8:
                raise ValueError(f"Expected uint8 packed indices, got {i_data.dtype}")
            v_data = val_list[i]
            idx_unpacked = unpack_12bit_indices(i_data.to(p.device), v_data.shape)
            unpacked_indices.append(idx_unpacked)

        idx_concat = torch.cat(unpacked_indices, dim=-1)
        val_concat = torch.cat(processed_vals, dim=-1).to(p.dtype)

        return self.decompress(
            p, idx_concat, val_concat, xshape, totalk, quantize_params=None
        )

    @torch.no_grad()
    def _quantize_values(self, val: torch.Tensor) -> tuple[torch.Tensor, QuantParamsT]:
        """Quantize tensor values to 8-bit integers."""
        offset = self.n_bins // 2
        shift = val.mean()
        centered = val - shift

        std = centered.norm() / math.sqrt(centered.numel() - 1)
        scale = self.range_in_sigmas * std / self.n_bins
        if scale == 0 or torch.isnan(scale) or torch.isinf(scale):
            scale = torch.tensor(1.0, dtype=centered.dtype, device=val.device)

        centered_fp32 = centered.to(torch.float32)
        qval = (
            (centered_fp32 / scale + offset)
            .round()
            .clamp(0, self.n_bins - 1)
            .to(torch.uint8)
        )

        device = qval.device
        sums = torch.zeros(self.n_bins, dtype=torch.float32, device=device)
        counts = torch.zeros(self.n_bins, dtype=torch.float32, device=device)

        sums.scatter_add_(0, qval.flatten().long(), centered_fp32.flatten())
        counts.scatter_add_(
            0, qval.flatten().long(), torch.ones_like(centered_fp32.flatten())
        )

        lookup = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))
        qparams: QuantParamsT = (shift, float(scale), offset, lookup, val.dtype)
        return qval, qparams

    @torch.no_grad()
    def _dequantize_values(
        self, val: torch.Tensor, qparams: QuantParamsT
    ) -> torch.Tensor:
        """Dequantize tensor values from 8-bit integers."""
        if val.dtype == torch.uint8:
            shift, _, _, lookup, orig_dtype = qparams
            lookup = (
                lookup.to(val.device) if isinstance(lookup, torch.Tensor) else lookup
            )
            deq = lookup[val.long()] + shift
            val = deq.to(orig_dtype)
        return val


def compress_gradients(
    model: torch.nn.Module,
    transformer: ChunkingTransformer,
    compressor: TopKCompressor,
    topk: int,
    use_dct: bool = True,
) -> dict:
    """
    Compress all gradients from a model.
    
    Args:
        model: The model with computed gradients.
        transformer: ChunkingTransformer instance.
        compressor: TopKCompressor instance.
        topk: Number of top-k values to keep.
        use_dct: Whether to use DCT transformation.
        
    Returns:
        Dictionary with compressed gradients.
    """
    compressed = {}
    
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
            
        grad = param.grad.detach().clone()
        
        # Encode with DCT
        encoded = transformer.encode(grad, use_dct=use_dct)
        
        # Compress
        result = compressor.compress(encoded, topk)
        
        if compressor.use_quantization:
            idx, val, xshape, totalk, qparams = result
            compressed[name + "idxs"] = idx.cpu()
            compressed[name + "vals"] = val.cpu()
            compressed[name + "xshape"] = xshape
            compressed[name + "totalk"] = totalk
            compressed[name + "quant_params"] = (
                qparams[0].cpu(),  # shift
                qparams[1],        # scale
                qparams[2],        # offset
                qparams[3].cpu(),  # lookup
                qparams[4],        # dtype
            )
        else:
            idx, val, xshape, totalk = result
            compressed[name + "idxs"] = idx.cpu()
            compressed[name + "vals"] = val.cpu()
            compressed[name + "xshape"] = xshape
            compressed[name + "totalk"] = totalk
            
    return compressed


def decompress_and_aggregate_gradients(
    model: torch.nn.Module,
    compressed_grads_list: list[dict],
    transformer: ChunkingTransformer,
    compressor: TopKCompressor,
    use_dct: bool = True,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Decompress and aggregate gradients from multiple workers.
    
    Args:
        model: The model to get parameter shapes from.
        compressed_grads_list: List of compressed gradient dicts from workers.
        transformer: ChunkingTransformer instance.
        compressor: TopKCompressor instance.
        use_dct: Whether DCT was used in compression.
        device: Device to perform decompression on.
        
    Returns:
        Dictionary mapping parameter names to aggregated gradients.
    """
    aggregated = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        idx_key = name + "idxs"
        val_key = name + "vals"
        
        # Collect compressed data from all workers
        all_idx = []
        all_val = []
        all_qparams = []
        xshape = None
        totalk = None
        
        for compressed in compressed_grads_list:
            if idx_key not in compressed:
                continue
                
            all_idx.append(compressed[idx_key].to(device))
            all_val.append(compressed[val_key].to(device))
            
            if name + "xshape" in compressed:
                xshape = compressed[name + "xshape"]
            if name + "totalk" in compressed:
                totalk = compressed[name + "totalk"]
                
            if name + "quant_params" in compressed:
                qp = compressed[name + "quant_params"]
                all_qparams.append((
                    qp[0].to(device) if isinstance(qp[0], torch.Tensor) else qp[0],
                    qp[1],
                    qp[2],
                    qp[3].to(device) if isinstance(qp[3], torch.Tensor) else qp[3],
                    qp[4],
                ))
                
        if not all_idx or xshape is None or totalk is None:
            continue
            
        # Batch decompress
        dummy_p = torch.zeros(xshape, device=device, dtype=param.dtype)
        
        if all_qparams:
            decompressed = compressor.batch_decompress(
                dummy_p, all_idx, all_val, xshape, totalk,
                quantize_params=all_qparams,
                clip_norm=True,
            )
        else:
            decompressed = compressor.batch_decompress(
                dummy_p, all_idx, all_val, xshape, totalk,
                clip_norm=True,
            )
            
        # Decode with inverse DCT
        gradient = transformer.decode(decompressed, use_dct=use_dct)
        
        # Average across workers
        gradient = gradient / len(compressed_grads_list)
        
        aggregated[name] = gradient
        
    return aggregated


def compute_gradient_fingerprint(gradients: dict[str, torch.Tensor]) -> dict:
    """
    Compute statistics about the gradients for logging.
    
    Args:
        gradients: Dictionary of gradient tensors.
        
    Returns:
        Dictionary with gradient statistics.
    """
    total_norm_sq = 0.0
    total_elements = 0
    param_norms = {}
    param_means = {}
    
    for name, grad in gradients.items():
        norm = grad.norm().item()
        param_norms[name] = norm
        param_means[name] = grad.mean().item()
        total_norm_sq += norm ** 2
        total_elements += grad.numel()
        
    return {
        "global_l2_norm": math.sqrt(total_norm_sq),
        "param_norms": param_norms,
        "param_means": param_means,
        "total_elements": total_elements,
    }





