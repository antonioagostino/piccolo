from dataclasses import dataclass
from pathlib import Path
from typing import Union, cast
import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint
import yaml


@dataclass(frozen=True)
class ModelConfig:
    """
    Hyperparameter configuration for the language model architecture.

    Attributes:
        vocab_size (int): Number of tokens in the vocabulary.
        sequence_length (int): Maximum context length in tokens.
        embedding_dim (int): Residual stream / embedding dimension D.
        n_decoder_blocks (int): Number of stacked transformer decoder blocks.
        n_heads (int): Number of query attention heads.
        n_kv_heads (int): Number of key/value heads (GQA; must divide n_heads).
        dropout_rate (float): Dropout probability applied to attention and FFN
            outputs.
        ffn_hidden_dim (int | None): Hidden dimension of the SwiGLU FFN. If
            None, defaults to 4 x embedding_dim.
    """

    vocab_size: int
    sequence_length: int
    embedding_dim: int
    n_decoder_blocks: int
    n_heads: int
    n_kv_heads: int
    dropout_rate: float = 0.0
    ffn_hidden_dim: int | None = None

    def __post_init__(self) -> None:
        """
        Validate that all field values are internally consistent.

        Raises:
            ValueError: If any field violates its constraint (e.g. non-positive
                dimensions, incompatible head counts, out-of-range dropout).
        """
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be greater than 0")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be greater than 0")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")
        if self.n_decoder_blocks <= 0:
            raise ValueError("n_decoder_blocks must be greater than 0")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be greater than 0")
        if self.n_kv_heads <= 0:
            raise ValueError("n_kv_heads must be greater than 0")
        if self.embedding_dim % self.n_heads != 0:
            raise ValueError("embedding_dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be between 0 and 1")
        if self.ffn_hidden_dim is not None and self.ffn_hidden_dim <= 0:
            raise ValueError("ffn_hidden_dim must be greater than 0")

    @property
    def resolved_ffn_hidden_dim(self) -> int:
        """
        FFN hidden dimension, falling back to 4 x embedding_dim when unset.

        Returns:
            int: The effective hidden dimension of the SwiGLU feed-forward
                block.
        """
        return self.ffn_hidden_dim if self.ffn_hidden_dim is not None else 4 * self.embedding_dim

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "ModelConfig":
        """
        Load a ModelConfig from a YAML file.

        The YAML file may contain either a top-level ``model`` key whose value
        is the config mapping, or the mapping at the root level.

        Args:
            config_path (str | Path): Path to the YAML configuration file.

        Returns:
            ModelConfig: The parsed model configuration.

        Raises:
            ValueError: If the file does not contain a valid YAML mapping or
                the ``model`` section is missing or malformed.
        """
        with Path(config_path).open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)

        if not isinstance(raw_config, dict):
            raise ValueError("The model configuration file must contain a YAML mapping")

        model_section = raw_config.get("model", raw_config)
        if not isinstance(model_section, dict):
            raise ValueError("The 'model' section must contain a YAML mapping")

        return cls(**model_section)

def get_supported_weights_precision(device: torch.device) -> torch.dtype:
    """
    Return the highest-precision dtype supported for AMP on the given device.

    Args:
        device (torch.device): The target compute device.

    Returns:
        torch.dtype: bfloat16 on CUDA if supported, float16 on other CUDA
            devices, float32 on CPU.
    """
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def chunked_cross_entropy_loss(
    hidden_states: torch.Tensor,
    embedding_weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """
    Cross-entropy loss without materializing the full (B*T, vocab_size) logits tensor.

    Splits the flattened token sequence into chunks of `chunk_size`, computes
    logits and cross-entropy for each chunk independently, then returns the
    mean loss over all tokens.

    Peak extra VRAM is O(chunk_size x vocab_size) instead of O(B x T x vocab_size).
    PyTorch's autograd processes the summed chunk losses in reverse order during
    backward, so at most one chunk's saved tensors are live simultaneously.
    """
    B, T, D = hidden_states.shape
    hidden_flat  = hidden_states.view(B * T, D)
    targets_flat = targets.contiguous().view(B * T)

    total_loss = hidden_states.new_zeros(())
    for start in range(0, B * T, chunk_size):
        end          = min(start + chunk_size, B * T)
        chunk_logits = F.linear(hidden_flat[start:end], embedding_weight)  # (chunk, V)
        chunk_loss   = F.cross_entropy(chunk_logits, targets_flat[start:end], reduction="sum")
        total_loss   = total_loss + chunk_loss
        del chunk_logits

    return total_loss / (B * T)


def language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss over a batch of next-token predictions.

    Args:
        logits (torch.Tensor): Raw model output of shape (B, T, vocab_size).
        targets (torch.Tensor): Ground-truth token IDs of shape (B, T).

    Returns:
        torch.Tensor: Scalar mean cross-entropy loss.
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1)
    )


def forward_backward_micro_step(
    language_model: nn.Module,
    scaler: torch.amp.GradScaler,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    loss_scale: float = 1.0,
) -> float:
    """
    Run one forward+backward pass for a single micro-batch.

    Multiplies the loss by loss_scale before backpropagation; set
    loss_scale = 1 / gradient_accumulation_steps so gradients average
    correctly across an accumulation window. Does not zero gradients or
    step the optimizer.

    Args:
        language_model (nn.Module): The model being trained.
        scaler (torch.amp.GradScaler): Gradient scaler for mixed-precision.
        inputs (torch.Tensor): Input token IDs of shape (B, T).
        targets (torch.Tensor): Target token IDs of shape (B, T).
        device (torch.device): Device that tensors are moved to.
        amp_dtype (torch.dtype): dtype used inside the autocast region.
        use_amp (bool): Whether to enable automatic mixed precision.
        loss_scale (float): Multiplier applied to the loss before backward.

    Returns:
        float: The raw (unscaled) scalar loss for this micro-batch.
    """
    language_model.train()
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        loss = language_model(inputs, targets=targets)

    # Scaling is only needed for CUDA float16. With bfloat16 or CPU,
    # GradScaler is disabled and these calls are no-ops/pass-throughs.
    scaler.scale(loss * loss_scale).backward()
    return float(loss.detach().item())


def optimizer_step(
    language_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    max_grad_norm: float | None = 1.0,
) -> None:
    """
    Unscale accumulated gradients, clip them, step the optimizer, and update the scaler.

    Args:
        language_model (nn.Module): The model whose parameters are updated.
        optimizer (torch.optim.Optimizer): The optimizer.
        scaler (torch.amp.GradScaler): Gradient scaler; updated after the step.
        max_grad_norm (float | None): Maximum L2 gradient norm for clipping.
            Pass None to disable clipping.
    """
    if max_grad_norm is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(language_model.parameters(), max_norm=max_grad_norm)
    scaler.step(optimizer)
    scaler.update()


def training_step(
    language_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    max_grad_norm: float | None = 1.0,
) -> float:
    """
    Run one forward-backward-optimizer step with AMP support.

    Convenience wrapper for a single accumulation step. For gradient
    accumulation, call forward_backward_micro_step and optimizer_step
    directly from the training loop.

    Args:
        language_model (nn.Module): The model being trained.
        optimizer (torch.optim.Optimizer): Optimizer holding the parameter groups.
        scaler (torch.amp.GradScaler): Gradient scaler for mixed-precision.
        inputs (torch.Tensor): Input token IDs of shape (B, T).
        targets (torch.Tensor): Target token IDs of shape (B, T).
        device (torch.device): Device that tensors are moved to.
        amp_dtype (torch.dtype): dtype used inside the autocast region.
        use_amp (bool): Whether to enable automatic mixed precision.
        max_grad_norm (float | None): Maximum gradient norm for clipping.
            Pass None to disable clipping.

    Returns:
        float: The scalar training loss for this step.
    """
    optimizer.zero_grad(set_to_none=True)
    loss = forward_backward_micro_step(
        language_model, scaler, inputs, targets, device, amp_dtype, use_amp
    )
    optimizer_step(language_model, optimizer, scaler, max_grad_norm)
    return loss


def compile_language_model(language_model: nn.Module, enabled: bool) -> nn.Module:
    """
    Optionally compile the model with torch.compile for faster execution.

    Args:
        language_model (nn.Module): The model to (optionally) compile.
        enabled (bool): If True, applies torch.compile; otherwise returns
            the model unchanged.

    Returns:
        nn.Module: The compiled or original model.
    """
    if enabled:
        return cast(nn.Module, torch.compile(language_model))
    return language_model

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation (no mean-subtraction)."""

    def __init__(self,
                 feature_size: int,
                 device: torch.device,
                 eps: float = 1e-8):
        """
        Initialise RMSNorm with a learnable gain vector.

        Args:
            feature_size (int): Number of features to normalise (last
                dimension).
            device (torch.device): Device on which the gain parameter is
                created.
            eps (float): Small constant added to the RMS for numerical
                stability.
        """
        super().__init__()
        self.feature_size = feature_size
        self.device = device
        self.eps = eps
        self.gain = nn.Parameter(torch.ones((feature_size,),
                                            device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalisation followed by element-wise gain scaling.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape
                (..., feature_size).

        Returns:
            torch.Tensor: Normalised and scaled tensor, same shape as input.
        """
        rec_rms = torch.rsqrt(torch.mean(x.square(), -1, keepdim=True) + self.eps)
        normalized = x * rec_rms * self.gain

        return normalized

class GroupedQueryAttention(nn.Module):
    """
    Multi-head causal self-attention with Grouped Query Attention (GQA).

    Query heads are split into groups that share a single key/value head,
    reducing KV-cache memory without significantly degrading quality.
    Rotary Position Embeddings (RoPE) are applied to queries and keys.
    """

    def __init__(self,
                 layer_idx: int,
                 embedding_dim: int,
                 sequence_length: int,
                 n_heads: int,
                 n_kv_heads: int,
                 head_size: int,
                 device: torch.device):
        """
        Initialise a single GQA attention layer.

        Args:
            layer_idx (int): Index of this layer within the decoder stack,
                used as the KV-cache key.
            embedding_dim (int): Residual stream dimension D.
            sequence_length (int): Maximum context length.
            n_heads (int): Number of query heads.
            n_kv_heads (int): Number of key/value heads (must divide n_heads).
            head_size (int): Dimension per attention head (D // n_heads).
            device (torch.device): Device for parameter initialisation.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.device = device
        self.head_size = head_size
        self.qkv_proj = nn.Linear(embedding_dim,
                                  embedding_dim + 2 * (self.head_size * n_kv_heads),
                                  bias=False)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def apply_rope(self,
                   x: torch.Tensor,
                   rope_cos: torch.Tensor,
                   rope_sin: torch.Tensor,
                   offset: int) -> torch.Tensor:
        """
        Apply Rotary Position Embeddings to a query or key tensor.

        Args:
            x (torch.Tensor): Tensor of shape (B, G, H//G, T, head_size) to
                rotate.
            rope_cos (torch.Tensor): Cosine component of shape
                (1, 1, 1, seq_len, head_size).
            rope_sin (torch.Tensor): Sine component of shape
                (1, 1, 1, seq_len, head_size).
            offset (int): Position offset for cached keys during inference.

        Returns:
            torch.Tensor: Rotated tensor, same shape as x.
        """
        rope_slice = slice(offset, offset + x.shape[-2])
        inv_x2 = x[..., ::2]     # (B, G, H // G, T or 1, HS)
        inv_x1 = -x[..., 1::2]
        x_rot = torch.stack((inv_x1, inv_x2), dim=-1).flatten(-2)

        return (x * rope_cos[:, :, :, rope_slice, :]) + (x_rot * rope_sin[:, :, :, rope_slice, :])

    def forward(self,
                embeddings: torch.Tensor,
                is_causal: bool,
                kv_cache: Union[dict[int, tuple[torch.Tensor, torch.Tensor]], None],
                rope_cos: torch.Tensor,
                rope_sin: torch.Tensor) -> torch.Tensor:
        """
        Compute grouped-query causal self-attention for a batch of sequences.

        Args:
            embeddings (torch.Tensor): Input residual of shape (B, T, D).
            is_causal (bool): Whether to apply a causal mask inside
                scaled_dot_product_attention. True during training and prefill;
                False for single-token autoregressive generation with KV-cache.
            kv_cache (dict[int, tuple[torch.Tensor, torch.Tensor]] | None):
                Optional KV-cache mapping layer index to (K, V) tensors.
                Updated in-place during inference.
            rope_cos (torch.Tensor): Cosine RoPE component.
            rope_sin (torch.Tensor): Sine RoPE component.

        Returns:
            torch.Tensor: Attention output of shape (B, T, D).
        """
        B, T, _ = embeddings.shape
        H = self.n_heads
        G = self.n_kv_heads
        D = self.embedding_dim
        q, k, v = self.qkv_proj(embeddings).split([D, self.head_size * G, self.head_size * G], dim=-1)
        q = q.view(B, T, G, H // G, self.head_size).permute(0, 2, 3, 1, 4)
        k = k.view(B, T, G, self.head_size).permute(0, 2, 1, 3).unsqueeze(2)
        v = v.view(B, T, G, self.head_size).permute(0, 2, 1, 3).unsqueeze(2)

        # Apply RoPE on query and key projections taking into account
        # querys and keys already rotated in the KV Cache
        offset = 0
        if not self.training and kv_cache is not None and self.layer_idx in kv_cache.keys():
            offset = kv_cache[self.layer_idx][0].shape[-2]

        q = self.apply_rope(q, rope_cos, rope_sin, offset)
        k = self.apply_rope(k, rope_cos, rope_sin, offset)

        # Collapse the GQA group dimension so SDPA sees standard 4-D tensors
        # [B, G, H//G, T, HS] → [B, H, T, HS]  and  [B, G, 1, T, HS] → [B, G, T, HS].
        # This is required for PyTorch to dispatch to FlashAttention; the 5-D
        # layout forces the slow math kernel and materialises the full O(T²) matrix.
        q = q.reshape(B, H, T, self.head_size)
        k = k.squeeze(2)   # [B, G, T, HS]
        v = v.squeeze(2)   # [B, G, T, HS]

        if not self.training and kv_cache is not None:
            if self.layer_idx in kv_cache.keys():
                past_k, past_v = kv_cache[self.layer_idx]
                k = torch.cat([past_k, k], dim=2)[:, :, -self.sequence_length:, :]
                v = torch.cat([past_v, v], dim=2)[:, :, -self.sequence_length:, :]

            kv_cache[self.layer_idx] = (k, v)

        outputs = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, enable_gqa=True)
        # [B, H, T, HS] → [B, T, H, HS] → [B, T, D]
        outputs = outputs.transpose(1, 2).contiguous().view(B, T, self.embedding_dim)
        y = self.output_proj(outputs)

        return y
    
class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward block: FFN with a gated activation.

    Uses a single fused linear projection for the gate and up-projection to
    halve memory-access overhead, then applies SiLU gating before the
    down-projection.
    """

    def __init__(self,
                 input_embedding_dim: int,
                 hidden_embedding_dim: int):
        """
        Initialise SwiGLU with fused gate/up projection and down projection.

        Args:
            input_embedding_dim (int): Input (and output) dimension D.
            hidden_embedding_dim (int): Hidden dimension H of the FFN.
        """
        super().__init__()
        self.linear_12 = nn.Linear(input_embedding_dim,
                             2 * hidden_embedding_dim, bias=False)
        self.linear_out = nn.Linear(hidden_embedding_dim,
                                    input_embedding_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation.

        Args:
            x (torch.Tensor): Input of shape (B, T, D).

        Returns:
            torch.Tensor: Output of shape (B, T, D).
        """
        # x -> (B, T, D)
        # A more efficient way of implementing the SwiGLU:
        # instead of accessing twice the memory we just have
        # a linear transformation (twice of the destination size)
        # and then we chunk it (chunk in Torch is a view of the
        # original tensors).
        x12 = self.linear_12(x) # (B, T, 2 * hidden_dim)
        x1, x2 = x12.chunk(2, dim=-1) # 2 * (B, T, hidden_dim)
        out = self.linear_out(F.silu(x1) * x2) # (B, T, D)

        return out

    
class TransformerDecoder(nn.Module):
    """
    Stacked causal transformer decoder with GQA, SwiGLU FFN, and RMSNorm.

    Manages the causal mask, RoPE sin/cos buffers, and the stacked attention
    and FFN blocks. Supports a KV-cache for efficient autoregressive decoding.
    """

    rope_freqs: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(self,
                 n_blocks: int,
                 sequence_length: int,
                 embedding_dim: int,
                 n_heads: int,
                 n_kv_heads: int,
                 ffn_hidden_dim: int | None,
                 kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
                 dropout_rate: float,
                 device: torch.device,
                 gradient_checkpointing: bool = False):
        """
        Initialise the transformer decoder stack.

        Args:
            n_blocks (int): Number of decoder blocks.
            sequence_length (int): Maximum context length in tokens.
            embedding_dim (int): Residual stream dimension D.
            n_heads (int): Number of query attention heads.
            n_kv_heads (int): Number of key/value heads.
            ffn_hidden_dim (int | None): SwiGLU hidden dimension; defaults to
                4 x embedding_dim when None.
            kv_cache (dict[int, tuple[torch.Tensor, torch.Tensor]]): Shared
                KV-cache dict updated in-place during inference. Pass an empty
                dict for training.
            dropout_rate (float): Dropout probability applied after attention
                and FFN outputs.
            device (torch.device): Device for all buffers and parameters.
            gradient_checkpointing (bool): If True, trades compute for memory
                by recomputing block activations during the backward pass.
        """
        super().__init__()
        self.kv_cache = kv_cache
        self.gradient_checkpointing = gradient_checkpointing
        self.dropout_rate = dropout_rate
        self.n_blocks = n_blocks
        self.device = device
        self.sequence_length = sequence_length
        self.head_size = embedding_dim // n_heads
        self.ffn_hidden_dim = ffn_hidden_dim if ffn_hidden_dim is not None else 4 * embedding_dim
        self.global_token_counter = 0
        # TODO: Add scaling RoPE for supporting long contexts
        rope_cos, rope_sin = self.build_rope_sin_cos(dtype=get_supported_weights_precision(device),
                                                     use_gqa=True)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)
        self.attentions = nn.ModuleList(
            [GroupedQueryAttention(
                layer_idx=i,
                embedding_dim=embedding_dim,
                sequence_length=sequence_length,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_size=self.head_size,
                device=device
            ) for i in range(n_blocks)]
        )

        self.norms_1 = nn.ModuleList(
            [RMSNorm(embedding_dim,
                     device=device) 
            for _ in range(n_blocks)]
        )

        self.norms_2 = nn.ModuleList(
            [RMSNorm(embedding_dim,
                     device=device) 
            for _ in range(n_blocks)]
        )

        self.ffns = nn.ModuleList(
            [SwiGLU(embedding_dim,
                    self.ffn_hidden_dim) 
            for _ in range(n_blocks)]
        )

    def build_rope_sin_cos(self,
                           dtype: torch.dtype,
                           use_gqa: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pre-compute RoPE cosine and sine tables for all positions.

        Args:
            dtype (torch.dtype): Floating-point dtype for the tables.
            use_gqa (bool): If True, adds GQA-specific batch dimensions
                (B, G, H, T, HS); otherwise uses (B, H, T, HS).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(cos, sin)`` tables with
                shape determined by ``use_gqa``.
        """
        assert self.head_size % 2 == 0

        # i goes from 0 to head_dim - 2
        freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_size, 2, device=self.device, dtype=dtype) / self.head_size)
        )

        self.register_buffer("rope_freqs", freq)
        positions = torch.arange(self.sequence_length, device=self.device, dtype=dtype)
        # (T, HS / 2)
        freqs = torch.outer(positions, self.rope_freqs)

        # (T, [<m*theta_0, m*theta_0>, ..., <m*theta_HS, m*theta_HS>])
        # (T, HS)
        emb = torch.cat([freqs, freqs], dim=-1)

        if use_gqa:
            cos = emb.cos()[None, None, None, :, :]  # (B, G, H, T, HS)
            sin = emb.sin()[None, None, None, :, :]
        else:
            cos = emb.cos()[None, None, :, :]  # (B, H, T, HS)
            sin = emb.sin()[None, None, :, :]

        return cos, sin

    def extend_rope_sin_cos(self,
                            extend_token_pos: int,
                            use_gqa: bool = True) -> None:
        """
        Append one new position to the RoPE tables for autoregressive decoding.

        Called during inference when the generated sequence exceeds the
        pre-computed context length.

        Args:
            extend_token_pos (int): Absolute position index of the new token.
            use_gqa (bool): Must match the value used in build_rope_sin_cos.
        """
        # During inference, if we are generating one token at time, in an
        # autoregressive mode, we need to compute new RoPE sin and cos.
        # (HS / 2)
        freqs = extend_token_pos * self.rope_freqs

        # (1, [<m*theta_0, m*theta_0>, ..., <m*theta_HS, m*theta_HS>])
        # (1, HS)
        emb = torch.cat([freqs, freqs], dim=-1).unsqueeze(dim=0)

        if use_gqa:
            cos = emb.cos()[None, None, None, :, :]  # (B, G, H, 1, HS)
            sin = emb.sin()[None, None, None, :, :]
        else:
            cos = emb.cos()[None, None, :, :]  # (B, H, 1, HS)
            sin = emb.sin()[None, None, :, :]

        self.rope_cos = torch.cat([self.rope_cos, cos], dim=-2)  # (B, G, H, T + 1, HS)
        self.rope_sin = torch.cat([self.rope_sin, sin], dim=-2)  # (B, G, H, T + 1, HS)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run all decoder blocks on an embedded sequence.

        Args:
            x (torch.Tensor): Embedded token sequence of shape (B, T, D).

        Returns:
            torch.Tensor: Output sequence of shape (B, T, D).
        """
        # x -> (B, T, D)
        _, T, _ = x.shape
        # Single-token generation with KV-cache is already causal by construction.
        is_causal = T > 1

        if not self.training:
            self.global_token_counter += T
            if self.global_token_counter > self.sequence_length:
                self.extend_rope_sin_cos(self.global_token_counter - 1, use_gqa=True)

        def run_block(x: torch.Tensor, block_idx: int) -> torch.Tensor:
            attn_out = self.attentions[block_idx](
                self.norms_1[block_idx](x),
                is_causal,
                self.kv_cache,
                self.rope_cos,
                self.rope_sin,
            )
            x = x + F.dropout(attn_out, self.dropout_rate, training=self.training)
            ffn_out = self.ffns[block_idx](self.norms_2[block_idx](x))
            return x + F.dropout(ffn_out, self.dropout_rate, training=self.training)

        for i in range(self.n_blocks):
            if self.gradient_checkpointing and self.training:
                x = _checkpoint(run_block, x, i, use_reentrant=False)
            else:
                x = run_block(x, i)

        return x  # (B, T, D)
    
class LanguageModel(nn.Module):
    """
    Decoder-only causal language model with tied input/output embeddings.
    """

    def __init__(self,
                 n_decoder_blocks: int,
                 sequence_length: int,
                 vocab_size: int,
                 embedding_dim: int,
                 n_heads: int,
                 n_kv_heads: int,
                 ffn_hidden_dim: int | None,
                 kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
                 dropout_rate: float,
                 device: torch.device,
                 gradient_checkpointing: bool = False):
        """
        Initialise the language model.

        Args:
            n_decoder_blocks (int): Number of decoder blocks.
            sequence_length (int): Maximum context length.
            vocab_size (int): Vocabulary size.
            embedding_dim (int): Residual stream dimension D.
            n_heads (int): Number of query attention heads.
            n_kv_heads (int): Number of key/value heads.
            ffn_hidden_dim (int | None): SwiGLU hidden dimension.
            kv_cache (dict[int, tuple[torch.Tensor, torch.Tensor]]): KV-cache
                dict shared with the decoder.
            dropout_rate (float): Dropout probability.
            device (torch.device): Device for all parameters and buffers.
            gradient_checkpointing (bool): If True, enables gradient
                checkpointing on the decoder blocks.
        """
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.embedding_matrix = nn.Embedding(vocab_size, embedding_dim)
        # N(0, 1/sqrt(D)) keeps logit variance ≈ 1 after the weight-tied LM
        # head, giving cross-entropy ≈ ln(vocab_size) at random init.
        nn.init.normal_(self.embedding_matrix.weight, mean=0.0,
                        std=1.0 / math.sqrt(embedding_dim))
        self.transformer_decoder = TransformerDecoder(
            n_blocks=n_decoder_blocks,
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            kv_cache=kv_cache,
            dropout_rate=dropout_rate,
            device=device,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.final_norm = RMSNorm(
            feature_size=embedding_dim,
            device=device
        )
        
    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute next-token logits or training loss for a batch of token sequences.

        Args:
            x (torch.Tensor): Integer token IDs of shape (B, T).
            targets (torch.Tensor | None): Target token IDs of shape (B, T).
                When provided, returns a scalar cross-entropy loss computed via
                chunked_cross_entropy_loss (no full (B,T,V) logits materialized).
                When None, returns logits of shape (B, T, vocab_size) for inference.

        Returns:
            torch.Tensor: Scalar loss when targets is given; logits (B, T, vocab_size)
                otherwise.
        """
        tokens = x                                 # (B, T)
        embeddings = self.embedding_matrix(tokens) # (B, T, D)
        out = self.transformer_decoder(embeddings) # (B, T, D)
        normed = self.final_norm(out)              # (B, T, D)

        if targets is not None:
            return chunked_cross_entropy_loss(normed, self.embedding_matrix.weight, targets)

        # Remember that torch.nn.functional.linear transposes the weight matrix
        # passed before applying the affine linear transformation, so we don't
        # need to compute the transpose of the embedding matrix to get back to
        # vocab space.
        return F.linear(normed, self.embedding_matrix.weight)  # (B, T, vocab_size)

    @classmethod
    def from_config(cls,
                    config: ModelConfig,
                    kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
                    device: torch.device,
                    gradient_checkpointing: bool = False) -> "LanguageModel":
        """
        Construct a LanguageModel from a ModelConfig.

        Args:
            config (ModelConfig): Model architecture configuration.
            kv_cache (dict[int, tuple[torch.Tensor, torch.Tensor]]): KV-cache
                dict to pass through to the decoder.
            device (torch.device): Target device.
            gradient_checkpointing (bool): If True, enables gradient
                checkpointing on the decoder blocks.

        Returns:
            LanguageModel: Fully initialised language model.
        """
        return cls(
            n_decoder_blocks=config.n_decoder_blocks,
            sequence_length=config.sequence_length,
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            ffn_hidden_dim=config.resolved_ffn_hidden_dim,
            kv_cache=kv_cache,
            dropout_rate=config.dropout_rate,
            device=device,
            gradient_checkpointing=gradient_checkpointing,
        )

