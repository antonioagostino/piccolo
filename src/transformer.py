from pathlib import Path
from typing import Union
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint
import yaml

def get_supported_weights_precision(device: torch.device) -> torch.dtype:
    """Return the highest-precision dtype supported for AMP on the given device."""
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
    Peak extra VRAM is O(chunk_size x vocab_size) instead of O(B x T x vocab_size).
    """
    B, T, D = hidden_states.shape
    hidden_flat  = hidden_states.view(B * T, D)
    targets_flat = targets.contiguous().view(B * T)

    total_loss = hidden_states.new_zeros(())
    for start in range(0, B * T, chunk_size):
        end = min(start + chunk_size, B * T)
        chunk_logits = F.linear(hidden_flat[start:end], embedding_weight)  # (chunk, V)
        chunk_loss = F.cross_entropy(chunk_logits, targets_flat[start:end], reduction="sum")
        total_loss = total_loss + chunk_loss
        del chunk_logits

    return total_loss / (B * T)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (no mean-subtraction)."""
    def __init__(self,
                 feature_size: int,
                 device: torch.device,
                 eps: float = 1e-8):
        super().__init__()
        self.feature_size = feature_size
        self.device = device
        self.eps = eps
        self.gain = nn.Parameter(torch.ones((feature_size,),
                                            device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        # [B, G, H//G, T, HS] -> [B, H, T, HS]  and  [B, G, 1, T, HS] -> [B, G, T, HS].
        # This is required for PyTorch to dispatch to FlashAttention; the 5-D
        # layout forces the slow math kernel and materialises the full O(T^2) matrix.
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
    """SwiGLU feed-forward block: FFN with a gated activation."""

    def __init__(self,
                 input_embedding_dim: int,
                 hidden_embedding_dim: int):
        super().__init__()
        self.linear_12 = nn.Linear(input_embedding_dim,
                             2 * hidden_embedding_dim, bias=False)
        self.linear_out = nn.Linear(hidden_embedding_dim,
                                    input_embedding_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """Transformer decoder with GQA, SwiGLU FFN, and RMSNorm."""
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
        # During inference, if we are generating one token at time, in an
        # autoregressive mode, we need to compute new RoPE sin and cos.
        # NOTE: the KV cache is truncated to sequence_length (sliding window),
        # but the RoPE offset is derived from the KV cache length. Once total
        # generated tokens exceed sequence_length the offset caps at sequence_length
        # and new tokens receive the wrong positional encoding. Generation beyond
        # the context window still works but quality degrades. It's a bug I've to solve.
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
    """Decoder-only causal language model with tied input/output embeddings."""
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
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        self.n_decoder_blocks = n_decoder_blocks
        self.n_kv_heads = n_kv_heads
        self.n_heads = n_heads
        self.ffn_hidden_dim = ffn_hidden_dim
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
                    config_path: Union[str, Path],
                    kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
                    device: torch.device,
                    gradient_checkpointing: bool = False) -> "LanguageModel":
        """Construct a LanguageModel from a YAML config file."""
        with Path(config_path).open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        assert isinstance(config, dict), "The model configuration file must contain a YAML mapping"
        assert isinstance(config['model'], dict), "The 'model' section must contain a YAML mapping"
        model_selection = config['model']
        
        if model_selection["vocab_size"] <= 0:
            raise ValueError("vocab_size must be greater than 0")
        if model_selection["sequence_length"] <= 0:
            raise ValueError("sequence_length must be greater than 0")
        if model_selection["embedding_dim"] <= 0:
            raise ValueError("embedding_dim must be greater than 0")
        if model_selection["n_decoder_blocks"] <= 0:
            raise ValueError("n_decoder_blocks must be greater than 0")
        if model_selection["n_heads"] <= 0:
            raise ValueError("n_heads must be greater than 0")
        if model_selection["n_kv_heads"] <= 0:
            raise ValueError("n_kv_heads must be greater than 0")
        if model_selection["embedding_dim"] % model_selection["n_heads"] != 0:
            raise ValueError("embedding_dim must be divisible by n_heads")
        if model_selection["n_heads"] % model_selection["n_kv_heads"] != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if not 0.0 <= model_selection["dropout_rate"] <= 1.0:
            raise ValueError("dropout_rate must be between 0 and 1")
        if model_selection["ffn_hidden_dim"] is not None and model_selection["ffn_hidden_dim"] <= 0:
            raise ValueError("ffn_hidden_dim must be greater than 0")

        return cls(
            n_decoder_blocks=model_selection["n_decoder_blocks"],
            sequence_length=model_selection["sequence_length"],
            vocab_size=model_selection["vocab_size"],
            embedding_dim=model_selection["embedding_dim"],
            n_heads=model_selection["n_heads"],
            n_kv_heads=model_selection["n_kv_heads"],
            ffn_hidden_dim=model_selection["ffn_hidden_dim"],
            kv_cache=kv_cache,
            dropout_rate=model_selection["dropout_rate"],
            device=device,
            gradient_checkpointing=gradient_checkpointing,
        )
