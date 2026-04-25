from dataclasses import dataclass
from pathlib import Path
from typing import Union, cast
import random
import torch
from torch import nn
from torch.nn import functional as F
import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    sequence_length: int
    embedding_dim: int
    n_decoder_blocks: int
    n_heads: int
    n_kv_heads: int
    dropout_rate: float = 0.0
    ffn_hidden_dim: int | None = None

    def __post_init__(self) -> None:
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
        return self.ffn_hidden_dim if self.ffn_hidden_dim is not None else 4 * self.embedding_dim

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "ModelConfig":
        with Path(config_path).open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)

        if not isinstance(raw_config, dict):
            raise ValueError("The model configuration file must contain a YAML mapping")

        model_section = raw_config.get("model", raw_config)
        if not isinstance(model_section, dict):
            raise ValueError("The 'model' section must contain a YAML mapping")

        return cls(**model_section)

def get_supported_weights_precision(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1)
    )


def training_step(
    language_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    max_grad_norm: float | None = 1.0
) -> float:
    language_model.train()
    optimizer.zero_grad(set_to_none=True)

    inputs, targets = inputs.to(device), targets.to(device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        logits = language_model(inputs)
        loss = language_model_loss(logits, targets)

    # Scaling is only needed for CUDA float16. With bfloat16 or CPU,
    # GradScaler is disabled and these calls become no-ops/pass-throughs.
    scaler.scale(loss).backward()

    if max_grad_norm is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(language_model.parameters(), max_norm=max_grad_norm)

    scaler.step(optimizer)
    scaler.update()

    return float(loss.detach().item())


def compile_language_model(language_model: nn.Module, enabled: bool) -> nn.Module:
    if enabled:
        return cast(nn.Module, torch.compile(language_model))
    return language_model

class RMSNorm(nn.Module):
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

    def forward(self, x: torch.Tensor):
        rec_rms = torch.rsqrt(torch.mean(x.square(), -1, keepdim=True) + self.eps)
        normalized = x * rec_rms * self.gain

        return normalized

class GroupedQueryAttention(nn.Module):
    def __init__(self,
                 layer_idx: int,
                 embedding_dim: int,
                 sequence_length: int,
                 n_heads: int,
                 n_kv_heads: int,
                 head_size: int,
                 device: torch.device):
        """
        Implementation of Grouped-Query Attention
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
                   offset: int):
        rope_slice = slice(offset, offset + x.shape[-2])
        inv_x2 = x[..., ::2]     # (B, G, H // G, T or 1, HS)
        inv_x1 = -x[..., 1::2]
        x_rot = torch.stack((inv_x1, inv_x2), dim=-1).flatten(-2)

        return (x * rope_cos[:, :, :, rope_slice, :]) + (x_rot * rope_sin[:, :, :, rope_slice, :])

    def forward(self,
                embeddings: torch.Tensor,
                mask: torch.Tensor,
                kv_cache: Union[dict[int, tuple[torch.Tensor, torch.Tensor]], None],
                rope_cos: torch.Tensor,
                rope_sin: torch.Tensor):

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

        if not self.training and kv_cache is not None:
            if self.layer_idx in kv_cache.keys():
                past_k, past_v = kv_cache[self.layer_idx]
                k = torch.cat([past_k, k], dim=3)[:, :, :, -self.sequence_length:, :]
                v = torch.cat([past_v, v], dim=3)[:, :, :, -self.sequence_length:, :]

            kv_cache[self.layer_idx] = (k, v)

        # INEFFICIENT CAUSAL ATTENTION
        # ==================================
        # logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        # if mask is not None:
        #    logits = logits + mask
            
        # weights = F.softmax(logits, dim=-1)
        # outputs = torch.matmul(weights, v)
        # ==================================
        outputs = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        outputs = outputs.permute(0, 3, 1, 2, 4).contiguous().view(B, T, self.embedding_dim)
        y = self.output_proj(outputs)

        return y
    
class SwiGLU(nn.Module):
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
    casual_mask: torch.Tensor
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
                 device: torch.device):
        super().__init__()
        mask_shape = (1, 1, 1, sequence_length, sequence_length)
        self.register_buffer("casual_mask",
                             torch.triu(torch.full(mask_shape,
                                                   -torch.inf,
                                                   device=device,
                                                   dtype=get_supported_weights_precision(device)),
                                        diagonal=1))
        self.kv_cache = kv_cache
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
        

    def forward(self, x: torch.Tensor):
        # x -> (B, T, D)
        _, T, _ = x.shape
        mask = self.casual_mask[:, :, :, :T, :T]
        if not self.training:
            self.global_token_counter += T

            # If we are in the case of autoregressive generation
            if self.global_token_counter > self.sequence_length:
                self.extend_rope_sin_cos(self.global_token_counter - 1,
                                         use_gqa=True)
            if self.kv_cache:
                total_tokens = min(self.global_token_counter, self.sequence_length)
                mask = self.casual_mask[:, :, :, total_tokens - T:total_tokens, :total_tokens]

        for i in range(self.n_blocks):
            attn_out = self.attentions[i](
                self.norms_1[i](x),
                mask,
                self.kv_cache,
                self.rope_cos,
                self.rope_sin
            )

            x = x + F.dropout(attn_out, self.dropout_rate, training=self.training) # (B, T, D)

            ffn_out = self.ffns[i](self.norms_2[i](x))
            x = x + F.dropout(ffn_out, self.dropout_rate, training=self.training)

        return x # (B, T, D)
    
class LanguageModel(nn.Module):
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
                 device: torch.device):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.embedding_matrix = nn.Embedding(vocab_size,
                                             embedding_dim)
        self.transformer_decoder = TransformerDecoder(
            n_blocks=n_decoder_blocks,
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            kv_cache=kv_cache,
            dropout_rate=dropout_rate,
            device=device
        )

        self.final_norm = RMSNorm(
            feature_size=embedding_dim,
            device=device
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x                                 # (B, T)
        embeddings = self.embedding_matrix(tokens) # (B, T, D)
        out = self.transformer_decoder(embeddings) # (B, T, D)
        # Remember that torch.nn.functional.linear transposes the weight matrix
        # passed before applying the affine linear transformation, so we don't
        # need to compute the transpose of the embedding matrix to get back to
        # vocab space.
        logits = F.linear(self.final_norm(out), self.embedding_matrix.weight) # (B, T, vocab_size)

        return logits

    @classmethod
    def from_config(cls,
                    config: ModelConfig,
                    kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
                    device: torch.device) -> "LanguageModel":
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
        )

if __name__ == "__main__":
    batch_size = 1
    config = ModelConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs" / "model.yaml")
    kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    supported_dtype = get_supported_weights_precision(device)
    use_amp = device.type == "cuda"
    use_grad_scaler = use_amp and supported_dtype == torch.float16

    language_model: nn.Module = LanguageModel.from_config(config, kv_cache=kv_cache, device=device).to(device)
    optimizer = torch.optim.AdamW(language_model.parameters(), lr=3e-4)

    # Python 3.14 is supported by torch.compile starting from PyTorch 2.10.
    # Keep compilation opt-in here because the first compile can dominate a tiny
    # local smoke run; enable it for real training benchmarks.
    use_compile = False
    language_model = compile_language_model(language_model, enabled=use_compile)
    
    batch_of_tokens: list[list[int]] = []
    for _ in range(batch_size):
        sequence: list[int] = []
        for _ in range(config.sequence_length + 1):
            sequence.append(random.randint(0, config.vocab_size - 1))

        batch_of_tokens.append(sequence)


    new_token = torch.tensor([[random.randint(0, config.vocab_size - 1)]], dtype=torch.long) # (B=1, T=1)
    new_token_target = torch.tensor([[random.randint(0, config.vocab_size - 1)]], dtype=torch.long)
        

    inputs = torch.tensor([sequence[:-1] for sequence in batch_of_tokens], dtype=torch.long) # (B, T)
    targets = torch.tensor([sequence[1:] for sequence in batch_of_tokens], dtype=torch.long)

    dataloader: list[tuple[torch.Tensor, torch.Tensor]] = [
        (inputs, targets),
        (new_token, new_token_target)
    ]

    scaler = torch.amp.GradScaler(device.type, enabled=use_grad_scaler)


    for inputs, targets in dataloader:
        loss = training_step(
            language_model=language_model,
            optimizer=optimizer,
            scaler=scaler,
            inputs=inputs,
            targets=targets,
            device=device,
            amp_dtype=supported_dtype,
            use_amp=use_amp,
            max_grad_norm=1.0
        )
        print(f"Training loss: {loss:.4f}")
