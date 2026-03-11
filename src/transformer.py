from typing import Tuple, Dict, Union
import random
import torch
from torch import nn
from torch.nn import functional as F

def get_supported_weights_precision(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    else:
        return torch.float16

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
    
        inv_x2 = x[..., ::2]     # (B, G, H // G, T or 1, HS)
        inv_x1 = -x[..., 1::2]
        x_rot = torch.stack((inv_x1, inv_x2), dim=-1).flatten(-2)

        return (x * rope_cos[:, :, :, offset:, :]) + (x_rot * rope_sin[:, :, :, offset:, :])

    def forward(self,
                embeddings: torch.Tensor,
                mask: torch.Tensor,
                kv_cache: Union[Dict[int, Tuple[torch.Tensor, torch.Tensor]], None],
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
        # TODO: Fix error with mask in eval mode
        breakpoint()
        outputs = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        y = self.output_proj(outputs.view(B, T, self.embedding_dim))

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
    def __init__(self,
                 n_blocks: int,
                 sequence_length: int,
                 embedding_dim: int,
                 n_heads: int,
                 n_kv_heads: int,
                 kv_cache: Dict[int, Tuple[torch.Tensor]],
                 dropout_rate: float,
                 device: torch.device):
        super().__init__()
        if self.training:
            mask_shape = (1, 1, 1, sequence_length, sequence_length)
            self.register_buffer("casual_mask",
                                torch.triu(torch.full(mask_shape,
                                                    -torch.inf,
                                                    device=device,
                                                    dtype=get_supported_weights_precision(device)),
                                            diagonal=1))
        else:
            self.casual_mask = None
        self.kv_cache = kv_cache
        self.dropout_rate = dropout_rate
        self.n_blocks = n_blocks
        self.device = device
        self.sequence_length = sequence_length
        self.head_size = embedding_dim // n_heads
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
                    4 * embedding_dim) 
            for _ in range(n_blocks)]
        )

    def build_rope_sin_cos(self,
                           dtype: torch.dtype,
                           use_gqa: bool = True):

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
                            use_gqa: bool = True):
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
        if not self.training:
            self.global_token_counter += T

            # If we are in the case of autoregressive generation
            if self.global_token_counter > self.sequence_length:
                self.extend_rope_sin_cos(self.global_token_counter,
                                         use_gqa=True)

        for i in range(self.n_blocks):
            attn_out = self.attentions[i](
                self.norms_1[i](x),
                self.casual_mask,
                self.kv_cache,
                self.rope_cos,
                self.rope_sin
            )

            x += F.dropout(attn_out, self.dropout_rate) # (B, T, D)

            ffn_out = self.ffns[i](self.norms_2[i](x))
            x += F.dropout(ffn_out, self.dropout_rate)

        return x # (B, T, D)
    
class LanguageModel(nn.Module):
    def __init__(self,
                 n_decoder_blocks: int,
                 sequence_length: int,
                 vocab_size: int,
                 embedding_dim: int,
                 n_heads: int,
                 n_kv_heads: int,
                 kv_cache: Dict[int, Tuple[torch.Tensor]],
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
            kv_cache=kv_cache,
            dropout_rate=dropout_rate,
            device=device
        )

        self.final_norm = RMSNorm(
            feature_size=embedding_dim,
            device=device
        )
        
    def forward(self, x: torch.Tensor):
        tokens = x                                 # (B, T)
        embeddings = self.embedding_matrix(tokens) # (B, T, D)
        out = self.transformer_decoder(embeddings) # (B, T, D)
        # Remember that torch.nn.functional.linear transposes the weight matrix
        # passed before applying the affine linear transformation, so we don't
        # need to compute the transpose of the embedding matrix to get back to
        # vocab space.
        logits = F.linear(self.final_norm(out), self.embedding_matrix.weight) # (B, T, vocab_size)

        return logits

if __name__ == "__main__":
    vocab_size = 20
    embedding_dim = 8
    batch_size = 1
    sequence_length = 4
    n_decoder_blocks = 4
    n_heads = 4
    n_kv_heads = 2
    dropout_rate = 0.1
    kv_cache = {}
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    supported_dtype = get_supported_weights_precision(device)

    language_model = LanguageModel(
        n_decoder_blocks=n_decoder_blocks,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        kv_cache=kv_cache,
        dropout_rate=dropout_rate,
        device=device
    )

    language_model.eval()

    # TODO: Change Python to a version different from 3.14 to support
    # torch.compile
    # language_model = torch.compile(language_model)
    
    batch_of_tokens = []
    for _ in range(batch_size):
        sequence = []
        for _ in range(sequence_length):
            sequence.append(random.randint(0, vocab_size - 1))

        batch_of_tokens.append(sequence)


    new_token = torch.tensor([[random.randint(0, vocab_size - 1)]], dtype=torch.long) # (B=1, T=1)
    new_token_target = new_token.clone()
        

    inputs = torch.tensor(batch_of_tokens, dtype=torch.long) # (B, T)
    targets = inputs.clone()

    dataloader = [
        (inputs, targets),
        (new_token, new_token_target)
    ]

    scaler = torch.amp.GradScaler(device)


    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        # TODO: Put optimizer here

        with torch.autocast(device_type=device.type, dtype=supported_dtype):
            logits = language_model(inputs)
            print(torch.sum((F.softmax(logits, dim=-1).detach()[0][0])))
            print(kv_cache[0][0].shape)
            # TODO: Put loss here

        # If bfloat16 is not supported it scales the loss and then
        # backprops gradients, otherwise (bfloat16 supported), it
        # calls the standard backward method.
        # TODO: scaler.scale(loss).backward()
        
        # Gradient clipping
        # TODO: scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(language_model.parameters(), max_norm=1.0)
        
        # TODO: Optimizer step
        #scaler.step(optimizer)
        #scaler.update()