import copy

import torch

from src.transformer import GroupedQueryAttention, LanguageModel, RMSNorm


def build_language_model(kv_cache=None):
    return LanguageModel(
        n_decoder_blocks=2,
        sequence_length=8,
        vocab_size=32,
        embedding_dim=16,
        n_heads=4,
        n_kv_heads=2,
        kv_cache={} if kv_cache is None else kv_cache,
        dropout_rate=0.0,
        device=torch.device("cpu"),
    )


def test_rms_norm_keeps_shape_and_dtype():
    norm = RMSNorm(feature_size=8, device=torch.device("cpu"))
    x = torch.randn(2, 3, 8, dtype=torch.float32)

    y = norm(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()


def test_grouped_query_attention_preserves_embedding_shape():
    attention = GroupedQueryAttention(
        layer_idx=0,
        embedding_dim=16,
        sequence_length=8,
        n_heads=4,
        n_kv_heads=2,
        head_size=4,
        device=torch.device("cpu"),
    )
    decoder = build_language_model().transformer_decoder
    embeddings = torch.randn(2, 5, 16)
    mask = decoder.casual_mask[:, :, :, :5, :5]

    outputs = attention(
        embeddings=embeddings,
        mask=mask,
        kv_cache=None,
        rope_cos=decoder.rope_cos,
        rope_sin=decoder.rope_sin,
    )

    assert outputs.shape == embeddings.shape
    assert torch.isfinite(outputs).all()


def test_language_model_forward_returns_vocab_logits():
    model = build_language_model()
    tokens = torch.randint(0, model.vocab_size, (2, 6), dtype=torch.long)

    logits = model(tokens)

    assert logits.shape == (2, 6, model.vocab_size)
    assert torch.isfinite(logits).all()


def test_cached_autoregressive_step_matches_full_forward_last_token():
    torch.manual_seed(0)
    tokens = torch.randint(0, 32, (1, 5), dtype=torch.long)

    cached_model = build_language_model(kv_cache={})
    full_model = build_language_model(kv_cache={})
    full_model.load_state_dict(copy.deepcopy(cached_model.state_dict()))

    cached_model.eval()
    full_model.eval()

    _ = cached_model(tokens[:, :-1])
    cached_logits = cached_model(tokens[:, -1:])
    full_logits = full_model(tokens)

    assert torch.allclose(
        cached_logits[:, -1, :],
        full_logits[:, -1, :],
        atol=1e-5,
        rtol=1e-4,
    )
    assert len(cached_model.transformer_decoder.kv_cache) == 2
