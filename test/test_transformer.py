import copy
from pathlib import Path

import torch

from src.transformer import (
    GroupedQueryAttention,
    LanguageModel,
    ModelConfig,
    RMSNorm,
    get_supported_weights_precision,
    language_model_loss,
    training_step,
)


def build_language_model(kv_cache=None):
    return LanguageModel(
        n_decoder_blocks=2,
        sequence_length=8,
        vocab_size=32,
        embedding_dim=16,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=64,
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


def test_language_model_loss_returns_scalar():
    model = build_language_model()
    tokens = torch.randint(0, model.vocab_size, (2, 6), dtype=torch.long)
    targets = torch.randint(0, model.vocab_size, (2, 6), dtype=torch.long)

    logits = model(tokens)
    loss = language_model_loss(logits, targets)

    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


def test_training_step_updates_model_parameters():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = build_language_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    tokens = torch.randint(0, model.vocab_size, (2, 6), dtype=torch.long)
    targets = torch.randint(0, model.vocab_size, (2, 6), dtype=torch.long)
    initial_embedding = model.embedding_matrix.weight.detach().clone()

    loss = training_step(
        language_model=model,
        optimizer=optimizer,
        scaler=scaler,
        inputs=tokens,
        targets=targets,
        device=device,
        amp_dtype=get_supported_weights_precision(device),
        use_amp=False,
    )

    assert loss > 0
    assert not torch.equal(initial_embedding, model.embedding_matrix.weight)


def test_model_config_loads_from_yaml(tmp_path: Path):
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  vocab_size: 64",
                "  sequence_length: 16",
                "  embedding_dim: 32",
                "  n_decoder_blocks: 3",
                "  n_heads: 4",
                "  n_kv_heads: 2",
                "  ffn_hidden_dim: 96",
                "  dropout_rate: 0.15",
            ]
        ),
        encoding="utf-8",
    )

    config = ModelConfig.from_yaml(config_path)

    assert config.vocab_size == 64
    assert config.sequence_length == 16
    assert config.resolved_ffn_hidden_dim == 96


def test_language_model_can_be_built_from_config():
    config = ModelConfig(
        vocab_size=64,
        sequence_length=16,
        embedding_dim=32,
        n_decoder_blocks=3,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=96,
        dropout_rate=0.15,
    )

    model = LanguageModel.from_config(config, kv_cache={}, device=torch.device("cpu"))

    assert model.vocab_size == config.vocab_size
    assert model.embedding_dim == config.embedding_dim
    assert model.transformer_decoder.ffn_hidden_dim == config.ffn_hidden_dim


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
