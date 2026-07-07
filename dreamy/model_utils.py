"""Small compatibility helpers for Hugging Face causal LM architectures."""


def get_layers(model):
    """Return the transformer block list for common decoder-only models."""
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not locate transformer layers on model")


def get_mlp_output_projection(block):
    """Return the MLP output projection whose input is the post-activation feature."""
    mlp = getattr(block, "mlp", None)
    if mlp is None:
        raise AttributeError("Transformer block has no mlp module")

    for name in ("dense_4h_to_h", "fc2", "down_proj", "c_proj"):
        if hasattr(mlp, name):
            return getattr(mlp, name)
    raise AttributeError("Could not locate MLP output projection on block")


def get_attention_module(block):
    """Return the attention module for common decoder-only model blocks."""
    for name in ("attention", "self_attn", "attn"):
        if hasattr(block, name):
            return getattr(block, name)
    raise AttributeError("Could not locate attention module on block")
