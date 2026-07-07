import torch
import numpy as np
from sklearn.linear_model import LogisticRegression

from prompt_suppression.model_utils import get_layers


@torch.no_grad()
def collect_residual_states(model, hook_layer: int, tok, texts, max_len=256, pooling="last"):
    outs = []

    def hk(mod, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        hk.h = hidden

    handle = get_layers(model)[hook_layer].register_forward_hook(hk)
    for batch in [texts[i:i+16] for i in range(0, len(texts), 16)]:
        ids = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(model.device)
        _ = model(**ids)
        if pooling == "last":
            last_idx = ids["attention_mask"].sum(dim=1) - 1
            H = hk.h[torch.arange(hk.h.shape[0], device=model.device), last_idx]
        elif pooling == "mean":
            mask = ids["attention_mask"].to(hk.h.dtype).unsqueeze(-1)
            H = (hk.h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        else:
            raise ValueError("pooling must be 'last' or 'mean'")
        outs.append(H.detach().cpu().numpy())  # [B, d]
    handle.remove()
    return np.vstack(outs)  # [N, d]


@torch.no_grad()
def collect_residual_means(model, hook_layer: int, tok, texts, max_len=256):
    outs = []

    def hk(mod, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        hk.h = hidden

    handle = get_layers(model)[hook_layer].register_forward_hook(hk)
    for batch in [texts[i:i+16] for i in range(0, len(texts), 16)]:
        ids = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(model.device)
        _ = model(**ids)
        mask = ids["attention_mask"].to(hk.h.dtype).unsqueeze(-1)
        H = (hk.h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        H = H.detach().cpu().numpy()
        outs.append(H)
    handle.remove()
    return np.vstack(outs)  # [N, d]


def fit_direction(model, tok, layer, eval_texts, noneval_texts, pooling="last"):
    Xe = collect_residual_states(model, layer, tok, eval_texts, pooling=pooling)
    Xn = collect_residual_states(model, layer, tok, noneval_texts, pooling=pooling)
    X = np.vstack([Xe, Xn])
    y = np.hstack([np.ones(len(Xe)), np.zeros(len(Xn))])
    clf = LogisticRegression(max_iter=500, C=1.0).fit(X, y)
    w = torch.tensor(clf.coef_[
                     0] / (np.linalg.norm(clf.coef_[0]) + 1e-8), dtype=model.dtype, device=model.device)
    return w  # unit vector
