import torch
import numpy as np
from sklearn.linear_model import LogisticRegression


@torch.no_grad()
def collect_residual_means(model, hook_layer: int, tok, texts, max_len=256):
    outs = []

    def hk(mod, inp, out):
        # out: hidden state after the block forward; shape [B, T, d]
        hk.h = out
    handle = model.gpt_neox.layers[hook_layer].register_forward_hook(hk)
    for batch in [texts[i:i+16] for i in range(0, len(texts), 16)]:
        ids = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(model.device)
        _ = model(**ids)
        H = hk.h[:, :, :].mean(dim=1).detach().cpu().numpy()  # [B, d]
        outs.append(H)
    handle.remove()
    return np.vstack(outs)  # [N, d]


def fit_direction(model, tok, layer, eval_texts, noneval_texts):
    Xe = collect_residual_means(model, layer, tok, eval_texts)
    Xn = collect_residual_means(model, layer, tok, noneval_texts)
    X = np.vstack([Xe, Xn])
    y = np.hstack([np.ones(len(Xe)), np.zeros(len(Xn))])
    clf = LogisticRegression(max_iter=500, C=1.0).fit(X, y)
    w = torch.tensor(clf.coef_[
                     0] / (np.linalg.norm(clf.coef_[0]) + 1e-8), dtype=model.dtype, device=model.device)
    return w  # unit vector
