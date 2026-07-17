"""Few-shot classifier heads over frozen embeddings.

Common interface — predict(Xs, ys, Xq) -> scores:
  Xs [B, S, D] support embeddings, ys [B, S] support labels in 0..C-1,
  Xq [B, Q, D] query embeddings; returns scores [B, Q, C] (argmax = prediction).
B is the episode batch (B=1 for the simple protocol). All heads are pure torch and
differentiable, so stage 2/3 can insert a Flow Matching module upstream unchanged.
"""
import torch
import torch.nn.functional as F

from .utils import load_config


def _prototypes(Xs, ys, n_classes):
    """Class-mean prototypes [B, C, D] from support embeddings."""
    B, S, D = Xs.shape
    proto = torch.zeros(B, n_classes, D, device=Xs.device, dtype=Xs.dtype)
    count = torch.zeros(B, n_classes, 1, device=Xs.device, dtype=Xs.dtype)
    proto.scatter_add_(1, ys.unsqueeze(-1).expand(-1, -1, D), Xs)
    count.scatter_add_(1, ys.unsqueeze(-1), torch.ones(B, S, 1, device=Xs.device, dtype=Xs.dtype))
    return proto / count.clamp(min=1)


class PrototypeClassifier:
    """Nearest class-mean. metric: 'cosine' (primary) or 'euclidean' (ablation)."""

    def __init__(self, metric="cosine"):
        assert metric in ("cosine", "euclidean")
        self.metric = metric

    def predict(self, Xs, ys, Xq):
        n_classes = int(ys.max().item()) + 1
        proto = _prototypes(Xs, ys, n_classes)
        if self.metric == "cosine":
            return F.normalize(Xq, dim=-1) @ F.normalize(proto, dim=-1).transpose(1, 2)
        return -torch.cdist(Xq, proto).pow(2)


class LinearProbe:
    """Linear head (weights + bias, as in nn.Linear) trained with CrossEntropyLoss
    + Adam on the support set.

    Training is vectorized over the episode batch: weights [B, C, D] are optimized
    jointly with one Adam instance. The loss uses reduction='sum' scaled by 1/S, so
    ∂L/∂W_b equals each episode's own mean-CE gradient — exactly equivalent to B
    independent heads (same gradients, same per-parameter Adam state and weight
    decay), but hundreds of times faster for 600-episode evaluation. Init is
    0.01·N(0,1) (fixed a priori), not nn.Linear's default Kaiming-uniform.
    `as_module()` returns a plain nn.Linear carrying one episode's weights for
    stage-3 end-to-end use.
    """

    def __init__(self, steps=None, lr=None, weight_decay=None, seed=0):
        cfg = load_config()["linear_probe"]
        self.steps = steps if steps is not None else cfg["steps"]
        self.lr = lr if lr is not None else cfg["lr"]
        self.weight_decay = weight_decay if weight_decay is not None else cfg["weight_decay"]
        self.seed = seed
        self.W = self.b = None

    def fit(self, Xs, ys, W0=None):
        B, S, D = Xs.shape
        n_classes = int(ys.max().item()) + 1
        if W0 is None:
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            W0 = 0.01 * torch.randn(B, n_classes, D, generator=g)
        W = W0.to(Xs.device).clone().requires_grad_(True)
        b = torch.zeros(B, n_classes, device=Xs.device, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=self.lr, weight_decay=self.weight_decay)
        flat_ys = ys.reshape(-1)
        for _ in range(self.steps):
            opt.zero_grad()
            logits = torch.einsum("bsd,bcd->bsc", Xs, W) + b.unsqueeze(1)
            # sum/S = per-episode mean CE summed over episodes -> each episode's
            # head gets exactly its independent-head gradient (not scaled by 1/B)
            loss = F.cross_entropy(logits.reshape(B * S, n_classes), flat_ys,
                                   reduction="sum") / S
            loss.backward()
            opt.step()
        self.W, self.b = W.detach(), b.detach()
        return self

    def predict(self, Xs, ys, Xq):
        self.fit(Xs, ys)  # always refit: no silent reuse of stale weights
        return torch.einsum("bqd,bcd->bqc", Xq, self.W) + self.b.unsqueeze(1)

    def as_module(self, episode=0):
        import torch.nn as nn

        C, D = self.W.shape[1], self.W.shape[2]
        lin = nn.Linear(D, C)
        with torch.no_grad():
            lin.weight.copy_(self.W[episode])
            lin.bias.copy_(self.b[episode])
        return lin


class ZeroShotCLIP:
    """Image-text cosine similarity against cached CLIP text embeddings.

    text_emb [C_all, D] must be L2-normalized (primary or ensemble variant).
    Support data is ignored (zero-shot); `class_subset` restricts scoring to the
    episode's classes, in episode label order.
    """

    def __init__(self, text_emb):
        self.text_emb = text_emb

    def predict(self, Xs, ys, Xq, class_subset=None):
        text = self.text_emb.to(Xq.device, Xq.dtype)
        if class_subset is not None:
            text = text[class_subset]  # [B, C, D] via advanced indexing
            return F.normalize(Xq, dim=-1) @ F.normalize(text, dim=-1).transpose(1, 2)
        return F.normalize(Xq, dim=-1) @ F.normalize(text, dim=-1).T
