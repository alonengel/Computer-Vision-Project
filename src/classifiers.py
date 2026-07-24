"""Classification heads over frozen cached features (spec: `_docs/stage_1.pdf`).

  LinearProbe        required baseline — s = W z + b, softmax cross-entropy,
                     AdamW, checkpoint selected by highest validation accuracy.
  PrototypeClassifier  branch A — image-derived class prototypes:
                     mu_c = normalize( mean_{i in S_c} normalize(z_i) ),
                     y = argmax_c cos(z, mu_c).
  ZeroShotCLIP       branch B — text-derived class prototypes from CLIP RN50:
                     y = argmax_c cos(z, t_c), no labeled training images.
"""
import copy

import torch
import torch.nn.functional as F

from .utils import get_device, load_config


class LinearProbe:
    """Multiclass linear classifier trained on frozen features.

    Only W and b are trained. Defaults come from config/config.json, which holds
    the spec's suggested configuration (AdamW, lr 1e-3, weight decay 1e-4,
    batch size 64, max 200 epochs, checkpoint = highest validation accuracy).
    `fit` records per-epoch training and validation loss so the training curves
    can be plotted, and restores the best-validation-accuracy weights at the end.
    """

    def __init__(self, n_classes, dim, seed=0, **overrides):
        cfg = {**load_config()["linear_probe"], **overrides}
        self.cfg = cfg
        self.n_classes = n_classes
        self.dim = dim
        self.seed = seed
        self.device = get_device()
        g = torch.Generator().manual_seed(seed)
        self.model = torch.nn.Linear(dim, n_classes)
        with torch.no_grad():  # seeded reinit so init_seed is meaningful
            bound = 1.0 / dim ** 0.5
            self.model.weight.copy_((torch.rand(n_classes, dim, generator=g) * 2 - 1) * bound)
            self.model.bias.copy_((torch.rand(n_classes, generator=g) * 2 - 1) * bound)
        self.model = self.model.to(self.device)
        self.history = None
        self.best = None

    def fit(self, Xtr, ytr, Xval, yval):
        cfg = self.cfg
        dev = self.device
        Xtr, ytr = Xtr.to(dev), ytr.to(dev)
        Xval, yval = Xval.to(dev), yval.to(dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"])
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        n, bs = len(Xtr), cfg["batch_size"]
        hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
        best_acc, best_state, best_epoch = -1.0, None, -1

        for epoch in range(cfg["max_epochs"]):
            self.model.train()
            perm = torch.randperm(n, generator=g).to(dev)
            total = 0.0
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                opt.zero_grad()
                loss = F.cross_entropy(self.model(Xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(Xval)
                val_loss = F.cross_entropy(logits, yval).item()
                val_acc = (logits.argmax(-1) == yval).float().mean().item()
            hist["epoch"].append(epoch)
            hist["train_loss"].append(total / n)
            hist["val_loss"].append(val_loss)
            hist["val_acc"].append(val_acc)
            if val_acc > best_acc:  # checkpoint selection: highest validation accuracy
                best_acc, best_epoch = val_acc, epoch
                best_state = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_state)
        self.model.eval()
        self.history = hist
        self.best = {"val_acc": best_acc, "epoch": best_epoch}
        return self

    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        return self.model(X.to(self.device)).argmax(-1).cpu()


class PrototypeClassifier:
    """Image-derived class prototypes (branch A), exactly as specified:

    mu_c = normalize( (1/|S_c|) * sum_{i in S_c} normalize(z_i) )
    y_hat = argmax_c cos(z, mu_c)

    Prototypes are computed from the selected training subset only.
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.prototypes = None

    def fit(self, Xtr, ytr):
        Z = F.normalize(Xtr.float(), dim=-1)
        protos = []
        for c in range(self.n_classes):
            mask = ytr == c
            assert bool(mask.any()), f"class {c} has no training samples in this subset"
            protos.append(F.normalize(Z[mask].mean(0), dim=-1))
        self.prototypes = torch.stack(protos)
        return self

    @torch.no_grad()
    def predict(self, X):
        Z = F.normalize(X.float(), dim=-1)
        return (Z @ self.prototypes.T).argmax(-1)

    def scores(self, X):
        Z = F.normalize(X.float(), dim=-1)
        return Z @ self.prototypes.T


class ZeroShotCLIP:
    """Text-derived class prototypes (branch B): y_hat = argmax_c cos(z, t_c).

    `text_prototypes` [C, D] must be L2-normalized. Uses no labeled training images.
    """

    def __init__(self, text_prototypes):
        self.prototypes = text_prototypes.float()

    @torch.no_grad()
    def predict(self, X):
        Z = F.normalize(X.float(), dim=-1)
        return (Z @ self.prototypes.T).argmax(-1)

    def scores(self, X):
        Z = F.normalize(X.float(), dim=-1)
        return Z @ self.prototypes.T
