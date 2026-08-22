"""Flow-matching decision layer over frozen cached features (spec: `_docs/stage_2.pdf`).

Two training modes over one shared architecture (ADR 0007):

  standard  z_t = (1 - t) z_i + t p_{y_i},  t ~ U(0,1)
            L_FM = || v_theta(z_t, t) - (p_{y_i} - z_i) ||^2
  rollout   run the exact T-step Euler inference sequence from z_i,
            L_roll = || z_hat_T - p_{y_i} ||^2, backprop through the sequence
            (training T == inference T, per the spec).

Inference (both modes): z_hat_{k+1} = z_hat_k + (1/T) v_theta(z_hat_k, k/T),
k = 0..T-1, starting from the test feature; z_hat_T is classified with the
Stage-1 cosine/prototype rule.

The layer operates on L2-normalized features — the space the Stage-1 cosine
classifier acts in; prototypes are unit-norm by construction (ADR 0007 §3).
Losses are per-sample squared L2 norms (summed over feature dimensions, averaged
over the batch), matching the spec's ||.||_2^2 exactly.
"""
import copy

import torch
import torch.nn.functional as F

from .utils import get_device, load_config


class VelocityMLP(torch.nn.Module):
    """v_theta(z, t): the spec's suggested MLP — hidden layers with SiLU,
    scalar time t concatenated to the input feature, output dim = feature dim."""

    def __init__(self, dim, hidden=(512, 512)):
        super().__init__()
        layers, prev = [], dim + 1
        for h in hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.SiLU()]
            prev = h
        layers.append(torch.nn.Linear(prev, dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, z, t):
        return self.net(torch.cat([z, t.unsqueeze(-1)], dim=-1))


def _seeded_init(model, generator):
    """PyTorch's default Linear init (U(-1/sqrt(in), 1/sqrt(in))) drawn from an
    explicit generator, so the init seed is meaningful (mirrors LinearProbe)."""
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            bound = 1.0 / m.in_features ** 0.5
            with torch.no_grad():
                m.weight.copy_((torch.rand(m.weight.shape, generator=generator) * 2 - 1) * bound)
                m.bias.copy_((torch.rand(m.bias.shape, generator=generator) * 2 - 1) * bound)


class FlowMatchingHead:
    """FM transport toward fixed class prototypes + Stage-1 cosine classification.

    `prototypes` [C, D]: the Stage-1 class prototypes for this exact setting
    (image-derived from the selected training subset, or CLIP text prototypes).
    Training config comes from config/config.json `stage2.training` — the Stage-1
    probe recipe; checkpoint = minimum-training-loss epoch (the ADR 0007 §7
    contingency, triggered), never validation- or test-based.
    """

    def __init__(self, prototypes, mode, T=None, seed=0, hidden=None,
                 normalize=True, **overrides):
        cfg = load_config()["stage2"]
        self.cfg = {**cfg["training"], **overrides}
        assert mode in ("standard", "rollout"), mode
        assert mode != "rollout" or T is not None, "rolled-out training needs its inference T"
        self.mode, self.T, self.seed = mode, T, seed
        # normalize=False runs the literal-spec raw-feature version of the grid
        # (`run_stage2.py --raw`, reported as the ‡ comparison version); every
        # primary published result uses True (ADR 0007 §3).
        self.normalize = normalize
        self.device = get_device()
        self.prototypes = F.normalize(prototypes.float(), dim=-1).to(self.device)
        self.dim = self.prototypes.shape[1]
        self.hidden = tuple(hidden) if hidden else tuple(cfg["velocity_net"]["hidden"])
        g = torch.Generator().manual_seed(seed)
        self.model = VelocityMLP(self.dim, self.hidden)
        _seeded_init(self.model, g)
        self.model = self.model.to(self.device)
        self.history = None
        self.best = None

    def _rollout(self, z, T):
        for k in range(T):
            t = torch.full((len(z),), k / T, device=z.device)
            z = z + self.model(z, t) / T
        return z

    def fit(self, Xtr, ytr):
        cfg = self.cfg
        dev = self.device
        Z = (F.normalize(Xtr.float(), dim=-1) if self.normalize
             else Xtr.float()).to(dev)
        P = self.prototypes[ytr.long().to(dev)]
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"])
        g = torch.Generator().manual_seed(self.seed)  # batch order + t draws
        n, bs = len(Z), cfg["batch_size"]
        hist = {"epoch": [], "train_loss": []}
        best_loss, best_state, best_epoch = float("inf"), None, -1

        for epoch in range(cfg["max_epochs"]):
            self.model.train()
            perm = torch.randperm(n, generator=g).to(dev)
            total = 0.0
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                z, p = Z[idx], P[idx]
                opt.zero_grad()
                if self.mode == "standard":
                    t = torch.rand(len(idx), generator=g).to(dev)
                    zt = (1 - t).unsqueeze(-1) * z + t.unsqueeze(-1) * p
                    loss = ((self.model(zt, t) - (p - z)) ** 2).sum(-1).mean()
                else:
                    loss = ((self._rollout(z, self.T) - p) ** 2).sum(-1).mean()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            hist["epoch"].append(epoch)
            hist["train_loss"].append(total / n)
            if total / n < best_loss:
                best_loss, best_epoch = total / n, epoch
                if cfg["checkpoint_selection"] == "min_train_loss":
                    best_state = copy.deepcopy(self.model.state_dict())

        # ADR 0007 §7 contingency (triggered 2026-08-09, see status note there):
        # the selected checkpoint is the minimum-training-loss epoch, uniformly
        # for every model. No validation or test data is involved.
        if cfg["checkpoint_selection"] == "min_train_loss" and best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.history = hist
        self.best = {"epoch": best_epoch, "train_loss": best_loss}
        return self

    @torch.no_grad()
    def transport(self, X, T, return_traj=False):
        """Euler-transport features; optionally return every intermediate state
        [T+1, B, D] (z_hat_0 .. z_hat_T) for the flow-trajectory figures."""
        self.model.eval()
        z = (F.normalize(X.float(), dim=-1) if self.normalize
             else X.float()).to(self.device)
        traj = [z.cpu()]
        for k in range(T):
            t = torch.full((len(z),), k / T, device=self.device)
            z = z + self.model(z, t) / T
            if return_traj:
                traj.append(z.cpu())
        return (z.cpu(), torch.stack(traj)) if return_traj else z.cpu()

    @torch.no_grad()
    def predict(self, X, T):
        z = self.transport(X, T).to(self.device)
        return (F.normalize(z, dim=-1) @ self.prototypes.T).argmax(-1).cpu()

    def save(self, path):
        torch.save({"state_dict": self.model.state_dict(), "mode": self.mode,
                    "T": self.T, "seed": self.seed, "dim": self.dim,
                    "hidden": list(self.hidden),
                    "prototypes": self.prototypes.cpu()}, path)


def load_fm_head(path):
    """Rebuild a trained FlowMatchingHead from `save()` output (for figures)."""
    d = torch.load(path, weights_only=True)
    head = FlowMatchingHead(d["prototypes"], d["mode"], T=d["T"], seed=d["seed"],
                            hidden=d["hidden"])
    head.model.load_state_dict({k: v.to(head.device) for k, v in d["state_dict"].items()})
    head.model.eval()
    return head
