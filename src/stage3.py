"""Stage 3: an FM transformation before the frozen Stage-1 linear probe.

    z (raw frozen feature) -> FM, T Euler steps -> z_hat -> frozen probe -> logits

Everything here follows docs/adr/0008 (pre-registered before any training run):
raw feature space end to end; exact identity initialization (final velocity
layer zero); two training strategies over one shared architecture and one
uniform training policy (AdamW, exactly `epochs` epochs, best-validation-
accuracy checkpoint with tie-breaks, objective NaN/Inf failure trigger with a
pre-registered fallback ladder handled by the runner):

  Strategy 1 ("Rolled")  backprop CE through the full rollout, FM parameters only,
              plus a RELATIVE displacement penalty
              lambda * mean_i ||z_hat_i - z_i||^2 / (||z_i||^2 + eps).
  Strategy 2 ("Guided")  classifier-guided targets + standard FM training. No CE gradient
              ever reaches the FM. Per epoch (two-phase): snapshot the FM,
              build per-sample targets with normalized CE-gradient steps
              projected after EVERY step (including step 0) onto the ball
              B(z_i, rho_i), rho_i = alpha*||z_i||, step eta_i = beta*rho_i,
              monotone acceptance (lowest projected-iterate CE); then train
              one epoch of standard FM against the fixed cached targets.
"""
import copy

import torch
import torch.nn.functional as F

from .flow_matching import VelocityMLP, _seeded_init
from .utils import get_device, load_config


class NonFiniteLoss(RuntimeError):
    """Raised on any NaN/Inf training loss — the runner restarts the run with
    the next pre-registered fallback (ADR 0008 §7)."""


def zero_init_last(model):
    """Zero the final Linear layer (weight AND bias) -> exact identity map."""
    last = model.net[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()


class Stage3FM:
    """The FM module + frozen classifier pipeline for one setting.

    `classifier` is the pinned Stage-1 probe (torch.nn.Linear); it is frozen
    here and never updated. `fallback` in {0,1,2,3} activates the ADR 0008
    ladder uniformly: 1 = grad clipping (max-norm 1.0), 2 = + lr 3e-4,
    3 = + internal input standardization (does not break identity — the final
    velocity layer is still zero at init).
    """

    def __init__(self, classifier, dim, seed=0, fallback=0, standardize_stats=None):
        cfg = load_config()["stage3"]
        self.cfg = cfg
        self.T = cfg["T"]
        self.eps = cfg["epsilon"]
        self.seed = seed
        self.fallback = fallback
        self.device = get_device()

        self.classifier = classifier.to(self.device).eval()
        for p in self.classifier.parameters():
            p.requires_grad_(False)

        g = torch.Generator().manual_seed(seed)
        hidden = tuple(load_config()["stage2"]["velocity_net"]["hidden"])
        self.model = VelocityMLP(dim, hidden)   # the Stage-2 design, per the PDF
        _seeded_init(self.model, g)
        zero_init_last(self.model)
        self.model = self.model.to(self.device)

        # fallback 3: internal conditioning only — Euler stays in raw space
        self.standardize = fallback >= 3
        if self.standardize:
            mu, sigma = standardize_stats
            self.mu = mu.to(self.device)
            self.sigma = sigma.to(self.device).clamp_min(1e-6)

        self.history = None
        self.best = None

    # ------------------------------------------------------------------ #

    def _velocity(self, z, t):
        if self.standardize:
            return self.sigma * self.model((z - self.mu) / self.sigma, t)
        return self.model(z, t)

    def rollout(self, z):
        """Differentiable T-step Euler rollout in RAW space."""
        for k in range(self.T):
            t = torch.full((len(z),), k / self.T, device=z.device)
            z = z + self._velocity(z, t) / self.T
        return z

    @torch.no_grad()
    def transport(self, X, return_traj=False):
        self.model.eval()
        z = X.float().to(self.device)
        traj = [z.cpu()]
        for k in range(self.T):
            t = torch.full((len(z),), k / self.T, device=self.device)
            z = z + self._velocity(z, t) / self.T
            if return_traj:
                traj.append(z.cpu())
        return (z.cpu(), torch.stack(traj)) if return_traj else z.cpu()

    @torch.no_grad()
    def logits(self, X):
        return self.classifier(self.transport(X).to(self.device)).cpu()

    @torch.no_grad()
    def predict(self, X):
        return self.logits(X).argmax(-1)

    # ------------------------------------------------------------------ #

    def _make_optimizer(self):
        lr = 3e-4 if self.fallback >= 2 else self.cfg["training"]["lr"]
        return torch.optim.AdamW(self.model.parameters(), lr=lr,
                                 weight_decay=self.cfg["training"]["weight_decay"])

    def _step(self, opt, loss):
        if not torch.isfinite(loss):
            raise NonFiniteLoss(f"non-finite loss at fallback level {self.fallback}")
        opt.zero_grad()
        loss.backward()
        if self.fallback >= 1:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        opt.step()

    @torch.no_grad()
    def _validate(self, Xval, yval):
        self.model.eval()
        zhat = self.rollout(Xval)
        logits = self.classifier(zhat)
        return (F.cross_entropy(logits, yval).item(),
                (logits.argmax(-1) == yval).float().mean().item())

    def _better(self, val_acc, val_ce, best):
        """Checkpoint tie-breaks: highest val acc -> lowest val CE -> earliest."""
        if best is None:
            return True
        if val_acc != best["val_acc"]:
            return val_acc > best["val_acc"]
        return val_ce < best["val_ce"]

    def _finish(self, hist, best, best_state):
        self.model.load_state_dict(best_state)
        self.model.eval()
        self.history = hist
        self.best = best
        return self

    # ------------------------------------------------------------------ #

    def fit_strategy1(self, Xtr, ytr, Xval, yval, lam):
        """End-to-end rolled-out classification training (ADR 0008 §6)."""
        cfg = self.cfg["training"]
        dev = self.device
        Z, y = Xtr.float().to(dev), ytr.long().to(dev)
        Xval, yval = Xval.float().to(dev), yval.long().to(dev)
        znorm2 = (Z ** 2).sum(-1) + self.eps
        opt = self._make_optimizer()
        g = torch.Generator().manual_seed(self.seed)
        n, bs = len(Z), cfg["batch_size"]
        hist = {k: [] for k in ("epoch", "train_loss", "train_ce", "train_penalty",
                                "mean_disp", "train_pipeline_ce", "train_pipeline_acc",
                                "val_ce", "val_acc")}
        self.history = hist   # kept current so a failed attempt's partial curve survives
        best, best_state = None, None

        for epoch in range(cfg["epochs"]):
            self.model.train()
            perm = torch.randperm(n, generator=g).to(dev)
            tot = tot_ce = tot_pen = tot_disp = 0.0
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                zhat = self.rollout(Z[idx])
                ce = F.cross_entropy(self.classifier(zhat), y[idx])
                disp2 = ((zhat - Z[idx]) ** 2).sum(-1)
                penalty = (disp2 / znorm2[idx]).mean()
                loss = ce + lam * penalty
                self._step(opt, loss)
                b = len(idx)
                tot += loss.item() * b
                tot_ce += ce.item() * b
                tot_pen += penalty.item() * b
                tot_disp += disp2.detach().sqrt().sum().item()
            tr_ce, tr_acc = self._validate(Z, y)   # end-to-end train metrics (no_grad)
            val_ce, val_acc = self._validate(Xval, yval)
            hist["epoch"].append(epoch)
            hist["train_loss"].append(tot / n)
            hist["train_ce"].append(tot_ce / n)
            hist["train_penalty"].append(tot_pen / n)
            hist["mean_disp"].append(tot_disp / n)
            hist["train_pipeline_ce"].append(tr_ce)
            hist["train_pipeline_acc"].append(tr_acc)
            hist["val_ce"].append(val_ce)
            hist["val_acc"].append(val_acc)
            if self._better(val_acc, val_ce, best):
                best = {"val_acc": val_acc, "val_ce": val_ce, "epoch": epoch}
                best_state = copy.deepcopy(self.model.state_dict())
        return self._finish(hist, best, best_state)

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _project(self, u, z, rho):
        d = u - z
        norm = d.norm(dim=-1, keepdim=True)
        scale = (rho.unsqueeze(-1) / norm.clamp_min(self.eps)).clamp(max=1.0)
        return z + d * scale

    def _build_targets(self, Z, y, beta, m):
        """Per-sample classifier-guided targets (ADR 0008 §8). All quantities
        per sample. Returns (targets, diagnostics-dict)."""
        alpha = self.cfg["strategy2"]["alpha_trust_region"]
        rho = alpha * Z.norm(dim=-1)                       # [N]
        eta = (beta * rho).unsqueeze(-1)                   # [N, 1]

        with torch.no_grad():
            zhat = self.rollout(Z)                         # FM snapshot output
            ce_unproj = F.cross_entropy(self.classifier(zhat), y, reduction="none")

        u = self._project(zhat, Z, rho)                    # u0: projected (R14)
        with torch.no_grad():
            best_ce = F.cross_entropy(self.classifier(u), y, reduction="none")
        best_u = u.clone()
        ce_before = best_ce.clone()                        # CE at projected u0

        for _ in range(m):
            u = u.detach().requires_grad_(True)
            ce = F.cross_entropy(self.classifier(u), y, reduction="sum")
            (grad,) = torch.autograd.grad(ce, u)
            with torch.no_grad():
                step = grad / (grad.norm(dim=-1, keepdim=True) + self.eps)
                u = self._project(u - eta * step, Z, rho)
                ce_i = F.cross_entropy(self.classifier(u), y, reduction="none")
                improved = ce_i < best_ce
                best_ce = torch.where(improved, ce_i, best_ce)
                best_u[improved] = u[improved]

        targets = best_u.detach()
        at_boundary = ((targets - Z).norm(dim=-1) >= rho * (1 - 1e-6))
        diag = {"target_ce_before": ce_before.mean().item(),
                "target_ce_after": best_ce.mean().item(),
                "ce_unprojected_zhat": ce_unproj.mean().item(),
                "hit_rate": at_boundary.float().mean().item(),
                "mean_target_disp": (targets - Z).norm(dim=-1).mean().item()}
        return targets, diag

    def fit_strategy2(self, Xtr, ytr, Xval, yval, beta, m):
        """Classifier-guided targets + standard FM training (ADR 0008 §8).
        No CE gradient ever reaches the FM."""
        cfg = self.cfg["training"]
        dev = self.device
        Z, y = Xtr.float().to(dev), ytr.long().to(dev)
        Xval, yval = Xval.float().to(dev), yval.long().to(dev)
        opt = self._make_optimizer()
        g = torch.Generator().manual_seed(self.seed)
        n, bs = len(Z), cfg["batch_size"]
        hist = {k: [] for k in ("epoch", "fm_loss", "target_ce_before",
                                "target_ce_after", "ce_unprojected_zhat",
                                "hit_rate", "mean_target_disp", "mean_disp",
                                "train_pipeline_ce", "train_pipeline_acc",
                                "val_ce", "val_acc")}
        self.history = hist   # kept current so a failed attempt's partial curve survives
        best, best_state = None, None

        for epoch in range(cfg["epochs"]):
            # phase 1: snapshot FM, build + cache targets for ALL samples
            self.model.eval()
            targets, diag = self._build_targets(Z, y, beta, m)
            # phase 2: one epoch of standard FM against the fixed cache
            self.model.train()
            perm = torch.randperm(n, generator=g).to(dev)
            tot = 0.0
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                z, tgt = Z[idx], targets[idx]
                t = torch.rand(len(idx), generator=g).to(dev)
                zt = (1 - t).unsqueeze(-1) * z + t.unsqueeze(-1) * tgt
                loss = ((self._velocity(zt, t) - (tgt - z)) ** 2).sum(-1).mean()
                self._step(opt, loss)
                tot += loss.item() * len(idx)
            with torch.no_grad():
                self.model.eval()
                mean_disp = (self.rollout(Z) - Z).norm(dim=-1).mean().item()
            tr_ce, tr_acc = self._validate(Z, y)   # end-to-end train metrics, post-update
            val_ce, val_acc = self._validate(Xval, yval)
            hist["epoch"].append(epoch)
            hist["fm_loss"].append(tot / n)
            for k, v in diag.items():
                hist[k].append(v)
            hist["mean_disp"].append(mean_disp)
            hist["train_pipeline_ce"].append(tr_ce)
            hist["train_pipeline_acc"].append(tr_acc)
            hist["val_ce"].append(val_ce)
            hist["val_acc"].append(val_acc)
            if self._better(val_acc, val_ce, best):
                best = {"val_acc": val_acc, "val_ce": val_ce, "epoch": epoch}
                best_state = copy.deepcopy(self.model.state_dict())
        return self._finish(hist, best, best_state)

    # ------------------------------------------------------------------ #

    def fit_joint(self, Xtr, ytr, Xval, yval, classifier_lr):
        """Optional extension (ADR 0008 §9): unfreeze a COPY of the classifier
        and optimize FM + classifier jointly with plain CE (upper reference).
        The pinned probe passed at construction is never mutated. Same policy:
        exactly `epochs` epochs, best-validation checkpoint (both modules),
        NaN/Inf -> NonFiniteLoss."""
        cfg = self.cfg["training"]
        dev = self.device
        self.classifier = copy.deepcopy(self.classifier).to(dev)
        for p in self.classifier.parameters():
            p.requires_grad_(True)
        Z, y = Xtr.float().to(dev), ytr.long().to(dev)
        Xval, yval = Xval.float().to(dev), yval.long().to(dev)
        lr = 3e-4 if self.fallback >= 2 else cfg["lr"]
        opt = torch.optim.AdamW([
            {"params": self.model.parameters(), "lr": lr},
            {"params": self.classifier.parameters(), "lr": classifier_lr}],
            weight_decay=cfg["weight_decay"])
        g = torch.Generator().manual_seed(self.seed)
        n, bs = len(Z), cfg["batch_size"]
        hist = {k: [] for k in ("epoch", "train_ce", "train_pipeline_ce",
                                "train_pipeline_acc", "mean_disp", "val_ce", "val_acc")}
        self.history = hist
        best, best_state = None, None

        for epoch in range(cfg["epochs"]):
            self.model.train()
            self.classifier.train()
            perm = torch.randperm(n, generator=g).to(dev)
            tot = tot_disp = 0.0
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                zhat = self.rollout(Z[idx])
                loss = F.cross_entropy(self.classifier(zhat), y[idx])
                if not torch.isfinite(loss):
                    raise NonFiniteLoss(f"non-finite loss at fallback {self.fallback}")
                opt.zero_grad()
                loss.backward()
                if self.fallback >= 1:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters())
                        + list(self.classifier.parameters()), 1.0)
                opt.step()
                tot += loss.item() * len(idx)
                tot_disp += (zhat - Z[idx]).detach().norm(dim=-1).sum().item()
            self.classifier.eval()
            tr_ce, tr_acc = self._validate(Z, y)
            val_ce, val_acc = self._validate(Xval, yval)
            hist["epoch"].append(epoch)
            hist["train_ce"].append(tot / n)
            hist["train_pipeline_ce"].append(tr_ce)
            hist["train_pipeline_acc"].append(tr_acc)
            hist["mean_disp"].append(tot_disp / n)
            hist["val_ce"].append(val_ce)
            hist["val_acc"].append(val_acc)
            if self._better(val_acc, val_ce, best):
                best = {"val_acc": val_acc, "val_ce": val_ce, "epoch": epoch}
                best_state = {"fm": copy.deepcopy(self.model.state_dict()),
                              "clf": copy.deepcopy(self.classifier.state_dict())}
        self.model.load_state_dict(best_state["fm"])
        self.classifier.load_state_dict(best_state["clf"])
        self.model.eval()
        self.classifier.eval()
        self.history = hist
        self.best = best
        return self

    def save(self, path):
        d = {"state_dict": self.model.state_dict(), "seed": self.seed,
             "fallback": self.fallback, "T": self.T}
        if self.standardize:
            d["mu"], d["sigma"] = self.mu.cpu(), self.sigma.cpu()
        torch.save(d, path)


def load_pinned_probe(path):
    """Rebuild the pinned frozen classifier saved by the Stage-3 runner."""
    d = torch.load(path, weights_only=True)
    clf = torch.nn.Linear(d["dim"], d["n_classes"])
    clf.load_state_dict(d["state_dict"])
    return clf.eval()


def load_stage3_fm(path, classifier):
    """Rebuild a trained Stage3FM from `save()` output + its pinned probe
    (for the figure/notebook code — no retraining)."""
    d = torch.load(path, weights_only=True)
    dim = d["state_dict"]["net.0.weight"].shape[1] - 1
    stats = (d["mu"], d["sigma"]) if "mu" in d else None
    fm = Stage3FM(classifier, dim, seed=d["seed"], fallback=d["fallback"],
                  standardize_stats=stats)
    fm.model.load_state_dict({k: v.to(fm.device) for k, v in d["state_dict"].items()})
    fm.model.eval()
    return fm


def identity_guard(fm, X):
    """ADR 0008 §5: at initialization the pipeline must EQUAL the direct probe —
    features bit-exact, logits equal, predictions identical. Train/val data only."""
    z = X.float()
    zhat = fm.transport(z)
    assert torch.equal(zhat, z), "identity guard: transported features differ"
    with torch.no_grad():
        direct = fm.classifier(z.to(fm.device)).cpu()
        piped = fm.classifier(zhat.to(fm.device)).cpu()
    assert torch.equal(direct, piped), "identity guard: logits differ"
    assert torch.equal(direct.argmax(-1), piped.argmax(-1)), \
        "identity guard: predictions differ"
    return True
