"""Focused unit tests for the FM transport APIs (forward + reverse).

Plain-Python asserts, no pytest dependency (the prebuilt ROCm venv must not be
pip-installed into). Run directly:

    python tests/test_flow_matching.py        (or `tasks.ps1 tests`)

Synthetic velocity fields (zero / constant) verify the Euler integration loops
independently of any trained checkpoint; the trained-checkpoint behaviour is
exercised by `scripts/make_figures_stage2.py` and the notebook's assertion
cells. A forward-then-reverse reconstruction diagnostic is included WITHOUT an
exactness assertion: reverse Euler is not the exact inverse of the discrete
forward-Euler map, and the learned field need not be bijective.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.flow_matching import FlowMatchingHead

D, C, B, T = 16, 4, 5, 12


class _Const(torch.nn.Module):
    """v_theta(z, t) = c, ignoring z and t."""

    def __init__(self, c):
        super().__init__()
        self.register_buffer("c", c)

    def forward(self, z, t):
        return self.c.expand(len(z), -1)


def make_head(model=None):
    protos = torch.nn.functional.normalize(torch.randn(C, D, generator=g), dim=-1)
    head = FlowMatchingHead(protos, "standard", seed=0)
    if model is not None:
        head.model = model.to(head.device)
    return head


def run():
    checks = []

    def ok(name, cond):
        checks.append((name, bool(cond)))
        print(f"  [{'OK' if cond else 'FAIL'}] {name}")

    z1 = torch.randn(B, D, generator=g)

    # ---- shapes, ordering, determinism, finiteness, gradients ----
    head = make_head()
    z0, traj, times = head.reverse_transport(z1, T, return_traj=True)
    ok("reverse output shape", z0.shape == (B, D))
    ok("reverse trajectory shape [T+1, B, D]", traj.shape == (T + 1, B, D))
    ok("times shape [T+1]", times.shape == (T + 1,))
    ok("times descend 1 -> 0",
       torch.allclose(times, torch.arange(T, -1, -1).float() / T)
       and times[0] == 1.0 and times[-1] == 0.0)
    ok("first reverse state exactly equals the supplied start (prototype)",
       torch.equal(traj[0], z1.float()))
    ok("trajectory endpoint equals returned z0", torch.equal(traj[-1], z0))
    ok("all reverse outputs finite",
       bool(torch.isfinite(traj).all()) and bool(torch.isfinite(z0).all()))
    ok("no gradients retained on reverse outputs",
       not z0.requires_grad and not traj.requires_grad)
    z0_again = head.reverse_transport(z1, T)
    ok("reverse is deterministic (one start -> one trajectory)",
       torch.equal(z0, z0_again))

    # ---- zero velocity field leaves the state unchanged ----
    head = make_head(_Const(torch.zeros(1, D)))
    z0, traj, _ = head.reverse_transport(z1, T, return_traj=True)
    ok("zero field: every state unchanged",
       torch.allclose(traj, z1.float().expand(T + 1, B, D), atol=0))

    # ---- constant field c: z0 = z1 - sum_k (1/T) c = z1 - c ----
    c = torch.randn(1, D, generator=g)
    head = make_head(_Const(c))
    z0 = head.reverse_transport(z1, T)
    ok("constant field: reverse endpoint z1 - c",
       torch.allclose(z0, z1.float() - c, atol=1e-5))
    # and the forward map adds it back: forward(reverse(z1)) = z1 for constant c
    fwd = head.transport(z0, T)
    norm_z0 = torch.nn.functional.normalize(z0, dim=-1)  # forward normalizes input
    ok("constant field: forward re-adds c (on the normalized start)",
       torch.allclose(fwd, norm_z0 + c, atol=1e-5))

    # ---- forward-transport contract (previously untested) ----
    head = make_head()
    zT, ftraj = head.transport(z1, T, return_traj=True)
    ok("forward trajectory shape [T+1, B, D]", ftraj.shape == (T + 1, B, D))
    ok("forward starts from the L2-normalized input",
       torch.allclose(ftraj[0], torch.nn.functional.normalize(z1.float(), dim=-1)))
    ok("forward outputs finite and gradient-free",
       bool(torch.isfinite(ftraj).all()) and not ftraj.requires_grad)
    pred = head.predict(z1, T)
    ok("predict returns one class index per input",
       pred.shape == (B,) and int(pred.max()) < C and int(pred.min()) >= 0)

    # ---- forward-then-reverse reconstruction: DIAGNOSTIC ONLY (no assert on
    # exactness — reverse Euler is not the exact inverse of forward Euler) ----
    zT = head.transport(z1, T)
    zrec = head.reverse_transport(zT, T)
    err = (zrec - torch.nn.functional.normalize(z1.float(), dim=-1)).norm(dim=-1)
    print(f"  [diag] forward-then-reverse reconstruction error: "
          f"mean {err.mean():.4f}, max {err.max():.4f} (not asserted)")

    failed = [n for n, c in checks if not c]
    print(f"\n{len(checks) - len(failed)} / {len(checks)} checks passed")
    if failed:
        sys.exit("FAILED: " + "; ".join(failed))
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)
    run()
