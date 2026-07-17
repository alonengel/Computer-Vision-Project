# Lab Notebook — Stage 1

Chronological record of every step: what was done, why, exact commands, errors and fixes.

---

## 2026-07-17 — Repo moved to D:, scaffold

**Repo relocation.** The project directory was moved from `C:\Users\Alon\Desktop\Computer-Vision-Project` to `D:\Computer-Vision-Project` (361 GB free on D:) so datasets and cached features can live next to the code. All code uses paths relative to the repo root (`src/utils.py:repo_path`), so nothing else changed.

**Environment verification.** Reusing the prebuilt ROCm venv (never pip-install into it):

```
C:\Users\Alon\Desktop\cv-ex2\rocm_win312\Scripts\python.exe -c "import torch; ..."
→ Python 3.12.3 | torch 2.9.1+rocm7.2.1 | cuda available | AMD Radeon RX 7900 XTX
```

**Scaffold created.** `config/config.json` (all tunables), `src/utils.py` (seeding, device, ROCm guards, provenance dumps), `tasks.ps1` (one-word task runner pinned to the venv interpreter), README, .gitignore, review agents (`.claude/agents/cv-expert.md`, `critical-reviewer.md`), ADRs 0001–0002.

**Known Windows/ROCm quirks carried over from cv-ex2** (guards already in place):
- `KMP_DUPLICATE_LIB_OK=TRUE` before torch import — otherwise the `clip` package triggers an OpenMP duplicate-runtime crash.
- CLIP model forced to `.float()` — fp16 weights misbehave on the ROCm stack.
- SSL unverified-context fallback available for dataset downloads (`utils.allow_insecure_downloads`).
- Free GPU memory (`torch.cuda.empty_cache()`) between backbones during extraction.
