"""GPU smoke test + runtime provenance dump. Run first (tasks.ps1 setup)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import dump_runtime_summary, gpu_smoke_test


def main():
    info = gpu_smoke_test()
    print(f"GPU smoke test: {info['device_name']} — matmul OK")
    summary = dump_runtime_summary(extra=info)
    print("runtime_summary.json written:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
