#!/usr/bin/env python3
"""Smoke-test imports after setup_room / bootstrap. Run inside activated .venv."""

from __future__ import annotations

import sys


def main() -> None:
    try:
        import loguru  # noqa: F401

        print(" OK  loguru")
        import torch

        print(f" OK  torch {torch.__version__} cuda={torch.cuda.is_available()}")
        import torchaudio  # noqa: F401

        print(" OK  torchaudio")
        import transformers  # noqa: F401

        print(" OK  transformers")
        import acestep.gpu_config  # noqa: F401

        print(" OK  acestep.gpu_config")
    except Exception as e:
        print(f" FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    print("\nAll checks passed. Try: python app.py --share --port 7870")


if __name__ == "__main__":
    main()
