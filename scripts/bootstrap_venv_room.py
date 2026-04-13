#!/usr/bin/env python3
"""
Create .venv and install a matching torch + torchaudio (CUDA 12 / cu124 wheels).

Use when system Python has apt torch (CUDA 12) but pip installed torchaudio
for CUDA 13 → libcudart.so.13 errors.

Usage (from repo root):
  python3 scripts/bootstrap_venv_room.py
  source .venv/bin/activate    # Windows: .venv\\Scripts\\activate
  python scripts/setup_room.py
  python app.py --share --port 7870
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def main() -> None:
    if not VENV.exists():
        print("[venv] Creating .venv ...")
        import venv

        venv.EnvBuilder(with_pip=True).create(VENV)

    py = str(venv_python())
    steps = [
        [py, "-m", "pip", "install", "-U", "pip", "wheel"],
        [py, "-m", "pip", "install", "numpy>=1.26,<2"],
        [
            py,
            "-m",
            "pip",
            "install",
            "torch==2.6.0",
            "torchaudio==2.6.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ],
        [py, "-m", "pip", "install", "-e", str(ROOT)],
        [py, "-m", "pip", "install", "gradio>=4.0"],
    ]
    for cmd in steps:
        print(f"  $ {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(ROOT))

    print()
    print("Done.")
    if sys.platform == "win32":
        print(r"  Activate:  .venv\Scripts\activate")
    else:
        print("  Activate:  source .venv/bin/activate")
    print("  Then:      python scripts/setup_room.py")
    print("  Then:      python app.py --share --port 7870")


if __name__ == "__main__":
    main()
