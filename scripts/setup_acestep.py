#!/usr/bin/env python3
"""
Set up ACE-Step 1.5 as MODEL-W's audio foundation model.

Steps:
  1. Clone the ACE-Step repo into models/ace-step/
  2. Install it as an editable package
  3. Verify imports work
  4. (Optional) Download model weights on first run

Usage:
  python scripts/setup_acestep.py
  python scripts/setup_acestep.py --skip-install   # if you already pip-installed
  python scripts/setup_acestep.py --dit xl-turbo   # pick a different DiT variant
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/ACE-Step/ACE-Step-1.5.git"
DEFAULT_TARGET = "models/ace-step"

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: str | None = None, check: bool = True):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=check)


def main():
    ap = argparse.ArgumentParser(description="Set up ACE-Step 1.5 for MODEL-W")
    ap.add_argument("--target", type=str, default=DEFAULT_TARGET,
                    help="Where to clone ACE-Step (relative to repo root)")
    ap.add_argument("--skip-install", action="store_true",
                    help="Skip pip install (if already installed)")
    ap.add_argument("--dit", type=str, default="turbo",
                    choices=["base", "sft", "turbo", "xl-base", "xl-sft", "xl-turbo"],
                    help="Which DiT variant to configure")
    ap.add_argument("--lm", type=str, default="1.7B",
                    choices=["0.6B", "1.7B", "4B"],
                    help="Which LM model size")
    args = ap.parse_args()

    target = ROOT / args.target

    # 1. Clone if needed
    if not (target / "pyproject.toml").exists():
        print(f"\n[1/4] Cloning ACE-Step 1.5 → {target}")
        target.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", REPO_URL, str(target)])
    else:
        print(f"\n[1/4] ACE-Step already present at {target}")

    # 2. Install
    if not args.skip_install:
        print(f"\n[2/4] Installing ACE-Step as editable package")
        run([sys.executable, "-m", "pip", "install", "-e", str(target)])
    else:
        print("\n[2/4] Skipping install (--skip-install)")

    # 3. Verify imports
    print("\n[3/4] Verifying ACE-Step imports")
    try:
        import acestep  # noqa: F401
        from acestep.handler import AceStepHandler  # noqa: F401
        from acestep.inference import GenerationParams  # noqa: F401
        print("  ✓ acestep imports OK")
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        print("  You may need to install manually: pip install -e models/ace-step")
        sys.exit(1)

    # 4. Write config hint
    dit_name = f"acestep-v15-{args.dit.replace('-', '-')}"
    lm_name = f"acestep-5Hz-lm-{args.lm}"

    config_path = ROOT / ".env.acestep"
    with open(config_path, "w") as f:
        f.write(f"ACESTEP_ROOT={target}\n")
        f.write(f"ACESTEP_DIT_CONFIG={dit_name}\n")
        f.write(f"ACESTEP_LM_MODEL={lm_name}\n")

    print(f"\n[4/4] Config written to {config_path}")
    print(f"  DiT: {dit_name}")
    print(f"  LM:  {lm_name}")

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  ACE-Step 1.5 is ready as MODEL-W's audio foundation model  ║
║                                                               ║
║  Models auto-download on first generation run.               ║
║                                                               ║
║  Quick test:                                                  ║
║    python -m modelw.acestep_bridge synthetic/sessions/corpus_200  ║
║                                                               ║
║  Generate audio from sessions:                               ║
║    python scripts/generate_audio.py \\                         ║
║      --sessions synthetic/sessions/corpus_200 \\               ║
║      --out output/audio                                       ║
╚═══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
