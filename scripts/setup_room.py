#!/usr/bin/env python3
"""
ROOM Setup — Download and install all model components.

Pulls:
  1. ACE-Step 1.5   (text → audio)
  2. OpenVoice V2   (voice cloning)
  3. Demucs         (stem separation)
  4. BasicPitch     (audio → MIDI)

Usage:
  python scripts/setup_room.py
  python scripts/setup_room.py --skip-acestep   (if already installed)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd=None, check=True):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=check)


def pip(*args):
    run([sys.executable, "-m", "pip", "install", *args])


def main():
    ap = argparse.ArgumentParser(description="Set up all ROOM model components")
    ap.add_argument("--skip-acestep", action="store_true")
    ap.add_argument("--skip-openvoice", action="store_true")
    ap.add_argument("--skip-demucs", action="store_true")
    ap.add_argument("--skip-basicpitch", action="store_true")
    args = ap.parse_args()

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    # ── 1. ACE-Step 1.5 ───────────────────────────────────────────────
    if not args.skip_acestep:
        acestep_dir = models_dir / "ace-step"
        print("\n[1/4] ACE-Step 1.5")
        if not (acestep_dir / "pyproject.toml").exists():
            print("  Cloning...")
            run(["git", "clone", "https://github.com/ACE-Step/ACE-Step-1.5.git", str(acestep_dir)])
        else:
            print("  Already cloned.")

        # Fix Python version pin
        pyproject = acestep_dir / "pyproject.toml"
        text = pyproject.read_text()
        if ">=3.11,<3.13" in text:
            pyproject.write_text(text.replace(">=3.11,<3.13", ">=3.10,<3.14"))
            print("  Fixed Python version pin.")

        pip("-e", str(acestep_dir), "--no-deps")
        print("  ACE-Step ready. Weights auto-download on first run.")
    else:
        print("\n[1/4] ACE-Step — skipped")

    # ── 2. OpenVoice V2 ──────────────────────────────────────────────
    if not args.skip_openvoice:
        openvoice_dir = models_dir / "openvoice"
        print("\n[2/4] OpenVoice V2")
        if not (openvoice_dir / "setup.py").exists() and not (openvoice_dir / "pyproject.toml").exists():
            print("  Cloning...")
            run(["git", "clone", "https://github.com/myshell-ai/OpenVoice.git", str(openvoice_dir)])
        else:
            print("  Already cloned.")

        pip("-e", str(openvoice_dir), "--no-deps")

        # Download checkpoints
        ckpt_dir = openvoice_dir / "checkpoints_v2"
        if not ckpt_dir.exists():
            print("  Downloading OpenVoice V2 checkpoints...")
            pip("huggingface-hub")
            run([
                sys.executable, "-c",
                "from huggingface_hub import snapshot_download; "
                f"snapshot_download('myshell-ai/OpenVoiceV2', local_dir='{ckpt_dir}')"
            ])
        else:
            print("  Checkpoints already downloaded.")

        # Install OpenVoice deps
        pip("librosa", "wavmark", "whisper-timestamped", "pydub")
        print("  OpenVoice ready.")
    else:
        print("\n[2/4] OpenVoice — skipped")

    # ── 3. Demucs ────────────────────────────────────────────────────
    if not args.skip_demucs:
        print("\n[3/4] Demucs")
        pip("demucs")
        print("  Demucs ready. Model downloads on first use (~80MB).")
    else:
        print("\n[3/4] Demucs — skipped")

    # ── 4. BasicPitch ────────────────────────────────────────────────
    if not args.skip_basicpitch:
        print("\n[4/4] BasicPitch")
        pip("basic-pitch")
        print("  BasicPitch ready.")
    else:
        print("\n[4/4] BasicPitch — skipped")

    # ── Config file ──────────────────────────────────────────────────
    env_path = ROOT / ".env.room"
    with open(env_path, "w") as f:
        f.write(f"ACESTEP_ROOT={models_dir / 'ace-step'}\n")
        f.write(f"OPENVOICE_ROOT={models_dir / 'openvoice'}\n")
        f.write(f"DEMUCS_MODEL=htdemucs\n")

    print(f"""
==========================================================
  ROOM Setup Complete
==========================================================

  ACE-Step 1.5 : {'READY' if not args.skip_acestep else 'SKIPPED'}
  OpenVoice V2 : {'READY' if not args.skip_openvoice else 'SKIPPED'}
  Demucs       : {'READY' if not args.skip_demucs else 'SKIPPED'}
  BasicPitch   : {'READY' if not args.skip_basicpitch else 'SKIPPED'}

  Config: {env_path}

  Quick test:
    python -m modelw.room "piano ballad, E minor, emotional" --stems --midi

  Full pipeline with voice cloning:
    python -m modelw.room "R&B love song, my style" --voice my_voice.wav --stems --midi

==========================================================
""")


if __name__ == "__main__":
    main()
