#!/usr/bin/env python3
"""
Auto-run on HF Spaces startup: clone ACE-Step and OpenVoice, download checkpoints.
Called from app.py before the Gradio UI launches.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"


def run(cmd, **kw):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kw)


def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *args], check=False)


def setup():
    MODELS.mkdir(exist_ok=True)

    # ── ACE-Step ──────────────────────────────────────────────────────
    acestep_dir = MODELS / "ace-step"
    if not (acestep_dir / "pyproject.toml").exists():
        print("[setup] Cloning ACE-Step 1.5...")
        run(["git", "clone", "https://github.com/ACE-Step/ACE-Step-1.5.git", str(acestep_dir)])
        pyproject = acestep_dir / "pyproject.toml"
        text = pyproject.read_text()
        if ">=3.11,<3.13" in text:
            pyproject.write_text(text.replace(">=3.11,<3.13", ">=3.10,<3.14"))
    else:
        print("[setup] ACE-Step already present.")

    pip("-e", str(acestep_dir), "--no-deps")

    # Patch sys.path so acestep is importable
    if str(acestep_dir) not in sys.path:
        sys.path.insert(0, str(acestep_dir))

    # ── OpenVoice ─────────────────────────────────────────────────────
    openvoice_dir = MODELS / "openvoice"
    if not openvoice_dir.exists():
        print("[setup] Cloning OpenVoice...")
        run(["git", "clone", "https://github.com/myshell-ai/OpenVoice.git", str(openvoice_dir)])
    pip("-e", str(openvoice_dir), "--no-deps")

    ckpt_dir = openvoice_dir / "checkpoints_v2"
    if not (ckpt_dir / "converter" / "checkpoint.pth").exists():
        print("[setup] Downloading OpenVoice V2 checkpoints...")
        from huggingface_hub import snapshot_download
        snapshot_download("myshell-ai/OpenVoiceV2", local_dir=str(ckpt_dir))
    else:
        print("[setup] OpenVoice checkpoints present.")

    print("[setup] Done.")


if __name__ == "__main__":
    setup()
