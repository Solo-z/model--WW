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
    # Suppress git/subprocess output — keeps internal repo URLs out of logs
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)


def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *args],
                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def setup():
    MODELS.mkdir(exist_ok=True)

    print("[ROOM] Preparing engine…")

    # ── Audio model ───────────────────────────────────────────────────
    acestep_dir = MODELS / "ace-step"
    if not (acestep_dir / "pyproject.toml").exists():
        run(["git", "clone", "https://github.com/ACE-Step/ACE-Step-1.5.git", str(acestep_dir)])
        pyproject = acestep_dir / "pyproject.toml"
        text = pyproject.read_text()
        if ">=3.11,<3.13" in text:
            pyproject.write_text(text.replace(">=3.11,<3.13", ">=3.10,<3.14"))

    pip("-e", str(acestep_dir), "--no-deps")

    if str(acestep_dir) not in sys.path:
        sys.path.insert(0, str(acestep_dir))

    # ── Voice model ───────────────────────────────────────────────────
    openvoice_dir = MODELS / "openvoice"
    if not openvoice_dir.exists():
        run(["git", "clone", "https://github.com/myshell-ai/OpenVoice.git", str(openvoice_dir)])
    pip("-e", str(openvoice_dir), "--no-deps")

    ckpt_dir = openvoice_dir / "checkpoints_v2"
    if not (ckpt_dir / "converter" / "checkpoint.pth").exists():
        # Suppress hub progress bars in case env var wasn't read in time
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from huggingface_hub import snapshot_download
        snapshot_download("myshell-ai/OpenVoiceV2", local_dir=str(ckpt_dir),
                          tqdm_class=None)

    # ── Stem separator + MIDI transcriber — preload weights ──────────
    # If we don't pre-download these, the first Generate call hits a
    # cold model fetch inside the GPU window and may time out silently.
    try:
        from demucs.pretrained import get_model
        get_model("htdemucs")
    except Exception as _e:
        print(f"[ROOM] stem model preload skipped: {_e}")

    try:
        # basic-pitch ships its model with the package, no network needed,
        # but importing it once warms up the import graph.
        from basic_pitch.inference import predict  # noqa: F401
    except Exception as _e:
        print(f"[ROOM] midi model preload skipped: {_e}")

    print("[ROOM] Engine ready.")


if __name__ == "__main__":
    setup()
