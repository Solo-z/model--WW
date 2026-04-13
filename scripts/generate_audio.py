#!/usr/bin/env python3
"""
Generate audio from MODEL-W session specs using ACE-Step 1.5.

End-to-end pipeline:
  session JSON → caption + metadata → ACE-Step DiT → rendered audio

Usage:
  python scripts/generate_audio.py --sessions synthetic/sessions/corpus_200 --out output/audio
  python scripts/generate_audio.py --session synthetic/sessions/example_trap_fullsong.json
  python scripts/generate_audio.py --caption "dark trap beat, D minor, 140 BPM" --duration 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modelw.acestep_bridge import (
    ACEStepBridge,
    ACEStepConfig,
    session_to_caption,
    preview_captions,
)


def load_env_config() -> dict:
    """Read .env.acestep if it exists."""
    env_file = _ROOT / ".env.acestep"
    cfg = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Generate audio from MODEL-W sessions via ACE-Step")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--sessions", type=str, help="Directory of session JSON files")
    grp.add_argument("--session", type=str, help="Single session JSON file")
    grp.add_argument("--caption", type=str, help="Direct text caption (no session file)")
    grp.add_argument("--preview", type=str, help="Preview captions without generating (no GPU)")

    ap.add_argument("--out", type=str, default="output/audio")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--bpm", type=int, default=120)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--dit", type=str, default=None, help="DiT config override")
    ap.add_argument("--lm", type=str, default=None, help="LM model override")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    if args.preview:
        preview_captions(args.preview, max_files=args.max_files or 20)
        return

    env = load_env_config()
    config = ACEStepConfig(
        acestep_root=env.get("ACESTEP_ROOT", str(_ROOT / "models/ace-step")),
        dit_config=args.dit or env.get("ACESTEP_DIT_CONFIG", "acestep-v15-turbo"),
        lm_model=args.lm or env.get("ACESTEP_LM_MODEL", "acestep-5Hz-lm-1.7B"),
        output_dir=args.out,
    )
    if args.device:
        config.device = args.device

    bridge = ACEStepBridge(config)
    bridge.initialize()

    if args.caption:
        result = bridge.generate_from_caption(
            caption=args.caption,
            bpm=args.bpm,
            duration=args.duration or 30.0,
            batch_size=args.batch_size,
            seed=args.seed,
            save_dir=args.out,
        )
        if result.success:
            for audio in result.audios:
                print(f"Generated: {audio['path']}")
        else:
            print(f"Error: {result.error}")

    elif args.session:
        result = bridge.generate_from_session_file(
            args.session,
            duration=args.duration,
            batch_size=args.batch_size,
            seed=args.seed,
            save_dir=args.out,
        )
        if result.success:
            for audio in result.audios:
                print(f"Generated: {audio['path']}")
        else:
            print(f"Error: {result.error}")

    elif args.sessions:
        bridge.batch_generate_corpus(
            sessions_dir=args.sessions,
            save_dir=args.out,
            max_files=args.max_files,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
