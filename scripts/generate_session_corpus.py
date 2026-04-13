#!/usr/bin/env python3
"""
Generate N synthetic session JSON files (daw_session_spec_v0.3) for training.

Outputs are compatible with modelw.dataset.SessionDataset (tracks + clip_library + arrangement).

Usage:
  python scripts/generate_session_corpus.py --count 200 --out synthetic/sessions --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# repo root on path for optional validation
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Keys as in project.key (tokenizer normalizes to KEY_*)
KEY_DISPLAY = [
    "C major",
    "C minor",
    "C# major",
    "C# minor",
    "D major",
    "D minor",
    "D# major",
    "D# minor",
    "E major",
    "E minor",
    "F major",
    "F minor",
    "F# major",
    "F# minor",
    "G major",
    "G minor",
    "G# major",
    "G# minor",
    "A major",
    "A minor",
    "A# major",
    "A# minor",
    "B major",
    "B minor",
]

# Semitone offset from C for root (MIDI pitch class of tonic)
_KEY_PC = {
    "C major": 0,
    "C minor": 0,
    "C# major": 1,
    "C# minor": 1,
    "D major": 2,
    "D minor": 2,
    "D# major": 3,
    "D# minor": 3,
    "E major": 4,
    "E minor": 4,
    "F major": 5,
    "F minor": 5,
    "F# major": 6,
    "F# minor": 6,
    "G major": 7,
    "G minor": 7,
    "G# major": 8,
    "G# minor": 8,
    "A major": 9,
    "A minor": 9,
    "A# major": 10,
    "A# minor": 10,
    "B major": 11,
    "B minor": 11,
}

STYLES = [
    "trap",
    "reggaeton",
    "house",
    "techno",
    "edm",
    "hiphop",
    "lofi",
    "ambient",
    "pop",
    "rnb",
    "drill",
    "cinematic",
]

MOOD_POOL = [
    ("dark", 0.35),
    ("aggressive", 0.25),
    ("energetic", 0.2),
    ("calm", 0.15),
    ("mysterious", 0.15),
    ("dreamy", 0.12),
    ("melancholic", 0.12),
    ("playful", 0.1),
]

SECTION_LAYOUT = [
    ("intro", 1, 8),
    ("verse", 9, 24),
    ("chorus", 25, 40),
    ("bridge", 41, 56),
    ("outro", 57, 64),
]


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def drum_notes_four_bars(r: random.Random) -> list[dict]:
    """Dense trap-style drums over 4 bars (16 beats) for token count."""
    notes: list[dict] = []
    kick, snare, ch, ohat = 36, 38, 42, 46
    for beat in range(16):
        t = float(beat)
        vel_k = r.randint(100, 127)
        vel_s = r.randint(95, 118)
        vel_h = r.randint(45, 95)
        # kick 1 and 3 in bar (beat % 4 in 0,2)
        if beat % 4 == 0:
            notes.append({"pitch": kick, "start_beat": t, "duration_beat": 0.25, "velocity": vel_k})
        elif beat % 4 == 2:
            notes.append({"pitch": kick, "start_beat": t, "duration_beat": 0.2, "velocity": vel_k - 10})
        # snare on 2 and 4
        if beat % 4 == 1:
            notes.append({"pitch": snare, "start_beat": t, "duration_beat": 0.18, "velocity": vel_s})
        # 8th hats
        notes.append({"pitch": ch, "start_beat": t, "duration_beat": 0.12, "velocity": vel_h})
        notes.append({"pitch": ch, "start_beat": t + 0.5, "duration_beat": 0.1, "velocity": vel_h - 15})
        # open hat every 4 beats
        if beat % 4 == 3:
            notes.append({"pitch": ohat, "start_beat": t + 0.75, "duration_beat": 0.2, "velocity": 85})
    return notes


def bass_notes_four_bars(root_pc: int, r: random.Random) -> list[dict]:
    """808-ish bass: root + fifth patterns, 8th notes."""
    notes: list[dict] = []
    root = 36 + root_pc % 12
    fifth = root + 7
    scale = [0, 3, 5, 7, 10]  # minor-ish walk
    for i in range(32):
        t = i * 0.5
        deg = scale[r.randint(0, len(scale) - 1)]
        pc = (root_pc + deg) % 12
        pitch = 36 + pc
        pitch = max(28, min(pitch, 55))
        notes.append(
            {
                "pitch": int(pitch),
                "start_beat": t,
                "duration_beat": 0.45,
                "velocity": r.randint(85, 115),
            }
        )
    return notes


def lead_notes_four_bars(root_pc: int, r: random.Random) -> list[dict]:
    """Busy lead line (16th-based) for token count."""
    notes: list[dict] = []
    base = 60 + root_pc % 12
    for i in range(64):
        t = i * 0.25
        step = r.choice([-2, 0, 2, 3, 5, 7, 12])
        pitch = base + (step + r.randint(-2, 2))
        pitch = max(55, min(pitch, 95))
        notes.append(
            {
                "pitch": int(pitch),
                "start_beat": t,
                "duration_beat": 0.22,
                "velocity": r.randint(70, 110),
            }
        )
    return notes


def pad_notes_four_bars(root_pc: int, r: random.Random) -> list[dict]:
    """Layered pads + extra tones."""
    notes: list[dict] = []
    r3 = root_pc % 12
    triad = [48 + r3, 52 + r3, 55 + r3, 60 + r3, 64 + r3, 67 + r3]
    for b in range(8):
        t = float(b * 2)
        for j, p in enumerate(triad):
            notes.append(
                {
                    "pitch": p + r.choice([0, 12]),
                    "start_beat": t,
                    "duration_beat": 1.9,
                    "velocity": 50 + j * 3 + r.randint(0, 8),
                }
            )
    return notes


def fx_notes_four_bars(r: random.Random) -> list[dict]:
    """Many short FX hits so sections pass min_seq_len."""
    notes: list[dict] = []
    for i in range(48):
        t = i * 0.25 + r.uniform(0, 0.05)
        notes.append(
            {
                "pitch": r.randint(60, 84),
                "start_beat": t,
                "duration_beat": 0.15,
                "velocity": r.randint(40, 90),
            }
        )
    return notes


def clip_shell(length_bars: int, notes: list[dict]) -> dict:
    return {"type": "midi", "ppq": 480, "timebase": "beats", "length_bars": length_bars, "notes": notes, "cc": []}


def build_session(index: int, r: random.Random) -> dict:
    style = r.choice(STYLES)
    key_disp = r.choice(KEY_DISPLAY)
    root_pc = _KEY_PC[key_disp]
    bpm = r.randint(82, 174)
    title = f"Synthetic {style.title()} — {key_disp} ({index})"

    mood_primary = r.choice(MOOD_POOL)[0]
    _cand = [m for m in MOOD_POOL if m[0] != mood_primary]
    mood_secondary = r.choice(_cand)[0]

    energy = round(r.uniform(0.35, 0.95), 2)
    tension = round(r.uniform(0.25, 0.85), 2)
    density = round(r.uniform(0.35, 0.9), 2)

    lib = {
        "pat_drums_main": clip_shell(4, drum_notes_four_bars(r)),
        "pat_bass_main": clip_shell(4, bass_notes_four_bars(root_pc, r)),
        "pat_lead_main": clip_shell(4, lead_notes_four_bars(root_pc, r)),
        "pat_pad_main": clip_shell(4, pad_notes_four_bars(root_pc, r)),
        "pat_fx_main": clip_shell(4, fx_notes_four_bars(r)),
    }

    timeline_drums = [
        {"type": "midi", "ref": "pat_drums_main", "start_bar": 1, "loop_count": 2},
        {"type": "midi", "ref": "pat_drums_main", "start_bar": 9, "loop_count": 4},
        {"type": "midi", "ref": "pat_drums_main", "start_bar": 25, "loop_count": 4},
        {"type": "midi", "ref": "pat_drums_main", "start_bar": 41, "loop_count": 4},
        {"type": "midi", "ref": "pat_drums_main", "start_bar": 57, "loop_count": 2},
    ]
    timeline_bass = [
        {"type": "midi", "ref": "pat_bass_main", "start_bar": 1, "loop_count": 2},
        {"type": "midi", "ref": "pat_bass_main", "start_bar": 9, "loop_count": 4},
        {"type": "midi", "ref": "pat_bass_main", "start_bar": 25, "loop_count": 4},
        {"type": "midi", "ref": "pat_bass_main", "start_bar": 41, "loop_count": 4},
        {"type": "midi", "ref": "pat_bass_main", "start_bar": 57, "loop_count": 2},
    ]
    timeline_lead = [
        {"type": "midi", "ref": "pat_lead_main", "start_bar": 1, "loop_count": 2},
        {"type": "midi", "ref": "pat_lead_main", "start_bar": 9, "loop_count": 4},
        {"type": "midi", "ref": "pat_lead_main", "start_bar": 25, "loop_count": 4},
        {"type": "midi", "ref": "pat_lead_main", "start_bar": 41, "loop_count": 4},
        {"type": "midi", "ref": "pat_lead_main", "start_bar": 57, "loop_count": 2},
    ]
    timeline_pad = [
        {"type": "midi", "ref": "pat_pad_main", "start_bar": 1, "loop_count": 2},
        {"type": "midi", "ref": "pat_pad_main", "start_bar": 9, "loop_count": 4},
        {"type": "midi", "ref": "pat_pad_main", "start_bar": 25, "loop_count": 4},
        {"type": "midi", "ref": "pat_pad_main", "start_bar": 41, "loop_count": 4},
        {"type": "midi", "ref": "pat_pad_main", "start_bar": 57, "loop_count": 2},
    ]
    timeline_fx = [
        {"type": "midi", "ref": "pat_fx_main", "start_bar": 1, "loop_count": 2},
        {"type": "midi", "ref": "pat_fx_main", "start_bar": 9, "loop_count": 4},
        {"type": "midi", "ref": "pat_fx_main", "start_bar": 25, "loop_count": 4},
        {"type": "midi", "ref": "pat_fx_main", "start_bar": 41, "loop_count": 4},
        {"type": "midi", "ref": "pat_fx_main", "start_bar": 57, "loop_count": 2},
    ]

    def track_block(tid: str, name: str, role: str, timeline: list[dict]) -> dict:
        return {
            "track_id": tid,
            "name": name,
            "role": role,
            "semantic_labels": {
                "mood": [[mood_primary, 0.4], [mood_secondary, 0.25], ["energetic", 0.15], ["calm", 0.1]],
                "timbre": [["warm", 0.3], ["bright", 0.25], ["wide", 0.25], ["punchy", 0.2]],
                "envelope": [["stable", 0.4], ["short_decay", 0.3], ["long_release", 0.2], ["swell", 0.1]],
                "space": [["stereo", 0.35], ["mono", 0.25], ["close", 0.2], ["far", 0.2]],
            },
            "instrument": {
                "plugin": "Synth.Generic",
                "preset": f"{role}.Generated",
                "macro_params": {"drive": {"__default__": 0.0, "__value__": round(r.uniform(0.0, 0.2), 2)}},
            },
            "mix": {
                "gain_db": {"__default__": 0.0, "__value__": 0.0},
                "volume_db": {"__default__": -6.0, "__value__": -6.0},
                "pan": {"__default__": 0.0, "__value__": 0.0},
                "mute": {"__default__": False, "__value__": False},
                "solo": {"__default__": False, "__value__": False},
            },
            "insert_fx": [],
            "sends": {},
            "timeline": timeline,
            "automation": [],
        }

    return {
        "session_id": f"synth_corpus_{index:04d}",
        "schema_version": "daw_session_spec_v0.3",
        "metadata": {
            "title": title,
            "prompt": f"{style} track, {key_disp}, {bpm} bpm, generated corpus",
            "style": style,
            "duration_bars": 64,
            "created_by": "scripts/generate_session_corpus.py",
        },
        "project": {
            "sample_rate_hz": 48000,
            "bit_depth": "24bit",
            "tempo_map": [{"bar": 1, "bpm": bpm}],
            "time_signature_map": [{"bar": 1, "numerator": 4, "denominator": 4}],
            "swing": {"__default__": 0.0, "__value__": round(r.uniform(0.0, 0.06), 3)},
            "key": key_disp,
            "render": {
                "normalize": {"__default__": False, "__value__": False},
                "dither": {"__default__": "none", "__value__": "none"},
            },
        },
        "semantic_song_labels": {
            "mood": [[mood_primary, 0.45], [mood_secondary, 0.3], ["energetic", 0.15], ["calm", 0.1]],
            "energy": {"__default__": 0.5, "__value__": energy},
            "tension": {"__default__": 0.5, "__value__": tension},
            "density": {"__default__": 0.5, "__value__": density},
        },
        "arrangement": {"sections": [{"name": n, "bar_start": a, "bar_end": b} for n, a, b in SECTION_LAYOUT]},
        "libraries": {"clip_library": lib, "automation_library": {}},
        "routing": {
            "buses": [
                {
                    "bus_id": "BUS_VERB",
                    "name": "Room",
                    "plugin_chain": [
                        {
                            "plugin": "FX.Reverb",
                            "params": {
                                "wet": {"__default__": 0.2, "__value__": 0.22},
                                "decay_s": {"__default__": 1.0, "__value__": 1.1},
                            },
                        }
                    ],
                }
            ],
            "sidechain": [{"from": "T1", "to": "T2", "type": "compressor", "amount": 0.5}],
        },
        "tracks": [
            track_block("T1", "Drums", "drums", timeline_drums),
            track_block("T2", "Bass", "bass", timeline_bass),
            track_block("T3", "Lead", "lead", timeline_lead),
            track_block("T4", "Pad", "pad", timeline_pad),
            track_block("T5", "FX", "fx", timeline_fx),
        ],
        "master": {
            "target_lufs": -9.0,
            "plugin_chain": [
                {
                    "plugin": "FX.Limiter",
                    "params": {"ceiling_db": {"__default__": -1.0, "__value__": -0.8}},
                }
            ],
        },
        "focus": {
            "touch_a_lot": ["groove", "low end"],
            "touch_some": ["width"],
            "leave_default": ["tempo ramps"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("synthetic/sessions"))
    ap.add_argument("--prefix", type=str, default="session_gen")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validate", action="store_true", help="Smoke-test SessionDataset on first file")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for i in range(1, args.count + 1):
        r = _rng(args.seed + i * 7919)
        session = build_session(i, r)
        path = args.out / f"{args.prefix}_{i:03d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2, ensure_ascii=False)
        if i == 1 or i == args.count:
            print(f"Wrote {path}")

    if args.validate:
        from modelw.dataset import SessionDataset, SessionDatasetConfig
        from modelw.tokenizer import MIDITokenizer, TokenizerConfig

        tok = MIDITokenizer(TokenizerConfig())
        first = args.out / f"{args.prefix}_001.json"
        cfg = SessionDatasetConfig(
            sessions_dir=str(args.out.resolve()),
            cache_dir=str((_ROOT / "cache/sessions_validate").resolve()),
            max_files=1,
            train_split=1.0,
        )
        ds = SessionDataset(cfg, tokenizer=tok, split="train", preprocess=True)
        print(f"Validation: {len(ds)} samples from {first.name}")

    print(f"Done: {args.count} files in {args.out}")


if __name__ == "__main__":
    main()
