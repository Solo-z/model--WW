"""
ACE-Step Bridge — connects MODEL-W's MIDI/session pipeline to ACE-Step 1.5.

Architecture:
  text prompt → ACE-Step LM planner → session spec / caption
  MODEL-W session spec → MIDI tokens → refined MIDI
  MIDI + caption → ACE-Step DiT → rendered audio

This module provides:
  - ACEStepBridge: high-level class wiring both systems
  - session_to_caption: convert a MODEL-W session JSON to an ACE-Step caption
  - midi_to_reference_audio: render MIDI to a temp wav for ACE-Step conditioning
  - generate_from_session: full pipeline session → audio
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add ACE-Step source to path if cloned locally
_ACESTEP_SRC = Path(__file__).resolve().parents[1] / "models" / "ace-step"
if _ACESTEP_SRC.exists() and str(_ACESTEP_SRC) not in sys.path:
    sys.path.insert(0, str(_ACESTEP_SRC))

try:
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music
    ACESTEP_AVAILABLE = True
except ImportError:
    ACESTEP_AVAILABLE = False


@dataclass
class ACEStepConfig:
    """Configuration for the ACE-Step integration."""

    # ACE-Step model paths (auto-downloaded on first use)
    acestep_root: str = "./models/ace-step"
    dit_config: str = "acestep-v15-turbo"
    lm_model: str = "acestep-5Hz-lm-1.7B"
    lm_backend: str = "vllm"

    # Generation defaults
    default_duration: float = 30.0
    default_batch_size: int = 1
    inference_steps: int = 8
    guidance_scale: float = 7.0
    audio_format: str = "flac"

    # Hardware
    device: str = "cuda"

    # Output
    output_dir: str = "./output/audio"


# ── Caption builder ────────────────────────────────────────────────────────

_STYLE_DESCRIPTORS = {
    "trap": "trap hip-hop with heavy 808 bass and rolling hi-hats",
    "reggaeton": "reggaeton with dembow rhythm and Latin percussion",
    "house": "house music with four-on-the-floor kick and synth stabs",
    "techno": "driving techno with industrial textures",
    "edm": "energetic EDM with big drops and synth leads",
    "hiphop": "hip-hop beat with boom-bap drums and deep bass",
    "lofi": "lo-fi chill beat with vinyl crackle and mellow keys",
    "ambient": "ambient soundscape with atmospheric pads",
    "pop": "pop production with catchy hooks and clean mix",
    "rnb": "smooth R&B with soulful chords and groove",
    "drill": "UK drill with sliding 808s and dark melodies",
    "cinematic": "cinematic orchestral score with dramatic tension",
}


def session_to_caption(session: dict) -> str:
    """Convert a MODEL-W session JSON to a rich ACE-Step text caption."""
    meta = session.get("metadata", {})
    project = session.get("project", {})
    labels = session.get("semantic_song_labels", {})

    style = meta.get("style", "electronic")
    key_disp = project.get("key", "")
    tempo_map = project.get("tempo_map", [{}])
    bpm = tempo_map[0].get("bpm", 120) if tempo_map else 120

    mood_list = labels.get("mood", [])
    moods = ", ".join(m[0] for m in mood_list[:3]) if mood_list else "neutral"

    energy = labels.get("energy", {}).get("__value__", 0.5)
    energy_word = "high-energy" if energy > 0.7 else "moderate-energy" if energy > 0.4 else "low-energy"

    style_desc = _STYLE_DESCRIPTORS.get(style, f"{style} music")

    parts = [
        style_desc,
        f"in {key_disp}" if key_disp else "",
        f"at {bpm} BPM",
        f"{energy_word}",
        f"mood: {moods}",
        "instrumental",
    ]
    return ", ".join(p for p in parts if p)


def session_to_params(session: dict, duration: Optional[float] = None) -> dict:
    """Extract ACE-Step GenerationParams fields from a session spec."""
    project = session.get("project", {})
    tempo_map = project.get("tempo_map", [{}])
    bpm = tempo_map[0].get("bpm", 120) if tempo_map else 120

    key_disp = project.get("key", "")
    key_scale = key_disp.replace(" ", " ") if key_disp else ""

    dur_bars = session.get("metadata", {}).get("duration_bars", 64)
    if duration is None:
        duration = (dur_bars * 4 * 60.0) / bpm

    return {
        "caption": session_to_caption(session),
        "bpm": int(bpm),
        "keyscale": key_scale,
        "duration": float(duration),
        "instrumental": True,
    }


# ── Bridge class ───────────────────────────────────────────────────────────

class ACEStepBridge:
    """
    High-level bridge between MODEL-W sessions and ACE-Step audio generation.

    Usage:
        bridge = ACEStepBridge(ACEStepConfig())
        bridge.initialize()
        results = bridge.generate_from_session(session_json)
    """

    def __init__(self, config: Optional[ACEStepConfig] = None):
        self.config = config or ACEStepConfig()
        self.dit_handler = None
        self.llm_handler = None
        self._initialized = False

    def initialize(self):
        """Load ACE-Step handlers. Call once before generation."""
        if not ACESTEP_AVAILABLE:
            raise ImportError(
                "ACE-Step is not installed. Run:\n"
                "  git clone https://github.com/ACE-Step/ACE-Step-1.5.git models/ace-step\n"
                "  pip install -e models/ace-step\n"
                "Or run: python scripts/setup_acestep.py"
            )

        self.dit_handler = AceStepHandler()
        self.dit_handler.initialize_service(
            project_root=self.config.acestep_root,
            config_path=self.config.dit_config,
            device=self.config.device,
        )

        self.llm_handler = LLMHandler()
        self.llm_handler.initialize(
            checkpoint_dir=self.config.acestep_root,
            lm_model_path=self.config.lm_model,
            backend=self.config.lm_backend,
            device=self.config.device,
        )
        self._initialized = True
        print(f"[ACEStepBridge] Initialized: DiT={self.config.dit_config}, LM={self.config.lm_model}")

    def generate_from_caption(
        self,
        caption: str,
        bpm: int = 120,
        duration: float = 30.0,
        keyscale: str = "",
        batch_size: int = 1,
        seed: int = -1,
        save_dir: Optional[str] = None,
    ) -> dict:
        """Generate audio directly from a text caption."""
        self._check_initialized()

        params = GenerationParams(
            caption=caption,
            bpm=bpm,
            keyscale=keyscale,
            duration=duration,
            inference_steps=self.config.inference_steps,
            guidance_scale=self.config.guidance_scale,
            seed=seed,
            instrumental=True,
        )
        config = GenerationConfig(
            batch_size=batch_size,
            audio_format=self.config.audio_format,
        )

        save = save_dir or self.config.output_dir
        os.makedirs(save, exist_ok=True)
        return generate_music(self.dit_handler, self.llm_handler, params, config, save_dir=save)

    def generate_from_session(
        self,
        session: dict,
        duration: Optional[float] = None,
        batch_size: int = 1,
        seed: int = -1,
        save_dir: Optional[str] = None,
    ) -> dict:
        """Full pipeline: MODEL-W session spec → ACE-Step audio."""
        self._check_initialized()

        sp = session_to_params(session, duration=duration)
        return self.generate_from_caption(
            caption=sp["caption"],
            bpm=sp["bpm"],
            duration=sp["duration"],
            keyscale=sp["keyscale"],
            batch_size=batch_size,
            seed=seed,
            save_dir=save_dir,
        )

    def generate_from_session_file(
        self,
        session_path: str | Path,
        **kwargs,
    ) -> dict:
        """Load a session JSON file and generate audio."""
        with open(session_path, encoding="utf-8") as f:
            session = json.load(f)
        return self.generate_from_session(session, **kwargs)

    def batch_generate_corpus(
        self,
        sessions_dir: str | Path,
        save_dir: str | Path,
        max_files: Optional[int] = None,
        batch_size: int = 1,
    ) -> list[dict]:
        """Generate audio for every session JSON in a directory."""
        self._check_initialized()

        sessions_dir = Path(sessions_dir)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(sessions_dir.glob("*.json"))
        if max_files:
            files = files[:max_files]

        results = []
        for i, f in enumerate(files):
            print(f"[{i + 1}/{len(files)}] {f.name}")
            try:
                result = self.generate_from_session_file(
                    f,
                    save_dir=str(save_dir / f.stem),
                    batch_size=batch_size,
                )
                results.append({"file": str(f), "success": True, "result": result})
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({"file": str(f), "success": False, "error": str(e)})

        succeeded = sum(1 for r in results if r["success"])
        print(f"\nDone: {succeeded}/{len(results)} sessions rendered to audio in {save_dir}")
        return results

    def _check_initialized(self):
        if not self._initialized:
            raise RuntimeError("Call bridge.initialize() before generating.")


# ── Standalone caption preview (no GPU needed) ─────────────────────────────

def preview_captions(sessions_dir: str, max_files: int = 5):
    """Print ACE-Step captions that would be generated from session files."""
    for f in sorted(Path(sessions_dir).glob("*.json"))[:max_files]:
        with open(f, encoding="utf-8") as fh:
            session = json.load(fh)
        caption = session_to_caption(session)
        print(f"{f.name}:")
        print(f"  {caption}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        preview_captions(sys.argv[1], max_files=int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    else:
        print("Usage: python -m modelw.acestep_bridge <sessions_dir> [max_files]")
