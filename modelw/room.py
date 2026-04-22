"""
ROOM Engine — Unified music production pipeline.

Orchestrates:
  - ACE-Step 1.5  (text → full audio)
  - OpenVoice V2  (voice cloning / timbre swap)
  - Demucs        (stem separation)
  - BasicPitch    (audio → MIDI transcription)

Usage:
    engine = RoomEngine(RoomConfig())
    engine.initialize()

    # Full pipeline: prompt + voice → audio with your voice + stems + MIDI
    result = engine.generate(
        prompt="piano ballad, E minor, 70 BPM, emotional",
        voice_ref="my_voice.wav",
        split_stems=True,
        extract_midi=True,
    )
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
_ACESTEP_SRC = _ROOT / "models" / "ace-step"
if _ACESTEP_SRC.exists() and str(_ACESTEP_SRC) not in sys.path:
    sys.path.insert(0, str(_ACESTEP_SRC))


def normalize_voice_ref_path(voice_ref: object) -> Optional[str]:
    """
    Resolve a voice file path from Gradio uploads.

    Gradio versions may pass a filesystem path (str), a dict with ``path`` / ``name``,
    or legacy tuple shapes — if we mis-handle this, cloning is skipped silently.
    """
    if voice_ref is None:
        return None
    if isinstance(voice_ref, str):
        p = voice_ref.strip()
        return p if p and os.path.isfile(p) else None
    if isinstance(voice_ref, dict):
        p = voice_ref.get("path") or voice_ref.get("name")
        if isinstance(p, str):
            p = p.strip()
            return p if p and os.path.isfile(p) else None
        return None
    if isinstance(voice_ref, (list, tuple)) and len(voice_ref) > 0:
        return normalize_voice_ref_path(voice_ref[0])
    return None


def caption_with_vocals_for_voice_clone(caption: str) -> str:
    """
    OpenVoice swaps timbre on *existing* singing/speech in the generated mix.
    If ACE-Step returns an instrumental, there is almost nothing to clone.
    """
    t = (caption or "").strip()
    if not t:
        return "song with lead vocals, singing, emotional delivery"
    low = t.lower()
    hints = (
        "vocal",
        "sing",
        "singing",
        "voice",
        "lyrics",
        "choir",
        "rap",
        "spoken",
        "lyric",
    )
    if any(h in low for h in hints):
        return t
    return f"{t}, prominent lead vocals, singing, close-mic vocal take"


@dataclass
class RoomConfig:
    """Configuration for the full ROOM pipeline."""

    # ACE-Step
    acestep_root: str = str(_ROOT / "models" / "ace-step")
    dit_config: str = "acestep-v15-turbo"
    lm_model: str = "acestep-5Hz-lm-1.7B"
    # "pt" avoids nano-vllm (fewer pip landmines on fresh venvs). Use "vllm" if you install it.
    lm_backend: str = "pt"

    # OpenVoice
    openvoice_root: str = str(_ROOT / "models" / "openvoice")
    # OpenVoice demos use ``cuda:0``; plain ``cuda`` can confuse some loads.
    openvoice_device: str = "cuda:0"

    # Demucs
    demucs_model: str = "htdemucs"

    # Output
    output_dir: str = str(_ROOT / "output")
    audio_format: str = "wav"

    # Hardware
    device: str = "cuda"


class RoomEngine:
    """
    Unified orchestrator for ROOM's multi-model music production pipeline.

    Lazily loads each sub-model only when needed for a given request.
    """

    def __init__(self, config: Optional[RoomConfig] = None):
        self.config = config or RoomConfig()
        self._acestep_dit = None
        self._acestep_llm = None
        self._openvoice = None
        self._demucs_model = None
        self._basicpitch = None
        self._initialized = False
        self._last_stems_error: Optional[str] = None
        self._last_midi_error: Optional[str] = None

    # ── Lazy model loading ─────────────────────────────────────────────

    def _ace_step_checkpoint_dir(self) -> str:
        """Match ``InitServiceOrchestratorMixin.initialize_service`` checkpoint paths."""
        if os.environ.get("ACESTEP_CHECKPOINTS_DIR"):
            from acestep.model_downloader import get_checkpoints_dir

            return str(get_checkpoints_dir())
        root = (self.config.acestep_root or "").strip()
        if root:
            return os.path.join(root, "checkpoints")
        from acestep.model_downloader import get_checkpoints_dir

        return str(get_checkpoints_dir())

    def _load_acestep(self):
        if self._acestep_dit is not None:
            return
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        ckpt_dir = self._ace_step_checkpoint_dir()

        print("[ROOM] Loading ACE-Step DiT...")
        self._acestep_dit = AceStepHandler()
        dit_msg, dit_ok = self._acestep_dit.initialize_service(
            project_root=self.config.acestep_root,
            config_path=self.config.dit_config,
            device=self.config.device,
        )
        if not dit_ok:
            raise RuntimeError(
                dit_msg
                or "ACE-Step DiT initialize_service failed. See Container logs for download/OOM errors."
            )

        print("[ROOM] Loading ACE-Step LM...")
        self._acestep_llm = LLMHandler()
        llm_msg, llm_ok = self._acestep_llm.initialize(
            checkpoint_dir=ckpt_dir,
            lm_model_path=self.config.lm_model,
            backend=self.config.lm_backend,
            device=self.config.device,
        )
        if not llm_ok:
            raise RuntimeError(
                llm_msg
                or f"ACE-Step LM failed to load from {ckpt_dir!r}. See Container logs."
            )

        h = self._acestep_dit
        if (
            h.model is None
            or h.vae is None
            or h.text_tokenizer is None
            or h.text_encoder is None
        ):
            raise RuntimeError(
                "ACE-Step DiT reported success but model/VAE/text weights are missing. "
                f"Expected assets under {ckpt_dir!r}."
            )

        print("[ROOM] ACE-Step ready.")

    def _load_openvoice(self):
        if self._openvoice is not None:
            return
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter

        print("[ROOM] Loading OpenVoice V2...")
        ckpt_root = Path(self.config.openvoice_root) / "checkpoints_v2"
        conv_dir = ckpt_root / "converter"
        config_json = conv_dir / "config.json"
        weights_pth = conv_dir / "checkpoint.pth"
        if not config_json.is_file():
            raise FileNotFoundError(
                f"OpenVoice config missing: {config_json} "
                f"(run: python scripts/setup_room.py — need HF snapshot in checkpoints_v2/)"
            )
        if not weights_pth.is_file():
            raise FileNotFoundError(
                f"OpenVoice weights missing: {weights_pth} "
                f"(run: python scripts/setup_room.py)"
            )

        # Official demo: ToneColorConverter(f'{ckpt}/config.json'); then load_ckpt('.pth').
        # We previously passed only the folder and never loaded weights → broken / silent VC.
        dev = self.config.openvoice_device
        if dev == "cuda":
            dev = "cuda:0"

        self._openvoice = ToneColorConverter(str(config_json), device=dev)
        self._openvoice.load_ckpt(str(weights_pth))
        self._openvoice_se_extractor = se_extractor
        self._openvoice_ckpt = ckpt_root
        print("[ROOM] OpenVoice ready (converter weights loaded).")

    def _load_demucs(self):
        if self._demucs_model is not None:
            return
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        print("[ROOM] Loading Demucs...")
        self._demucs_model = get_model(self.config.demucs_model)
        self._demucs_model.to(self.config.device)
        self._demucs_apply = apply_model
        print("[ROOM] Demucs ready.")

    def _load_basicpitch(self):
        if self._basicpitch is not None:
            return
        from basic_pitch.inference import predict as bp_predict

        print("[ROOM] BasicPitch loaded.")
        self._basicpitch = bp_predict

    # ── Core pipeline ──────────────────────────────────────────────────

    def initialize(self):
        """Pre-load ACE-Step (always needed). Other models load on demand."""
        self._load_acestep()
        self._initialized = True
        print("[ROOM] Engine initialized. Other models load on demand.")

    def generate(
        self,
        prompt: str,
        voice_ref: Optional[str] = None,
        split_stems: bool = False,
        extract_midi: bool = False,
        duration: float = 30.0,
        seed: int = -1,
        inference_steps: int = 8,
        guidance_scale: float = 7.0,
        instrumental: bool = False,
        save_dir: Optional[str] = None,
    ) -> dict:
        """
        Full ROOM pipeline.

        Args:
            prompt: Text description of the music.
            voice_ref: Path to a voice reference wav (enables voice cloning).
            split_stems: If True, run Demucs to split into tracks.
            extract_midi: If True, transcribe stems to MIDI via BasicPitch.
            duration: Song length in seconds.
            seed: Random seed (-1 = random).
            instrumental: If True, generate without vocals.

        Returns:
            dict with keys: audio_path, voice_cloned_path, voice_clone_error, stems, midis, metadata
        """
        if not self._initialized:
            self.initialize()

        voice_ref = normalize_voice_ref_path(voice_ref)
        caption = caption_with_vocals_for_voice_clone(prompt) if voice_ref else (prompt or "").strip()

        out_dir = Path(save_dir or self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())

        # Reset per-call error state so the result reflects this run only.
        self._last_stems_error = None
        self._last_midi_error = None

        result = {
            "audio_path": None,
            "voice_cloned_path": None,
            "voice_clone_error": None,
            "stems": {},
            "midis": {},
            "stems_error": None,
            "midi_error": None,
            "metadata": {
                "prompt": prompt,
                "caption_for_acestep": caption,
                "duration": duration,
                "seed": seed,
                "voice_ref_normalized": voice_ref,
            },
        }

        # ── Step 1: Generate audio with ACE-Step ──────────────────────
        print(f"\n[ROOM] Step 1/4: Generating audio...")
        audio_path, audio_err = self._generate_audio(
            prompt=caption,
            duration=duration,
            seed=seed,
            steps=inference_steps,
            guidance=guidance_scale,
            instrumental=instrumental or (voice_ref is None),
            save_dir=str(out_dir),
        )
        result["audio_path"] = audio_path
        print(f"[ROOM] Audio: {audio_path}")
        if not audio_path:
            raise RuntimeError(
                audio_err
                or "ACE-Step produced no audio file. On Hugging Face, check Container logs "
                "(CUDA, OOM, missing weights under models/ace-step)."
            )

        # ── Step 2: Voice cloning (if reference provided) ─────────────
        source_audio = audio_path
        if voice_ref and audio_path:
            print(f"[ROOM] Step 2/4: Cloning voice from {voice_ref}...")
            cloned_path, cerr = self._clone_voice(
                source_audio=audio_path,
                voice_ref=voice_ref,
                output_path=str(out_dir / f"room_cloned_{ts}.wav"),
            )
            result["voice_cloned_path"] = cloned_path
            result["voice_clone_error"] = cerr
            source_audio = cloned_path or audio_path
            if cerr:
                print(f"[ROOM] Voice clone failed, using ACE output: {cerr}")
            else:
                print(f"[ROOM] Voice cloned: {cloned_path}")
        elif voice_ref and not audio_path:
            result["voice_clone_error"] = "No ACE-Step audio to clone onto."
            print("[ROOM] Step 2/4: Skipped (audio generation failed)")
        else:
            print("[ROOM] Step 2/4: Skipped (no voice reference)")

        # ── Step 3: Stem separation (if requested) ────────────────────
        if split_stems and source_audio:
            print(f"[ROOM] Step 3/4: Splitting stems...")
            stems = self._split_stems(
                audio_path=source_audio,
                output_dir=str(out_dir / "stems"),
            )
            result["stems"] = stems
            print(f"[ROOM] Stems: {list(stems.keys())}")
        else:
            print("[ROOM] Step 3/4: Skipped (stems not requested)")

        # ── Step 4: MIDI extraction (if requested) ────────────────────
        if extract_midi and result["stems"]:
            print(f"[ROOM] Step 4/4: Extracting MIDI from stems...")
            midis = self._extract_midi(
                stems=result["stems"],
                output_dir=str(out_dir / "midi"),
            )
            result["midis"] = midis
            print(f"[ROOM] MIDI files: {list(midis.keys())}")
        elif extract_midi and source_audio:
            print(f"[ROOM] Step 4/4: Extracting MIDI from full audio...")
            midis = self._extract_midi(
                stems={"full": source_audio},
                output_dir=str(out_dir / "midi"),
            )
            result["midis"] = midis
        else:
            print("[ROOM] Step 4/4: Skipped (MIDI not requested)")

        # Surface any silent failures so the UI can display them
        result["stems_error"] = self._last_stems_error
        result["midi_error"] = self._last_midi_error

        print(f"\n[ROOM] Done. Output: {out_dir}")
        return result

    # ── Sub-model operations ───────────────────────────────────────────

    def _generate_audio(
        self, prompt, duration, seed, steps, guidance, instrumental, save_dir
    ) -> tuple[Optional[str], Optional[str]]:
        """Returns (wav_path, error_message). error_message is set when path is None."""
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        params = GenerationParams(
            caption=prompt,
            duration=float(duration),
            seed=int(seed),
            inference_steps=int(steps),
            guidance_scale=float(guidance),
            instrumental=instrumental,
        )
        config = GenerationConfig(batch_size=1, audio_format=self.config.audio_format)

        result = generate_music(
            self._acestep_dit, self._acestep_llm, params, config, save_dir=save_dir
        )

        if result.success and result.audios:
            return result.audios[0]["path"], None
        err = str(result.error) if result.error else "unknown error"
        print(f"[ROOM] Audio generation failed: {err}")
        return None, err

    def _clone_voice(
        self, source_audio: str, voice_ref: str, output_path: str
    ) -> tuple[Optional[str], Optional[str]]:
        try:
            self._load_openvoice()

            target_se, _ = self._openvoice_se_extractor.get_se(
                voice_ref, self._openvoice, vad=True
            )

            source_se, _ = self._openvoice_se_extractor.get_se(
                source_audio, self._openvoice, vad=True
            )

            self._openvoice.convert(
                audio_src_path=source_audio,
                src_se=source_se,
                tgt_se=target_se,
                output_path=output_path,
            )
            return output_path, None
        except Exception as e:
            print(f"[ROOM] Voice cloning failed: {e}")
            return None, str(e)

    def _split_stems(self, audio_path: str, output_dir: str) -> dict[str, str]:
        try:
            self._load_demucs()
            import torch
            import torchaudio

            os.makedirs(output_dir, exist_ok=True)

            wav, sr = torchaudio.load(audio_path)
            wav = wav.to(self.config.device)

            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)

            # Normalise per channel — earlier code used `wav.mean()` (scalar)
            # then called `.std()` on that scalar producing NaN, which silently
            # destroyed the audio before Demucs even ran.
            ref_mean = wav.mean()
            ref_std = wav.std().clamp_min(1e-8)
            wav_n = (wav - ref_mean) / ref_std
            wav_n = wav_n.unsqueeze(0)

            sources = self._demucs_apply(
                self._demucs_model, wav_n, device=self.config.device
            )

            # Undo normalisation so the stems sound correct
            sources = sources * ref_std + ref_mean

            stem_names = self._demucs_model.sources
            stems = {}
            for i, name in enumerate(stem_names):
                stem_path = os.path.join(output_dir, f"{name}.wav")
                stem_audio = sources[0, i].cpu()
                torchaudio.save(stem_path, stem_audio, sr)
                stems[name] = stem_path

            self._last_stems_error = None
            return stems
        except Exception as e:
            import traceback
            err = f"{type(e).__name__}: {e}"
            self._last_stems_error = err
            print(f"[ROOM] Stem separation failed: {err}", flush=True)
            print(traceback.format_exc(), flush=True)
            return {}

    def _extract_midi(self, stems: dict[str, str], output_dir: str) -> dict[str, str]:
        try:
            self._load_basicpitch()
            os.makedirs(output_dir, exist_ok=True)

            midis = {}
            for name, audio_path in stems.items():
                if name == "drums":
                    continue
                midi_path = os.path.join(output_dir, f"{name}.mid")
                model_output, midi_data, note_events = self._basicpitch(audio_path)
                midi_data.write(midi_path)
                midis[name] = midi_path

            self._last_midi_error = None
            return midis
        except Exception as e:
            import traceback
            err = f"{type(e).__name__}: {e}"
            self._last_midi_error = err
            print(f"[ROOM] MIDI extraction failed: {err}", flush=True)
            print(traceback.format_exc(), flush=True)
            return {}

    # ── Standalone utilities ───────────────────────────────────────────

    def clone_voice(self, audio_path: str, voice_ref: str, output_path: Optional[str] = None) -> Optional[str]:
        """Standalone voice cloning without generating new audio."""
        if output_path is None:
            output_path = str(Path(self.config.output_dir) / "voice_cloned.wav")
        path, err = self._clone_voice(audio_path, voice_ref, output_path)
        if err:
            print(f"[ROOM] clone_voice: {err}")
        return path

    def split_stems(self, audio_path: str, output_dir: Optional[str] = None) -> dict[str, str]:
        """Standalone stem separation."""
        if output_dir is None:
            output_dir = str(Path(self.config.output_dir) / "stems")
        return self._split_stems(audio_path, output_dir)

    def extract_midi(self, audio_path: str, output_dir: Optional[str] = None) -> dict[str, str]:
        """Standalone MIDI extraction."""
        if output_dir is None:
            output_dir = str(Path(self.config.output_dir) / "midi")
        return self._extract_midi({"full": audio_path}, output_dir)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="ROOM Engine CLI")
    ap.add_argument("prompt", type=str, help="Music description")
    ap.add_argument("--voice", type=str, default=None, help="Voice reference wav")
    ap.add_argument("--stems", action="store_true", help="Split output into stems")
    ap.add_argument("--midi", action="store_true", help="Extract MIDI from stems")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--out", type=str, default="output/room")
    args = ap.parse_args()

    engine = RoomEngine()
    engine.initialize()
    result = engine.generate(
        prompt=args.prompt,
        voice_ref=args.voice,
        split_stems=args.stems,
        extract_midi=args.midi,
        duration=args.duration,
        seed=args.seed,
        save_dir=args.out,
    )

    import json
    print("\n" + json.dumps({k: v for k, v in result.items() if k != "metadata"}, indent=2))
