#!/usr/bin/env python3
"""
MODEL-W / ROOM — Music Production AI

Full pipeline: text prompt + voice reference → audio + stems + MIDI.

Usage:
  python app.py
  python app.py --share
  python app.py --port 8080
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

import gradio as gr

# ZeroGPU (HF Spaces) — wrap generate with spaces.GPU so the Space gets a GPU
# only during inference. The PyPI `spaces` package must expose `.GPU`; another
# module named `spaces` may exist locally without it.
try:
    import spaces

    _ZEROGPU = hasattr(spaces, "GPU")
except ImportError:
    _ZEROGPU = False

_ROOT = Path(__file__).resolve().parent
_ACESTEP_SRC = _ROOT / "models" / "ace-step"

# On HF Spaces the models folder won't exist until we clone them.
# Run setup_spaces.py once at import time (safe to re-run; it's idempotent).
if os.environ.get("SPACE_ID"):  # only on HF Spaces
    try:
        import setup_spaces
        setup_spaces.setup()
    except Exception as _se:
        print(f"[setup_spaces] Warning: {_se}")

if _ACESTEP_SRC.exists() and str(_ACESTEP_SRC) not in sys.path:
    sys.path.insert(0, str(_ACESTEP_SRC))

# ── Engine loading (lazy) ──────────────────────────────────────────────

_engine = None


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    try:
        from modelw.room import RoomConfig, RoomEngine

        _engine = RoomEngine(RoomConfig())
        _engine.initialize()
        return _engine
    except Exception as e:
        print(f"[ROOM] Engine init failed: {e}")
        raise gr.Error(f"Engine not ready: {e}")


def _check_available() -> bool:
    try:
        import acestep  # noqa: F401
        return True
    except ImportError:
        return False


AVAILABLE = _check_available()


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def _pick_server_port(preferred: int, max_offsets: int = 64) -> int:
    """Use preferred port if free; otherwise scan upward (common when 7860 is stuck)."""
    for offset in range(max_offsets):
        p = preferred + offset
        if _port_is_free(p):
            return p
    raise OSError(
        f"No free TCP port in range {preferred}-{preferred + max_offsets - 1}. "
        "Kill the old process: fuser -k 7860/tcp  OR  lsof -i :7860"
    )


# ── Generate ───────────────────────────────────────────────────────────

def _generate_impl(prompt, voice_ref, split_stems, extract_midi, duration, seed, steps, guidance):
    """Core generation logic — separated so ZeroGPU decorator can wrap it."""
    if not AVAILABLE:
        raise gr.Error("ROOM not installed. Run: python scripts/setup_room.py")

    from modelw.room import normalize_voice_ref_path

    engine = _get_engine()

    out_dir = str(_ROOT / "output" / "room")
    os.makedirs(out_dir, exist_ok=True)

    voice_path = normalize_voice_ref_path(voice_ref)
    ref_problem = None
    if voice_ref is not None and voice_path is None:
        ref_problem = (
            "Voice reference could not be read (need a saved upload path). Voice cloning was skipped."
        )

    result = engine.generate(
        prompt=prompt,
        voice_ref=voice_path,
        split_stems=split_stems,
        extract_midi=extract_midi,
        duration=float(duration),
        seed=int(seed),
        inference_steps=int(steps),
        guidance_scale=float(guidance),
        save_dir=out_dir,
    )

    # Primary audio output
    audio_out = result.get("voice_cloned_path") or result.get("audio_path")

    # Stem files for download
    stem_files = []
    for name, path in result.get("stems", {}).items():
        if path and os.path.exists(path):
            stem_files.append(path)

    # MIDI files for download
    midi_files = []
    for name, path in result.get("midis", {}).items():
        if path and os.path.exists(path):
            midi_files.append(path)

    all_files = stem_files + midi_files

    info_parts = []
    if ref_problem:
        info_parts.append(ref_problem)
    meta = result.get("metadata") or {}
    cap = meta.get("caption_for_acestep")
    if voice_path and cap and cap.strip() != (prompt or "").strip():
        info_parts.append("Prompt auto-expanded so the mix includes lead vocals (needed for cloning).")
    if result.get("voice_clone_error"):
        info_parts.append(f"Voice clone failed: {result['voice_clone_error']}")
    elif result.get("voice_cloned_path"):
        info_parts.append("Voice timbre applied (OpenVoice).")
    elif voice_path:
        info_parts.append("Voice ref set but no cloned output file was produced.")
    if result.get("stems"):
        info_parts.append(f"Stems: {', '.join(result['stems'].keys())}")
    if result.get("midis"):
        info_parts.append(f"MIDI: {', '.join(result['midis'].keys())}")
    info = " | ".join(info_parts) if info_parts else "Generated"

    return audio_out, all_files if all_files else None, info


# Wrap with ZeroGPU decorator when running on HF Spaces; no-op otherwise.
if _ZEROGPU:
    generate = spaces.GPU(_generate_impl)
else:
    generate = _generate_impl


# ── UI ─────────────────────────────────────────────────────────────────

CSS = """
.main-header { text-align: center; padding: 24px 0 8px 0; }
.main-header h1 {
    font-size: 2.6em; font-weight: 800;
    background: linear-gradient(135deg, #7c3aed, #a855f7, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.main-header p { color: #94a3b8; font-size: 1.05em; margin-top: 2px; }
.prompt-box textarea { font-size: 1.1em !important; }
footer { display: none !important; }
"""

EXAMPLES = [
    ["dark aggressive trap, D minor, 140 BPM, heavy 808 bass, rolling hi-hats"],
    ["piano ballad, E minor, 70 BPM, emotional, soft vocals"],
    ["lo-fi chill beat, A minor, 85 BPM, vinyl crackle, mellow piano"],
    ["epic cinematic orchestral, G minor, 100 BPM, dramatic strings"],
    ["smooth R&B, Eb major, 90 BPM, soulful chords, warm keys"],
    ["house music, F# minor, 126 BPM, deep bass, synth stabs"],
]


def build_ui():
    # Gradio 6+: theme/css belong on launch(), not Blocks()
    with gr.Blocks(title="ROOM") as demo:

        gr.HTML("""
        <div class="main-header">
            <h1>ROOM</h1>
            <p>Describe the music. Upload your voice. We handle the rest.</p>
        </div>
        """)

        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="piano ballad, E minor, emotional, soft vocals...",
                lines=3,
                elem_classes=["prompt-box"],
            )

            voice_ref = gr.Audio(
                label="Voice Reference (optional —10–30s dry speech or singing; we match timbre to vocals in the generated track)",
                type="filepath",
            )

            with gr.Row():
                split_stems = gr.Checkbox(value=False, label="Split to stems (vocals, drums, bass, other)")
                extract_midi = gr.Checkbox(value=False, label="Extract MIDI from stems")

            generate_btn = gr.Button("Generate", variant="primary", size="lg")

            audio_out = gr.Audio(label="Output", type="filepath")
            download_files = gr.File(label="Stems + MIDI files", file_count="multiple")
            info = gr.Markdown("")

            with gr.Accordion("Settings", open=False):
                with gr.Row():
                    duration = gr.Slider(10, 300, value=30, step=5, label="Duration (seconds)")
                    seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                with gr.Row():
                    steps = gr.Slider(4, 50, value=8, step=1, label="Inference steps")
                    guidance = gr.Slider(1.0, 15.0, value=7.0, step=0.5, label="Guidance scale")

            gr.Examples(examples=EXAMPLES, inputs=[prompt], label="Try these")

        generate_btn.click(
            fn=generate,
            inputs=[prompt, voice_ref, split_stems, extract_midi, duration, seed, steps, guidance],
            outputs=[audio_out, download_files, info],
        )

        gr.Markdown("""
        ---
        <center style="color: #64748b; font-size: 0.85em;">
        ROOM v0.1 — A foundation model for music production
        </center>
        """)

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        help="Preferred port; if busy, next free port is used (override with GRADIO_SERVER_PORT).",
    )
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    os.makedirs(_ROOT / "output", exist_ok=True)

    status = "READY" if AVAILABLE else "NOT INSTALLED (run: python scripts/setup_room.py)"
    print(f"\n  ROOM v0.1 | Engine: {status}\n")

    preferred = args.port
    server_port = _pick_server_port(preferred)
    if server_port != preferred:
        print(
            f"  [ROOM] Port {preferred} busy; using {server_port} "
            f"(free the old server with: fuser -k {preferred}/tcp)\n"
        )

    demo = build_ui()
    _theme = gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    )
    demo.launch(
        server_port=server_port,
        share=args.share,
        server_name="0.0.0.0",
        theme=_theme,
        css=CSS,
    )


if __name__ == "__main__":
    main()
