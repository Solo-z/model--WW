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
import logging
import os
import socket
import sys
from pathlib import Path

# ── Silence internal library logs BEFORE any heavy imports ────────────
# Hides model names, file paths, and progress bars from the runtime log
# stream (which other people might glimpse on a public Space).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("ACESTEP_DISABLE_TQDM", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

# Stdlib logging — cap noisy third-party loggers
for _name in ("transformers", "diffusers", "accelerate", "demucs",
              "openvoice", "basic_pitch", "matplotlib", "PIL",
              "urllib3", "huggingface_hub", "filelock"):
    logging.getLogger(_name).setLevel(logging.ERROR)

# Loguru (used by ACE-Step) — disable entirely for these namespaces
try:
    from loguru import logger as _loguru_logger
    for _ns in ("acestep", "openvoice", "demucs"):
        _loguru_logger.disable(_ns)
except ImportError:
    pass

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


def _default_room_config():
    """HF Spaces: use 1.7B LM + PyTorch backend (the only LM shipped by ACE-Step)."""
    from modelw.room import RoomConfig

    if os.environ.get("SPACE_ID"):
        print("[ROOM] Space runtime: lm_model=acestep-5Hz-lm-1.7B, lm_backend=pt", flush=True)
        return RoomConfig(lm_model="acestep-5Hz-lm-1.7B", lm_backend="pt")
    return RoomConfig()


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    try:
        from modelw.room import RoomEngine

        _engine = RoomEngine(_default_room_config())
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

def _friendly_error(exc: Exception) -> str:
    """Translate cryptic stack traces into something a human can act on,
    without leaking internal model names, file paths, or library internals."""
    msg = str(exc).lower()
    if "duration" in msg and ("maximum" in msg or "larger than" in msg):
        return ("Generation needs more GPU time than your account allows. "
                "Try a shorter duration or fewer steps, or sign in to HuggingFace for a higher quota.")
    if "out of memory" in msg or "cuda oom" in msg or "outofmemory" in msg:
        return "Ran out of memory. Try a shorter duration or fewer steps."
    if "model not fully initialized" in msg or "not fully initialized" in msg:
        return "ROOM is still warming up. Wait 30 seconds and try again."
    if "no module named" in msg or "modulenotfounderror" in msg:
        return "ROOM is not fully ready on this Space yet. Try again in a minute."
    if "cuda" in msg and ("not available" in msg or "no device" in msg):
        return "GPU is busy right now. Try again in a minute."
    if "timeout" in msg or "timed out" in msg:
        return "Request timed out. Try a shorter duration or simpler prompt."
    if "engine not ready" in msg:
        return "ROOM is still starting up. Try again in a moment."
    return "Generation failed. Try again, or simplify the prompt."


def _generate_impl(prompt, outputs_select,
                   duration, seed, steps, guidance,
                   progress=gr.Progress()):
    """Core generation logic — separated so ZeroGPU decorator can wrap it."""
    import traceback

    if not AVAILABLE:
        raise gr.Error("ROOM is not ready on this Space yet.")

    if not (prompt or "").strip():
        raise gr.Error("Add a prompt first — describe the music you want.")

    selected = outputs_select or []
    split_stems = "Stems" in selected
    extract_midi = "MIDI" in selected

    try:
        progress(0.05, desc="Composing")
        engine = _get_engine()

        out_dir = str(_ROOT / "output" / "room")
        os.makedirs(out_dir, exist_ok=True)

        progress(0.20, desc="Generating")
        result = engine.generate(
            prompt=prompt,
            voice_ref=None,
            split_stems=split_stems,
            extract_midi=extract_midi,
            duration=float(duration),
            seed=int(seed),
            inference_steps=int(steps),
            guidance_scale=float(guidance),
            save_dir=out_dir,
        )
        progress(0.92, desc="Mastering")

        audio_out = result.get("audio_path")

        stem_files = []
        for name, path in result.get("stems", {}).items():
            if path and os.path.exists(path):
                stem_files.append(path)

        midi_files = []
        for name, path in result.get("midis", {}).items():
            if path and os.path.exists(path):
                midi_files.append(path)

        all_files = stem_files + midi_files

        info_parts = []
        if stem_files:
            info_parts.append(f"{len(stem_files)} stems")
        elif split_stems:
            stems_err = result.get("stems_error") or "unknown error"
            info_parts.append(f"Stems failed: {stems_err}")
        if midi_files:
            info_parts.append(f"{len(midi_files)} MIDI")
        elif extract_midi:
            midi_err = result.get("midi_error") or "stems failed first" if split_stems and not stem_files else (result.get("midi_error") or "unknown error")
            info_parts.append(f"MIDI failed: {midi_err}")
        info = "Ready · " + " · ".join(info_parts) if info_parts else "Ready"

        progress(1.0, desc="Ready")
        # Show "Download All" button only if there are files to download
        has_files = bool(all_files) or bool(audio_out)
        download_all_update = gr.update(visible=has_files)
        return audio_out, all_files if all_files else None, download_all_update, info
    except gr.Error:
        raise
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        raise gr.Error(_friendly_error(e))


# ZeroGPU: anonymous/free-tier users have a low cap (~120s).
# ACE-Step turbo (8 steps) on H200 finishes well within this.
_ZEROGPU_GPU_SECONDS = 120

# Wrap with ZeroGPU decorator when running on HF Spaces; no-op otherwise.
if _ZEROGPU:
    generate = spaces.GPU(duration=_ZEROGPU_GPU_SECONDS)(_generate_impl)
else:
    generate = _generate_impl


# ── UI ─────────────────────────────────────────────────────────────────

CSS = """
/* ── Full-bleed tree as page background ──────────────────────────── */
html, body, gradio-app {
    background: #000 !important;
    color: #f5f5f5 !important;
    font-family: 'Inter', -apple-system, sans-serif;
    margin: 0 !important;
    padding: 0 !important;
}
body::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image: url('/gradio_api/file=assets/room_tree.png');
    background-repeat: no-repeat;
    background-position: center center;
    background-size: contain;
    background-color: #000;
    opacity: 0.55;
    mix-blend-mode: screen;
    z-index: 0;
    pointer-events: none;
}
body::after {
    content: "";
    position: fixed;
    inset: 0;
    background: radial-gradient(ellipse at center, rgba(0,0,0,0) 30%, rgba(0,0,0,0.85) 100%);
    z-index: 1;
    pointer-events: none;
}
.gradio-container {
    position: relative;
    z-index: 10 !important;
    max-width: 760px !important;
    margin: 0 auto !important;
    padding: 5vh 24px 80px !important;
    background: transparent !important;
}

/* ── Hero text — sits over the tree ──────────────────────────────── */
.room-hero { text-align: center; padding: 8vh 0 6vh 0; }
.room-hero h1 {
    font-size: clamp(5rem, 12vw, 11rem);
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 0.95;
    margin: 0 0 16px 0;
    color: #fff;
    user-select: none;
    text-shadow: 0 4px 24px rgba(0,0,0,0.7);
}
.room-hero p {
    font-size: clamp(0.7rem, 1vw, 1rem);
    font-weight: 300;
    letter-spacing: 0.5em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.85);
    text-shadow: 0 2px 12px rgba(0,0,0,0.8);
    margin: 0;
    padding-left: 0.5em;
}

/* Intro paragraph */
.room-intro {
    max-width: 560px;
    margin: 0 auto 56px auto;
    text-align: center;
    color: rgba(255,255,255,0.65);
    font-size: 0.95rem;
    line-height: 1.6;
    font-weight: 300;
    text-shadow: 0 2px 8px rgba(0,0,0,0.7);
}

/* ── Inputs are nearly invisible — tree shows through ────────────── */
.gradio-container .block,
.gradio-container .form,
.gradio-container .gr-box,
.gradio-container .gr-panel {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Prompt bar — single hairline rectangle, see-through */
.prompt-box {
    margin: 0 auto 24px auto !important;
}
.prompt-box textarea {
    font-size: 1.05em !important;
    background: rgba(0,0,0,0.35) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 4px !important;
    color: #fff !important;
    text-align: center;
    padding: 18px 20px !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: border-color 0.25s ease, background 0.25s ease;
}
.prompt-box textarea:focus {
    background: rgba(0,0,0,0.55) !important;
    border-color: rgba(255,255,255,0.5) !important;
    outline: none !important;
}
.prompt-box textarea::placeholder {
    color: rgba(255,255,255,0.45) !important;
    text-transform: lowercase;
    letter-spacing: 0.02em;
}

/* Generic input styling — exclude checkboxes/radios so they stay clickable */
input:not([type="checkbox"]):not([type="radio"]),
textarea, select {
    background: rgba(0,0,0,0.35) !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: #f5f5f5 !important;
}

/* Toggle row — centered, minimal styling so Gradio's clicks still work */
.toggles-row {
    justify-content: center !important;
    margin: 8px auto 24px auto !important;
}

/* ── Labels — tiny uppercase, fashion-brand feel ─────────────────── */
label, .gr-input-label, span[data-testid="block-label"] {
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.7em !important;
    font-weight: 500 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
}

/* ── Generate button — large, dramatic, pulses on hover ──────────── */
.generate-btn, .gr-button-primary, button.primary, .gradio-container button.lg {
    background: #fff !important;
    color: #000 !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 0.3em !important;
    text-transform: uppercase !important;
    font-size: 0.95em !important;
    padding: 22px 36px !important;
    border-radius: 4px !important;
    margin: 16px 0 !important;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s cubic-bezier(.2,.9,.3,1.2),
                box-shadow 0.25s ease,
                letter-spacing 0.25s ease;
    box-shadow: 0 4px 24px rgba(255,255,255,0.08);
}
.generate-btn:hover, .gr-button-primary:hover, button.primary:hover {
    background: #fff !important;
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 8px 32px rgba(255,255,255,0.18);
    letter-spacing: 0.35em !important;
}
.generate-btn:active { transform: translateY(0) scale(0.99); }

/* While generating, button glows softly */
.generate-btn:disabled,
.gr-button-primary:disabled {
    background: #fff !important;
    color: #000 !important;
    opacity: 1 !important;
    animation: room-pulse 1.6s ease-in-out infinite;
}
@keyframes room-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,255,255,0.4); }
    50%      { box-shadow: 0 0 28px 6px rgba(255,255,255,0.18); }
}

/* ── Generation visuals — chunky animated bar + clear status text ─ */
.progress,
.progress-text,
[class*="Progress"],
[class*="progress"],
.gr-progress {
    color: #fff !important;
    background: transparent !important;
}

/* The progress bar fill — bold, full-width, glowing */
.progress-bar,
[class*="progressBar"],
.gr-progress > div,
div[role="progressbar"] {
    background: linear-gradient(90deg,
        rgba(255,255,255,0.15),
        rgba(255,255,255,1),
        rgba(255,255,255,0.15)) !important;
    background-size: 200% 100% !important;
    animation: room-shimmer 1.2s linear infinite !important;
    height: 8px !important;
    min-height: 8px !important;
    border-radius: 4px !important;
    box-shadow: 0 0 24px rgba(255,255,255,0.4) !important;
    margin: 12px 0 !important;
    width: 100% !important;
    display: block !important;
}

/* Bar background (the track) */
.gr-progress, [class*="progress-bar-container"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 6px !important;
    padding: 4px !important;
    margin: 16px 0 !important;
    width: 100% !important;
}

@keyframes room-shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* Loading status text — big, centered, letter-spaced */
.gr-progress-text,
.progress-text,
[class*="progressText"] {
    text-align: center !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.4em !important;
    text-transform: uppercase !important;
    color: #fff !important;
    padding: 20px 0 12px 0 !important;
    text-shadow: 0 2px 8px rgba(0,0,0,0.6);
    animation: room-text-fade 1.6s ease-in-out infinite;
}

@keyframes room-text-fade {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.55; }
}

/* ── Output panels ───────────────────────────────────────────────── */
.audio-out audio {
    width: 100% !important;
    margin-top: 16px;
    filter: grayscale(100%) invert(0%);
}

/* Big "Download All" button — sits above individual file list */
.download-all-btn, .download-all-btn button {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.4) !important;
    color: #fff !important;
    font-weight: 600 !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase !important;
    font-size: 0.85em !important;
    padding: 18px 24px !important;
    border-radius: 6px !important;
    margin-top: 24px !important;
    width: 100% !important;
    cursor: pointer !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: background 0.2s ease, border-color 0.2s ease, transform 0.15s ease;
}
.download-all-btn:hover, .download-all-btn button:hover {
    background: rgba(255,255,255,0.22) !important;
    border-color: #fff !important;
    transform: translateY(-1px);
}

/* Hidden file source — kept off-screen but still in the DOM so JS can read URLs */
.files-out-hidden {
    position: absolute !important;
    left: -9999px !important;
    top: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
}

/* Downloads panel — clearly visible when files exist, empty state hidden */
.files-out {
    margin-top: 24px !important;
    background: rgba(0,0,0,0.55) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 6px !important;
    padding: 18px 20px !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    pointer-events: auto !important;
}
.files-out * { pointer-events: auto !important; }

.files-out > label,
.files-out [data-testid="block-label"],
.files-out span[data-testid="block-label"] {
    color: #fff !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase !important;
    margin-bottom: 12px !important;
    display: block !important;
    opacity: 1 !important;
}

/* Hide Gradio's giant empty-state icon + "Drop file here" prompt */
.files-out [data-testid="upload"],
.files-out .upload-container,
.files-out [class*="upload-text"],
.files-out [class*="UploadText"],
.files-out [class*="empty"],
.files-out [class*="Empty"],
.files-out svg.feather-file,
.files-out svg[class*="upload"] {
    display: none !important;
}
/* Collapse any oversized icons */
.files-out svg {
    max-width: 18px !important;
    max-height: 18px !important;
    color: #fff !important;
    opacity: 1 !important;
}

/* File rows / pills */
.files-out a,
.files-out button,
.files-out [data-testid="file-name"],
.files-out [class*="file-preview"],
.files-out [class*="filePreview"] {
    color: #fff !important;
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 4px !important;
    padding: 10px 14px !important;
    margin: 4px 6px 4px 0 !important;
    text-decoration: none !important;
    font-size: 0.85em !important;
    cursor: pointer !important;
    display: inline-flex !important;
    align-items: center;
    gap: 8px;
    transition: background 0.15s ease;
}
.files-out a:hover,
.files-out button:hover {
    background: rgba(255,255,255,0.18) !important;
}

.info-line {
    margin-top: 12px !important;
}
.info-line p {
    text-align: center;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase;
    color: rgba(255,255,255,0.55) !important;
}

/* ── Accordion ───────────────────────────────────────────────────── */
.gr-accordion { background: rgba(10,10,10,0.45) !important; }

/* ── Footer / chrome ─────────────────────────────────────────────── */
footer, .footer, .built-with { display: none !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Examples styling ────────────────────────────────────────────── */
.gr-examples button, .gr-sample-textbox {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: rgba(255,255,255,0.75) !important;
}

/* ── Strict monochrome — purge any residual color from Gradio ────── */
.gradio-container *:not(svg):not(path) {
    --color-accent: #ffffff !important;
    --color-accent-soft: rgba(255,255,255,0.12) !important;
    --primary-50: #f5f5f5 !important;
    --primary-100: #e5e5e5 !important;
    --primary-200: #d4d4d4 !important;
    --primary-300: #a3a3a3 !important;
    --primary-400: #737373 !important;
    --primary-500: #525252 !important;
    --primary-600: #404040 !important;
    --primary-700: #262626 !important;
    --primary-800: #171717 !important;
    --primary-900: #0a0a0a !important;
}

/* Sliders — grayscale only */
input[type="range"] { accent-color: #fff !important; }
.gr-slider .noUi-connect { background: #fff !important; }
.gr-slider .noUi-handle { background: #fff !important; border: none !important; }

/* Checkboxes — grayscale */
input[type="checkbox"] { accent-color: #fff !important; }

/* Progress bar — white */
.progress-bar, .progress-text { color: #fff !important; }
[class*="progress"] [style*="background"] { background: #fff !important; }

/* Audio player */
.gr-audio, audio { filter: grayscale(100%); }

/* Selection */
::selection { background: rgba(255,255,255,0.2); color: #fff; }
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
    with gr.Blocks(title="ROOM — A Foundation Model for Music Production") as demo:

        gr.HTML("""
        <div class="room-hero">
            <h1>ROOM</h1>
            <p>A Foundation Model for Music Production</p>
        </div>

        <div class="room-intro">
            ROOM turns a written description into a finished track.
            Upload a voice reference and the lead vocal will match your timbre.
            Optional stem separation and MIDI extraction for further production.
        </div>
        """)

        with gr.Column():
            prompt = gr.Textbox(
                label="",
                placeholder="describe the track — genre, key, BPM, mood, instruments…",
                lines=2,
                elem_classes=["prompt-box"],
                show_label=False,
            )

            outputs_select = gr.CheckboxGroup(
                choices=["Stems", "MIDI"],
                value=["Stems", "MIDI"],
                label="Outputs",
                elem_classes=["toggles-row"],
            )

            generate_btn = gr.Button("⏵  Generate", variant="primary", size="lg",
                                     elem_classes=["generate-btn"])

            audio_out = gr.Audio(label="", type="filepath", show_label=False,
                                 elem_classes=["audio-out"])

            download_all = gr.Button(
                "⬇  Download All",
                visible=False,
                elem_classes=["download-all-btn"],
                elem_id="download-all-btn",
            )

            # Hidden via CSS only — needs to be rendered in the DOM so its
            # file <a> tags exist for the Download All JS to click.
            download_files = gr.File(
                file_count="multiple",
                elem_classes=["files-out-hidden"],
                elem_id="room-hidden-files",
                show_label=False,
            )
            info = gr.Markdown("", elem_classes=["info-line"])

            with gr.Accordion("Advanced", open=False, elem_classes=["advanced-acc"]):
                with gr.Row():
                    duration = gr.Slider(10, 300, value=30, step=5, label="Duration (s)")
                    seed = gr.Number(value=-1, label="Seed", precision=0)
                with gr.Row():
                    steps = gr.Slider(4, 50, value=8, step=1, label="Steps")
                    guidance = gr.Slider(1.0, 15.0, value=7.0, step=0.5, label="Guidance")

            gr.Examples(examples=EXAMPLES, inputs=[prompt], label="Examples")

        generate_btn.click(
            fn=generate,
            inputs=[prompt, outputs_select, duration, seed, steps, guidance],
            outputs=[audio_out, download_files, download_all, info],
        )

        # No Python work — JS finds the served file links inside the hidden
        # File panel and triggers each as a separate browser download.
        # The first time this fires, browsers usually ask the user to allow
        # multi-file downloads from the page; after that they just go through.
        download_all.click(
            fn=lambda: None,
            inputs=[],
            outputs=[],
            js="""
            () => {
                console.log('[ROOM] Download All clicked');
                const container = document.getElementById('room-hidden-files');
                if (!container) {
                    console.warn('[ROOM] hidden file container not found');
                    return;
                }
                // Try multiple selectors — Gradio renders downloads variously
                // across versions. Anchors with download attr or /file= href
                // are the classic; otherwise look for any anchor with href.
                let targets = Array.from(container.querySelectorAll(
                    'a[download], a[href*="/file="], a[href*="gradio_api/file="]'
                ));
                if (targets.length === 0) {
                    targets = Array.from(container.querySelectorAll('a[href]'));
                }
                if (targets.length === 0) {
                    // Fallback: Gradio v6 sometimes renders the download as a
                    // <button> with an href on a child anchor or as a click
                    // handler. Find any descendant anchor.
                    targets = Array.from(container.getElementsByTagName('a'));
                }
                console.log('[ROOM] Found ' + targets.length + ' download targets');
                if (targets.length === 0) return;

                targets.forEach((el, i) => setTimeout(() => {
                    try {
                        const href = el.getAttribute('href');
                        if (href) {
                            // Force a fresh anchor with download attr so the
                            // browser saves rather than navigates.
                            const a = document.createElement('a');
                            a.href = href;
                            a.download = (el.getAttribute('download') ||
                                          href.split('/').pop().split('?')[0] ||
                                          'room_file');
                            a.style.display = 'none';
                            document.body.appendChild(a);
                            a.click();
                            setTimeout(() => document.body.removeChild(a), 100);
                        } else {
                            el.click();
                        }
                    } catch (e) { console.warn('[ROOM] download err', e); }
                }, i * 300));
            }
            """,
        )

        gr.HTML("""
        <div style="text-align:center; margin-top:48px; padding:24px 0;
                    color:#525252; font-size:0.75em; letter-spacing:0.2em;
                    text-transform:uppercase; border-top:1px solid #1f1f1f;">
            ROOM v0.1 &nbsp;·&nbsp; A FOUNDATION MODEL FOR MUSIC PRODUCTION
        </div>
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

    _theme = gr.themes.Monochrome(
        font=gr.themes.GoogleFont("Inter"),
    )
    demo.launch(
        server_port=server_port,
        share=args.share,
        server_name="0.0.0.0",
        theme=_theme,
        css=CSS,
        ssr_mode=False,
        allowed_paths=[str(_ROOT / "assets")],
    )


# Allow Gradio to serve the hero image (works in both local & HF Space launches).
try:
    gr.set_static_paths(paths=[str(_ROOT / "assets")])
except Exception:
    pass

# Hugging Face mounts `demo` at import time; module-level avoids "demo not found in __main__".
demo = build_ui()

if __name__ == "__main__":
    main()
