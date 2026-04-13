#!/usr/bin/env python3
"""
MODEL-W — Music Production AI

A clean wrapper around ACE-Step 1.5. Type a prompt, get music.

Usage:
  python app.py
  python app.py --share       (public link for demos)
  python app.py --port 8080
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import gradio as gr

_ROOT = Path(__file__).resolve().parent

# Add ACE-Step source to path (no pip install needed)
_ACESTEP_SRC = _ROOT / "models" / "ace-step"
if _ACESTEP_SRC.exists() and str(_ACESTEP_SRC) not in sys.path:
    sys.path.insert(0, str(_ACESTEP_SRC))

# ── ACE-Step loading (lazy) ────────────────────────────────────────────────

_handler = None
_llm = None


def _load_acestep():
    global _handler, _llm
    if _handler is not None:
        return _handler, _llm

    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    env = {}
    env_file = _ROOT / ".env.acestep"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()

    root = env.get("ACESTEP_ROOT", str(_ROOT / "models" / "ace-step"))
    dit = env.get("ACESTEP_DIT_CONFIG", "acestep-v15-turbo")
    lm = env.get("ACESTEP_LM_MODEL", "acestep-5Hz-lm-1.7B")
    backend = env.get("ACESTEP_LM_BACKEND", "vllm")

    _handler = AceStepHandler()
    _handler.initialize_service(project_root=root, config_path=dit, device="cuda")

    _llm = LLMHandler()
    _llm.initialize(checkpoint_dir=root, lm_model_path=lm, backend=backend, device="cuda")

    return _handler, _llm


def _check_acestep() -> bool:
    try:
        import acestep  # noqa: F401
        return True
    except ImportError:
        return False


ACESTEP_INSTALLED = _check_acestep()


# ── Generate ───────────────────────────────────────────────────────────────

def generate(prompt, duration, seed, steps, guidance):
    """Text prompt in, audio out."""
    if not ACESTEP_INSTALLED:
        raise gr.Error(
            "ACE-Step not installed. Run:  python scripts/setup_acestep.py"
        )

    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    dit, llm = _load_acestep()

    out_dir = str(_ROOT / "output" / "audio")
    os.makedirs(out_dir, exist_ok=True)

    params = GenerationParams(
        caption=prompt,
        duration=float(duration),
        seed=int(seed),
        inference_steps=int(steps),
        guidance_scale=float(guidance),
        instrumental=True,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav")

    result = generate_music(dit, llm, params, config, save_dir=out_dir)

    if result.success and result.audios:
        path = result.audios[0]["path"]
        return path, f"Seed: {result.audios[0]['params'].get('seed', '?')}"
    else:
        raise gr.Error(f"Generation failed: {result.error}")


# ── UI ─────────────────────────────────────────────────────────────────────

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
    ["lo-fi chill beat, A minor, 85 BPM, vinyl crackle, mellow piano, rainy mood"],
    ["epic cinematic orchestral, G minor, 100 BPM, dramatic strings, war drums"],
    ["house music, F# minor, 126 BPM, deep bass, synth stabs, four-on-the-floor"],
    ["smooth R&B, Eb major, 90 BPM, soulful chords, warm keys, late night vibe"],
    ["UK drill, C# minor, 140 BPM, sliding 808s, dark pads, aggressive energy"],
]


def build_ui():
    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.violet,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CSS,
        title="MODEL-W",
    ) as demo:

        gr.HTML("""
        <div class="main-header">
            <h1>MODEL-W</h1>
            <p>Describe the music you want. We'll make it.</p>
        </div>
        """)

        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="dark trap beat, D minor, 140 BPM, heavy 808s, aggressive...",
                lines=3,
                elem_classes=["prompt-box"],
            )

            generate_btn = gr.Button("Generate", variant="primary", size="lg")

            audio_out = gr.Audio(label="Output", type="filepath")
            info = gr.Markdown("")

            with gr.Accordion("Settings", open=False):
                with gr.Row():
                    duration = gr.Slider(10, 300, value=30, step=5, label="Duration (seconds)")
                    seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                with gr.Row():
                    steps = gr.Slider(4, 50, value=8, step=1, label="Inference steps")
                    guidance = gr.Slider(1.0, 15.0, value=7.0, step=0.5, label="Guidance scale")

            gr.Examples(
                examples=EXAMPLES,
                inputs=[prompt],
                label="Try these",
            )

        generate_btn.click(
            fn=generate,
            inputs=[prompt, duration, seed, steps, guidance],
            outputs=[audio_out, info],
        )

        gr.Markdown("""
        ---
        <center style="color: #64748b; font-size: 0.85em;">
        MODEL-W v0.2.0
        </center>
        """)

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    os.makedirs(_ROOT / "output" / "audio", exist_ok=True)

    status = "READY" if ACESTEP_INSTALLED else "NOT INSTALLED (run: python scripts/setup_acestep.py)"
    print(f"\n  MODEL-W v0.2.0 | Audio engine: {status}\n")

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
