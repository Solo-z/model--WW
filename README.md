---
title: MODEL-W
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
python_version: "3.10"
---

**Space build note:** README `python_version` is `3.10` so Linux installs `basic-pitch` with `tflite-runtime` instead of full TensorFlow (Python3.11+ on Linux pulls TensorFlow and conflicts with Demucs/PyTorch).

# MODEL-W

**Music Foundation Model — MIDI + Audio Generation**

Combines a custom MIDI transformer (symbolic music) with [ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5) (audio generation) for end-to-end controllable music production.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.1+-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Architecture

```
text prompt
    │
    ▼
ACE-Step LM planner (fine-tuned on session specs)
    │
    ▼
MODEL-W session JSON (style, key, tracks, arrangement, MIDI clips)
    │
    ├──► MIDI tokens ──► MODEL-W transformer ──► refined MIDI ──► DAW
    │                                                │
    ▼                                                ▼
ACE-Step DiT (conditioned on structure) ──────────► audio
```

**Two output paths:**
- **MIDI path**: Editable, per-track, per-note control — import into any DAW
- **Audio path**: Rendered audio via ACE-Step's diffusion transformer — instant playback

## Features

- **ACE-Step 1.5 as main model**: MIT-licensed audio foundation model, runs on <4GB VRAM
- **Session specs**: Structured JSON format (`daw_session_spec_v0.3`) describing full songs with tracks, arrangement, clips, mixing, and routing
- **MIDI transformer**: GPT-style model with RoPE, SwiGLU, style/key/role/section conditioning
- **200+ synthetic sessions**: Generated training corpus covering 12 styles, 24 keys, varied tempos
- **End-to-end pipeline**: Text prompt → session spec → MIDI → audio
- **LoRA fine-tuning**: Train ACE-Step's LM planner on your own session data
- **DAW ready**: Standard MIDI output compatible with any DAW

## Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/MODEL-W.git
cd MODEL-W
pip install -e .

# Set up ACE-Step as the audio backend
python scripts/setup_acestep.py
```

### 2. Preview captions from session specs (no GPU)

```bash
python -m modelw.acestep_bridge synthetic/sessions/corpus_200
```

### 3. Generate audio from a text prompt

```bash
python scripts/generate_audio.py \
  --caption "dark trap beat, D minor, 140 BPM, aggressive" \
  --duration 60 --out output/audio
```

### 4. Generate audio from session specs

```bash
# Single session
python scripts/generate_audio.py \
  --session synthetic/sessions/example_trap_fullsong.json \
  --out output/audio

# Batch (all 200 corpus files)
python scripts/generate_audio.py \
  --sessions synthetic/sessions/corpus_200 \
  --out output/audio --max-files 10
```

### 5. Generate MIDI only (MODEL-W transformer)

```python
from modelw import MIDIGenerator, MIDITransformer, MIDITokenizer

tokenizer = MIDITokenizer.load("./checkpoints/tokenizer")
model = MIDITransformer.load("./checkpoints/best_model.pt")
generator = MIDIGenerator(model, tokenizer)

generator.generate_dataset(
    num_samples=10,
    output_dir="./output/midi",
    prompts=[
        {"tempo": 140, "instrument": "drums", "mood": "dark", "style": "trap"},
    ]
)
```

## Training

### MIDI Transformer

```bash
# Train with Lakh MIDI + session corpus blended
python -m modelw.trainer \
  --data_dir ./data/lakh_midi \
  --session_dir ./synthetic/sessions/corpus_200 \
  --session_blend_ratio 0.5 \
  --model_size base \
  --batch_size 32

# Or from a YAML config
python -m modelw.trainer \
  --config_file configs/train_base.yaml \
  --session_dir ./synthetic/sessions/corpus_200
```

### ACE-Step LM Fine-tune (LoRA)

Fine-tune the ACE-Step planner to generate MODEL-W session specs from text prompts:

```bash
# See configs/acestep_finetune.yaml for full config
# Uses ACE-Step's built-in LoRA training — ~1 hour on a 3090
```

### Generate more training data

```bash
python scripts/generate_session_corpus.py --count 1000 --out synthetic/sessions/corpus_1k
```

## Model Zoo

### MIDI Transformer (MODEL-W)

| Size | Params | Layers | d_model | Heads | d_ff |
|------|--------|--------|---------|-------|------|
| Small | 45M | 8 | 512 | 8 | 2048 |
| Base | 125M | 12 | 768 | 12 | 3072 |
| Large | 350M | 24 | 1024 | 16 | 4096 |
| XL | 770M | 32 | 1536 | 24 | 6144 |

### ACE-Step 1.5 (Audio Foundation)

| Component | Model | Params | VRAM |
|-----------|-------|--------|------|
| DiT (turbo) | `acestep-v15-turbo` | 2B | <4GB |
| DiT (XL turbo) | `acestep-v15-xl-turbo` | 4B | 12-20GB |
| LM planner | `acestep-5Hz-lm-0.6B` | 0.6B | ~2GB |
| LM planner | `acestep-5Hz-lm-1.7B` | 1.7B | ~4GB |
| LM planner | `acestep-5Hz-lm-4B` | 4B | ~8GB |

## Project Structure

```
MODEL-W/
├── modelw/
│   ├── __init__.py
│   ├── tokenizer.py          # MIDI tokenization (REMI-style)
│   ├── model.py              # Transformer (RoPE, SwiGLU, Flash Attn)
│   ├── dataset.py            # Lakh + SessionDataset + blending
│   ├── trainer.py            # DDP training + YAML config loading
│   ├── generate.py           # MIDI generation farm
│   ├── acestep_bridge.py     # ACE-Step integration layer
│   ├── api.py                # High-level API
│   └── eval_metrics.py       # Quality metrics
├── scripts/
│   ├── setup_acestep.py      # ACE-Step installation
│   ├── generate_audio.py     # Session → audio pipeline
│   ├── generate_session_corpus.py  # Synthetic session generator
│   ├── train_lambda.sh       # Lambda Cloud training
│   └── ...
├── configs/
│   ├── train_base.yaml
│   ├── acestep_finetune.yaml # LoRA fine-tune config
│   └── ...
├── synthetic/
│   └── sessions/
│       ├── example_trap_fullsong.json
│       └── corpus_200/       # 200 generated sessions
├── tests/
│   └── test_core.py          # 14 tests (tokenizer, model, dataset)
├── models/
│   └── ace-step/             # ACE-Step 1.5 (cloned here)
└── requirements.txt
```

## Session Spec Format

Sessions use `daw_session_spec_v0.3` — a structured JSON format:

```json
{
  "metadata": { "style": "trap", "duration_bars": 64 },
  "project": { "tempo_map": [{"bar": 1, "bpm": 140}], "key": "D minor" },
  "arrangement": {
    "sections": [
      {"name": "intro", "bar_start": 1, "bar_end": 8},
      {"name": "verse", "bar_start": 9, "bar_end": 24}
    ]
  },
  "libraries": {
    "clip_library": {
      "pat_drums_main": { "notes": [...], "length_bars": 4 }
    }
  },
  "tracks": [
    { "role": "drums", "timeline": [{"ref": "pat_drums_main", "start_bar": 1}] }
  ]
}
```

## Tests

```bash
pytest tests/ -v
```

14 tests covering tokenizer, model forward/generation, and SessionDataset.

## License

MIT — both MODEL-W and ACE-Step 1.5 are MIT licensed.

## Acknowledgments

- [ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5) by ACE Studio & StepFun
- [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) by Colin Raffel
- Inspired by [REMI](https://arxiv.org/abs/2002.00212) and [Music Transformer](https://magenta.tensorflow.org/music-transformer)
