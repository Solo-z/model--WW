"""
Core tests for MODEL-W: tokenizer, model, dataset, and session pipeline.

Run:  pytest tests/ -v
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modelw.tokenizer import MIDITokenizer, TokenizerConfig
from modelw.model import MIDITransformer, MIDITransformerConfig, create_model


@pytest.fixture(scope="module")
def tokenizer():
    return MIDITokenizer(TokenizerConfig())


@pytest.fixture(scope="module")
def tiny_model(tokenizer):
    cfg = MIDITransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=256,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
    )
    return MIDITransformer(cfg)


# ── Tokenizer ──────────────────────────────────────────────────────────────

class TestTokenizer:
    def test_vocab_size_positive(self, tokenizer):
        assert tokenizer.vocab_size > 200

    def test_special_tokens_exist(self, tokenizer):
        for tok in ["<BOS>", "<EOS>", "<PAD>", "<SEP>", "<BAR>"]:
            assert tok in tokenizer.token_to_id, f"Missing {tok}"

    def test_conditioning_tokens(self, tokenizer):
        t2i = tokenizer.token_to_id
        assert "<STYLE_TRAP>" in t2i
        assert "<KEY_D_MINOR>" in t2i
        assert "<ROLE_DRUMS>" in t2i
        assert "<SECTION_CHORUS>" in t2i
        assert "<MOOD_DARK>" in t2i

    def test_encode_decode_roundtrip_length(self, tokenizer, tmp_path):
        """Encode a MIDI file and check decode produces PrettyMIDI."""
        import pretty_midi

        pm_in = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0, name="Piano")
        for pitch, start, end in [(60, 0.0, 0.5), (64, 0.5, 1.0), (67, 1.0, 1.5)]:
            inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))
        pm_in.instruments.append(inst)
        midi_path = tmp_path / "roundtrip.mid"
        pm_in.write(str(midi_path))

        tokens = tokenizer.encode(str(midi_path), tempo=120.0)
        assert len(tokens) > 5
        pm_out = tokenizer.decode(tokens)
        assert pm_out is not None

    def test_pad_sequence(self, tokenizer):
        seq = [1, 2, 3]
        padded = tokenizer.pad_sequence(seq, 8)
        assert len(padded) == 8
        assert padded[:3] == seq
        assert all(t == tokenizer.pad_id for t in padded[3:])

    def test_tempo_token_in_range(self, tokenizer):
        tok = tokenizer._tempo_to_token(120.0)
        assert tok.startswith("<TEMPO_")


# ── Model ──────────────────────────────────────────────────────────────────

class TestModel:
    def test_forward_shape(self, tiny_model, tokenizer):
        bsz, seq = 2, 32
        ids = torch.randint(0, tokenizer.vocab_size, (bsz, seq))
        out = tiny_model(ids)
        assert out["logits"].shape == (bsz, seq, tokenizer.vocab_size)
        assert out["loss"] is None

    def test_forward_with_labels(self, tiny_model, tokenizer):
        bsz, seq = 2, 32
        ids = torch.randint(0, tokenizer.vocab_size, (bsz, seq))
        labels = torch.randint(0, tokenizer.vocab_size, (bsz, seq))
        out = tiny_model(ids, labels=labels)
        assert out["loss"] is not None
        assert out["loss"].ndim == 0
        assert out["loss"].item() > 0

    def test_generate_produces_tokens(self, tiny_model, tokenizer):
        prompt = torch.tensor([[tokenizer.bos_id]])
        generated = tiny_model.generate(prompt, max_length=20, temperature=1.0, top_k=10)
        assert generated.shape[1] >= 2

    def test_create_model_presets(self, tokenizer):
        for size in ["tiny", "small"]:
            m = create_model(size, vocab_size=tokenizer.vocab_size)
            assert isinstance(m, MIDITransformer)

    def test_rope_q_k_different_positions(self, tiny_model, tokenizer):
        """With KV-cache, q and k should get different position embeddings."""
        bsz = 1
        prompt = torch.randint(0, tokenizer.vocab_size, (bsz, 16))
        out1 = tiny_model(prompt, use_cache=True)
        past = out1["past_key_values"]

        next_tok = torch.randint(0, tokenizer.vocab_size, (bsz, 1))
        out2 = tiny_model(next_tok, use_cache=True, past_key_values=past)
        assert out2["logits"].shape == (bsz, 1, tokenizer.vocab_size)


# ── SessionDataset ─────────────────────────────────────────────────────────

class TestSessionDataset:
    @pytest.fixture
    def session_dir(self, tmp_path):
        """Write a minimal valid session JSON and return its parent dir."""
        session = {
            "session_id": "test_001",
            "schema_version": "daw_session_spec_v0.3",
            "metadata": {"title": "Test", "style": "trap", "duration_bars": 16},
            "project": {
                "tempo_map": [{"bar": 1, "bpm": 120}],
                "key": "C minor",
            },
            "semantic_song_labels": {
                "mood": [["dark", 0.5], ["calm", 0.3]],
            },
            "arrangement": {
                "sections": [
                    {"name": "intro", "bar_start": 1, "bar_end": 8},
                    {"name": "chorus", "bar_start": 9, "bar_end": 16},
                ]
            },
            "libraries": {
                "clip_library": {
                    "pat_drums": {
                        "type": "midi",
                        "ppq": 480,
                        "timebase": "beats",
                        "length_bars": 4,
                        "notes": [
                            {"pitch": 36, "start_beat": float(i), "duration_beat": 0.25, "velocity": 100}
                            for i in range(16)
                        ] + [
                            {"pitch": 42, "start_beat": i * 0.5, "duration_beat": 0.1, "velocity": 70}
                            for i in range(32)
                        ],
                        "cc": [],
                    }
                }
            },
            "tracks": [
                {
                    "track_id": "T1",
                    "name": "Drums",
                    "role": "drums",
                    "timeline": [
                        {"type": "midi", "ref": "pat_drums", "start_bar": 1, "loop_count": 2},
                        {"type": "midi", "ref": "pat_drums", "start_bar": 9, "loop_count": 2},
                    ],
                }
            ],
        }
        p = tmp_path / "sessions"
        p.mkdir()
        with open(p / "test.json", "w") as f:
            json.dump(session, f)
        return str(p)

    def test_session_dataset_loads(self, session_dir, tokenizer, tmp_path):
        from modelw.dataset import SessionDataset, SessionDatasetConfig

        cfg = SessionDatasetConfig(
            sessions_dir=session_dir,
            cache_dir=str(tmp_path / "cache"),
            train_split=1.0,
        )
        ds = SessionDataset(cfg, tokenizer, split="train", preprocess=True)
        assert len(ds) > 0

    def test_session_sample_shape(self, session_dir, tokenizer, tmp_path):
        from modelw.dataset import SessionDataset, SessionDatasetConfig

        cfg = SessionDatasetConfig(
            sessions_dir=session_dir,
            cache_dir=str(tmp_path / "cache_shape"),
            max_seq_len=512,
            train_split=1.0,
        )
        ds = SessionDataset(cfg, tokenizer, split="train", preprocess=True)
        sample = ds[0]
        assert sample["input_ids"].shape == sample["labels"].shape
        assert sample["input_ids"].shape[0] == 511  # max_seq_len - 1

    def test_labels_mask_padding(self, session_dir, tokenizer, tmp_path):
        from modelw.dataset import SessionDataset, SessionDatasetConfig

        cfg = SessionDatasetConfig(
            sessions_dir=session_dir,
            cache_dir=str(tmp_path / "cache_pad"),
            max_seq_len=512,
            train_split=1.0,
        )
        ds = SessionDataset(cfg, tokenizer, split="train", preprocess=True)
        sample = ds[0]
        assert (sample["labels"] == -100).any(), "Should have padding masked with -100"
