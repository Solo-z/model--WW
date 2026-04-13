"""
MODEL-W: Music Foundation Model

Combines a MIDI transformer (symbolic music) with ACE-Step 1.5 (audio
generation) for end-to-end controllable music production.

Heavy imports (torch, ACE-Step) are lazy so ``from modelw.room import RoomEngine``
works without pulling the full stack until you call into it.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = [
    "MIDITokenizer",
    "MIDITransformer",
    "MIDITransformerConfig",
    "LakhMIDIDataset",
    "SessionDataset",
    "SessionDatasetConfig",
    "MIDIGenerator",
    "MIDIEvaluator",
    "EvaluationConfig",
    "ACEStepBridge",
    "ACEStepConfig",
    "session_to_caption",
    "RoomEngine",
    "RoomConfig",
]


def __getattr__(name: str):
    if name == "MIDITokenizer":
        from modelw.tokenizer import MIDITokenizer

        return MIDITokenizer
    if name in ("MIDITransformer", "MIDITransformerConfig"):
        from modelw import model as _model

        return getattr(_model, name)
    if name in ("LakhMIDIDataset", "SessionDataset", "SessionDatasetConfig"):
        from modelw import dataset as _dataset

        return getattr(_dataset, name)
    if name == "MIDIGenerator":
        from modelw.generate import MIDIGenerator

        return MIDIGenerator
    if name in ("MIDIEvaluator", "EvaluationConfig"):
        from modelw import eval_metrics as _eval

        return getattr(_eval, name)
    if name in ("ACEStepBridge", "ACEStepConfig", "session_to_caption"):
        from modelw import acestep_bridge as _bridge

        return getattr(_bridge, name)
    if name in ("RoomEngine", "RoomConfig"):
        from modelw import room as _room

        return getattr(_room, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
