"""
MODEL-W: Music Foundation Model

Combines a MIDI transformer (symbolic music) with ACE-Step 1.5 (audio
generation) for end-to-end controllable music production.

  text prompt → ACE-Step LM planner → session spec
  session spec → MIDI transformer → refined MIDI
  MIDI + caption → ACE-Step DiT → rendered audio

Designed for DAW integration, synthetic data generation, and
fine-tuning on custom styles via LoRA.
"""

__version__ = "0.2.0"

from modelw.tokenizer import MIDITokenizer
from modelw.model import MIDITransformer, MIDITransformerConfig
from modelw.dataset import LakhMIDIDataset, SessionDataset, SessionDatasetConfig
from modelw.generate import MIDIGenerator
from modelw.eval_metrics import MIDIEvaluator, EvaluationConfig
from modelw.acestep_bridge import ACEStepBridge, ACEStepConfig, session_to_caption
from modelw.room import RoomEngine, RoomConfig

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

