"""
Simple API for MIDI Generation

Provides a clean interface for DAW integration and batch generation.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from modelw.tokenizer import MIDITokenizer
from modelw.model import MIDITransformer, MIDITransformerConfig
from modelw.generate import MIDIGenerator, GenerationConfig


@dataclass 
class MIDIPrompt:
    """Prompt specification for MIDI generation."""
    
    tempo: int = 120
    instrument: str = "piano"
    mood: str = "happy"
    
    # Optional fine-grained control
    time_signature: str = "4/4"
    key: Optional[str] = None
    duration_bars: int = 16


class ModelW:
    """
    Main API for MODEL-W MIDI generation.
    
    Example:
        model = ModelW.load("./checkpoints")
        midi = model.generate(tempo=120, instrument="piano", mood="happy")
        midi.write("output.mid")
    """
    
    def __init__(
        self,
        model: MIDITransformer,
        tokenizer: MIDITokenizer,
        device: str = "auto",
        default_config: Optional[GenerationConfig] = None,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = default_config or GenerationConfig()
        self._generator = MIDIGenerator(model, tokenizer, self.config, device)
    
    @classmethod
    def load(
        cls,
        checkpoint_path: Union[str, Path],
        device: str = "auto",
    ) -> "ModelW":
        """
        Load a trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory or .pt file
            device: Device to load model on ("auto", "cuda", "cpu")
            
        Returns:
            ModelW instance ready for generation
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Handle directory or file
        if checkpoint_path.is_dir():
            model_path = checkpoint_path / "best_model.pt"
            tokenizer_path = checkpoint_path / "tokenizer"
        else:
            model_path = checkpoint_path
            tokenizer_path = checkpoint_path.parent / "tokenizer"
        
        # Load tokenizer
        tokenizer = MIDITokenizer.load(tokenizer_path)
        
        # Load model
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint = torch.load(model_path, map_location=device)
        
        config_dict = checkpoint.get("config", {})
        if isinstance(config_dict, MIDITransformerConfig):
            config = config_dict
        else:
            config = MIDITransformerConfig(**config_dict)
        config.vocab_size = tokenizer.vocab_size
        
        model = MIDITransformer(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return cls(model, tokenizer, device)
    
    def generate(
        self,
        tempo: int = 120,
        instrument: str = "piano",
        mood: str = "happy",
        max_length: int = 1024,
        temperature: float = 0.9,
        top_p: float = 0.92,
        output_path: Optional[str] = None,
    ):
        """
        Generate a single MIDI file.
        
        Args:
            tempo: BPM (40-240)
            instrument: Instrument type (piano, guitar, bass, strings, etc.)
            mood: Mood/emotion (happy, sad, energetic, calm, etc.)
            max_length: Maximum sequence length
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            output_path: Optional path to save MIDI file
            
        Returns:
            pretty_midi.PrettyMIDI object
        """
        # Update config
        self.config.max_length = max_length
        self.config.temperature = temperature
        self.config.top_p = top_p
        
        # Generate
        results = self._generator.generate_batch([
            {"tempo": tempo, "instrument": instrument, "mood": mood}
        ], show_progress=False)
        
        # Decode
        midi = self.tokenizer.decode(results[0]["tokens"], output_path)
        
        return midi
    
    def generate_batch(
        self,
        prompts: list[dict],
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> list:
        """
        Generate multiple MIDI files.
        
        Args:
            prompts: List of prompt dicts with tempo, instrument, mood
            output_dir: Directory to save MIDI files
            **kwargs: Generation parameters
            
        Returns:
            List of pretty_midi.PrettyMIDI objects
        """
        # Update config
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        
        # Generate
        results = self._generator.generate_batch(prompts)
        
        # Decode
        midis = []
        output_path = Path(output_dir) if output_dir else None
        
        for i, result in enumerate(results):
            save_path = None
            if output_path:
                output_path.mkdir(parents=True, exist_ok=True)
                save_path = output_path / f"generated_{i:04d}.mid"
            
            midi = self.tokenizer.decode(result["tokens"], save_path)
            midis.append(midi)
        
        return midis
    
    def generate_dataset(
        self,
        num_samples: int,
        output_dir: str,
        **kwargs,
    ) -> dict:
        """
        Generate a large dataset of MIDI files.
        
        Args:
            num_samples: Number of samples to generate
            output_dir: Output directory
            **kwargs: Generation parameters
            
        Returns:
            Generation statistics
        """
        return self._generator.generate_dataset(
            num_samples=num_samples,
            output_dir=output_dir,
            **kwargs,
        )
    
    @property
    def available_instruments(self) -> list[str]:
        """List of available instrument types."""
        return [
            t[6:-1].lower() for t in self.tokenizer.instrument_tokens
        ]
    
    @property
    def available_moods(self) -> list[str]:
        """List of available mood types."""
        return [
            t[6:-1].lower() for t in self.tokenizer.mood_tokens
        ]
    
    def __repr__(self) -> str:
        num_params = self.model.get_num_params()
        return f"ModelW(params={num_params/1e6:.1f}M, device={self.device})"


# Simple functional API
_default_model: Optional[ModelW] = None


def load_model(checkpoint_path: str, device: str = "auto") -> ModelW:
    """Load a MODEL-W checkpoint."""
    global _default_model
    _default_model = ModelW.load(checkpoint_path, device)
    return _default_model


def generate(
    tempo: int = 120,
    instrument: str = "piano", 
    mood: str = "happy",
    output_path: Optional[str] = None,
    **kwargs,
):
    """
    Generate MIDI with the default loaded model.
    
    Must call load_model() first.
    """
    if _default_model is None:
        raise RuntimeError("No model loaded. Call load_model() first.")
    
    return _default_model.generate(
        tempo=tempo,
        instrument=instrument,
        mood=mood,
        output_path=output_path,
        **kwargs,
    )


if __name__ == "__main__":
    # Quick test
    import fire
    
    def main(
        checkpoint: str,
        tempo: int = 120,
        instrument: str = "piano",
        mood: str = "happy",
        output: str = "output.mid",
    ):
        """Generate MIDI from command line."""
        model = ModelW.load(checkpoint)
        print(f"Loaded {model}")
        
        print(f"\nGenerating: tempo={tempo}, instrument={instrument}, mood={mood}")
        midi = model.generate(tempo, instrument, mood, output_path=output)
        
        print(f"Saved to {output}")
        print(f"Notes: {len(midi.instruments[0].notes)}")
        print(f"Duration: {midi.get_end_time():.1f}s")
    
    fire.Fire(main)

