"""
MIDI Generation Module

High-throughput generation for synthetic data creation.
Supports batch generation, parallel processing, and quality filtering.
"""

import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from modelw.tokenizer import MIDITokenizer
from modelw.model import MIDITransformer, MIDITransformerConfig


@dataclass
class GenerationConfig:
    """Configuration for MIDI generation."""
    
    # Generation parameters
    max_length: int = 1024
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.92
    repetition_penalty: float = 1.15
    
    # Conditioning ranges (for random sampling)
    tempo_range: tuple = (60, 180)
    instruments: list = field(default_factory=lambda: [
        "piano", "guitar", "bass", "strings", "synth_lead", "synth_pad"
    ])
    moods: list = field(default_factory=lambda: [
        "happy", "sad", "energetic", "calm", "dark", "bright", "epic", "mysterious"
    ])
    
    # Batch generation
    batch_size: int = 8
    num_workers: int = 4
    
    # Quality filtering
    min_notes: int = 20
    max_notes: int = 2000
    min_unique_pitches: int = 5
    min_bars: int = 4


class MIDIGenerator:
    """
    High-throughput MIDI generator for synthetic data creation.
    
    Features:
    - Batch generation with conditioning
    - Parallel MIDI decoding
    - Quality filtering
    - Progress tracking and statistics
    """
    
    def __init__(
        self,
        model: MIDITransformer,
        tokenizer: MIDITokenizer,
        config: GenerationConfig = None,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self.device = device
        
        self.model.eval()
    
    def _create_prompt(
        self,
        tempo: Optional[int] = None,
        instrument: Optional[str] = None,
        mood: Optional[str] = None,
    ) -> list[int]:
        """Create conditioning prompt tokens."""
        tokens = ["<BOS>"]
        
        # Tempo
        if tempo is None:
            tempo = random.randint(*self.config.tempo_range)
        tempo_token = f"<TEMPO_{tempo}>"
        if tempo_token in self.tokenizer.token_to_id:
            tokens.append(tempo_token)
        else:
            # Find closest
            for t in self.tokenizer.tempo_tokens:
                if abs(int(t[7:-1]) - tempo) < 10:
                    tokens.append(t)
                    break
        
        # Instrument
        if instrument is None:
            instrument = random.choice(self.config.instruments)
        inst_token = f"<INST_{instrument.upper()}>"
        if inst_token in self.tokenizer.token_to_id:
            tokens.append(inst_token)
        
        # Mood
        if mood is None:
            mood = random.choice(self.config.moods)
        mood_token = f"<MOOD_{mood.upper()}>"
        if mood_token in self.tokenizer.token_to_id:
            tokens.append(mood_token)
        
        tokens.append("<SEP>")
        tokens.append("<BAR>")  # Start first bar
        
        return [self.tokenizer.token_to_id.get(t, self.tokenizer.unk_id) for t in tokens]
    
    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[dict],
        show_progress: bool = True,
    ) -> list[dict]:
        """
        Generate a batch of MIDI sequences.
        
        Args:
            prompts: List of dicts with tempo, instrument, mood
            show_progress: Show progress bar
            
        Returns:
            List of generation results with tokens and metadata
        """
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(prompts), self.config.batch_size), 
                      disable=not show_progress, desc="Generating"):
            batch_prompts = prompts[i:i + self.config.batch_size]
            
            # Create input tensors
            prompt_tokens = [self._create_prompt(**p) for p in batch_prompts]
            max_prompt_len = max(len(p) for p in prompt_tokens)
            
            # Pad prompts
            padded = [
                p + [self.tokenizer.pad_id] * (max_prompt_len - len(p))
                for p in prompt_tokens
            ]
            input_ids = torch.tensor(padded, device=self.device)
            
            # Generate
            output_ids = self.model.generate(
                input_ids,
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                eos_token_id=self.tokenizer.eos_id,
                pad_token_id=self.tokenizer.pad_id,
            )
            
            # Collect results
            for j, (tokens, prompt) in enumerate(zip(output_ids, batch_prompts)):
                results.append({
                    "tokens": tokens.cpu().tolist(),
                    "prompt": prompt,
                    "length": len(tokens),
                })
        
        return results
    
    def _quality_check(self, token_ids: list[int]) -> bool:
        """Check if generated sequence meets quality criteria."""
        tokens = [self.tokenizer.id_to_token.get(t, "") for t in token_ids]
        
        # Count notes
        note_count = sum(1 for t in tokens if t.startswith("<PITCH_"))
        if note_count < self.config.min_notes or note_count > self.config.max_notes:
            return False
        
        # Count unique pitches
        pitches = set()
        for t in tokens:
            if t.startswith("<PITCH_"):
                try:
                    pitches.add(int(t[7:-1]))
                except:
                    pass
        if len(pitches) < self.config.min_unique_pitches:
            return False
        
        # Count bars
        bar_count = sum(1 for t in tokens if t == "<BAR>")
        if bar_count < self.config.min_bars:
            return False
        
        return True
    
    def generate_dataset(
        self,
        num_samples: int,
        output_dir: str,
        save_midi: bool = True,
        save_tokens: bool = True,
        quality_filter: bool = True,
        max_retries: int = 3,
    ) -> dict:
        """
        Generate a large dataset of MIDI files.
        
        Args:
            num_samples: Number of samples to generate
            output_dir: Output directory
            save_midi: Save decoded MIDI files
            save_tokens: Save token sequences
            quality_filter: Apply quality filtering
            max_retries: Max retries for quality-failed generations
            
        Returns:
            Generation statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if save_midi:
            (output_path / "midi").mkdir(exist_ok=True)
        if save_tokens:
            (output_path / "tokens").mkdir(exist_ok=True)
        
        stats = {
            "total_generated": 0,
            "passed_quality": 0,
            "failed_quality": 0,
            "save_errors": 0,
            "start_time": time.time(),
        }
        
        metadata = []
        generated = 0
        retry_count = 0
        
        pbar = tqdm(total=num_samples, desc="Generating dataset")
        
        while generated < num_samples:
            # Generate batch of prompts
            batch_size = min(self.config.batch_size * 4, num_samples - generated)
            prompts = [
                {
                    "tempo": random.randint(*self.config.tempo_range),
                    "instrument": random.choice(self.config.instruments),
                    "mood": random.choice(self.config.moods),
                }
                for _ in range(batch_size)
            ]
            
            # Generate
            results = self.generate_batch(prompts, show_progress=False)
            
            for result in results:
                if generated >= num_samples:
                    break
                
                stats["total_generated"] += 1
                
                # Quality check
                if quality_filter and not self._quality_check(result["tokens"]):
                    stats["failed_quality"] += 1
                    retry_count += 1
                    
                    if retry_count >= max_retries * batch_size:
                        retry_count = 0  # Reset to avoid infinite loop
                    continue
                
                stats["passed_quality"] += 1
                
                # Save
                sample_id = f"gen_{generated:08d}"
                
                try:
                    if save_tokens:
                        token_path = output_path / "tokens" / f"{sample_id}.json"
                        with open(token_path, "w") as f:
                            json.dump({
                                "id": sample_id,
                                "tokens": result["tokens"],
                                "prompt": result["prompt"],
                            }, f)
                    
                    if save_midi:
                        midi_path = output_path / "midi" / f"{sample_id}.mid"
                        self.tokenizer.decode(result["tokens"], midi_path)
                    
                    metadata.append({
                        "id": sample_id,
                        "prompt": result["prompt"],
                        "length": result["length"],
                    })
                    
                    generated += 1
                    pbar.update(1)
                    
                except Exception as e:
                    stats["save_errors"] += 1
        
        pbar.close()
        
        # Save metadata
        stats["end_time"] = time.time()
        stats["duration_seconds"] = stats["end_time"] - stats["start_time"]
        stats["samples_per_second"] = num_samples / stats["duration_seconds"]
        
        with open(output_path / "metadata.json", "w") as f:
            json.dump({
                "stats": stats,
                "samples": metadata,
            }, f, indent=2)
        
        print(f"\n✓ Generated {generated} samples in {stats['duration_seconds']:.1f}s")
        print(f"  Rate: {stats['samples_per_second']:.2f} samples/sec")
        print(f"  Quality pass rate: {stats['passed_quality']/stats['total_generated']*100:.1f}%")
        
        return stats


class GenerationFarm:
    """
    Distributed generation farm for producing millions of synthetic MIDI files.
    
    Supports:
    - Multi-GPU generation
    - Job queue with Redis (optional)
    - Checkpoint and resume
    - Automatic scaling
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        output_dir: str,
        num_gpus: int = None,
        config: GenerationConfig = None,
    ):
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.output_dir = Path(output_dir)
        self.config = config or GenerationConfig()
        
        # Detect GPUs
        if num_gpus is None:
            self.num_gpus = torch.cuda.device_count()
        else:
            self.num_gpus = min(num_gpus, torch.cuda.device_count())
        
        print(f"Generation Farm initialized with {self.num_gpus} GPUs")
    
    def _worker(self, gpu_id: int, num_samples: int, worker_output_dir: Path) -> dict:
        """Worker process for generation on a single GPU."""
        import torch
        
        device = f"cuda:{gpu_id}"
        
        # Load model and tokenizer
        tokenizer = MIDITokenizer.load(self.tokenizer_path)
        
        checkpoint = torch.load(self.model_path, map_location=device)
        config = checkpoint.get("config", MIDITransformerConfig())
        if isinstance(config, dict):
            config = MIDITransformerConfig(**config)
        config.vocab_size = tokenizer.vocab_size
        
        model = MIDITransformer(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        
        # Create generator
        generator = MIDIGenerator(model, tokenizer, self.config, device)
        
        # Generate
        stats = generator.generate_dataset(
            num_samples=num_samples,
            output_dir=str(worker_output_dir),
            save_midi=True,
            save_tokens=True,
            quality_filter=True,
        )
        
        return stats
    
    def run(self, total_samples: int, resume: bool = True) -> dict:
        """
        Run the generation farm.
        
        Args:
            total_samples: Total number of samples to generate
            resume: Resume from checkpoint if exists
            
        Returns:
            Aggregated statistics
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing progress
        checkpoint_path = self.output_dir / "farm_checkpoint.json"
        start_sample = 0
        
        if resume and checkpoint_path.exists():
            with open(checkpoint_path) as f:
                checkpoint = json.load(f)
                start_sample = checkpoint.get("completed", 0)
                print(f"Resuming from sample {start_sample}")
        
        remaining = total_samples - start_sample
        samples_per_gpu = remaining // self.num_gpus
        
        print(f"Generating {remaining} samples across {self.num_gpus} GPUs")
        print(f"  {samples_per_gpu} samples per GPU")
        
        # Run workers
        all_stats = []
        
        with ProcessPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = []
            
            for gpu_id in range(self.num_gpus):
                worker_dir = self.output_dir / f"worker_{gpu_id}"
                num_samples = samples_per_gpu
                
                # Last worker gets remainder
                if gpu_id == self.num_gpus - 1:
                    num_samples += remaining % self.num_gpus
                
                future = executor.submit(
                    self._worker, gpu_id, num_samples, worker_dir
                )
                futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    stats = future.result()
                    all_stats.append(stats)
                except Exception as e:
                    print(f"Worker error: {e}")
        
        # Aggregate stats
        total_stats = {
            "total_generated": sum(s["total_generated"] for s in all_stats),
            "passed_quality": sum(s["passed_quality"] for s in all_stats),
            "failed_quality": sum(s["failed_quality"] for s in all_stats),
            "save_errors": sum(s["save_errors"] for s in all_stats),
            "duration_seconds": max(s["duration_seconds"] for s in all_stats),
        }
        
        total_stats["samples_per_second"] = total_stats["passed_quality"] / total_stats["duration_seconds"]
        
        # Save checkpoint
        with open(checkpoint_path, "w") as f:
            json.dump({
                "completed": start_sample + total_stats["passed_quality"],
                "target": total_samples,
                "stats": total_stats,
            }, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Generation Farm Complete")
        print(f"  Total generated: {total_stats['passed_quality']}")
        print(f"  Duration: {total_stats['duration_seconds']:.1f}s")
        print(f"  Rate: {total_stats['samples_per_second']:.2f} samples/sec")
        print(f"{'='*60}")
        
        return total_stats


def generate_cli():
    """CLI interface for generation."""
    import fire
    
    def generate(
        model_path: str,
        tokenizer_path: str = "./checkpoints/tokenizer",
        output_dir: str = "./generated",
        num_samples: int = 1000,
        batch_size: int = 8,
        temperature: float = 0.9,
        top_p: float = 0.92,
        device: str = "cuda",
    ):
        """Generate MIDI files from trained model."""
        
        # Load tokenizer
        tokenizer = MIDITokenizer.load(tokenizer_path)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get("config", MIDITransformerConfig())
        if isinstance(config, dict):
            config = MIDITransformerConfig(**config)
        config.vocab_size = tokenizer.vocab_size
        
        model = MIDITransformer(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Create generator
        gen_config = GenerationConfig(
            batch_size=batch_size,
            temperature=temperature,
            top_p=top_p,
        )
        generator = MIDIGenerator(model, tokenizer, gen_config, device)
        
        # Generate
        generator.generate_dataset(
            num_samples=num_samples,
            output_dir=output_dir,
        )
    
    fire.Fire(generate)


if __name__ == "__main__":
    generate_cli()

