#!/usr/bin/env python3
"""
MODEL-W Demo Script

Quick demonstration of MIDI generation capabilities.
"""

import argparse
from pathlib import Path

import torch

from modelw import MIDITokenizer, MIDITransformer, MIDITransformerConfig, MIDIGenerator
from modelw.generate import GenerationConfig


def demo_tokenizer():
    """Demonstrate MIDI tokenization."""
    print("\n" + "="*60)
    print("MIDI Tokenizer Demo")
    print("="*60)
    
    tokenizer = MIDITokenizer()
    
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print(f"\nSpecial tokens:")
    for token in tokenizer.special_tokens:
        print(f"  {token}: {tokenizer.token_to_id[token]}")
    
    print(f"\nTempo tokens: {len(tokenizer.tempo_tokens)}")
    print(f"  Range: {tokenizer.tempo_tokens[0]} to {tokenizer.tempo_tokens[-1]}")
    
    print(f"\nInstrument tokens: {len(tokenizer.instrument_tokens)}")
    for token in tokenizer.instrument_tokens[:5]:
        print(f"  {token}")
    print("  ...")
    
    print(f"\nMood tokens: {len(tokenizer.mood_tokens)}")
    for token in tokenizer.mood_tokens[:5]:
        print(f"  {token}")
    print("  ...")
    
    return tokenizer


def demo_model(tokenizer: MIDITokenizer):
    """Demonstrate model creation."""
    print("\n" + "="*60)
    print("MIDI Transformer Demo")
    print("="*60)
    
    # Create small model for demo
    config = MIDITransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        max_seq_len=512,
    )
    
    model = MIDITransformer(config)
    
    num_params = model.get_num_params()
    print(f"\nModel created:")
    print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"  d_model: {config.d_model}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Heads: {config.n_heads}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 64
    
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
    
    outputs = model(input_ids, labels=labels)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    
    return model


def demo_generation(model: MIDITransformer, tokenizer: MIDITokenizer, output_dir: str):
    """Demonstrate MIDI generation."""
    print("\n" + "="*60)
    print("MIDI Generation Demo")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    model = model.to(device)
    
    gen_config = GenerationConfig(
        max_length=256,
        temperature=0.9,
        top_p=0.92,
        batch_size=2,
    )
    
    generator = MIDIGenerator(model, tokenizer, gen_config, device)
    
    # Generate with different prompts
    prompts = [
        {"tempo": 120, "instrument": "piano", "mood": "happy"},
        {"tempo": 60, "instrument": "strings", "mood": "sad"},
        {"tempo": 140, "instrument": "synth_lead", "mood": "energetic"},
        {"tempo": 80, "instrument": "guitar", "mood": "calm"},
    ]
    
    print(f"\nGenerating {len(prompts)} samples...")
    
    results = generator.generate_batch(prompts)
    
    for i, (result, prompt) in enumerate(zip(results, prompts)):
        print(f"\n  Sample {i+1}:")
        print(f"    Prompt: tempo={prompt['tempo']}, instrument={prompt['instrument']}, mood={prompt['mood']}")
        print(f"    Length: {result['length']} tokens")
        
        # Decode to MIDI
        midi_path = output_path / f"demo_{i+1}.mid"
        pm = tokenizer.decode(result["tokens"], midi_path)
        print(f"    Saved: {midi_path}")
        print(f"    Notes: {len(pm.instruments[0].notes)}")
    
    print(f"\n✓ Generated {len(prompts)} MIDI files in {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MODEL-W Demo")
    parser.add_argument(
        "--output",
        type=str,
        default="./demo_output",
        help="Output directory for generated MIDI files",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation demo (useful without GPU)",
    )
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗      ██╗    ██╗   ║
    ║   ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║      ██║    ██║   ║
    ║   ██╔████╔██║██║   ██║██║  ██║█████╗  ██║█████╗██║ █╗ ██║   ║
    ║   ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║╚════╝██║███╗██║   ║
    ║   ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗ ╚███╔███╔╝   ║
    ║   ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝  ╚══╝╚══╝    ║
    ║                                                              ║
    ║              MIDI Generation Foundation Model                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Demo tokenizer
    tokenizer = demo_tokenizer()
    
    # Demo model
    model = demo_model(tokenizer)
    
    # Demo generation
    if not args.skip_generation:
        demo_generation(model, tokenizer, args.output)
    else:
        print("\n(Skipping generation demo)")
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("""
Next steps:
    1. Download the Lakh MIDI dataset:
       python scripts/download_lakh.py --output ./data/lakh_midi
    
    2. Train the model:
       python -m modelw.trainer --data_dir=./data/lakh_midi
    
    3. Generate MIDI:
       python -m modelw.generate --model_path=./checkpoints/best_model.pt
    """)


if __name__ == "__main__":
    main()

