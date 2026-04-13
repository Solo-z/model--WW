#!/usr/bin/env python3
"""
Test MIDI Tokenizer

See how the tokenizer encodes your MIDI files.
Shows the actual token sequences and reconstructs MIDI.
"""

import argparse
import sys
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelw.tokenizer import MIDITokenizer


def test_single_file(tokenizer: MIDITokenizer, midi_path: str, verbose: bool = True):
    """Test tokenization on a single file."""
    
    print(f"\n{'─'*60}")
    print(f"File: {Path(midi_path).name}")
    print(f"{'─'*60}")
    
    try:
        # Encode
        token_ids = tokenizer.encode(midi_path)
        tokens = [tokenizer.id_to_token.get(tid, "<UNK>") for tid in token_ids]
        
        print(f"✓ Encoded: {len(token_ids)} tokens")
        
        # Show token breakdown
        token_types = {
            "special": 0,
            "tempo": 0,
            "instrument": 0,
            "mood": 0,
            "bar": 0,
            "position": 0,
            "pitch": 0,
            "velocity": 0,
            "duration": 0,
        }
        
        for t in tokens:
            if t.startswith("<TEMPO"):
                token_types["tempo"] += 1
            elif t.startswith("<INST"):
                token_types["instrument"] += 1
            elif t.startswith("<MOOD"):
                token_types["mood"] += 1
            elif t == "<BAR>":
                token_types["bar"] += 1
            elif t.startswith("<POS"):
                token_types["position"] += 1
            elif t.startswith("<PITCH"):
                token_types["pitch"] += 1
            elif t.startswith("<VEL"):
                token_types["velocity"] += 1
            elif t.startswith("<DUR"):
                token_types["duration"] += 1
            else:
                token_types["special"] += 1
        
        print(f"\nToken breakdown:")
        for ttype, count in token_types.items():
            if count > 0:
                print(f"  {ttype:12s}: {count:4d}")
        
        # Show first N tokens
        if verbose:
            print(f"\nFirst 50 tokens:")
            for i, t in enumerate(tokens[:50]):
                print(f"  {i:3d}: {t}")
            if len(tokens) > 50:
                print(f"  ... ({len(tokens) - 50} more)")
        
        # Decode back
        print(f"\nDecoding back to MIDI...")
        pm = tokenizer.decode(token_ids)
        
        if pm.instruments:
            notes = pm.instruments[0].notes
            print(f"✓ Reconstructed: {len(notes)} notes, {pm.get_end_time():.1f}s duration")
        else:
            print("⚠️  No instruments in reconstruction")
        
        return True, len(token_ids)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False, 0


def test_dataset(data_dir: str, num_samples: int = 10, verbose: bool = False):
    """Test tokenization on multiple files."""
    
    print(f"\n{'='*60}")
    print("MIDI Tokenizer Test")
    print(f"{'='*60}")
    
    # Create tokenizer
    tokenizer = MIDITokenizer()
    print(f"\nTokenizer vocab size: {tokenizer.vocab_size}")
    
    # Find MIDI files
    data_path = Path(data_dir)
    midi_files = list(data_path.rglob("*.mid")) + list(data_path.rglob("*.midi"))
    
    print(f"Found {len(midi_files)} MIDI files")
    
    if len(midi_files) == 0:
        print("⚠️  No MIDI files found!")
        return
    
    # Sample files
    if len(midi_files) > num_samples:
        midi_files = random.sample(midi_files, num_samples)
    
    print(f"Testing {len(midi_files)} files...")
    
    # Test each file
    success = 0
    total_tokens = 0
    
    for midi_path in midi_files:
        ok, num_tokens = test_single_file(tokenizer, str(midi_path), verbose=verbose)
        if ok:
            success += 1
            total_tokens += num_tokens
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Success rate: {success}/{len(midi_files)} ({success/len(midi_files)*100:.0f}%)")
    print(f"Avg tokens: {total_tokens/success:.0f}" if success > 0 else "N/A")
    print(f"{'='*60}")


def show_vocab(tokenizer: MIDITokenizer):
    """Show the vocabulary."""
    
    print(f"\n{'='*60}")
    print("Tokenizer Vocabulary")
    print(f"{'='*60}")
    print(f"Total vocab size: {tokenizer.vocab_size}\n")
    
    print("Special tokens:")
    for t in tokenizer.special_tokens:
        print(f"  {tokenizer.token_to_id[t]:4d}: {t}")
    
    print(f"\nTempo tokens ({len(tokenizer.tempo_tokens)}):")
    print(f"  {tokenizer.tempo_tokens[0]} ... {tokenizer.tempo_tokens[-1]}")
    
    print(f"\nInstrument tokens ({len(tokenizer.instrument_tokens)}):")
    for t in tokenizer.instrument_tokens:
        print(f"  {tokenizer.token_to_id[t]:4d}: {t}")
    
    print(f"\nMood tokens ({len(tokenizer.mood_tokens)}):")
    for t in tokenizer.mood_tokens:
        print(f"  {tokenizer.token_to_id[t]:4d}: {t}")
    
    print(f"\nPitch tokens: {len(tokenizer.pitch_tokens)}")
    print(f"  {tokenizer.pitch_tokens[0]} (MIDI {tokenizer.config.pitch_range[0]}) to")
    print(f"  {tokenizer.pitch_tokens[-1]} (MIDI {tokenizer.config.pitch_range[1]-1})")
    
    print(f"\nVelocity tokens: {len(tokenizer.velocity_tokens)}")
    print(f"Duration tokens: {len(tokenizer.duration_tokens)}")
    print(f"Position tokens: {len(tokenizer.position_tokens)}")


def main():
    parser = argparse.ArgumentParser(description="Test MIDI tokenizer")
    parser.add_argument(
        "data_dir",
        type=str,
        nargs="?",
        default=None,
        help="Directory containing MIDI files (optional)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Test a specific MIDI file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of files to sample",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full token sequences",
    )
    parser.add_argument(
        "--vocab",
        action="store_true",
        help="Show vocabulary",
    )
    args = parser.parse_args()
    
    tokenizer = MIDITokenizer()
    
    if args.vocab:
        show_vocab(tokenizer)
        return
    
    if args.file:
        test_single_file(tokenizer, args.file, verbose=True)
        return
    
    if args.data_dir:
        test_dataset(args.data_dir, args.num_samples, args.verbose)
    else:
        # Just show vocab if no data
        show_vocab(tokenizer)
        print("\n💡 Usage:")
        print("  python scripts/test_tokenizer.py ./data/lakh_midi")
        print("  python scripts/test_tokenizer.py --file song.mid -v")


if __name__ == "__main__":
    main()

