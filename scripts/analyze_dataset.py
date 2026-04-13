#!/usr/bin/env python3
"""
Analyze MIDI Dataset

Quick analysis of your MIDI files to see:
- Instrument distribution (piano, guitar, drums, etc.)
- Tempo distribution
- Note statistics
- File quality
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random

import numpy as np
import pretty_midi
from tqdm import tqdm


# GM Instrument families
INSTRUMENT_FAMILIES = {
    range(0, 8): "Piano",
    range(8, 16): "Chromatic Percussion", 
    range(16, 24): "Organ",
    range(24, 32): "Guitar",
    range(32, 40): "Bass",
    range(40, 48): "Strings",
    range(48, 56): "Ensemble",
    range(56, 64): "Brass",
    range(64, 72): "Reed",
    range(72, 80): "Pipe",
    range(80, 88): "Synth Lead",
    range(88, 96): "Synth Pad",
    range(96, 104): "Synth Effects",
    range(104, 112): "Ethnic",
    range(112, 120): "Percussive",
    range(120, 128): "Sound Effects",
}


def get_instrument_family(program: int) -> str:
    """Get instrument family name from GM program number."""
    for prog_range, name in INSTRUMENT_FAMILIES.items():
        if program in prog_range:
            return name
    return "Unknown"


def analyze_midi(midi_path: str) -> dict:
    """Analyze a single MIDI file."""
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        return {"valid": False, "error": str(e)}
    
    result = {
        "valid": True,
        "instruments": [],
        "has_drums": False,
        "note_count": 0,
        "duration": pm.get_end_time(),
        "tempos": [],
        "time_signatures": [],
    }
    
    # Get tempo
    tempo_times, tempos = pm.get_tempo_changes()
    result["tempos"] = tempos.tolist() if len(tempos) > 0 else [120.0]
    result["avg_tempo"] = np.mean(result["tempos"])
    
    # Analyze instruments
    for inst in pm.instruments:
        if inst.is_drum:
            result["has_drums"] = True
            result["instruments"].append("Drums")
        else:
            family = get_instrument_family(inst.program)
            result["instruments"].append(family)
        
        result["note_count"] += len(inst.notes)
    
    # Pitch statistics
    all_pitches = []
    all_velocities = []
    for inst in pm.instruments:
        for note in inst.notes:
            all_pitches.append(note.pitch)
            all_velocities.append(note.velocity)
    
    if all_pitches:
        result["pitch_range"] = (min(all_pitches), max(all_pitches))
        result["avg_velocity"] = np.mean(all_velocities)
    
    return result


def analyze_dataset(
    data_dir: str,
    max_files: int = None,
    sample: bool = True,
):
    """Analyze entire dataset."""
    
    data_path = Path(data_dir)
    
    # Find all MIDI files
    midi_files = list(data_path.rglob("*.mid")) + list(data_path.rglob("*.midi"))
    midi_files += list(data_path.rglob("*.MID")) + list(data_path.rglob("*.MIDI"))
    
    print(f"\n{'='*60}")
    print(f"MIDI Dataset Analysis")
    print(f"{'='*60}")
    print(f"Directory: {data_dir}")
    print(f"Total files found: {len(midi_files)}")
    
    if len(midi_files) == 0:
        print("\n⚠️  No MIDI files found!")
        return
    
    # Sample if too many
    if sample and max_files and len(midi_files) > max_files:
        print(f"Sampling {max_files} files for analysis...")
        midi_files = random.sample(midi_files, max_files)
    elif max_files:
        midi_files = midi_files[:max_files]
    
    print(f"Analyzing {len(midi_files)} files...\n")
    
    # Counters
    instrument_counter = Counter()
    tempo_bins = Counter()
    valid_count = 0
    invalid_count = 0
    total_notes = 0
    durations = []
    has_drums_count = 0
    piano_only_count = 0
    multi_instrument_count = 0
    
    for midi_path in tqdm(midi_files, desc="Analyzing"):
        result = analyze_midi(midi_path)
        
        if not result["valid"]:
            invalid_count += 1
            continue
        
        valid_count += 1
        total_notes += result["note_count"]
        durations.append(result["duration"])
        
        # Count instruments
        for inst in result["instruments"]:
            instrument_counter[inst] += 1
        
        if result["has_drums"]:
            has_drums_count += 1
        
        # Check if piano only
        non_drum_instruments = [i for i in result["instruments"] if i != "Drums"]
        if non_drum_instruments == ["Piano"]:
            piano_only_count += 1
        if len(set(non_drum_instruments)) > 1:
            multi_instrument_count += 1
        
        # Bin tempo
        tempo = result["avg_tempo"]
        if tempo < 60:
            tempo_bins["Very Slow (<60)"] += 1
        elif tempo < 90:
            tempo_bins["Slow (60-90)"] += 1
        elif tempo < 120:
            tempo_bins["Medium (90-120)"] += 1
        elif tempo < 150:
            tempo_bins["Fast (120-150)"] += 1
        else:
            tempo_bins["Very Fast (>150)"] += 1
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    
    print(f"\n📊 File Statistics:")
    print(f"   Valid files:    {valid_count} ({valid_count/len(midi_files)*100:.1f}%)")
    print(f"   Invalid files:  {invalid_count}")
    print(f"   Total notes:    {total_notes:,}")
    print(f"   Avg notes/file: {total_notes/valid_count:.0f}")
    
    if durations:
        print(f"\n⏱️  Duration:")
        print(f"   Average: {np.mean(durations):.1f}s")
        print(f"   Median:  {np.median(durations):.1f}s")
        print(f"   Range:   {min(durations):.1f}s - {max(durations):.1f}s")
    
    print(f"\n🎹 Instrument Distribution:")
    for inst, count in instrument_counter.most_common(15):
        pct = count / valid_count * 100
        bar = "█" * int(pct / 2)
        print(f"   {inst:20s} {count:6d} ({pct:5.1f}%) {bar}")
    
    print(f"\n🎼 Composition Types:")
    print(f"   Piano only:       {piano_only_count:6d} ({piano_only_count/valid_count*100:.1f}%)")
    print(f"   With drums:       {has_drums_count:6d} ({has_drums_count/valid_count*100:.1f}%)")
    print(f"   Multi-instrument: {multi_instrument_count:6d} ({multi_instrument_count/valid_count*100:.1f}%)")
    
    print(f"\n🎵 Tempo Distribution:")
    for tempo_range, count in sorted(tempo_bins.items()):
        pct = count / valid_count * 100
        bar = "█" * int(pct / 2)
        print(f"   {tempo_range:20s} {count:6d} ({pct:5.1f}%) {bar}")
    
    print(f"\n{'='*60}")
    print("✅ Analysis complete!")
    print(f"{'='*60}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    top_instrument = instrument_counter.most_common(1)[0][0]
    print(f"   • Most common instrument: {top_instrument}")
    
    if piano_only_count / valid_count > 0.3:
        print(f"   • Lots of piano-only files - good for piano generation")
    
    if has_drums_count / valid_count > 0.5:
        print(f"   • Strong drum presence - rhythm patterns will train well")
    
    if multi_instrument_count / valid_count > 0.4:
        print(f"   • Many multi-instrument files - good arrangement learning")


def main():
    parser = argparse.ArgumentParser(description="Analyze MIDI dataset")
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing MIDI files",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5000,
        help="Max files to analyze (default: 5000, use -1 for all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all files (may take a while for 190k files)",
    )
    args = parser.parse_args()
    
    max_files = None if args.all else args.max_files
    if args.max_files == -1:
        max_files = None
    
    analyze_dataset(args.data_dir, max_files=max_files)


if __name__ == "__main__":
    main()

