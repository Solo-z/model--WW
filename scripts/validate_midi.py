#!/usr/bin/env python3
"""
Validate generated MIDI files for quality.

Checks:
- File integrity (can be parsed)
- Note count and distribution
- Pitch diversity
- Temporal structure
- Velocity patterns
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pretty_midi
from tqdm import tqdm


def analyze_midi(midi_path: str) -> dict:
    """Analyze a single MIDI file."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        return {"valid": False, "error": str(e)}
    
    # Collect all notes
    all_notes = []
    for inst in pm.instruments:
        all_notes.extend(inst.notes)
    
    if len(all_notes) == 0:
        return {"valid": False, "error": "No notes found"}
    
    # Extract features
    pitches = [n.pitch for n in all_notes]
    velocities = [n.velocity for n in all_notes]
    durations = [n.end - n.start for n in all_notes]
    starts = [n.start for n in all_notes]
    
    # Pitch analysis
    unique_pitches = len(set(pitches))
    pitch_range = max(pitches) - min(pitches)
    pitch_mean = np.mean(pitches)
    pitch_std = np.std(pitches)
    
    # Velocity analysis
    velocity_mean = np.mean(velocities)
    velocity_std = np.std(velocities)
    
    # Duration analysis
    duration_mean = np.mean(durations)
    duration_std = np.std(durations)
    
    # Temporal analysis
    total_duration = pm.get_end_time()
    note_density = len(all_notes) / total_duration if total_duration > 0 else 0
    
    # Repetition analysis
    pitch_counter = Counter(pitches)
    most_common_pitch_ratio = pitch_counter.most_common(1)[0][1] / len(pitches)
    
    return {
        "valid": True,
        "note_count": len(all_notes),
        "unique_pitches": unique_pitches,
        "pitch_range": pitch_range,
        "pitch_mean": round(pitch_mean, 2),
        "pitch_std": round(pitch_std, 2),
        "velocity_mean": round(velocity_mean, 2),
        "velocity_std": round(velocity_std, 2),
        "duration_mean": round(duration_mean, 4),
        "duration_std": round(duration_std, 4),
        "total_duration": round(total_duration, 2),
        "note_density": round(note_density, 2),
        "most_common_pitch_ratio": round(most_common_pitch_ratio, 4),
        "num_instruments": len(pm.instruments),
    }


def validate_batch(
    midi_dir: str,
    output_path: str = None,
    min_notes: int = 20,
    max_notes: int = 2000,
    min_unique_pitches: int = 5,
    min_duration: float = 5.0,
    max_repetition_ratio: float = 0.5,
):
    """Validate a batch of MIDI files."""
    
    midi_dir = Path(midi_dir)
    midi_files = list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi"))
    
    print(f"Found {len(midi_files)} MIDI files")
    
    results = {
        "total": len(midi_files),
        "valid": 0,
        "invalid": 0,
        "passed_quality": 0,
        "failed_quality": 0,
        "files": [],
    }
    
    quality_failures = Counter()
    
    for midi_path in tqdm(midi_files, desc="Validating"):
        analysis = analyze_midi(str(midi_path))
        analysis["path"] = str(midi_path)
        
        if not analysis.get("valid"):
            results["invalid"] += 1
            analysis["quality_passed"] = False
            quality_failures["parse_error"] += 1
        else:
            results["valid"] += 1
            
            # Quality checks
            failed = []
            
            if analysis["note_count"] < min_notes:
                failed.append("too_few_notes")
            if analysis["note_count"] > max_notes:
                failed.append("too_many_notes")
            if analysis["unique_pitches"] < min_unique_pitches:
                failed.append("low_pitch_diversity")
            if analysis["total_duration"] < min_duration:
                failed.append("too_short")
            if analysis["most_common_pitch_ratio"] > max_repetition_ratio:
                failed.append("too_repetitive")
            
            if failed:
                results["failed_quality"] += 1
                analysis["quality_passed"] = False
                analysis["quality_failures"] = failed
                for f in failed:
                    quality_failures[f] += 1
            else:
                results["passed_quality"] += 1
                analysis["quality_passed"] = True
        
        results["files"].append(analysis)
    
    # Summary stats
    valid_analyses = [f for f in results["files"] if f.get("valid")]
    
    if valid_analyses:
        results["summary"] = {
            "avg_notes": round(np.mean([f["note_count"] for f in valid_analyses]), 1),
            "avg_unique_pitches": round(np.mean([f["unique_pitches"] for f in valid_analyses]), 1),
            "avg_duration": round(np.mean([f["total_duration"] for f in valid_analyses]), 1),
            "avg_note_density": round(np.mean([f["note_density"] for f in valid_analyses]), 2),
        }
    
    results["quality_failure_counts"] = dict(quality_failures)
    
    # Print summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    print(f"Total files:        {results['total']}")
    print(f"Valid (parseable):  {results['valid']} ({results['valid']/results['total']*100:.1f}%)")
    print(f"Invalid:            {results['invalid']}")
    print(f"Passed quality:     {results['passed_quality']} ({results['passed_quality']/results['total']*100:.1f}%)")
    print(f"Failed quality:     {results['failed_quality']}")
    
    if valid_analyses:
        print(f"\nValid file statistics:")
        print(f"  Avg notes:          {results['summary']['avg_notes']}")
        print(f"  Avg unique pitches: {results['summary']['avg_unique_pitches']}")
        print(f"  Avg duration:       {results['summary']['avg_duration']}s")
        print(f"  Avg note density:   {results['summary']['avg_note_density']} notes/s")
    
    if quality_failures:
        print(f"\nQuality failure breakdown:")
        for reason, count in quality_failures.most_common():
            print(f"  {reason}: {count}")
    
    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate MIDI files")
    parser.add_argument(
        "midi_dir",
        type=str,
        help="Directory containing MIDI files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--min-notes",
        type=int,
        default=20,
        help="Minimum note count",
    )
    parser.add_argument(
        "--max-notes",
        type=int,
        default=2000,
        help="Maximum note count",
    )
    args = parser.parse_args()
    
    validate_batch(
        args.midi_dir,
        args.output,
        min_notes=args.min_notes,
        max_notes=args.max_notes,
    )


if __name__ == "__main__":
    main()

