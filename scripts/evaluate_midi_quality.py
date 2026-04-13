#!/usr/bin/env python3
"""
Evaluate generated MIDI folders with one-shot quality metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelw.eval_metrics import EvaluationConfig, MIDIEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate MIDI quality metrics")
    parser.add_argument("midi_dir", type=str, help="Directory containing MIDI files")
    parser.add_argument(
        "--output",
        type=str,
        default="midi_quality_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional metadata.json path with prompt information",
    )
    parser.add_argument("--min-notes", type=int, default=20, help="Minimum note count")
    parser.add_argument("--max-notes", type=int, default=2000, help="Maximum note count")
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=5.0,
        help="Minimum file duration",
    )
    parser.add_argument(
        "--acceptance-threshold",
        type=float,
        default=0.75,
        help="Composite score threshold for acceptance",
    )
    args = parser.parse_args()

    config = EvaluationConfig(
        min_notes=args.min_notes,
        max_notes=args.max_notes,
        min_duration_seconds=args.min_duration_seconds,
        acceptance_threshold=args.acceptance_threshold,
    )
    evaluator = MIDIEvaluator(config)
    results = evaluator.evaluate_directory(
        midi_dir=args.midi_dir,
        metadata_path=args.metadata,
        output_path=args.output,
    )

    summary = results["summary"]
    print("\n" + "=" * 60)
    print("MIDI Quality Evaluation")
    print("=" * 60)
    print(f"Files:                 {summary['total_files']}")
    print(f"Valid:                 {summary['valid_files']}")
    print(f"Hard validity rate:    {summary['hard_validity_rate']:.2%}")
    print(f"Acceptance rate:       {summary['acceptance_rate']:.2%}")
    print(f"Composite score:       {summary['mean_composite_score']:.4f}")
    print(f"Prompt match score:    {summary['mean_prompt_match_score']}")
    print(f"Key/scale adherence:   {summary['mean_key_scale_adherence']:.4f}")
    print(f"Rhythm grid accuracy:  {summary['mean_rhythm_grid_accuracy']:.4f}")
    print(f"Velocity expression:   {summary['mean_velocity_expressiveness']:.4f}")
    print(f"Repeat/variation:      {summary['mean_repetition_variation_balance']:.4f}")
    print(f"Section coherence:     {summary['mean_section_coherence']:.4f}")
    print(f"Track role integrity:  {summary['mean_track_role_integrity']}")
    print(f"Saved JSON:            {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
