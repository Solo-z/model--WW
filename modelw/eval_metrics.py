"""
MIDI quality evaluation utilities.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pretty_midi

MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}
DEFAULT_WEIGHTS = {
    "prompt_match_score": 0.22,
    "key_scale_adherence": 0.14,
    "rhythm_grid_accuracy": 0.14,
    "velocity_expressiveness": 0.10,
    "repetition_variation_balance": 0.12,
    "section_coherence": 0.10,
    "track_role_integrity": 0.08,
    "hard_validity_score": 0.10,
}


@dataclass
class EvaluationConfig:
    min_notes: int = 20
    max_notes: int = 2000
    min_duration_seconds: float = 5.0
    max_overlap_ratio: float = 0.15
    grid_subdivision: int = 4
    grid_tolerance_ms: float = 40.0
    acceptance_threshold: float = 0.75


@dataclass
class PromptSpec:
    tempo: Optional[float] = None
    instrument: Optional[str] = None
    mood: Optional[str] = None


@dataclass
class FileMetrics:
    path: str
    valid: bool
    error: Optional[str] = None
    note_count: int = 0
    duration_seconds: float = 0.0
    estimated_tempo: float = 120.0
    estimated_key: Optional[str] = None
    hard_validity_score: float = 0.0
    key_scale_adherence: float = 0.0
    rhythm_grid_accuracy: float = 0.0
    velocity_expressiveness: float = 0.0
    repetition_variation_balance: float = 0.0
    section_coherence: float = 0.0
    track_role_integrity: Optional[float] = None
    prompt_match_score: Optional[float] = None
    composite_score: float = 0.0
    accepted: bool = False
    prompt: Optional[dict] = None
    hard_failures: list[str] | None = None


class MIDIEvaluator:
    """Evaluate generated MIDI with heuristic musical quality metrics."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()

    def evaluate_directory(
        self,
        midi_dir: str | Path,
        metadata_path: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None,
    ) -> dict:
        midi_dir = Path(midi_dir)
        midi_files = sorted(list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi")))
        prompt_map = self._load_prompt_map(midi_dir, metadata_path)

        file_metrics = [
            self.evaluate_file(midi_path, prompt=prompt_map.get(midi_path.stem))
            for midi_path in midi_files
        ]
        valid_files = [m for m in file_metrics if m.valid]
        accepted_files = [m for m in valid_files if m.accepted]

        hard_failure_counts = Counter()
        for metrics in file_metrics:
            for failure in metrics.hard_failures or []:
                hard_failure_counts[failure] += 1

        result = {
            "config": asdict(self.config),
            "summary": {
                "total_files": len(file_metrics),
                "valid_files": len(valid_files),
                "invalid_files": len(file_metrics) - len(valid_files),
                "hard_validity_rate": self._safe_ratio(
                    sum(1 for m in valid_files if m.hard_validity_score >= 1.0),
                    len(file_metrics),
                ),
                "acceptance_rate": self._safe_ratio(len(accepted_files), len(valid_files)),
                "mean_composite_score": self._mean([m.composite_score for m in valid_files]),
                "mean_prompt_match_score": self._mean_optional(
                    [m.prompt_match_score for m in valid_files]
                ),
                "mean_key_scale_adherence": self._mean(
                    [m.key_scale_adherence for m in valid_files]
                ),
                "mean_rhythm_grid_accuracy": self._mean(
                    [m.rhythm_grid_accuracy for m in valid_files]
                ),
                "mean_velocity_expressiveness": self._mean(
                    [m.velocity_expressiveness for m in valid_files]
                ),
                "mean_repetition_variation_balance": self._mean(
                    [m.repetition_variation_balance for m in valid_files]
                ),
                "mean_section_coherence": self._mean([m.section_coherence for m in valid_files]),
                "mean_track_role_integrity": self._mean_optional(
                    [m.track_role_integrity for m in valid_files]
                ),
                "hard_failure_counts": dict(hard_failure_counts),
            },
            "files": [asdict(m) for m in file_metrics],
        }

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)

        return result

    def evaluate_file(
        self,
        midi_path: str | Path,
        prompt: Optional[PromptSpec | dict] = None,
    ) -> FileMetrics:
        midi_path = Path(midi_path)
        prompt_spec = self._normalize_prompt(prompt)
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as exc:
            return FileMetrics(
                path=str(midi_path),
                valid=False,
                error=str(exc),
                hard_failures=["parse_error"],
            )

        notes = self._collect_notes(pm)
        if not notes:
            return FileMetrics(
                path=str(midi_path),
                valid=False,
                error="No notes found",
                hard_failures=["empty_midi"],
            )

        estimated_tempo = self._estimate_tempo(pm)
        duration_seconds = float(pm.get_end_time())
        estimated_key, key_scale_adherence = self._estimate_key_and_adherence(notes)
        hard_failures = self._hard_failures(notes, duration_seconds)
        hard_validity_score = 1.0 if not hard_failures else 0.0
        rhythm_grid_accuracy = self._rhythm_grid_accuracy(notes, estimated_tempo)
        velocity_expressiveness = self._velocity_expressiveness(notes)
        repetition_variation_balance = self._repetition_variation_balance(notes, estimated_tempo)
        section_coherence = self._section_coherence(notes, estimated_tempo)
        track_role_integrity = self._track_role_integrity(notes, prompt_spec.instrument)
        prompt_match_score = self._prompt_match_score(
            prompt_spec, notes, estimated_tempo, estimated_key
        )

        composite_score = self._composite_score(
            hard_validity_score=hard_validity_score,
            key_scale_adherence=key_scale_adherence,
            rhythm_grid_accuracy=rhythm_grid_accuracy,
            velocity_expressiveness=velocity_expressiveness,
            repetition_variation_balance=repetition_variation_balance,
            section_coherence=section_coherence,
            track_role_integrity=track_role_integrity,
            prompt_match_score=prompt_match_score,
        )

        return FileMetrics(
            path=str(midi_path),
            valid=True,
            note_count=len(notes),
            duration_seconds=round(duration_seconds, 3),
            estimated_tempo=round(estimated_tempo, 2),
            estimated_key=estimated_key,
            hard_validity_score=round(hard_validity_score, 4),
            key_scale_adherence=round(key_scale_adherence, 4),
            rhythm_grid_accuracy=round(rhythm_grid_accuracy, 4),
            velocity_expressiveness=round(velocity_expressiveness, 4),
            repetition_variation_balance=round(repetition_variation_balance, 4),
            section_coherence=round(section_coherence, 4),
            track_role_integrity=self._round_optional(track_role_integrity),
            prompt_match_score=self._round_optional(prompt_match_score),
            composite_score=round(composite_score, 4),
            accepted=composite_score >= self.config.acceptance_threshold and not hard_failures,
            prompt=asdict(prompt_spec) if prompt_spec else None,
            hard_failures=hard_failures,
        )

    def _load_prompt_map(
        self,
        midi_dir: Path,
        metadata_path: Optional[str | Path],
    ) -> dict[str, PromptSpec]:
        prompt_map: dict[str, PromptSpec] = {}
        candidate_roots = [midi_dir, midi_dir.parent]
        candidate_metadata = []
        if metadata_path is not None:
            candidate_metadata.append(Path(metadata_path))
        for root in candidate_roots:
            candidate_metadata.append(root / "metadata.json")

        for metadata_file in candidate_metadata:
            if not metadata_file.exists():
                continue
            try:
                with metadata_file.open(encoding="utf-8") as handle:
                    payload = json.load(handle)
                for sample in payload.get("samples", []):
                    sample_id = sample.get("id")
                    if sample_id:
                        prompt_map[sample_id] = self._normalize_prompt(sample.get("prompt"))
            except Exception:
                continue

        for root in candidate_roots:
            token_dir = root / "tokens"
            if not token_dir.exists():
                continue
            for token_file in token_dir.glob("*.json"):
                if token_file.stem in prompt_map:
                    continue
                try:
                    with token_file.open(encoding="utf-8") as handle:
                        payload = json.load(handle)
                    prompt_map[token_file.stem] = self._normalize_prompt(payload.get("prompt"))
                except Exception:
                    continue

        return prompt_map

    def _normalize_prompt(self, prompt: Optional[PromptSpec | dict]) -> PromptSpec:
        if prompt is None:
            return PromptSpec()
        if isinstance(prompt, PromptSpec):
            return prompt
        if isinstance(prompt, dict):
            return PromptSpec(
                tempo=prompt.get("tempo"),
                instrument=prompt.get("instrument"),
                mood=prompt.get("mood"),
            )
        return PromptSpec()

    def _collect_notes(self, pm: pretty_midi.PrettyMIDI) -> list[dict]:
        notes = []
        for instrument in pm.instruments:
            for note in instrument.notes:
                notes.append(
                    {
                        "pitch": int(note.pitch),
                        "start": float(note.start),
                        "end": float(note.end),
                        "duration": float(note.end - note.start),
                        "velocity": int(note.velocity),
                        "is_drum": bool(instrument.is_drum),
                    }
                )
        notes.sort(key=lambda item: (item["start"], item["pitch"], item["duration"]))
        return notes

    def _estimate_tempo(self, pm: pretty_midi.PrettyMIDI) -> float:
        _, tempos = pm.get_tempo_changes()
        return float(tempos[0]) if len(tempos) else 120.0

    def _estimate_key_and_adherence(self, notes: list[dict]) -> tuple[str, float]:
        pitch_classes = [note["pitch"] % 12 for note in notes if not note["is_drum"]]
        if not pitch_classes:
            return "unknown", 1.0

        pitch_counter = Counter(pitch_classes)
        best_name = "C major"
        best_score = -1.0
        best_scale = {(0 + degree) % 12 for degree in MAJOR_SCALE}

        for tonic in range(12):
            for mode_name, scale in (("major", MAJOR_SCALE), ("minor", MINOR_SCALE)):
                absolute_scale = {(tonic + degree) % 12 for degree in scale}
                score = sum(count for pc, count in pitch_counter.items() if pc in absolute_scale)
                if score > best_score:
                    best_score = float(score)
                    best_scale = absolute_scale
                    best_name = f"{self._pc_name(tonic)} {mode_name}"

        adherence = sum(1 for pc in pitch_classes if pc in best_scale) / max(1, len(pitch_classes))
        return best_name, adherence

    def _rhythm_grid_accuracy(self, notes: list[dict], tempo: float) -> float:
        beat_duration = 60.0 / max(tempo, 1.0)
        grid = beat_duration / self.config.grid_subdivision
        tolerance = min(self.config.grid_tolerance_ms / 1000.0, grid * 0.35)
        distances = []
        for note in notes:
            remainder = note["start"] % grid
            distances.append(min(remainder, grid - remainder))
        if not distances:
            return 0.0
        on_grid_ratio = sum(1 for d in distances if d <= tolerance) / len(distances)
        mean_distance = float(np.mean(distances))
        mean_penalty = max(0.0, 1.0 - mean_distance / max(tolerance, 1e-6))
        return 0.7 * on_grid_ratio + 0.3 * mean_penalty

    def _velocity_expressiveness(self, notes: list[dict]) -> float:
        velocities = [note["velocity"] for note in notes]
        if len(velocities) < 2:
            return 0.0
        velocity_std = float(np.std(velocities))
        if velocity_std < 3.0:
            return 0.05
        if velocity_std < 8.0:
            return velocity_std / 8.0
        if velocity_std <= 28.0:
            center = 18.0
            return max(0.0, 1.0 - abs(velocity_std - center) / center)
        return max(0.0, 1.0 - (velocity_std - 28.0) / 40.0)

    def _repetition_variation_balance(self, notes: list[dict], tempo: float) -> float:
        signatures = self._bar_signatures(notes, tempo)
        if len(signatures) < 2:
            return 0.5
        counts = Counter(signatures)
        dominant_ratio = counts.most_common(1)[0][1] / len(signatures)
        unique_ratio = len(counts) / len(signatures)
        dominant_score = max(0.0, 1.0 - abs(dominant_ratio - 0.45) / 0.45)
        unique_score = max(0.0, 1.0 - abs(unique_ratio - 0.45) / 0.45)
        return 0.5 * dominant_score + 0.5 * unique_score

    def _section_coherence(self, notes: list[dict], tempo: float) -> float:
        beat_duration = 60.0 / max(tempo, 1.0)
        bar_duration = beat_duration * 4.0
        window_duration = bar_duration * 4.0
        total_duration = max(note["end"] for note in notes)
        window_count = max(1, math.ceil(total_duration / window_duration))
        features = []
        for window_idx in range(window_count):
            start = window_idx * window_duration
            end = start + window_duration
            window_notes = [note for note in notes if start <= note["start"] < end]
            if not window_notes:
                features.append(np.array([0.0, 0.0, 0.0, 0.0], dtype=float))
                continue
            features.append(
                np.array(
                    [
                        len(window_notes) / 4.0,
                        float(np.mean([n["pitch"] for n in window_notes])) / 127.0,
                        float(np.mean([n["velocity"] for n in window_notes])) / 127.0,
                        float(np.mean([n["duration"] for n in window_notes])) / max(bar_duration, 1e-6),
                    ],
                    dtype=float,
                )
            )
        if len(features) < 2:
            return 0.5
        sims = []
        for left, right in zip(features, features[1:]):
            left_norm = np.linalg.norm(left)
            right_norm = np.linalg.norm(right)
            if left_norm == 0.0 or right_norm == 0.0:
                sims.append(0.5)
                continue
            cosine = float(np.dot(left, right) / (left_norm * right_norm))
            sims.append((cosine + 1.0) / 2.0)
        variance = float(np.mean(np.std(np.stack(features), axis=0)))
        variation_score = max(0.0, 1.0 - abs(variance - 0.18) / 0.18)
        return 0.65 * float(np.mean(sims)) + 0.35 * variation_score

    def _track_role_integrity(self, notes: list[dict], instrument: Optional[str]) -> Optional[float]:
        if not instrument:
            return None
        name = instrument.lower()
        pitches = [note["pitch"] for note in notes]
        durations = [note["duration"] for note in notes]
        mean_pitch = float(np.mean(pitches))

        if "drum" in name:
            drum_pitch_ratio = sum(1 for pitch in pitches if 35 <= pitch <= 81) / max(1, len(pitches))
            short_note_ratio = sum(1 for dur in durations if dur <= 0.35) / max(1, len(durations))
            return 0.6 * drum_pitch_ratio + 0.4 * short_note_ratio
        if "bass" in name:
            return 1.0 if mean_pitch <= 55 else max(0.0, 1.0 - (mean_pitch - 55.0) / 24.0)
        if "pad" in name or "strings" in name:
            return sum(1 for dur in durations if dur >= 0.35) / max(1, len(durations))
        if "lead" in name or "guitar" in name or "piano" in name:
            return 1.0 if 55 <= mean_pitch <= 90 else max(
                0.0, 1.0 - min(abs(mean_pitch - 72.0), 36.0) / 36.0
            )
        return 1.0

    def _hard_failures(self, notes: list[dict], duration_seconds: float) -> list[str]:
        failures = []
        if len(notes) < self.config.min_notes:
            failures.append("too_few_notes")
        if len(notes) > self.config.max_notes:
            failures.append("too_many_notes")
        if duration_seconds < self.config.min_duration_seconds:
            failures.append("too_short")
        if self._overlap_ratio(notes) > self.config.max_overlap_ratio:
            failures.append("too_many_overlaps")
        if len({note["pitch"] for note in notes}) < 3:
            failures.append("low_pitch_diversity")
        return failures

    def _overlap_ratio(self, notes: list[dict]) -> float:
        overlaps = 0
        comparisons = 0
        last_end_by_pitch = {}
        for note in notes:
            pitch = note["pitch"]
            if pitch in last_end_by_pitch:
                comparisons += 1
                if note["start"] < last_end_by_pitch[pitch]:
                    overlaps += 1
            last_end_by_pitch[pitch] = max(last_end_by_pitch.get(pitch, 0.0), note["end"])
        return self._safe_ratio(overlaps, comparisons)

    def _prompt_match_score(
        self,
        prompt_spec: PromptSpec,
        notes: list[dict],
        tempo: float,
        estimated_key: str,
    ) -> Optional[float]:
        subscores = []
        if prompt_spec.tempo is not None:
            delta = abs(float(prompt_spec.tempo) - tempo)
            subscores.append(max(0.0, 1.0 - delta / 20.0))
        if prompt_spec.instrument:
            role_score = self._track_role_integrity(notes, prompt_spec.instrument)
            if role_score is not None:
                subscores.append(role_score)
        if prompt_spec.mood:
            subscores.append(self._mood_match_score(prompt_spec.mood, notes, tempo, estimated_key))
        if not subscores:
            return None
        return float(np.mean(subscores))

    def _mood_match_score(
        self,
        mood: str,
        notes: list[dict],
        tempo: float,
        estimated_key: str,
    ) -> float:
        mood = mood.lower()
        pitch_mean = float(np.mean([note["pitch"] for note in notes]))
        velocity_mean = float(np.mean([note["velocity"] for note in notes]))
        note_density = len(notes) / max(1.0, max(note["end"] for note in notes))
        is_minor = "minor" in (estimated_key or "")

        heuristics = {
            "energetic": min(1.0, (tempo / 160.0) * 0.5 + (note_density / 10.0) * 0.5),
            "calm": max(0.0, 1.0 - min(1.0, tempo / 150.0)),
            "dark": 0.65 * float(is_minor) + 0.35 * max(0.0, 1.0 - pitch_mean / 84.0),
            "bright": 0.5 * max(0.0, pitch_mean / 84.0) + 0.5 * min(1.0, velocity_mean / 110.0),
            "happy": 0.7 * float(not is_minor) + 0.3 * min(1.0, velocity_mean / 110.0),
            "sad": 0.7 * float(is_minor) + 0.3 * max(0.0, 1.0 - tempo / 160.0),
            "epic": min(1.0, note_density / 9.0) * 0.4 + min(1.0, velocity_mean / 100.0) * 0.6,
            "mysterious": 0.5 * float(is_minor) + 0.5 * max(0.0, 1.0 - tempo / 150.0),
            "aggressive": min(1.0, velocity_mean / 115.0) * 0.6 + min(1.0, tempo / 170.0) * 0.4,
            "playful": min(1.0, tempo / 150.0) * 0.5 + max(0.0, pitch_mean / 90.0) * 0.5,
        }
        return float(heuristics.get(mood, 0.5))

    def _bar_signatures(self, notes: list[dict], tempo: float) -> list[tuple]:
        beat_duration = 60.0 / max(tempo, 1.0)
        bar_duration = beat_duration * 4.0
        if bar_duration <= 0:
            return []
        bars: dict[int, list[tuple]] = {}
        for note in notes:
            bar_idx = int(note["start"] // bar_duration)
            pos_in_bar = note["start"] - bar_idx * bar_duration
            step = int(round(pos_in_bar / (bar_duration / 16.0)))
            entry = (step, note["pitch"] % 12 if not note["is_drum"] else note["pitch"])
            bars.setdefault(bar_idx, []).append(entry)
        signatures = []
        for bar_idx in sorted(bars):
            signatures.append(tuple(sorted(bars[bar_idx])[:64]))
        return signatures

    def _composite_score(self, **metrics: Optional[float]) -> float:
        weighted_sum = 0.0
        active_weight = 0.0
        for name, weight in DEFAULT_WEIGHTS.items():
            value = metrics.get(name)
            if value is None:
                continue
            weighted_sum += weight * float(value)
            active_weight += weight
        if active_weight == 0.0:
            return 0.0
        return weighted_sum / active_weight

    def _pc_name(self, pitch_class: int) -> str:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return names[pitch_class % 12]

    def _mean(self, values: list[float]) -> float:
        if not values:
            return 0.0
        return round(float(np.mean(values)), 4)

    def _mean_optional(self, values: list[Optional[float]]) -> Optional[float]:
        present = [value for value in values if value is not None]
        if not present:
            return None
        return round(float(np.mean(present)), 4)

    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return round(float(numerator) / float(denominator), 4)

    def _round_optional(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return round(float(value), 4)
