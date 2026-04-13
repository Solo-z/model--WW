"""
Dataset Pipeline for MIDI / Session training data.

Supports two data sources:
  1. LakhMIDIDataset  — raw .mid files (Lakh etc.), heuristic labels
  2. SessionDataset   — structured session JSON specs with full style/key/role/section labels
  3. MixedDataset     — combines both with a configurable blend ratio
"""

import hashlib
import json
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, ConcatDataset
from tqdm import tqdm

from modelw.tokenizer import MIDITokenizer, TokenizerConfig


@dataclass
class DatasetConfig:
    """Configuration for MIDI dataset."""
    
    data_dir: str = "./data/lakh_midi"
    cache_dir: str = "./cache/tokenized"
    max_seq_len: int = 2048
    min_seq_len: int = 64
    max_files: Optional[int] = None  # Limit for testing
    train_split: float = 0.95
    seed: int = 42
    
    # Preprocessing
    num_workers: int = 8
    chunk_size: int = 1000
    
    # Augmentation
    tempo_augment: bool = True
    tempo_augment_range: tuple = (0.8, 1.2)
    pitch_augment: bool = True
    pitch_augment_range: tuple = (-6, 6)  # Semitones


class LakhMIDIDataset(Dataset):
    """
    PyTorch Dataset for Lakh MIDI files.
    
    Supports:
    - Lazy loading from cache
    - Data augmentation (tempo, pitch shifting)
    - Conditioning labels (mood estimation from musical features)
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: MIDITokenizer,
        split: str = "train",
        preprocess: bool = True,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        
        self.data_dir = Path(config.data_dir)
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all MIDI files
        self.midi_files = self._find_midi_files()
        
        # Split train/val
        random.seed(config.seed)
        random.shuffle(self.midi_files)
        
        split_idx = int(len(self.midi_files) * config.train_split)
        if split == "train":
            self.midi_files = self.midi_files[:split_idx]
        else:
            self.midi_files = self.midi_files[split_idx:]
        
        # Load or build cache
        self.cache_index = self._load_or_build_cache(preprocess)
        
        print(f"[{split}] Loaded {len(self.cache_index)} sequences from {len(self.midi_files)} files")
    
    def _find_midi_files(self) -> list[Path]:
        """Find all MIDI files in data directory."""
        extensions = [".mid", ".midi", ".MID", ".MIDI"]
        files = []
        
        for ext in extensions:
            files.extend(self.data_dir.rglob(f"*{ext}"))
        
        if self.config.max_files:
            files = files[:self.config.max_files]
        
        return sorted(files)
    
    def _get_cache_path(self, midi_path: Path) -> Path:
        """Get cache file path for a MIDI file."""
        # Use hash of path for cache filename
        path_hash = hashlib.md5(str(midi_path).encode()).hexdigest()[:16]
        return self.cache_dir / f"{path_hash}.pkl"
    
    def _load_or_build_cache(self, preprocess: bool) -> list[dict]:
        """Load tokenized data from cache or preprocess."""
        index_path = self.cache_dir / f"index_{self.split}.json"
        
        if index_path.exists() and not preprocess:
            with open(index_path) as f:
                return json.load(f)
        
        # Preprocess all files
        index = []
        failed = 0
        
        print(f"Preprocessing {len(self.midi_files)} MIDI files...")
        
        for midi_path in tqdm(self.midi_files, desc="Tokenizing"):
            try:
                # Estimate mood from filename/path (heuristic)
                mood = self._estimate_mood(midi_path)
                
                # Tokenize
                token_ids = self.tokenizer.encode(
                    midi_path,
                    mood=mood
                )
                
                # Skip if too short or too long
                if len(token_ids) < self.config.min_seq_len:
                    continue
                if len(token_ids) > self.config.max_seq_len:
                    token_ids = token_ids[:self.config.max_seq_len]
                
                # Cache
                cache_path = self._get_cache_path(midi_path)
                with open(cache_path, "wb") as f:
                    pickle.dump({
                        "token_ids": token_ids,
                        "midi_path": str(midi_path),
                        "mood": mood,
                    }, f)
                
                index.append({
                    "cache_path": str(cache_path),
                    "length": len(token_ids),
                    "midi_path": str(midi_path),
                })
                
            except Exception as e:
                failed += 1
                continue
        
        print(f"Preprocessed {len(index)} files, {failed} failed")
        
        # Save index
        with open(index_path, "w") as f:
            json.dump(index, f)
        
        return index
    
    def _estimate_mood(self, midi_path: Path) -> Optional[str]:
        """Estimate mood from filename (heuristic for labeling)."""
        name = midi_path.stem.lower()
        
        mood_keywords = {
            "happy": ["happy", "joy", "bright", "cheerful", "fun"],
            "sad": ["sad", "melancholy", "sorrow", "grief", "cry"],
            "energetic": ["energetic", "fast", "upbeat", "dance", "party"],
            "calm": ["calm", "peaceful", "relax", "gentle", "soft"],
            "dark": ["dark", "evil", "sinister", "ominous", "doom"],
            "epic": ["epic", "grand", "heroic", "battle", "war"],
            "romantic": ["love", "romantic", "heart", "passion"],
            "mysterious": ["mystery", "mysterious", "secret", "enigma"],
        }
        
        for mood, keywords in mood_keywords.items():
            if any(kw in name for kw in keywords):
                return mood
        
        return None
    
    def __len__(self) -> int:
        return len(self.cache_index)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a tokenized sequence."""
        entry = self.cache_index[idx]
        
        with open(entry["cache_path"], "rb") as f:
            data = pickle.load(f)
        
        token_ids = data["token_ids"]
        
        # Apply augmentation
        if self.split == "train":
            token_ids = self._augment(token_ids)
        
        # Pad to max length
        token_ids = self.tokenizer.pad_sequence(token_ids, self.config.max_seq_len)
        
        return {
            "input_ids": torch.tensor(token_ids[:-1], dtype=torch.long),
            "labels": torch.tensor(
                [
                    token if token != self.tokenizer.pad_id else -100
                    for token in token_ids[1:]
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [1 if t != self.tokenizer.pad_id else 0 for t in token_ids[:-1]],
                dtype=torch.long
            ),
        }
    
    def _augment(self, token_ids: list[int]) -> list[int]:
        """Apply data augmentation."""
        if not self.config.pitch_augment:
            return token_ids
        
        # Pitch shift
        if random.random() < 0.5:
            shift = random.randint(*self.config.pitch_augment_range)
            token_ids = self._pitch_shift(token_ids, shift)
        
        return token_ids
    
    def _pitch_shift(self, token_ids: list[int], shift: int) -> list[int]:
        """Shift all pitch tokens by semitones."""
        result = []
        
        for tid in token_ids:
            token = self.tokenizer.id_to_token.get(tid, "")
            
            if token.startswith("<PITCH_"):
                try:
                    pitch = int(token[7:-1])
                    new_pitch = pitch + shift
                    
                    # Keep in range
                    if self.tokenizer.config.pitch_range[0] <= new_pitch < self.tokenizer.config.pitch_range[1]:
                        new_token = f"<PITCH_{new_pitch}>"
                        tid = self.tokenizer.token_to_id.get(new_token, tid)
                except:
                    pass
            
            result.append(tid)
        
        return result


@dataclass
class SessionDatasetConfig:
    """Configuration for the structured session JSON dataset."""

    sessions_dir: str = "./synthetic/sessions"
    cache_dir: str = "./cache/sessions"
    max_seq_len: int = 4096
    min_seq_len: int = 64
    train_split: float = 0.95
    seed: int = 42
    max_files: Optional[int] = None


class SessionDataset(Dataset):
    """
    PyTorch Dataset built from structured session JSON specs.

    Each JSON file describes a full song with:
      - style, key, tempo, mood
      - multiple tracks with role + section labels
      - clip_library with actual MIDI notes

    Produces token sequences with full STYLE/KEY/ROLE/SECTION conditioning,
    one track per sequence (multi-track files generate one sample per track).
    """

    def __init__(
        self,
        config: SessionDatasetConfig,
        tokenizer: MIDITokenizer,
        split: str = "train",
        preprocess: bool = True,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split

        self.sessions_dir = Path(config.sessions_dir)
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session_files = self._find_session_files()
        random.seed(config.seed)
        random.shuffle(self.session_files)

        split_idx = int(len(self.session_files) * config.train_split)
        if split == "train":
            self.session_files = self.session_files[:split_idx]
        else:
            self.session_files = self.session_files[split_idx:]

        self.samples = self._load_or_build_cache(preprocess)
        print(f"[SessionDataset/{split}] {len(self.samples)} track samples from {len(self.session_files)} sessions")

    def _find_session_files(self) -> list[Path]:
        files = sorted(self.sessions_dir.rglob("*.json"))
        if self.config.max_files:
            files = files[: self.config.max_files]
        return files

    def _load_or_build_cache(self, preprocess: bool) -> list[dict]:
        index_path = self.cache_dir / f"index_{self.split}.json"
        if index_path.exists() and not preprocess:
            with open(index_path, encoding="utf-8") as handle:
                return json.load(handle)

        samples = []
        failed = 0

        print(f"Processing {len(self.session_files)} session files...")
        for session_path in tqdm(self.session_files, desc="Encoding sessions"):
            try:
                new_samples = self._encode_session(session_path)
                samples.extend(new_samples)
            except Exception:
                failed += 1

        print(f"Built {len(samples)} track samples, {failed} sessions failed")
        with open(index_path, "w", encoding="utf-8") as handle:
            json.dump(samples, handle)
        return samples

    def _encode_session(self, session_path: Path) -> list[dict]:
        """Encode one session JSON into one token sequence per track."""
        with open(session_path, encoding="utf-8") as handle:
            session = json.load(handle)

        meta = session.get("metadata", {})
        project = session.get("project", {})
        labels = session.get("semantic_song_labels", {})
        clip_library = session.get("libraries", {}).get("clip_library", {})
        arrangement = session.get("arrangement", {})
        sections = arrangement.get("sections", [])

        style = meta.get("style", "")
        raw_key = project.get("key", "")
        normalized_key = raw_key.upper().replace(" ", "_").replace("-", "_")
        tempo_list = project.get("tempo_map", [{}])
        tempo = float(tempo_list[0].get("bpm", 120)) if tempo_list else 120.0
        mood = self._top_label(labels.get("mood", []))

        samples = []
        for track in session.get("tracks", []):
            role = track.get("role", "")
            timeline = track.get("timeline", [])
            if not timeline:
                continue

            for section_entry in self._iter_sections(timeline, sections):
                notes = self._expand_timeline_section(
                    section_entry["refs"], clip_library, section_entry["length_bars"]
                )
                if not notes:
                    continue

                token_ids = self._notes_to_token_ids(
                    notes=notes,
                    tempo=tempo,
                    style=style,
                    key=normalized_key,
                    role=role,
                    section=section_entry["section_name"],
                    mood=mood,
                    instrument=self._role_to_instrument(role),
                )

                if len(token_ids) < self.config.min_seq_len:
                    continue
                if len(token_ids) > self.config.max_seq_len:
                    token_ids = token_ids[: self.config.max_seq_len]

                cache_key = hashlib.md5(
                    f"{session_path}|{role}|{section_entry['section_name']}|{len(samples)}".encode()
                ).hexdigest()[:16]
                cache_path = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_path, "wb") as f:
                    pickle.dump({"token_ids": token_ids}, f)

                samples.append({
                    "cache_path": str(cache_path),
                    "length": len(token_ids),
                    "style": style,
                    "role": role,
                    "section": section_entry["section_name"],
                })

        return samples

    def _iter_sections(self, timeline: list[dict], sections: list[dict]):
        """
        Yield per-section clips grouped by which arrangement section they fall in.
        If there are no arrangement sections, yield all clips as one group.
        """
        if not sections:
            yield {
                "section_name": "full",
                "refs": [entry["ref"] for entry in timeline],
                "length_bars": sum(
                    entry.get("loop_count", 1) * 4 for entry in timeline
                ),
            }
            return

        for section in sections:
            name = section.get("name", "verse")
            bar_start = section.get("bar_start", 1)
            bar_end = section.get("bar_end", bar_start + 8)
            refs_in_section = []
            total_bars = 0
            for entry in timeline:
                start = entry.get("start_bar", 1)
                loops = entry.get("loop_count", 1)
                # roughly check overlap
                if start >= bar_start and start < bar_end:
                    refs_in_section.append(entry["ref"])
                    total_bars += loops * 4
            if refs_in_section:
                yield {
                    "section_name": name,
                    "refs": refs_in_section,
                    "length_bars": max(1, total_bars),
                }

    def _expand_timeline_section(
        self,
        refs: list[str],
        clip_library: dict,
        length_bars: int,
    ) -> list[dict]:
        """Collect raw note dicts for all clips referenced in this section."""
        notes = []
        beat_offset = 0.0
        for ref in refs:
            clip = clip_library.get(ref)
            if not clip:
                continue
            clip_bars = clip.get("length_bars", 4)
            beats_per_bar = 4.0
            for note in clip.get("notes", []):
                notes.append({
                    "start_beat": float(note["start_beat"]) + beat_offset,
                    "duration_beat": float(note["duration_beat"]),
                    "pitch": int(note["pitch"]),
                    "velocity": int(note["velocity"]),
                })
            beat_offset += clip_bars * beats_per_bar
        return notes

    def _notes_to_token_ids(
        self,
        notes: list[dict],
        tempo: float,
        style: str,
        key: str,
        role: str,
        section: str,
        mood: Optional[str],
        instrument: Optional[str],
    ) -> list[int]:
        """Convert note dicts directly to token IDs without going through a MIDI file."""
        tok = self.tokenizer
        t2i = tok.token_to_id
        unk = tok.unk_id
        beats_per_bar = 4.0
        pos_resolution = tok.config.position_resolution  # 32
        ticks_per_beat = pos_resolution  # 1 tick == 1 position slot
        ticks_per_bar = int(beats_per_bar * ticks_per_beat)

        tokens = [tok.bos_id]

        def add(name: str):
            tokens.append(t2i.get(name, unk))

        add(tok._tempo_to_token(tempo))

        if style:
            s_tok = f"<STYLE_{style.upper()}>"
            if s_tok in t2i:
                add(s_tok)

        if key:
            k_tok = f"<KEY_{key}>"
            if k_tok in t2i:
                add(k_tok)

        if mood:
            m_tok = f"<MOOD_{mood.upper()}>"
            if m_tok in t2i:
                add(m_tok)

        if role:
            r_tok = f"<ROLE_{role.upper()}>"
            if r_tok in t2i:
                add(r_tok)

        if section:
            sec_tok = f"<SECTION_{section.upper()}>"
            if sec_tok in t2i:
                add(sec_tok)

        if instrument:
            inst_tok = f"<INST_{instrument.upper()}>"
            if inst_tok in t2i:
                add(inst_tok)

        add("<SEP>")

        sorted_notes = sorted(notes, key=lambda n: (n["start_beat"], n["pitch"]))
        current_bar = -1

        for note in sorted_notes:
            tick = int(round(note["start_beat"] * ticks_per_beat))
            bar = tick // ticks_per_bar
            pos = tick % ticks_per_bar

            while current_bar < bar:
                current_bar += 1
                add("<BAR>")

            pos = min(pos, ticks_per_bar - 1)
            add(f"<POS_{pos}>")

            pitch = max(tok.config.pitch_range[0], min(note["pitch"], tok.config.pitch_range[1] - 1))
            add(f"<PITCH_{pitch}>")

            vel_bin = tok._velocity_to_bin(note["velocity"])
            add(f"<VEL_{vel_bin}>")

            dur_ticks = int(round(note["duration_beat"] * ticks_per_beat))
            dur_bin = min(dur_ticks, tok.config.duration_bins - 1)
            add(f"<DUR_{dur_bin}>")

            if len(tokens) >= self.config.max_seq_len - 1:
                break

        tokens.append(tok.eos_id)
        return tokens

    def _top_label(self, mood_list: list) -> Optional[str]:
        if not mood_list:
            return None
        try:
            return max(mood_list, key=lambda pair: pair[1])[0]
        except Exception:
            return None

    def _role_to_instrument(self, role: str) -> Optional[str]:
        mapping = {
            "drums": "DRUMS",
            "bass": "BASS",
            "lead": "SYNTH_LEAD",
            "pad": "SYNTH_PAD",
            "fx": "SYNTH_FX",
            "chords": "PIANO",
            "melody": "SYNTH_LEAD",
        }
        return mapping.get(role.lower())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        entry = self.samples[idx]
        with open(entry["cache_path"], "rb") as f:
            data = pickle.load(f)

        token_ids = data["token_ids"]
        token_ids = self.tokenizer.pad_sequence(token_ids, self.config.max_seq_len)

        return {
            "input_ids": torch.tensor(token_ids[:-1], dtype=torch.long),
            "labels": torch.tensor(
                [t if t != self.tokenizer.pad_id else -100 for t in token_ids[1:]],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [1 if t != self.tokenizer.pad_id else 0 for t in token_ids[:-1]],
                dtype=torch.long,
            ),
        }


def create_dataloaders(
    config: DatasetConfig,
    tokenizer: MIDITokenizer,
    batch_size: int = 32,
    num_workers: int = 4,
    distributed: bool = False,
    session_config: Optional[SessionDatasetConfig] = None,
    session_blend_ratio: float = 0.5,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    If session_config is provided, a SessionDataset is built from structured
    session JSON files and blended with the Lakh MIDI dataset via ConcatDataset.
    session_blend_ratio controls what fraction of the final dataset comes from
    sessions (0.0 = all Lakh, 1.0 = all sessions).
    """
    lakh_train = LakhMIDIDataset(config, tokenizer, split="train")
    lakh_val = LakhMIDIDataset(config, tokenizer, split="val", preprocess=False)

    if session_config is not None:
        sess_train = SessionDataset(session_config, tokenizer, split="train")
        sess_val = SessionDataset(session_config, tokenizer, split="val", preprocess=False)

        # Upsample the smaller dataset so the blend ratio is respected
        if len(sess_train) > 0 and len(lakh_train) > 0:
            target_sess = int(
                (len(lakh_train) + len(sess_train)) * session_blend_ratio
            )
            repeats = max(1, target_sess // len(sess_train))
            sess_train_repeated = ConcatDataset([sess_train] * repeats)
            train_dataset = ConcatDataset([lakh_train, sess_train_repeated])
        else:
            train_dataset = ConcatDataset([lakh_train, sess_train])

        val_dataset = ConcatDataset([lakh_val, sess_val]) if len(sess_val) > 0 else lakh_val
    else:
        train_dataset = lakh_train
        val_dataset = lakh_val

    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def download_lakh_midi(output_dir: str = "./data/lakh_midi"):
    """
    Instructions for downloading Lakh MIDI dataset.
    
    The Lakh MIDI Dataset contains 176,581 unique MIDI files.
    Download from: https://colinraffel.com/projects/lmd/
    
    Files:
    - lmd_full.tar.gz (32GB) - Full dataset
    - lmd_matched.tar.gz - Matched to Million Song Dataset
    - lmd_aligned.tar.gz - Aligned versions
    """
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           LAKH MIDI DATASET DOWNLOAD INSTRUCTIONS            ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  Download from: https://colinraffel.com/projects/lmd/        ║
    ║                                                              ║
    ║  Recommended: lmd_full.tar.gz (176,581 MIDI files, 32GB)     ║
    ║                                                              ║
    ║  Commands:                                                   ║
    ║  $ wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
    ║  $ tar -xzf lmd_full.tar.gz -C ./data/lakh_midi              ║
    ║                                                              ║
    ║  Or use the included download script:                        ║
    ║  $ python scripts/download_lakh.py --output ./data/lakh_midi ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

