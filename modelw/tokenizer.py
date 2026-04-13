"""
MIDI Tokenizer - REMI-style encoding for MIDI files

Converts MIDI files to/from token sequences that can be processed by
the transformer model. Supports conditioning tokens for tempo, instrument,
mood, style, key, role, and section labels.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pretty_midi

# Vocabulary ranges
PITCH_RANGE = (21, 109)  # Piano range A0-C8
VELOCITY_BINS = 32
DURATION_BINS = 64
POSITION_RESOLUTION = 32  # 32nd notes for rolls and finer groove
MAX_BARS = 256
TEMPO_BINS = 32
TEMPO_RANGE = (40, 240)
STYLE_TOKENS = [
    "trap", "reggaeton", "house", "techno", "edm", "hiphop", "lofi",
    "ambient", "pop", "rnb", "drill", "cinematic"
]
KEY_TOKENS = [
    "C_MAJOR", "C_MINOR", "C#_MAJOR", "C#_MINOR", "D_MAJOR", "D_MINOR",
    "D#_MAJOR", "D#_MINOR", "E_MAJOR", "E_MINOR", "F_MAJOR", "F_MINOR",
    "F#_MAJOR", "F#_MINOR", "G_MAJOR", "G_MINOR", "G#_MAJOR", "G#_MINOR",
    "A_MAJOR", "A_MINOR", "A#_MAJOR", "A#_MINOR", "B_MAJOR", "B_MINOR",
]
ROLE_TOKENS = ["drums", "bass", "lead", "pad", "fx", "chords", "melody"]
SECTION_TOKENS = ["intro", "verse", "chorus", "bridge", "drop", "breakdown", "outro"]


@dataclass
class TokenizerConfig:
    """Configuration for MIDI tokenizer."""
    
    pitch_range: tuple = PITCH_RANGE
    velocity_bins: int = VELOCITY_BINS
    duration_bins: int = DURATION_BINS
    position_resolution: int = POSITION_RESOLUTION
    max_bars: int = MAX_BARS
    tempo_bins: int = TEMPO_BINS
    tempo_range: tuple = TEMPO_RANGE
    max_seq_len: int = 2048
    
    # Conditioning
    use_tempo_condition: bool = True
    use_instrument_condition: bool = True
    use_mood_condition: bool = True
    use_style_condition: bool = True
    use_key_condition: bool = True
    use_role_condition: bool = True
    use_section_condition: bool = True


@dataclass
class MIDITokenizer:
    """
    REMI-style MIDI tokenizer with conditioning support.
    
    Token structure:
    - Special tokens: PAD, BOS, EOS, SEP, MASK
    - Conditioning tokens: TEMPO_*, INST_*, MOOD_*
    - Bar tokens: BAR
    - Position tokens: POS_0 to POS_{resolution-1}
    - Pitch tokens: PITCH_21 to PITCH_108
    - Velocity tokens: VEL_0 to VEL_{bins-1}
    - Duration tokens: DUR_0 to DUR_{bins-1}
    """
    
    config: TokenizerConfig = field(default_factory=TokenizerConfig)
    
    def __post_init__(self):
        self._build_vocab()
    
    def _build_vocab(self):
        """Build the vocabulary mapping."""
        self.token_to_id = {}
        self.id_to_token = {}
        
        idx = 0
        
        # Special tokens
        self.special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<SEP>", "<MASK>", "<UNK>"]
        for token in self.special_tokens:
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        self.pad_id = self.token_to_id["<PAD>"]
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.sep_id = self.token_to_id["<SEP>"]
        self.mask_id = self.token_to_id["<MASK>"]
        self.unk_id = self.token_to_id["<UNK>"]
        
        self.tempo_tokens = []
        self.instrument_tokens = []
        self.mood_tokens = []
        self.style_tokens = []
        self.key_tokens = []
        self.role_tokens = []
        self.section_tokens = []

        # Tempo conditioning tokens (quantized BPM)
        if self.config.use_tempo_condition:
            tempo_min, tempo_max = self.config.tempo_range
            for i in range(self.config.tempo_bins):
                bpm = tempo_min + (tempo_max - tempo_min) * i / (self.config.tempo_bins - 1)
                token = f"<TEMPO_{int(bpm)}>"
                self.tempo_tokens.append(token)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                idx += 1
        
        # Instrument conditioning tokens (GM program categories)
        if self.config.use_instrument_condition:
            self.instrument_tokens = [
                "<INST_PIANO>", "<INST_CHROMATIC>", "<INST_ORGAN>", "<INST_GUITAR>",
                "<INST_BASS>", "<INST_STRINGS>", "<INST_ENSEMBLE>", "<INST_BRASS>",
                "<INST_REED>", "<INST_PIPE>", "<INST_SYNTH_LEAD>", "<INST_SYNTH_PAD>",
                "<INST_SYNTH_FX>", "<INST_ETHNIC>", "<INST_PERCUSSIVE>", "<INST_SFX>",
                "<INST_DRUMS>"
            ]
            for token in self.instrument_tokens:
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                idx += 1
        
        # Mood conditioning tokens
        if self.config.use_mood_condition:
            self.mood_tokens = [
                "<MOOD_HAPPY>", "<MOOD_SAD>", "<MOOD_ENERGETIC>", "<MOOD_CALM>",
                "<MOOD_DARK>", "<MOOD_BRIGHT>", "<MOOD_AGGRESSIVE>", "<MOOD_PEACEFUL>",
                "<MOOD_MYSTERIOUS>", "<MOOD_ROMANTIC>", "<MOOD_EPIC>", "<MOOD_PLAYFUL>",
                "<MOOD_MELANCHOLIC>", "<MOOD_TRIUMPHANT>", "<MOOD_TENSE>", "<MOOD_DREAMY>"
            ]
            for token in self.mood_tokens:
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                idx += 1

        if self.config.use_style_condition:
            self.style_tokens = [f"<STYLE_{style.upper()}>" for style in STYLE_TOKENS]
            for token in self.style_tokens:
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                idx += 1

        if self.config.use_key_condition:
            self.key_tokens = [f"<KEY_{key}>" for key in KEY_TOKENS]
            for token in self.key_tokens:
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                idx += 1

        if self.config.use_role_condition:
            self.role_tokens = [f"<ROLE_{role.upper()}>" for role in ROLE_TOKENS]
            for token in self.role_tokens:
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                idx += 1

        if self.config.use_section_condition:
            self.section_tokens = [f"<SECTION_{section.upper()}>" for section in SECTION_TOKENS]
            for token in self.section_tokens:
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                idx += 1
        
        # Bar token
        self.bar_token = "<BAR>"
        self.token_to_id[self.bar_token] = idx
        self.id_to_token[idx] = self.bar_token
        idx += 1
        
        # Position tokens (within bar)
        self.position_tokens = []
        for i in range(self.config.position_resolution * 4):  # 4 beats per bar
            token = f"<POS_{i}>"
            self.position_tokens.append(token)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        # Pitch tokens
        self.pitch_tokens = []
        for pitch in range(self.config.pitch_range[0], self.config.pitch_range[1]):
            token = f"<PITCH_{pitch}>"
            self.pitch_tokens.append(token)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        # Velocity tokens
        self.velocity_tokens = []
        for i in range(self.config.velocity_bins):
            token = f"<VEL_{i}>"
            self.velocity_tokens.append(token)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        # Duration tokens
        self.duration_tokens = []
        for i in range(self.config.duration_bins):
            token = f"<DUR_{i}>"
            self.duration_tokens.append(token)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1
        
        self.vocab_size = idx
    
    def _tempo_to_token(self, bpm: float) -> str:
        """Convert BPM to tempo token."""
        tempo_min, tempo_max = self.config.tempo_range
        bpm = np.clip(bpm, tempo_min, tempo_max)
        bin_idx = int((bpm - tempo_min) / (tempo_max - tempo_min) * (self.config.tempo_bins - 1))
        return self.tempo_tokens[bin_idx]
    
    def _token_to_tempo(self, token: str) -> float:
        """Convert tempo token back to BPM."""
        if token in self.tempo_tokens:
            idx = self.tempo_tokens.index(token)
            tempo_min, tempo_max = self.config.tempo_range
            return tempo_min + (tempo_max - tempo_min) * idx / (self.config.tempo_bins - 1)
        return 120.0  # Default
    
    def _program_to_instrument(self, program: int, is_drum: bool = False) -> str:
        """Convert MIDI program number to instrument category token."""
        if is_drum:
            return "<INST_DRUMS>"
        
        categories = [
            "<INST_PIANO>", "<INST_CHROMATIC>", "<INST_ORGAN>", "<INST_GUITAR>",
            "<INST_BASS>", "<INST_STRINGS>", "<INST_ENSEMBLE>", "<INST_BRASS>",
            "<INST_REED>", "<INST_PIPE>", "<INST_SYNTH_LEAD>", "<INST_SYNTH_PAD>",
            "<INST_SYNTH_FX>", "<INST_ETHNIC>", "<INST_PERCUSSIVE>", "<INST_SFX>"
        ]
        return categories[program // 8]
    
    def _velocity_to_bin(self, velocity: int) -> int:
        """Quantize velocity to bin."""
        return min(velocity * self.config.velocity_bins // 128, self.config.velocity_bins - 1)
    
    def _bin_to_velocity(self, bin_idx: int) -> int:
        """Convert bin back to velocity."""
        return int((bin_idx + 0.5) * 128 / self.config.velocity_bins)
    
    def _duration_to_bin(self, duration: float, ticks_per_beat: int = 480) -> int:
        """Quantize duration to bin (in ticks)."""
        # Duration bins: 0 = 1/64 note, 63 = whole note tied
        tick_per_bin = ticks_per_beat // 8  # 1/32 note base
        bin_idx = int(duration / tick_per_bin)
        return min(bin_idx, self.config.duration_bins - 1)
    
    def _bin_to_duration(self, bin_idx: int, ticks_per_beat: int = 480) -> float:
        """Convert bin back to duration in ticks."""
        tick_per_bin = ticks_per_beat // 8
        return (bin_idx + 0.5) * tick_per_bin
    
    def encode(
        self,
        midi_path: str | Path,
        tempo: Optional[float] = None,
        instrument: Optional[str] = None,
        mood: Optional[str] = None,
        style: Optional[str] = None,
        key: Optional[str] = None,
        role: Optional[str] = None,
        section: Optional[str] = None,
    ) -> list[int]:
        """
        Encode a MIDI file to token sequence.
        
        Args:
            midi_path: Path to MIDI file
            tempo: Optional tempo override (detected from file if None)
            instrument: Optional instrument hint
            mood: Optional mood label
            style: Optional style label
            key: Optional key label, e.g. "D_MINOR"
            role: Optional track role
            section: Optional section label
            
        Returns:
            List of token IDs
        """
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as e:
            raise ValueError(f"Failed to load MIDI: {e}")
        
        tokens = ["<BOS>"]
        
        # Add conditioning tokens
        if self.config.use_tempo_condition:
            if tempo is None:
                # Estimate tempo from file
                tempo_times, tempos = pm.get_tempo_changes()
                tempo = tempos[0] if len(tempos) > 0 else 120.0
            tokens.append(self._tempo_to_token(tempo))
        
        if self.config.use_instrument_condition and instrument:
            inst_token = f"<INST_{instrument.upper()}>"
            if inst_token in self.token_to_id:
                tokens.append(inst_token)
        elif self.config.use_instrument_condition and len(pm.instruments) > 0:
            # Use first non-drum instrument
            for inst in pm.instruments:
                if not inst.is_drum:
                    tokens.append(self._program_to_instrument(inst.program))
                    break
        
        if self.config.use_mood_condition and mood:
            mood_token = f"<MOOD_{mood.upper()}>"
            if mood_token in self.token_to_id:
                tokens.append(mood_token)

        if self.config.use_style_condition and style:
            style_token = f"<STYLE_{style.upper()}>"
            if style_token in self.token_to_id:
                tokens.append(style_token)

        if self.config.use_key_condition and key:
            normalized_key = key.upper().replace(" ", "_")
            key_token = f"<KEY_{normalized_key}>"
            if key_token in self.token_to_id:
                tokens.append(key_token)

        if self.config.use_role_condition and role:
            role_token = f"<ROLE_{role.upper()}>"
            if role_token in self.token_to_id:
                tokens.append(role_token)

        if self.config.use_section_condition and section:
            section_token = f"<SECTION_{section.upper()}>"
            if section_token in self.token_to_id:
                tokens.append(section_token)
        
        tokens.append("<SEP>")
        
        # Collect all notes with timing
        all_notes = []
        ticks_per_beat = 480  # Standard
        
        for inst in pm.instruments:
            for note in inst.notes:
                # Convert time to ticks
                start_tick = int(note.start * ticks_per_beat * (tempo or 120) / 60)
                duration_tick = int(note.duration * ticks_per_beat * (tempo or 120) / 60)
                
                all_notes.append({
                    'start': start_tick,
                    'pitch': note.pitch,
                    'velocity': note.velocity,
                    'duration': duration_tick,
                    'is_drum': inst.is_drum
                })
        
        # Sort by start time
        all_notes.sort(key=lambda x: (x['start'], x['pitch']))
        
        # Convert to tokens
        ticks_per_bar = ticks_per_beat * 4  # 4/4 time
        ticks_per_pos = ticks_per_bar // (self.config.position_resolution * 4)
        
        current_bar = -1
        
        for note in all_notes:
            # Skip notes outside pitch range (except drums)
            if not note['is_drum']:
                if note['pitch'] < self.config.pitch_range[0] or note['pitch'] >= self.config.pitch_range[1]:
                    continue
            
            bar = note['start'] // ticks_per_bar
            pos = (note['start'] % ticks_per_bar) // ticks_per_pos
            
            # Add bar tokens if needed
            while current_bar < bar:
                current_bar += 1
                tokens.append("<BAR>")
                if current_bar >= self.config.max_bars:
                    break
            
            if current_bar >= self.config.max_bars:
                break
            
            # Position
            pos = min(pos, len(self.position_tokens) - 1)
            tokens.append(f"<POS_{pos}>")
            
            # Pitch
            pitch = np.clip(note['pitch'], self.config.pitch_range[0], self.config.pitch_range[1] - 1)
            tokens.append(f"<PITCH_{pitch}>")
            
            # Velocity
            vel_bin = self._velocity_to_bin(note['velocity'])
            tokens.append(f"<VEL_{vel_bin}>")
            
            # Duration
            dur_bin = self._duration_to_bin(note['duration'], ticks_per_beat)
            tokens.append(f"<DUR_{dur_bin}>")
            
            # Limit sequence length
            if len(tokens) >= self.config.max_seq_len - 1:
                break
        
        tokens.append("<EOS>")
        
        # Convert to IDs
        token_ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]
        
        return token_ids
    
    def decode(self, token_ids: list[int], output_path: Optional[str | Path] = None) -> pretty_midi.PrettyMIDI:
        """
        Decode token sequence back to MIDI.
        
        Args:
            token_ids: List of token IDs
            output_path: Optional path to save MIDI file
            
        Returns:
            PrettyMIDI object
        """
        tokens = [self.id_to_token.get(tid, "<UNK>") for tid in token_ids]
        
        # Parse conditioning
        tempo = 120.0
        for token in tokens:
            if token.startswith("<TEMPO_"):
                tempo = self._token_to_tempo(token)
                break
        
        # Create MIDI
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=0)  # Piano default
        
        ticks_per_beat = 480
        ticks_per_bar = ticks_per_beat * 4
        ticks_per_pos = ticks_per_bar // (self.config.position_resolution * 4)
        
        current_bar = -1
        current_pos = 0
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token == "<BAR>":
                current_bar += 1
                i += 1
                continue
            
            if token.startswith("<POS_"):
                try:
                    current_pos = int(token[5:-1])
                except:
                    pass
                i += 1
                continue
            
            if token.startswith("<PITCH_"):
                try:
                    pitch = int(token[7:-1])
                    
                    # Look for velocity and duration
                    velocity = 80
                    duration_ticks = ticks_per_beat
                    
                    if i + 1 < len(tokens) and tokens[i + 1].startswith("<VEL_"):
                        vel_bin = int(tokens[i + 1][5:-1])
                        velocity = self._bin_to_velocity(vel_bin)
                        i += 1
                    
                    if i + 1 < len(tokens) and tokens[i + 1].startswith("<DUR_"):
                        dur_bin = int(tokens[i + 1][5:-1])
                        duration_ticks = self._bin_to_duration(dur_bin, ticks_per_beat)
                        i += 1
                    
                    # Calculate time
                    start_tick = max(0, current_bar) * ticks_per_bar + current_pos * ticks_per_pos
                    start_time = start_tick * 60 / (tempo * ticks_per_beat)
                    duration_time = duration_ticks * 60 / (tempo * ticks_per_beat)
                    
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=start_time + duration_time
                    )
                    instrument.notes.append(note)
                except:
                    pass
            
            i += 1
        
        pm.instruments.append(instrument)
        
        if output_path:
            pm.write(str(output_path))
        
        return pm
    
    def encode_batch(self, midi_paths: list[str | Path], **kwargs) -> list[list[int]]:
        """Encode multiple MIDI files."""
        return [self.encode(p, **kwargs) for p in midi_paths]
    
    def pad_sequence(self, token_ids: list[int], max_len: Optional[int] = None) -> list[int]:
        """Pad or truncate sequence to max length."""
        max_len = max_len or self.config.max_seq_len
        if len(token_ids) > max_len:
            return token_ids[:max_len]
        return token_ids + [self.pad_id] * (max_len - len(token_ids))
    
    def save(self, path: str | Path):
        """Save tokenizer to file."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / "tokenizer_config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        with open(path / "vocab.json", "w") as f:
            json.dump(self.token_to_id, f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> "MIDITokenizer":
        """Load tokenizer from file."""
        path = Path(path)
        
        with open(path / "tokenizer_config.json") as f:
            config_dict = json.load(f)
        
        config = TokenizerConfig(**config_dict)
        return cls(config=config)

