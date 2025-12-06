# game: py_blockworld
# dev: gpt5.1-agent
# prompting: github.com/zeittresor

from __future__ import annotations

import logging
import math
import os
import random
import struct
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
from PIL import Image

try:
    import pygame
    from pygame import locals as pyloc
    import moderngl
except ImportError as exc:
    raise ImportError(
        "Dieses Skript benötigt die Module 'pygame' und 'moderngl'. "
        "Bitte installieren Sie sie z.B. mit 'pip install pygame moderngl pillow numpy noise'."
    ) from exc

def perspective_matrix(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
       
    f = 1.0 / math.tan(math.radians(fov) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m

def look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
       
    f = target - eye
    f = f / np.linalg.norm(f)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m

def ensure_directories(base_path: Path) -> None:
       
    for sub in ("textures", "sounds", "music", "screenshots"):
        (base_path / sub).mkdir(parents=True, exist_ok=True)

    (base_path / "backup").mkdir(parents=True, exist_ok=True)

def backup_generated_file(file_path: Path, base_path: Path) -> None:
       
    try:
                                                     
        if not file_path.exists():
            return
                             
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = base_path / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_name = f"{timestamp}_{file_path.name}"
        backup_path = backup_dir / backup_name
                               
        with open(file_path, "rb") as src, open(backup_path, "wb") as dst:
            dst.write(src.read())
    except Exception as exc:
        logging.error(f"Konnte Backup für {file_path} nicht erstellen: {exc}")

def generate_texture(path: Path, color: Tuple[int, int, int], size: int = 16) -> None:
       
    base_r, base_g, base_b = color
    img = Image.new("RGB", (size, size))
    pixels = img.load()
    for x in range(size):
        for y in range(size):
                                           
            check = ((x // 4) + (y // 4)) % 2
                                         
            variation = random.randint(-20, 20)
            r = max(0, min(255, base_r + variation + (check * 10)))
            g = max(0, min(255, base_g + variation + (check * 10)))
            b = max(0, min(255, base_b + variation + (check * 10)))
            pixels[x, y] = (r, g, b)
    img.save(path, format="PNG")

def generate_gradient_texture(path: Path, top_color: Tuple[int, int, int], bottom_color: Tuple[int, int, int], size: int = 16) -> None:
       
    img = Image.new("RGB", (size, size))
    pixels = img.load()
    top_r, top_g, top_b = top_color
    bot_r, bot_g, bot_b = bottom_color
    for x in range(size):
        for y in range(size):
                                                              
            t = y / (size - 1)
                                                  
            r = int((1 - t) * top_r + t * bot_r)
            g = int((1 - t) * top_g + t * bot_g)
            b = int((1 - t) * top_b + t * bot_b)
                             
            variation = random.randint(-15, 15)
            r = max(0, min(255, r + variation))
            g = max(0, min(255, g + variation))
            b = max(0, min(255, b + variation))
            pixels[x, y] = (r, g, b)
    img.save(path, format="PNG")

def generate_textures(data_dir: Path) -> Dict[str, Path]:
       
    textures: Dict[str, Path] = {}
    tex_dir = data_dir / "textures"

    grass_top_color = (34, 139, 34)
                                                                     
    grass_side_top = (34, 139, 34)
    grass_side_bottom = (139, 69, 19)
          
    dirt_color = (139, 69, 19)
           
    stone_color = (128, 128, 128)
                                                       
    texture_definitions = {
        "grass_top": (grass_top_color, None),             
        "grass_side": (grass_side_top, grass_side_bottom),            
        "dirt": (dirt_color, None),
        "stone": (stone_color, None),
    }
    for name, params in texture_definitions.items():
        path = tex_dir / f"{name}.png"
        if not path.exists():
                                                                                                      
            if params[1] is None:
                                 
                generate_texture(path, params[0], size=128)
            else:
                generate_gradient_texture(path, params[0], params[1], size=128)
                                         
            backup_generated_file(path, data_dir)
        textures[name] = path
    return textures

def generate_sine_wave(
    filename: Path,
    frequency: float = 440.0,
    duration: float = 3.0,
    volume: float = 0.5,
    sample_rate: int = 44100,
    chord: Tuple[float, ...] | None = None,
) -> None:
       
    n_samples = int(sample_rate * duration)
    frames = bytearray()
    if chord is None:
        chord = (frequency,)
                                                                       
    amplitudes = [1.0 / len(chord) for _ in chord]
    for i in range(n_samples):
        t = i / sample_rate
        sample_val = 0.0
        for amp, freq in zip(amplitudes, chord):
            sample_val += amp * math.sin(2.0 * math.pi * freq * t)
                                                 
        sample_int = int(sample_val * 32767 * volume)
        frames += struct.pack("<h", sample_int)
    with wave.open(str(filename), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)

def generate_sounds(data_dir: Path) -> Dict[str, Path]:
       
    sounds: Dict[str, Path] = {}
    snd_dir = data_dir / "sounds"
    mus_dir = data_dir / "music"

    click_path = snd_dir / "click.wav"
    if not click_path.exists():
                                    
        generate_sine_wave(click_path, chord=(600.0, 800.0), duration=0.15, volume=0.6)
        backup_generated_file(click_path, data_dir)
    sounds["click"] = click_path

    menu_music = mus_dir / "menu_music.wav"
    if not menu_music.exists():
        try:
                                                                           
            generate_structured_music(menu_music, total_duration=60.0)
        except Exception as e:
            logging.error(f"Konnte Menümusik nicht generieren: {e}")
                                                                           
            generate_sine_wave(menu_music, duration=20.0, volume=0.3, chord=(220.0, 261.63, 329.63))
        backup_generated_file(menu_music, data_dir)
    sounds["menu_music"] = menu_music
                                                                                    
    game_music = mus_dir / "game_music.wav"
    if not game_music.exists():
        try:
            generate_classical_music(game_music, total_duration=60.0)
        except Exception as e:
            logging.error(f"Konnte Spielmusik nicht generieren: {e}")
            generate_sine_wave(game_music, duration=20.0, volume=0.4, chord=(220.0, 261.63, 329.63))
        backup_generated_file(game_music, data_dir)
    sounds["game_music"] = game_music
                                                                                   
    menu_extra = mus_dir / "menu_extra.wav"
    if not menu_extra.exists():
        try:
            generate_structured_music(menu_extra, total_duration=45.0)
        except Exception as e:
            logging.error(f"Konnte Menü‑Zusatzmusik nicht generieren: {e}")
        backup_generated_file(menu_extra, data_dir)
    sounds["menu_extra"] = menu_extra
    return sounds

def generate_classical_music(filename: Path, total_duration: float = 60.0, sample_rate: int = 44100) -> None:
       
    melody = [
        (659.25, 0.4),       
        (622.25, 0.4),              
        (659.25, 0.4),       
        (622.25, 0.4),        
        (659.25, 0.4),       
        (493.88, 0.4),       
        (587.33, 0.4),       
        (523.25, 0.4),       
        (440.00, 0.8),                         
    ]
    pattern_duration = sum(dur for (_, dur) in melody)
    loops = int(math.ceil(total_duration / pattern_duration))
    n_total_samples = int(total_duration * sample_rate)
                                                                          
    base_frequencies = (440.00, 523.25, 659.25)
                    
    frames = bytearray()
    current_time = 0.0
    sample_count = 0
                                                        
    melody_amp = 0.6
    base_amp = 0.2
                   
    for _ in range(loops):
        for freq, dur in melody:
            n_samples = int(dur * sample_rate)
            for i in range(n_samples):
                t = current_time
                               
                m_sample = math.sin(2.0 * math.pi * freq * t)
                                      
                b_sample = 0.0
                for bf in base_frequencies:
                    b_sample += math.sin(2.0 * math.pi * bf * t)
                b_sample /= len(base_frequencies)
                                          
                sample_val = melody_amp * m_sample + base_amp * b_sample
                                      
                sample_int = int(max(-1.0, min(1.0, sample_val)) * 32767)
                frames += struct.pack("<h", sample_int)
                current_time += 1.0 / sample_rate
                sample_count += 1
                                                               
                if sample_count >= n_total_samples:
                    break
            if sample_count >= n_total_samples:
                break
        if sample_count >= n_total_samples:
            break
                  
    with wave.open(str(filename), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)

def generate_procedural_music(filename: Path, total_duration: float = 90.0, sample_rate: int = 44100) -> None:
       
    scales: List[Tuple[float, List[int]]] = [
        (261.63, [0, 2, 4, 5, 7, 9, 11, 12]),         
        (220.00, [0, 2, 3, 5, 7, 8, 10, 12]),         
        (293.66, [0, 2, 3, 5, 7, 8, 10, 12]),         
        (196.00, [0, 2, 4, 5, 7, 9, 11, 12]),        
        (164.81, [0, 2, 4, 5, 7, 9, 11, 12]),         
    ]
                                                              
    root_freq, intervals = random.choice(scales)

    chord_progression: List[int] = [0, 7, 9, 5]

    phrase_pool: List[List[int]] = [
        [0, 2, 4, 2],                                                       
        [0, 3, 5, 3],                                                   
        [0, 4, 7, 4],                                                            
        [0, 2, 1, 2],                                  
        [0, -1, -2, -1],                      
        [0, -3, -5, -3],                            
        [0, 5, 7, 5],                                           
        [0, 2, 5, 2],                                        
        [0, 4, 5, 7],                                   
        [0, 7, 5, 4],                                                  
    ]

    tempo = random.choice([80, 90, 100, 110, 120])
    beat_duration = 60.0 / tempo
                                                                               
    total_beats = int(total_duration / beat_duration)
                                          
    bars = max(1, total_beats // 4)

    def chord_frequencies(degree: int) -> List[float]:
        intervals_major = [0, 4, 7]
        intervals_minor = [0, 3, 7]
                                                                       
        is_minor = (intervals[2] == 3)
        chord_ints = intervals_minor if is_minor else intervals_major
        freqs: List[float] = []
        for ci in chord_ints:
            total_iv = degree + ci
                                                               
            octave_shift = total_iv // len(intervals)
            scale_index = total_iv % len(intervals)
                                
            freq = root_freq * (2 ** ((intervals[scale_index] + 12 * octave_shift) / 12.0))
            freqs.append(freq)
        return freqs

    sequence: List[Tuple[float, float, List[float]]] = []
    for bar_idx in range(bars):
                                                            
        degree = chord_progression[bar_idx % len(chord_progression)]
                                                       
        chord_freqs = chord_frequencies(degree)
                                                        
        phrase = random.choice(phrase_pool)
                                                
        for offset in phrase:
                                           
            idx = degree + offset
            octave_shift = idx // len(intervals)
            scale_idx = idx % len(intervals)
            freq = root_freq * (2 ** ((intervals[scale_idx] + 12 * octave_shift) / 12.0))
            sequence.append((freq, beat_duration, chord_freqs))

    elapsed_time = bars * 4 * beat_duration
    if elapsed_time < total_duration:
        last_degree = chord_progression[bars % len(chord_progression)]
        last_chord = chord_frequencies(last_degree)
        base_freq = root_freq * (2 ** (intervals[last_degree % len(intervals)] / 12.0))
        remaining = total_duration - elapsed_time
        sequence.append((base_freq, remaining, last_chord))

    frames = bytearray()
    current_time = 0.0
    max_samples = int(total_duration * sample_rate)
    sample_count = 0
                         
    melody_amp = 0.5
    chord_amp = 0.25
    second_voice_amp = 0.3
                                                  
    attack_fraction = 0.05
    release_fraction = 0.1
    for freq, dur, chord_freqs in sequence:
                                       
        n_samples = int(min(dur, total_duration - (sample_count / sample_rate)) * sample_rate)
                                                                       
        use_second_voice = (random.random() < 0.25)
        for i in range(n_samples):
            pos_in_note = i / float(n_samples)
                                 
            if pos_in_note < attack_fraction:
                env = pos_in_note / attack_fraction
            elif pos_in_note > 1.0 - release_fraction:
                env = (1.0 - pos_in_note) / release_fraction
            else:
                env = 1.0
                        
            m_sample = math.sin(2.0 * math.pi * freq * current_time)
                                                
            v_sample = 0.0
            if use_second_voice:
                v_sample = math.sin(2.0 * math.pi * (freq / 2.0) * current_time)
                                        
            b_sample = 0.0
            for bf in chord_freqs:
                b_sample += math.sin(2.0 * math.pi * bf * current_time)
            b_sample /= len(chord_freqs)
                                               
            sample_val = env * (melody_amp * (m_sample + second_voice_amp * v_sample) + chord_amp * b_sample)
                                                   
            sample_int = int(max(-1.0, min(1.0, sample_val)) * 32767)
            frames += struct.pack("<h", sample_int)
            current_time += 1.0 / sample_rate
            sample_count += 1
            if sample_count >= max_samples:
                break
        if sample_count >= max_samples:
            break
                                                     
    with wave.open(str(filename), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)

def generate_markov_music(filename: Path, total_duration: float = 90.0, sample_rate: int = 44100) -> None:
       
    scales: List[Tuple[float, List[int]]] = [
        (261.63, [0, 2, 4, 5, 7, 9, 11, 12]),         
        (220.00, [0, 2, 3, 5, 7, 8, 10, 12]),         
        (196.00, [0, 2, 4, 5, 7, 9, 11, 12]),        
        (174.61, [0, 2, 3, 5, 7, 8, 10, 12]),               
    ]
    root_freq, intervals = random.choice(scales)
                                                
    chord_progression: List[int] = [0, 7, 9, 5]
                                                           
    def chord_frequencies(degree: int) -> List[float]:
                                      
        is_minor = (intervals[2] == 3)
        intervals_major = [0, 4, 7]
        intervals_minor = [0, 3, 7]
        chord_ints = intervals_minor if is_minor else intervals_major
        freqs: List[float] = []
        for ci in chord_ints:
            total_iv = degree + ci
            octave_shift = total_iv // len(intervals)
            scale_idx = total_iv % len(intervals)
            freq = root_freq * (2 ** ((intervals[scale_idx] + 12 * octave_shift) / 12.0))
            freqs.append(freq)
        return freqs
                                                                      
    states = list(range(-4, 5))
                                                                                    
    transition_prob: Dict[int, List[Tuple[int, float]]] = {}
    for s in states:
        probs: List[Tuple[int, float]] = []
                                                                         
        probs.append((s, 0.2))
                           
        if s < max(states):
            probs.append((s + 1, 0.3))
        else:
            probs.append((s - 1, 0.3))
                            
        if s > min(states):
            probs.append((s - 1, 0.3))
        else:
            probs.append((s + 1, 0.3))
                                                    
        leaps = []
        for step in [2, 3]:
            if s + step <= max(states):
                leaps.append((s + step, 0.05))
            elif s - step >= min(states):
                leaps.append((s - step, 0.05))
                                                                                                       
        if not leaps:
                                                                 
            total_extra = 0.1
                                                 
            new_probs = []
            for target, p in probs:
                new_probs.append((target, p + total_extra * (p / sum(p for _, p in probs))))
            probs = new_probs
        else:
            probs.extend(leaps)
                                           
        total = sum(p for _, p in probs)
        transition_prob[s] = [(t, p / total) for t, p in probs]
                     
    tempo = random.choice([80, 90, 100, 110])
    beat_duration = 60.0 / tempo
                           
    total_beats = int(total_duration / beat_duration)
    bars = max(1, total_beats // 4)
                                                              
    current_state = 0
    sequence: List[Tuple[float, float, List[float]]] = []
                      
    for bar_idx in range(bars):
        degree = chord_progression[bar_idx % len(chord_progression)]
        chord_freqs = chord_frequencies(degree)
                                                                                 
        beat = 0
        while beat < 4:
                                                                        
            rand_val = random.random()
            cumulative = 0.0
            next_state = current_state
            for target, prob in transition_prob[current_state]:
                cumulative += prob
                if rand_val <= cumulative:
                    next_state = target
                    break
            current_state = next_state
                                                   
            idx = degree + current_state
            octave_shift = idx // len(intervals)
            scale_idx = idx % len(intervals)
            freq = root_freq * (2 ** ((intervals[scale_idx] + 12 * octave_shift) / 12.0))
                                                                        
            dur_beats = 2 if random.random() < 0.25 else 1
                                                                       
            if beat + dur_beats > 4:
                dur_beats = 4 - beat
            sequence.append((freq, dur_beats * beat_duration, chord_freqs))
            beat += dur_beats
                                                
    elapsed = sum(d for _, d, _ in sequence)
    if elapsed < total_duration:
        remaining = total_duration - elapsed
        last_deg = chord_progression[bars % len(chord_progression)]
        last_chord = chord_frequencies(last_deg)
        base_freq = root_freq * (2 ** (intervals[last_deg % len(intervals)] / 12.0))
        sequence.append((base_freq, remaining, last_chord))
                    
    frames = bytearray()
    current_time = 0.0
    max_samples = int(total_duration * sample_rate)
    sample_count = 0
    melody_amp = 0.45
    chord_amp = 0.25
    second_voice_amp = 0.25
    attack_fraction = 0.05
    release_fraction = 0.1
    for freq, dur, chord_freqs in sequence:
        n_samples = int(min(dur, total_duration - (sample_count / sample_rate)) * sample_rate)
                                                                      
        use_second_voice = (random.random() < 0.3)
        for i in range(n_samples):
            pos_in_note = i / float(n_samples)
            if pos_in_note < attack_fraction:
                env = pos_in_note / attack_fraction
            elif pos_in_note > 1.0 - release_fraction:
                env = (1.0 - pos_in_note) / release_fraction
            else:
                env = 1.0
            m_sample = math.sin(2.0 * math.pi * freq * current_time)
            v_sample = 0.0
            if use_second_voice:
                v_sample = math.sin(2.0 * math.pi * (freq / 2.0) * current_time)
            b_sample = 0.0
            for bf in chord_freqs:
                b_sample += math.sin(2.0 * math.pi * bf * current_time)
            b_sample /= len(chord_freqs)
            sample_val = env * (melody_amp * (m_sample + second_voice_amp * v_sample) + chord_amp * b_sample)
            sample_int = int(max(-1.0, min(1.0, sample_val)) * 32767)
            frames += struct.pack("<h", sample_int)
            current_time += 1.0 / sample_rate
            sample_count += 1
            if sample_count >= max_samples:
                break
        if sample_count >= max_samples:
            break
    with wave.open(str(filename), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)

def generate_structured_music(filename: Path, total_duration: float = 90.0, sample_rate: int = 44100) -> None:
       
    scales: List[Tuple[float, List[int]]] = [
        (261.63, [0, 2, 4, 5, 7, 9, 11, 12]),         
        (220.00, [0, 2, 3, 5, 7, 8, 10, 12]),         
        (293.66, [0, 2, 4, 5, 7, 9, 11, 12]),        
        (196.00, [0, 2, 4, 5, 7, 9, 11, 12]),        
        (174.61, [0, 2, 3, 5, 7, 8, 10, 12]),         
    ]
    root_freq, intervals = random.choice(scales)

    progressions: List[List[int]] = [
        [0, 5, 7, 0],             
        [0, 7, 9, 5],                                
        [0, 3, 4, 5],                                
        [0, 9, 5, 7],              
    ]
    chord_progression = random.choice(progressions)

    motifs: List[Dict[str, List[int] | List[float]]] = [
        {"notes": [0, 0, 1, 2, 2, 1, 0, -1], "durations": [0.5]*8},                     
        {"notes": [0, -1, 0, -1, 0, -3, -2, -4], "durations": [0.5]*8},                   
        {"notes": [0, 2, 4, 5, 7, 5, 4, 2], "durations": [0.5]*8},                      
        {"notes": [0, 3, 5, 3, 0, 3, 7, 5], "durations": [0.5]*8},            
        {"notes": [0, 2, 0, -2, 0, 3, 0, -3], "durations": [0.5]*8},                  
        {"notes": [0, 4, 7, 4, 0, 5, 7, 5], "durations": [0.5]*8},               
    ]

    tempo = random.choice([80, 90, 100, 110, 120])
    beat_duration = 60.0 / tempo
                                                         
    eigth_duration = beat_duration / 2.0
                                          
    bar_duration = 4 * beat_duration
                                                    
    bars = max(1, int(total_duration / bar_duration))

    def chord_frequencies(degree: int) -> List[float]:
                                                           
        is_minor = (intervals[2] == 3)
        intervals_major = [0, 4, 7]
        intervals_minor = [0, 3, 7]
        chord_ints = intervals_minor if is_minor else intervals_major
        freqs: List[float] = []
        for ci in chord_ints:
            total_iv = degree + ci
            octave_shift = total_iv // len(intervals)
            scale_idx = total_iv % len(intervals)
            freq = root_freq * (2 ** ((intervals[scale_idx] + 12 * octave_shift) / 12.0))
            freqs.append(freq)
        return freqs

    sequence: List[Tuple[float, float, List[float]]] = []
    for bar_idx in range(bars):
                                        
        degree = chord_progression[bar_idx % len(chord_progression)]
        chord_freqs = chord_frequencies(degree)
                                               
        motif = random.choice(motifs)
        notes = motif["notes"]                           
        durations = motif["durations"]                           
                                           
        for offset, dur in zip(notes, durations):
                                                       
            idx = degree + offset
            octave_shift = idx // len(intervals)
            scale_idx = idx % len(intervals)
            freq = root_freq * (2 ** ((intervals[scale_idx] + 12 * octave_shift) / 12.0))
                                                                                    
            sequence.append((freq, dur * beat_duration, chord_freqs))

    elapsed = sum(d for _, d, _ in sequence)
    if elapsed < total_duration:
        remaining = total_duration - elapsed
        last_degree = chord_progression[bars % len(chord_progression)]
        last_chord = chord_frequencies(last_degree)
                                                                    
        base_freq = root_freq * (2 ** (intervals[last_degree % len(intervals)] / 12.0))
        sequence.append((base_freq, remaining, last_chord))

    frames = bytearray()
    current_time = 0.0
    max_samples = int(total_duration * sample_rate)
    sample_count = 0
                                                                
    melody_amp = 0.45
    second_voice_amp = 0.3
    chord_amp = 0.25
                       
    attack_fraction = 0.05
    release_fraction = 0.1
    for freq, dur, chord_freqs in sequence:
        n_samples = int(min(dur, total_duration - (sample_count / sample_rate)) * sample_rate)
                                                                       
        use_second_voice = random.random() < 0.4
                                                                                
        second_factor = 0.5 if random.random() < 0.5 else 2.0
        for i in range(n_samples):
            pos_in_note = i / float(n_samples)
                                                            
            if pos_in_note < attack_fraction:
                env = pos_in_note / attack_fraction
            elif pos_in_note > 1.0 - release_fraction:
                env = (1.0 - pos_in_note) / release_fraction
            else:
                env = 1.0
                         
            m_sample = math.sin(2.0 * math.pi * freq * current_time)
                           
            v_sample = 0.0
            if use_second_voice:
                v_sample = math.sin(2.0 * math.pi * (freq * second_factor) * current_time)
                              
            b_sample = 0.0
            for cf in chord_freqs:
                b_sample += math.sin(2.0 * math.pi * cf * current_time)
            b_sample /= len(chord_freqs)
            sample_val = env * (melody_amp * (m_sample + second_voice_amp * v_sample) + chord_amp * b_sample)
            sample_int = int(max(-1.0, min(1.0, sample_val)) * 32767)
            frames += struct.pack("<h", sample_int)
            current_time += 1.0 / sample_rate
            sample_count += 1
            if sample_count >= max_samples:
                break
        if sample_count >= max_samples:
            break
                        
    with wave.open(str(filename), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)

def generate_skybox_textures(data_dir: Path, size: int = 256) -> Dict[str, Path]:
       
    sky_dir = data_dir / "skybox"
    sky_dir.mkdir(parents=True, exist_ok=True)
    faces = [
        "right",      
        "left",       
        "top",        
        "bottom",     
        "front",      
        "back",       
    ]
    base_color = np.array([135, 206, 235], dtype=np.float32)              
    cloud_color = np.array([255, 255, 255], dtype=np.float32)        
                                                                                             
    random.seed(42)
    base_phases: Dict[str, Tuple[float, float]] = {}
    for name in faces:
                                                                  
        phi1 = random.uniform(0.0, 2.0 * math.pi)
        phi2 = random.uniform(0.0, 2.0 * math.pi)
        base_phases[name] = (phi1, phi2)
    textures: Dict[str, Path] = {}
    for name in faces:
        path = sky_dir / f"{name}.png"
        if path.exists():
            textures[name] = path
            continue
                      
        img = Image.new("RGB", (size, size))
        pixels = img.load()
        phi_u, phi_v = base_phases[name]
        for x in range(size):
            for y in range(size):
                u = x / float(size)
                v = y / float(size)
                                                                                           
                val = math.sin(2.0 * math.pi * (u * 2.0 + phi_u)) * math.sin(2.0 * math.pi * (v * 2.0 + phi_v))
                val += 0.5 * math.sin(2.0 * math.pi * (u * 4.0 - phi_u)) * math.sin(2.0 * math.pi * (v * 4.0 - phi_v))
                val = (val * 0.5) + 0.5
                val = val * val
                color = base_color * (1.0 - val) + cloud_color * val
                r, g, b = int(color[0]), int(color[1]), int(color[2])
                pixels[x, y] = (r, g, b)
        img.save(path, format="PNG")
                                      
        backup_generated_file(path, data_dir)
        textures[name] = path
    return textures

BLOCK_TEXTURE_MAP: Dict[Tuple[int, int], int] | None = None

@dataclass
class Block:
                                                               
    position: Tuple[int, int, int]
    type_id: int

class World:
       
    def __init__(self, chunk_size: int = 32, max_height: int = 6, seed: int | None = None) -> None:
                                                
        self.chunk_size = chunk_size
                                                 
        self.max_height = max_height
                                                                                 
        self.blocks: Dict[Tuple[int, int, int], int] = {}
                                                                                       
        self.chunk_blocks: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
                                                                                           
        self.loaded_chunks: Dict[Tuple[int, int], bool] = {}
                                          
        if seed is not None:
            random.seed(seed)
                                              
        self.generate_chunk(0, 0)

    def generate_chunk(self, cx: int, cz: int) -> None:
           
        if (cx, cz) in self.loaded_chunks:
            return
        block_list: List[Tuple[int, int, int]] = []
                                               
        freq_hills = 0.05               
        freq_mountains = 0.1                       
        for local_x in range(self.chunk_size):
            for local_z in range(self.chunk_size):
                world_x = cx * self.chunk_size + local_x
                world_z = cz * self.chunk_size + local_z
                                                                   
                hills = math.sin((world_x) * freq_hills) * math.cos((world_z) * freq_hills)
                mountains = abs(math.sin(world_x * freq_mountains) * math.sin(world_z * freq_mountains))
                                              
                base_height = self.max_height * 0.4 * hills + self.max_height * 0.5 * mountains + self.max_height * 0.4
                                                                                
                hash_val = ((world_x * 73856093) ^ (world_z * 19349663)) & 0xFFFFFFFF
                rand_val = ((hash_val >> 13) ^ hash_val) & 0xFFFF
                variation = (rand_val / 65535.0 - 0.5) * 2.0                  
                base_height += variation
                height = int(max(1.0, min(self.max_height, base_height)))
                                                                              
                if (hash_val & 0xFF) < 8:
                    height = 0
                for y in range(height):
                    if y == height - 1 and height > 0:
                        block_type = 0                  
                    elif y >= height - 3:
                        block_type = 1        
                    else:
                        block_type = 2         
                    pos = (world_x, y, world_z)
                    self.blocks[pos] = block_type
                    block_list.append(pos)
                                                              
        self.loaded_chunks[(cx, cz)] = True
        self.chunk_blocks[(cx, cz)] = block_list

    def unload_chunk(self, cx: int, cz: int) -> None:
           
        if (cx, cz) not in self.loaded_chunks:
            return
                                      
        for pos in self.chunk_blocks.get((cx, cz), []):
            self.blocks.pop(pos, None)
                            
        self.loaded_chunks.pop((cx, cz), None)
        self.chunk_blocks.pop((cx, cz), None)

    def update_chunks(self, camera_pos: np.ndarray, radius: int = 1) -> bool:
           
        changed = False
        cx = int(math.floor(camera_pos[0] / self.chunk_size))
        cz = int(math.floor(camera_pos[2] / self.chunk_size))
                                 
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                nx = cx + dx
                nz = cz + dz
                if (nx, nz) not in self.loaded_chunks:
                    self.generate_chunk(nx, nz)
                    changed = True
                                   
        to_remove: List[Tuple[int, int]] = []
        for (lcx, lcz) in list(self.loaded_chunks.keys()):
            if abs(lcx - cx) > radius or abs(lcz - cz) > radius:
                to_remove.append((lcx, lcz))
        for (rcx, rcz) in to_remove:
            self.unload_chunk(rcx, rcz)
            changed = True
        return changed

    def is_block_at(self, x: int, y: int, z: int) -> bool:
                                                                            
        return (x, y, z) in self.blocks

    def get_highest_y(self, x: int, z: int) -> int:
           
        max_y = -1
                                                                     
        for (bx, by, bz), btype in self.blocks.items():
            if bx == x and bz == z:
                if by > max_y:
                    max_y = by
        return max_y

    def build_mesh(self) -> np.ndarray:
           
        vertices: List[float] = []
                                                                                        
        face_vertices = [
                              
            [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
                            
            [(1, 0, 0), (0, 0, 0), (0, 1, 0), (1, 1, 0)],
                        
            [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
                         
            [(1, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)],
                       
            [(0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0)],
                        
            [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        ]
                            
        face_normals = [
            (0, 0, 1),                
            (0, 0, -1),             
            (-1, 0, 0),         
            (1, 0, 0),           
            (0, 1, 0),         
            (0, -1, 0),         
        ]
                                                 
        neighbor_offsets = [
            (0, 0, 1),
            (0, 0, -1),
            (-1, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
        ]
                        
        uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
                                                                     
        global BLOCK_TEXTURE_MAP
        for (x, y, z), block_type in self.blocks.items():
            for face_idx, quad in enumerate(face_vertices):
                nx, ny, nz = neighbor_offsets[face_idx]
                if self.is_block_at(x + nx, y + ny, z + nz):
                                    
                    continue
                normal = face_normals[face_idx]
                                                                                        
                if BLOCK_TEXTURE_MAP is not None:
                    tex_index = BLOCK_TEXTURE_MAP.get((block_type, face_idx), block_type)
                else:
                    tex_index = block_type
                                          
                indices = [0, 1, 2, 2, 3, 0]
                for idx in indices:
                    px, py, pz = quad[idx]
                    world_pos = (x + px, y + py, z + pz)
                    u, v = uvs[idx % 4]
                    vertices.extend([
                        float(world_pos[0]), float(world_pos[1]), float(world_pos[2]),
                        u, v,
                        float(tex_index),
                        float(normal[0]), float(normal[1]), float(normal[2]),
                    ])
        return np.array(vertices, dtype=np.float32)

class blockworldClone:
                                                                       
    def __init__(self) -> None:
                                                                          
        logging.basicConfig(
            filename="blockworld_clone.log",
            level=logging.ERROR,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
                                                  
        self.base_path = Path(__file__).resolve().parent
        self.data_path = self.base_path / "data"
        ensure_directories(self.data_path)
                               
        self.textures = generate_textures(self.data_path)
        self.sounds = generate_sounds(self.data_path)
                                    
        self.skybox_textures = generate_skybox_textures(self.data_path)
                                                                                               
        self.texture_keys = ["grass_top", "grass_side", "dirt", "stone"]
                               
        pygame.init()
                                                                             
        try:
            pygame.mixer.init()
        except pygame.error as e:
            logging.error(f"Audio initialisierung fehlgeschlagen: {e}")
                          
        pygame.font.init()
        self.font = pygame.font.Font(None, 48)
                                                     
        info = pygame.display.Info()
        self.width = info.current_w
        self.height = info.current_h
        self.clock = pygame.time.Clock()
                     
        self.state = "menu"
                             
        self.camera_pos = np.array([16.0, 10.0, 16.0], dtype=np.float32)
        self.camera_yaw = 0.0                              
        self.camera_pitch = 0.0                            
        self.move_speed = 10.0                         
        self.mouse_sensitivity = 0.1                                      
                                    
        self.flight_mode: bool = False                                              
        self.gravity: float = -25.0                                             
        self.vertical_velocity: float = 0.0                                            
                                                                                                    
        self.jump_speed: float = math.sqrt(2.0 * abs(self.gravity) * 1.0)
        self.on_ground: bool = False                                              
                                                                        
        self.time_of_day: float = 0.0
        self.day_length: float = 60.0                                                  
                                                       
        self.world: World | None = None
        self.ctx: moderngl.Context | None = None
        self.prog = None
        self.vbo = None
        self.vao = None
        self.atlas_texture = None

    def run_menu(self) -> None:
                                                                       
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
        screen = pygame.display.set_mode((self.width, self.height), pyloc.FULLSCREEN)
        pygame.display.set_caption("blockworld Clone – Menü")
                                                                                       
        try:
                                                                                                               
            play_list: List[Path] = []
            menu_path = self.sounds.get("menu_music")
            if menu_path:
                play_list.append(menu_path)
                                     
            extra_path = self.sounds.get("menu_extra")
            if extra_path:
                play_list.append(extra_path)
            if play_list:
                                                      
                pygame.mixer.music.load(str(play_list[0]))
                pygame.mixer.music.play()
                                                                               
                if len(play_list) > 1:
                    pygame.mixer.music.queue(str(play_list[1]))
            else:
                                                                             
                pygame.mixer.music.load(str(self.sounds["menu_music"]))
                pygame.mixer.music.play(-1)
        except Exception as e:
            logging.error(f"Menümusik konnte nicht geladen werden: {e}")
        while self.state == "menu":
            for event in pygame.event.get():
                if event.type == pyloc.QUIT:
                    self.state = "quit"
                elif event.type == pyloc.KEYDOWN:
                    if event.key == pyloc.K_RETURN:
                        self.state = "game"
                    elif event.key in (pyloc.K_ESCAPE, pyloc.K_q):
                        self.state = "quit"
                           
            screen.fill((10, 20, 30))
            title_surface = self.font.render("blockworld Clone", True, (255, 255, 255))
            start_surface = self.font.render("Drücke ENTER zum Starten", True, (200, 200, 200))
            quit_surface = self.font.render("Drücke ESC zum Beenden", True, (200, 200, 200))
            info_surface = self.font.render("Drücke F zum Flugmodus im Spiel", True, (180, 180, 180))
                           
            screen.blit(title_surface, title_surface.get_rect(center=(self.width // 2, self.height // 3)))
            screen.blit(start_surface, start_surface.get_rect(center=(self.width // 2, self.height // 2)))
            screen.blit(info_surface, info_surface.get_rect(center=(self.width // 2, self.height // 2 + 50)))
            screen.blit(quit_surface, quit_surface.get_rect(center=(self.width // 2, self.height // 2 + 100)))
            pygame.display.flip()
            self.clock.tick(30)
                       
        pygame.mixer.music.stop()

    def _setup_opengl(self) -> None:
                                                                                
        pygame.display.set_mode(
            (self.width, self.height), pyloc.OPENGL | pyloc.DOUBLEBUF | pyloc.FULLSCREEN
        )
                                             
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
                                    
        self.ctx = moderngl.create_context()
                                                                                 
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
                                       
        self.world = World(chunk_size=32, max_height=12, seed=42)
                                                               
        global BLOCK_TEXTURE_MAP
        BLOCK_TEXTURE_MAP = {}
                                                                                                
        for block_type in (0, 1, 2):
            for face_idx in range(6):
                if block_type == 0:
                               
                    if face_idx == 4:        
                        tex_name = "grass_top"
                    elif face_idx == 5:         
                        tex_name = "dirt"
                    else:          
                        tex_name = "grass_side"
                elif block_type == 1:
                                                           
                    tex_name = "dirt"
                else:
                           
                    tex_name = "stone"
                                                     
                try:
                    tex_index = self.texture_keys.index(tex_name)
                except ValueError:
                    tex_index = 0
                BLOCK_TEXTURE_MAP[(block_type, face_idx)] = tex_index
                                                     
        vertices = self.world.build_mesh()
                                                                   
        vertex_shader_source = """
            #version 330
            in vec3 in_position;
            in vec2 in_uv;
            in float in_tex_index;
            in vec3 in_normal;

            uniform mat4 mvp;
            out vec2 v_uv;
            flat out int v_tex_index;
            out vec3 v_normal;

            void main() {
                gl_Position = mvp * vec4(in_position, 1.0);
                v_uv = in_uv;
                v_tex_index = int(in_tex_index + 0.5);
                v_normal = in_normal;
            }
        """
        fragment_shader_source = """
            #version 330
            in vec2 v_uv;
            flat in int v_tex_index;
            in vec3 v_normal;
            out vec4 fragColor;

            uniform sampler2D atlas;
            uniform int atlas_cols;
            uniform vec3 light_dir;

            void main() {
                float cols = float(atlas_cols);
                // Texturkoordinaten für den jeweiligen Blocktyp berechnen
                vec2 uv = vec2(v_uv.x / cols + float(v_tex_index) / cols, v_uv.y);
                vec4 tex_color = texture(atlas, uv);
                if (tex_color.a < 0.1) {
                    discard;
                }
                // Beleuchtung: Lambert + Ambient
                vec3 N = normalize(v_normal);
                vec3 L = normalize(light_dir);
                float diff = max(dot(N, -L), 0.2); // Minimalwert als Umgebungslicht
                vec3 color = tex_color.rgb * diff;
                fragColor = vec4(color, tex_color.a);
            }
        """
        self.prog = self.ctx.program(vertex_shader=vertex_shader_source, fragment_shader=fragment_shader_source)
                                                                      
        self.vbo = self.ctx.buffer(vertices.tobytes())
        vao_content = [
            (self.vbo, "3f 2f 1f 3f", "in_position", "in_uv", "in_tex_index", "in_normal"),
        ]
        self.vao = self.ctx.vertex_array(self.prog, vao_content)
                                         
        atlas_img, atlas_cols = self._create_atlas()
        self.atlas_texture = self.ctx.texture(atlas_img.size, 3, atlas_img.tobytes())
        self.atlas_texture.build_mipmaps()
        self.atlas_texture.use(location=0)
                         
        self.prog["atlas"].value = 0
        self.prog["atlas_cols"].value = atlas_cols
                                                                         
        self.atlas_cols = atlas_cols
                                       
        self.prog["light_dir"].value = (0.5, -1.0, 0.5)

        sky_vertices = np.array([
                              
            -1.0, -1.0,  1.0,
             1.0, -1.0,  1.0,
             1.0,  1.0,  1.0,
             1.0,  1.0,  1.0,
            -1.0,  1.0,  1.0,
            -1.0, -1.0,  1.0,
                            
            -1.0, -1.0, -1.0,
            -1.0,  1.0, -1.0,
             1.0,  1.0, -1.0,
             1.0,  1.0, -1.0,
             1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0,
                        
            -1.0,  1.0, -1.0,
            -1.0,  1.0,  1.0,
            -1.0, -1.0,  1.0,
            -1.0, -1.0,  1.0,
            -1.0, -1.0, -1.0,
            -1.0,  1.0, -1.0,
                         
             1.0,  1.0, -1.0,
             1.0, -1.0, -1.0,
             1.0, -1.0,  1.0,
             1.0, -1.0,  1.0,
             1.0,  1.0,  1.0,
             1.0,  1.0, -1.0,
                       
            -1.0,  1.0, -1.0,
             1.0,  1.0, -1.0,
             1.0,  1.0,  1.0,
             1.0,  1.0,  1.0,
            -1.0,  1.0,  1.0,
            -1.0,  1.0, -1.0,
                        
            -1.0, -1.0, -1.0,
            -1.0, -1.0,  1.0,
             1.0, -1.0,  1.0,
             1.0, -1.0,  1.0,
             1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0,
        ], dtype='f4')
        self.sky_vbo = self.ctx.buffer(sky_vertices.tobytes())
                                                                                 
        try:
                                                                          
            face_order = ["right", "left", "top", "bottom", "front", "back"]
                                            
            face_images = [Image.open(str(self.skybox_textures[name])).convert("RGB") for name in face_order]
            cube_size = face_images[0].size[0]
                              
            self.sky_cubemap = self.ctx.texture_cube((cube_size, cube_size), 3)
            for i, img in enumerate(face_images):
                                                                               
                self.sky_cubemap.write(i, img.tobytes())
            self.sky_cubemap.build_mipmaps()
                                           
            sky_vertex_shader = """
                #version 330
                in vec3 in_position;
                uniform mat4 mvp;
                out vec3 v_dir;
                void main() {
                    v_dir = in_position;
                    gl_Position = mvp * vec4(in_position, 1.0);
                }
            """
            sky_fragment_shader = """
                #version 330
                in vec3 v_dir;
                out vec4 fragColor;
                uniform samplerCube skybox;
                void main() {
                    // Invertiere die Richtung, damit die Texturen bei einer Skybox, die von innen betrachtet wird,
                    // korrekt angezeigt werden. Ohne Invertierung könnten die Bilder nach außen zeigen.
                    vec3 dir = normalize(v_dir);
                    fragColor = texture(skybox, -dir);
                }
            """
            self.sky_prog = self.ctx.program(vertex_shader=sky_vertex_shader, fragment_shader=sky_fragment_shader)
            self.sky_vao = self.ctx.vertex_array(self.sky_prog, [(self.sky_vbo, '3f', 'in_position')])
                                                                 
            self.sky_cubemap.use(location=1)
            self.sky_prog["skybox"].value = 1
            self.using_cubemap = True
        except Exception as e:
                                                                                  
            logging.error(f"Skybox Cubemap konnte nicht geladen werden: {e}")
                                             
            sky_vertex_shader = """
                #version 330
                in vec3 in_position;
                uniform mat4 mvp;
                out vec3 v_dir;
                void main() {
                    v_dir = in_position;
                    gl_Position = mvp * vec4(in_position, 1.0);
                }
            """
            sky_fragment_shader = """
                #version 330
                in vec3 v_dir;
                out vec4 fragColor;
                void main() {
                    vec3 dir = normalize(v_dir);
                    // einfacher Verlauf: heller am Horizont, dunkler am Zenit
                    vec3 topColor = vec3(0.5, 0.7, 0.95);
                    vec3 bottomColor = vec3(0.9, 0.95, 1.0);
                    float t = dir.y * 0.5 + 0.5;
                    vec3 col = mix(bottomColor, topColor, t);
                    fragColor = vec4(col, 1.0);
                }
            """
            self.sky_prog = self.ctx.program(vertex_shader=sky_vertex_shader, fragment_shader=sky_fragment_shader)
            self.sky_vao = self.ctx.vertex_array(self.sky_prog, [(self.sky_vbo, '3f', 'in_position')])
            self.using_cubemap = False

    def _create_atlas(self) -> Tuple[Image.Image, int]:
           
        if hasattr(self, "texture_keys"):
            keys = self.texture_keys
        else:
            keys = sorted(self.textures.keys())
        images = [Image.open(str(self.textures[name])).convert("RGB") for name in keys]
        tile_w, tile_h = images[0].size
        cols = len(images)
        atlas = Image.new("RGB", (tile_w * cols, tile_h))
        for idx, img in enumerate(images):
            atlas.paste(img, (idx * tile_w, 0))
        return atlas, cols

    def run_game(self) -> None:
                                           
        try:
                                                                            
            try:
                play_list: List[Path] = []
                missing: List[Tuple[int, Path]] = []
                                                                                 
                for i in range(1, 6):
                    song_key = f"song{i}"
                    song_path = self.data_path / "music" / f"song{i}.wav"
                                                          
                    self.sounds[song_key] = song_path
                    play_list.append(song_path)
                    if not song_path.exists():
                        missing.append((i, song_path))
                                                             
                if missing:
                    loading_screen = pygame.display.set_mode((self.width, self.height), pyloc.FULLSCREEN)
                    loading_font = pygame.font.Font(None, 48)
                    total_missing = len(missing)
                    for idx, (num, spath) in enumerate(missing, start=1):
                        loading_screen.fill((20, 30, 40))
                        text = loading_font.render(f"Erzeuge Musik {idx}/{total_missing} ...", True, (220, 220, 220))
                        subtext = loading_font.render("Bitte warten", True, (150, 150, 150))
                        rect = text.get_rect(center=(self.width // 2, self.height // 2))
                        loading_screen.blit(text, rect)
                        loading_screen.blit(subtext, subtext.get_rect(center=(self.width // 2, self.height // 2 + 50)))
                        pygame.display.flip()
                        try:
                            generate_structured_music(spath, total_duration=30.0)
                                                                
                            backup_generated_file(spath, self.data_path)
                        except Exception as ex:
                            logging.error(f"Konnte Musikstück {spath.name} nicht generieren: {ex}")
                                           
                    loading_screen.fill((0, 0, 0))
                    pygame.display.flip()
                                                                                 
                self.current_song_index = 0
                                                                   
                valid_songs = [p for p in play_list if p.exists()]
                if valid_songs:
                    try:
                        pygame.mixer.music.load(str(valid_songs[self.current_song_index]))
                        pygame.mixer.music.play()
                                                                 
                        self.MUSIC_END_EVENT = pygame.USEREVENT + 10
                        pygame.mixer.music.set_endevent(self.MUSIC_END_EVENT)
                    except Exception as e:
                        logging.error(f"Spielmusik konnte nicht geladen werden: {e}")
                                                                      
                    self.song_list = valid_songs
                else:
                                                                               
                    try:
                        pygame.mixer.music.load(str(self.sounds["game_music"]))
                        pygame.mixer.music.play(-1)
                    except Exception as e:
                        logging.error(f"Spielmusik konnte nicht geladen werden: {e}")
                    self.song_list = []
            except Exception as e:
                logging.error(f"Spielmusik konnte nicht geladen werden: {e}")
                                   
            self._setup_opengl()
                                           
            aspect = self.width / float(self.height)
            proj = perspective_matrix(75.0, aspect, 0.1, 200.0)
                                         
            self.vertical_velocity = 0.0
            self.on_ground = False
            self.time_of_day = 0.0
                        
            running = True
                                                                         
            last_chunk_coord: Tuple[int, int] | None = None
            while running and self.state == "game":
                dt = self.clock.tick(60) / 1000.0
                                                             
                self.time_of_day = (self.time_of_day + dt / self.day_length) % 1.0
                                                                     
                sun_angle = self.time_of_day * 2.0 * math.pi
                                                                                                
                light_dir = np.array([
                    math.cos(sun_angle),
                    -abs(math.sin(sun_angle)),
                    math.sin(sun_angle) * 0.5
                ], dtype=np.float32)
                                              
                self.prog["light_dir"].value = tuple(light_dir)
                                        
                take_screenshot = False
                                                                        
                mouse_break = False
                for event in pygame.event.get():
                                                                      
                    if hasattr(self, "MUSIC_END_EVENT") and event.type == getattr(self, "MUSIC_END_EVENT", -1):
                                                                                             
                        if hasattr(self, "song_list") and self.song_list:
                            self.current_song_index = (self.current_song_index + 1) % len(self.song_list)
                            try:
                                pygame.mixer.music.load(str(self.song_list[self.current_song_index]))
                                pygame.mixer.music.play()
                                pygame.mixer.music.set_endevent(self.MUSIC_END_EVENT)
                            except Exception as e:
                                logging.error(f"Konnte nächsten Song nicht laden: {e}")
                        continue
                    if event.type == pyloc.QUIT:
                        running = False
                        self.state = "quit"
                    elif event.type == pyloc.KEYDOWN:
                        if event.key in (pyloc.K_ESCAPE, pyloc.K_q):
                            running = False
                            self.state = "quit"
                        elif event.key == pyloc.K_f:
                                                  
                            self.flight_mode = not self.flight_mode
                        elif event.key == pyloc.K_SPACE:
                                                    
                            if not self.flight_mode and self.on_ground:
                                self.vertical_velocity = self.jump_speed
                                self.on_ground = False
                        elif event.key == pyloc.K_F12:
                                                 
                            take_screenshot = True
                    elif event.type == pyloc.MOUSEBUTTONDOWN:
                        if event.button == 1:
                                                                                         
                            mouse_break = True
                                                       
                            try:
                                sound = pygame.mixer.Sound(str(self.sounds["click"]))
                                sound.play()
                            except Exception as e:
                                logging.error(f"Sound konnte nicht abgespielt werden: {e}")
                                                                                 
                mx, my = pygame.mouse.get_rel()
                self.camera_yaw += mx * self.mouse_sensitivity
                self.camera_pitch -= my * self.mouse_sensitivity
                self.camera_pitch = max(-89.9, min(89.9, self.camera_pitch))
                                    
                yaw_rad = math.radians(self.camera_yaw)
                pitch_rad = math.radians(self.camera_pitch)
                front = np.array([
                    math.cos(pitch_rad) * math.cos(yaw_rad),
                    math.sin(pitch_rad),
                    math.cos(pitch_rad) * math.sin(yaw_rad),
                ], dtype=np.float32)
                front_norm = front / np.linalg.norm(front)
                right = np.cross(front_norm, np.array([0.0, 1.0, 0.0], dtype=np.float32))
                right_norm = right / np.linalg.norm(right)
                up = np.cross(right_norm, front_norm)
                                
                keys = pygame.key.get_pressed()

                if self.flight_mode:
                                                                     
                    if keys[pyloc.K_w]:
                        self.camera_pos += front_norm * self.move_speed * dt
                    if keys[pyloc.K_s]:
                        self.camera_pos -= front_norm * self.move_speed * dt
                    if keys[pyloc.K_a]:
                        self.camera_pos -= right_norm * self.move_speed * dt
                    if keys[pyloc.K_d]:
                        self.camera_pos += right_norm * self.move_speed * dt
                    if keys[pyloc.K_SPACE]:
                        self.camera_pos += up * self.move_speed * dt
                    if keys[pyloc.K_LCTRL] or keys[pyloc.K_RCTRL]:
                        self.camera_pos -= up * self.move_speed * dt
                                                    
                    self.vertical_velocity = 0.0
                    self.on_ground = False
                else:
                                                                                                            
                    def collides(candidate_pos: np.ndarray) -> bool:
                           
                        if self.world is None:
                            return False
                                                                                                            
                        radius = 0.4
                                                                                                                      
                        foot_y = int(math.floor(candidate_pos[1] - 1.8)) + 1
                        head_y = int(math.floor(candidate_pos[1] - 0.1))
                                                                                  
                        for dx in (-radius, radius):
                            for dz in (-radius, radius):
                                check_x = int(math.floor(candidate_pos[0] + dx))
                                check_z = int(math.floor(candidate_pos[2] + dz))
                                for y in range(foot_y, head_y + 1):
                                    if self.world.is_block_at(check_x, y, check_z):
                                        return True
                        return False

                    horiz_front = np.array([front_norm[0], 0.0, front_norm[2]], dtype=np.float32)
                                                        
                    if np.linalg.norm(horiz_front) > 0.0001:
                        horiz_front /= np.linalg.norm(horiz_front)
                                                
                    horiz_right = np.array([right_norm[0], 0.0, right_norm[2]], dtype=np.float32)
                    if np.linalg.norm(horiz_right) > 0.0001:
                        horiz_right /= np.linalg.norm(horiz_right)
                                                                                                                          
                    move_dx = 0.0
                    move_dz = 0.0
                                                                              
                    if keys[pyloc.K_w]:
                        move_dx += horiz_front[0] * self.move_speed * dt
                        move_dz += horiz_front[2] * self.move_speed * dt
                    if keys[pyloc.K_s]:
                        move_dx -= horiz_front[0] * self.move_speed * dt
                        move_dz -= horiz_front[2] * self.move_speed * dt
                    if keys[pyloc.K_a]:
                        move_dx -= horiz_right[0] * self.move_speed * dt
                        move_dz -= horiz_right[2] * self.move_speed * dt
                    if keys[pyloc.K_d]:
                        move_dx += horiz_right[0] * self.move_speed * dt
                        move_dz += horiz_right[2] * self.move_speed * dt
                                                                                                                 
                    if abs(move_dx) > 1e-5:
                                                            
                        candidate = np.array([
                            self.camera_pos[0] + move_dx,
                            self.camera_pos[1],
                            self.camera_pos[2],
                        ], dtype=np.float32)
                        if not collides(candidate):
                                                               
                            self.camera_pos[0] += move_dx
                        else:
                                                                      
                            candidate_step = candidate.copy()
                            candidate_step[1] += 1.0
                            if not collides(candidate_step):
                                self.camera_pos[0] += move_dx
                                self.camera_pos[1] += 1.0
                                               
                    if abs(move_dz) > 1e-5:
                        candidate = np.array([
                            self.camera_pos[0],
                            self.camera_pos[1],
                            self.camera_pos[2] + move_dz,
                        ], dtype=np.float32)
                        if not collides(candidate):
                            self.camera_pos[2] += move_dz
                        else:
                            candidate_step = candidate.copy()
                            candidate_step[1] += 1.0
                            if not collides(candidate_step):
                                self.camera_pos[2] += move_dz
                                self.camera_pos[1] += 1.0
                                                                      
                    self.vertical_velocity += self.gravity * dt
                    next_y = self.camera_pos[1] + self.vertical_velocity * dt
                                         
                    col_x = int(math.floor(self.camera_pos[0]))
                    col_z = int(math.floor(self.camera_pos[2]))
                    top_y = self.world.get_highest_y(col_x, col_z)
                                                                               
                    ground_level = (top_y + 1.8) if top_y >= 0 else 0.0
                    if next_y < ground_level:
                                              
                        next_y = ground_level
                        self.vertical_velocity = 0.0
                        self.on_ground = True
                    else:
                        self.on_ground = False
                    self.camera_pos[1] = next_y

                if mouse_break:
                    def break_block(dir_vec: np.ndarray) -> None:
                                         
                        if self.world is None:
                            return
                        max_dist = 6.0
                        step = 0.1
                        origin = self.camera_pos.copy()
                                                                                             
                        for i in range(1, int(max_dist / step)):
                            sample = origin + dir_vec * (i * step)
                            bx = int(math.floor(sample[0]))
                            by = int(math.floor(sample[1]))
                            bz = int(math.floor(sample[2]))
                            if self.world.is_block_at(bx, by, bz):
                                                 
                                try:
                                    self.world.blocks.pop((bx, by, bz))
                                except KeyError:
                                    pass
                                                                    
                                cx = int(math.floor(bx / self.world.chunk_size))
                                cz = int(math.floor(bz / self.world.chunk_size))
                                if (cx, cz) in self.world.chunk_blocks:
                                    try:
                                        self.world.chunk_blocks[(cx, cz)].remove((bx, by, bz))
                                    except ValueError:
                                        pass
                                                        
                                new_vertices = self.world.build_mesh()
                                self.vbo = self.ctx.buffer(new_vertices.tobytes())
                                vao_content = [
                                    (self.vbo, "3f 2f 1f 3f", "in_position", "in_uv", "in_tex_index", "in_normal"),
                                ]
                                self.vao = self.ctx.vertex_array(self.prog, vao_content)
                                                              
                                self.prog["atlas"].value = 0
                                self.prog["atlas_cols"].value = getattr(self, "atlas_cols", 1)
                                return
                                                                                                         
                    break_block(front_norm)
                                       
                    mouse_break = False
                                                                                                            
                if self.world:
                                                                         
                    curr_cx = int(math.floor(self.camera_pos[0] / self.world.chunk_size))
                    curr_cz = int(math.floor(self.camera_pos[2] / self.world.chunk_size))
                    curr_coord = (curr_cx, curr_cz)
                    if curr_coord != last_chunk_coord:
                                                      
                        if self.world.update_chunks(self.camera_pos, radius=1):
                                                                               
                            new_vertices = self.world.build_mesh()
                                                       
                            self.vbo = self.ctx.buffer(new_vertices.tobytes())
                            vao_content = [
                                (self.vbo, "3f 2f 1f 3f", "in_position", "in_uv", "in_tex_index", "in_normal"),
                            ]
                            self.vao = self.ctx.vertex_array(self.prog, vao_content)
                                                                        
                            self.prog["atlas"].value = 0
                            self.prog["atlas_cols"].value = getattr(self, "atlas_cols", 1)
                        last_chunk_coord = curr_coord
                                          
                target = self.camera_pos + front_norm
                view = look_at_matrix(self.camera_pos, target, up)
                                              
                mvp = (proj @ view).T
                self.prog["mvp"].write(mvp.astype('f4').tobytes())
                                                                      
                view_no_trans = view.copy()
                view_no_trans[0, 3] = 0.0
                view_no_trans[1, 3] = 0.0
                view_no_trans[2, 3] = 0.0
                sky_mvp = (proj @ view_no_trans).T
                self.sky_prog["mvp"].write(sky_mvp.astype('f4').tobytes())
                                                            
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                                                                                 
                self.ctx.disable(moderngl.DEPTH_TEST)
                self.ctx.disable(moderngl.CULL_FACE)
                                                                     
                if hasattr(self, "sky_cubemap"):
                    self.sky_cubemap.use(location=1)
                self.sky_vao.render(moderngl.TRIANGLES)
                                                                      
                self.ctx.enable(moderngl.DEPTH_TEST)
                self.ctx.enable(moderngl.CULL_FACE)
                                                                         
                self.vao.render()
                                                                                            
                if take_screenshot:
                    try:
                                                                                           
                        data = self.ctx.screen.read(components=3, alignment=1)
                        img = Image.frombytes('RGB', (self.width, self.height), data)
                        img = img.transpose(Image.FLIP_TOP_BOTTOM)
                                                                    
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_dir = self.data_path / "screenshots"
                        screenshot_dir.mkdir(parents=True, exist_ok=True)
                        filename = screenshot_dir / f"screenshot_{timestamp}.png"
                        img.save(filename)
                                                    
                        logging.info(f"Screenshot gespeichert: {filename}")
                    except Exception as e:
                        logging.error(f"Screenshot konnte nicht gespeichert werden: {e}")
                                 
                pygame.display.flip()
                           
            pygame.mixer.music.stop()
        except Exception as e:
            logging.exception("Fehler in der Spielschleife")
                                                          
            pygame.time.wait(2000)

    def run(self) -> None:
                                                                           
        while self.state not in ("quit", "exit"):
            if self.state == "menu":
                self.run_menu()
            elif self.state == "game":
                self.run_game()
        pygame.quit()

if __name__ == "__main__":
    game = blockworldClone()
    game.run()