import numpy as np
import scipy.io.wavfile as wav
import librosa
import soundfile as sf
import os
import random
import json
from datetime import datetime
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import time


class EnhancedBinauralBeatsGenerator:
    def __init__(self, sample_rate: int = 44100):
        """
        Verbeterde AI Music Generator voor rustgevende 40Hz Gamma Binaural Beats
        """
        self.sample_rate = sample_rate
        self.gamma_frequency = 40  # 40Hz gamma waves voor focus

        # Zachte, rustgevende basis frequenties (lagere tonen)
        self.base_frequencies = {
            'deep_focus': [110, 123.47, 146.83, 164.81],  # Warme, diepe tonen
            'creative': [174.61, 196, 220, 246.94],  # Creatieve frequenties
            'meditation': [136.1, 144, 151.91, 161.26],  # Meditatie frequenties
            'study': [185.93, 207.65, 233.08, 261.63],  # Study frequenties
            'ambient': [98, 103.83, 110, 116.54],  # Zeer zachte ambient
            'healing': [128, 136.1, 144, 152.56],  # Helende frequenties
            'relaxed': [103.83, 110, 123.47, 138.59]  # Ontspannen frequenties
        }

        # Meer variatie in natuurlijke geluiden
        self.nature_patterns = {
            'gentle_rain': {'freq_range': (200, 4000), 'intensity': 0.08, 'modulation': 0.2},
            'soft_ocean': {'freq_range': (80, 1500), 'intensity': 0.12, 'modulation': 0.15},
            'forest_breeze': {'freq_range': (300, 5000), 'intensity': 0.06, 'modulation': 0.25},
            'distant_thunder': {'freq_range': (40, 800), 'intensity': 0.10, 'modulation': 0.1},
            'mountain_wind': {'freq_range': (100, 2500), 'intensity': 0.07, 'modulation': 0.3},
            'babbling_brook': {'freq_range': (400, 6000), 'intensity': 0.09, 'modulation': 0.4}
        }

        # Verschillende fade/overgang patronen
        self.transition_patterns = [
            'linear', 'exponential', 'sine_wave', 'logarithmic', 'double_curve'
        ]

    def generate_smooth_binaural_beat(self, base_freq: float, duration: int = 35,
                                      beat_freq: float = 40.0,
                                      frequency_drift: bool = True) -> np.ndarray:
        """
        Genereer vloeiende binaural beat met frequentie drift voor natuurlijkheid
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)

        # Voeg subtiele frequentie drift toe voor natuurlijkheid
        if frequency_drift:
            # Zeer langzame frequentie modulatie (0.01-0.05 Hz)
            drift_freq = random.uniform(0.01, 0.05)
            drift_amount = random.uniform(0.5, 2.0)  # Maximaal 2Hz drift

            frequency_modulation = drift_amount * np.sin(2 * np.pi * drift_freq * t)
            left_freq_modulated = base_freq + frequency_modulation
            right_freq_modulated = base_freq + beat_freq + frequency_modulation
        else:
            left_freq_modulated = base_freq
            right_freq_modulated = base_freq + beat_freq

        # Genereer zachte sinusgolven
        left_channel = np.sin(2 * np.pi * left_freq_modulated * t)
        right_channel = np.sin(2 * np.pi * right_freq_modulated * t)

        # Voeg zeer subtiele amplitude modulatie toe
        amp_modulation = 1 + 0.05 * np.sin(2 * np.pi * 0.1 * t)
        left_channel *= amp_modulation
        right_channel *= amp_modulation

        stereo = np.column_stack([left_channel, right_channel])
        return stereo

    def save_track(self, audio: np.ndarray, metadata: Dict, filename: str, output_dir: str) -> str:
        """Sla audio track op als WAV bestand"""
        os.makedirs(output_dir, exist_ok=True)

        # Zorg ervoor dat filename eindigt op .wav
        if not filename.endswith('.wav'):
            filename += '.wav'

        file_path = os.path.join(output_dir, filename)

        # Normaliseer audio naar 16-bit integer bereik
        if audio.dtype != np.int16:
            audio_normalized = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        else:
            audio_normalized = audio

        # Sla op als WAV
        sf.write(file_path, audio_normalized, self.sample_rate)

        # Sla metadata op als JSON
        metadata_file = file_path.replace('.wav', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        return file_path
    def generate_harmonic_layers(self, base_freq: float, duration: int,
                                 num_layers: int = None) -> np.ndarray:
        """Genereer meerdere harmonische lagen met verschillende overgangen"""
        if num_layers is None:
            num_layers = random.randint(2, 5)

        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        combined_signal = np.zeros(len(t))

        # Harmonische verhoudingen die goed klinken samen
        harmonic_ratios = [1.0, 1.2, 1.5, 2.0, 2.4, 3.0]
        selected_ratios = random.sample(harmonic_ratios, min(num_layers, len(harmonic_ratios)))

        for i, ratio in enumerate(selected_ratios):
            freq = base_freq * ratio

            # Verschillende amplitude curves per laag
            if i == 0:  # Basis laag - meest prominent
                amplitude = 0.6
            else:  # Harmonische lagen - afnemend
                amplitude = 0.4 / (i + 1)

            # Voeg per laag verschillende fade-in/out patronen toe
            layer_signal = amplitude * np.sin(2 * np.pi * freq * t)

            # Random fade patroon per laag
            fade_pattern = random.choice(self.transition_patterns)
            layer_signal = self.apply_transition_pattern(layer_signal, fade_pattern)

            combined_signal += layer_signal

        return combined_signal

    def apply_transition_pattern(self, signal: np.ndarray, pattern: str) -> np.ndarray:
        """Pas verschillende overgangspatronen toe"""
        length = len(signal)
        fade_length = int(length * 0.1)  # 10% van de lengte voor fade

        if pattern == 'linear':
            # Standaard lineaire fade
            pass  # Wordt later toegepast in apply_envelope

        elif pattern == 'exponential':
            # Exponentiële fade-in
            fade_in = np.exp(np.linspace(-3, 0, fade_length))
            fade_in = fade_in / fade_in[-1]  # Normaliseer naar 1
            signal[:fade_length] *= fade_in

        elif pattern == 'sine_wave':
            # Sine wave fade
            fade_in = np.sin(np.linspace(0, np.pi / 2, fade_length))
            signal[:fade_length] *= fade_in

        elif pattern == 'logarithmic':
            # Logaritmische fade
            fade_in = np.log(np.linspace(1, np.e, fade_length))
            signal[:fade_length] *= fade_in

        elif pattern == 'double_curve':
            # S-curve (double exponential)
            x = np.linspace(0, 1, fade_length)
            fade_in = 1 / (1 + np.exp(-10 * (x - 0.5)))
            signal[:fade_length] *= fade_in

        return signal

    def generate_evolving_ambient(self, duration: int, evolution_points: int = None) -> np.ndarray:
        """
        Genereer evoluerende ambient laag die verandert over tijd
        """
        if evolution_points is None:
            evolution_points = random.randint(3, 7)

        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)

        # Basis ambient noise
        base_noise = np.random.normal(0, 0.03, samples)

        # Verschillende evolutie punten
        section_length = samples // evolution_points
        evolved_noise = np.zeros(samples)

        for i in range(evolution_points):
            start_idx = i * section_length
            end_idx = min((i + 1) * section_length, samples)
            section_samples = end_idx - start_idx

            # Verschillende karakteristieken per sectie
            if i % 2 == 0:
                # Even secties: warmer, dieper
                section_noise = self.generate_warm_noise(section_samples)
            else:
                # Oneven secties: breder, opener
                section_noise = self.generate_bright_noise(section_samples)

            # Crossfade tussen secties
            if i > 0:
                crossfade_length = min(section_samples // 4, 2048)
                fade_out = np.linspace(1, 0, crossfade_length)
                fade_in = np.linspace(0, 1, crossfade_length)

                # Overlap
                evolved_noise[start_idx:start_idx + crossfade_length] *= fade_out
                evolved_noise[start_idx:start_idx + crossfade_length] += section_noise[:crossfade_length] * fade_in
                evolved_noise[start_idx + crossfade_length:end_idx] = section_noise[crossfade_length:]
            else:
                evolved_noise[start_idx:end_idx] = section_noise

        return evolved_noise

    def generate_warm_noise(self, samples: int) -> np.ndarray:
        """Genereer warme, diepe ambient noise"""
        noise = np.random.normal(0, 0.04, samples)
        # Filter voor warmere karakteristiek (low-pass)
        from scipy import signal
        b, a = signal.butter(3, 0.1, btype='low')
        warm_noise = signal.filtfilt(b, a, noise)
        return warm_noise

    def generate_bright_noise(self, samples: int) -> np.ndarray:
        """Genereer heldere, open ambient noise"""
        noise = np.random.normal(0, 0.03, samples)
        # Filter voor heldere karakteristiek (high-pass + band-pass)
        from scipy import signal
        b, a = signal.butter(2, [0.15, 0.6], btype='band')
        bright_noise = signal.filtfilt(b, a, noise)
        return bright_noise

    def generate_enhanced_nature_sound(self, duration: int, nature_type: str) -> np.ndarray:
        """Verbeterde natuurgeluiden met meer realisme"""
        if nature_type not in self.nature_patterns:
            nature_type = 'gentle_rain'

        pattern = self.nature_patterns[nature_type]
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)

        # Basis natuurgeluid
        base_intensity = pattern['intensity']
        noise = np.random.normal(0, base_intensity, len(t))

        # Frequentie filtering
        from scipy import signal
        nyquist = self.sample_rate / 2
        low = pattern['freq_range'][0] / nyquist
        high = pattern['freq_range'][1] / nyquist

        b, a = signal.butter(4, [low, high], btype='band')
        filtered_noise = signal.filtfilt(b, a, noise)

        # Meerdere modulatie lagen voor realisme
        mod_freq1 = random.uniform(0.05, 0.15)  # Zeer langzame variatie
        mod_freq2 = random.uniform(0.3, 0.8)  # Snellere details
        mod_freq3 = random.uniform(1.5, 3.0)  # Fijne textuur

        modulation1 = 1 + pattern['modulation'] * 0.5 * np.sin(2 * np.pi * mod_freq1 * t)
        modulation2 = 1 + pattern['modulation'] * 0.3 * np.sin(2 * np.pi * mod_freq2 * t)
        modulation3 = 1 + pattern['modulation'] * 0.2 * np.sin(2 * np.pi * mod_freq3 * t)

        # Combineer modulaties
        total_modulation = modulation1 * modulation2 * modulation3

        return filtered_noise * total_modulation

    def apply_advanced_envelope(self, audio: np.ndarray, fade_type: str = None) -> np.ndarray:
        """Geavanceerde envelope met verschillende fade types"""
        if fade_type is None:
            fade_type = random.choice(self.transition_patterns)

        samples = len(audio) if len(audio.shape) == 1 else len(audio)
        is_stereo = len(audio.shape) == 2

        # Langere, zachtere fades
        fade_duration = random.uniform(2.0, 4.0)
        fade_samples = int(fade_duration * self.sample_rate)
        fade_samples = min(fade_samples, samples // 4)  # Max 25% van track

        # Genereer fade curves gebaseerd op type
        if fade_type == 'exponential':
            fade_in_curve = 1 - np.exp(-5 * np.linspace(0, 1, fade_samples))
            fade_out_curve = np.exp(-5 * np.linspace(0, 1, fade_samples))
        elif fade_type == 'sine_wave':
            fade_in_curve = np.sin(np.linspace(0, np.pi / 2, fade_samples))
            fade_out_curve = np.cos(np.linspace(0, np.pi / 2, fade_samples))
        elif fade_type == 'logarithmic':
            fade_in_curve = np.log(1 + np.linspace(0, np.e - 1, fade_samples))
            fade_out_curve = np.log(1 + np.linspace(np.e - 1, 0, fade_samples))
        else:  # linear of andere
            fade_in_curve = np.linspace(0, 1, fade_samples)
            fade_out_curve = np.linspace(1, 0, fade_samples)

        # Pas toe op audio
        if is_stereo:
            fade_in_curve = fade_in_curve.reshape(-1, 1)
            fade_out_curve = fade_out_curve.reshape(-1, 1)

        audio[:fade_samples] *= fade_in_curve
        audio[-fade_samples:] *= fade_out_curve

        return audio

    def generate_focus_masterpiece(self, style: str = 'deep_focus', duration: int = 35,
                                   include_nature: bool = True, nature_type: str = None,
                                   complexity: str = 'medium') -> Tuple[np.ndarray, Dict]:
        """
        Genereer een complete, rustgevende focus masterpiece
        """
        # Random stijl selectie als niet gespecificeerd
        if style not in self.base_frequencies:
            style = random.choice(list(self.base_frequencies.keys()))

        # Random natuur geluid
        if include_nature and nature_type is None:
            nature_type = random.choice(list(self.nature_patterns.keys()))

        # Variatie in duur
        duration = duration + random.randint(-3, 7)  # ±3-7 seconden variatie

        base_freqs = self.base_frequencies[style]
        main_freq = random.choice(base_freqs)

        print(f"🎵 Creëren: {style} masterpiece ({duration}s) - {main_freq:.2f}Hz")

        # 1. Basis binaural beat met drift
        binaural = self.generate_smooth_binaural_beat(
            main_freq, duration, self.gamma_frequency, frequency_drift=True
        )

        # 2. Harmonische lagen met variërende complexiteit
        if complexity == 'simple':
            num_harmonics = random.randint(2, 3)
        elif complexity == 'complex':
            num_harmonics = random.randint(4, 6)
        else:  # medium
            num_harmonics = random.randint(3, 4)

        harmonics = self.generate_harmonic_layers(main_freq, duration, num_harmonics)
        harmonic_stereo = np.column_stack([harmonics, harmonics])

        # 3. Evoluerende ambient laag
        ambient_evolution = self.generate_evolving_ambient(duration)
        ambient_stereo = np.column_stack([ambient_evolution, ambient_evolution])

        # 4. Natuurgeluiden indien gewenst
        nature_stereo = None
        if include_nature and nature_type:
            nature_sound = self.generate_enhanced_nature_sound(duration, nature_type)
            nature_stereo = np.column_stack([nature_sound, nature_sound])

        # 5. Mix alle lagen met dynamische balans
        mix_weights = self.generate_dynamic_mix_weights(duration)

        # Basis mix
        combined = binaural * 0.5  # Binaural beat als basis
        combined += harmonic_stereo * 0.25  # Harmonische lagen
        combined += ambient_stereo * 0.15  # Ambient evolutie

        if nature_stereo is not None:
            # Variërend volume voor natuurgeluiden
            nature_weight = 0.1 + 0.1 * np.sin(2 * np.pi * 0.02 * np.linspace(0, duration, len(nature_stereo)))
            if len(nature_stereo.shape) == 2:
                nature_weight = nature_weight.reshape(-1, 1)
            combined += nature_stereo * nature_weight

        # 6. Geavanceerde envelope
        fade_type = random.choice(self.transition_patterns)
        combined = self.apply_advanced_envelope(combined, fade_type)

        # 7. Dynamische normalisatie
        target_level = random.uniform(-8.0, -5.0)  # Variërende loudness levels
        combined = self.normalize_audio(combined, target_level)

        # Metadata met meer details
        metadata = {
            'style': style,
            'duration': duration,
            'base_frequency': round(main_freq, 2),
            'binaural_frequency': self.gamma_frequency,
            'nature_sound': nature_type,
            'complexity': complexity,
            'fade_type': fade_type,
            'harmonic_layers': num_harmonics,
            'target_level_db': target_level,
            'generated_at': datetime.now().isoformat(),
            'sample_rate': self.sample_rate,
            'version': '2.0_enhanced'
        }

        return combined, metadata

    def generate_dynamic_mix_weights(self, duration: int) -> Dict:
        """Genereer dynamische mix gewichten die veranderen over tijd"""
        # Voor toekomstige uitbreiding - nu nog eenvoudig
        return {'binaural': 0.5, 'harmonics': 0.25, 'ambient': 0.15, 'nature': 0.1}

    def normalize_audio(self, audio: np.ndarray, target_level: float = -6.0) -> np.ndarray:
        """Verbeterde audio normalisatie"""
        if len(audio.shape) == 2:
            max_amplitude = np.max(np.abs(audio))
        else:
            max_amplitude = np.max(np.abs(audio))

        if max_amplitude > 0:
            current_level = 20 * np.log10(max_amplitude)
            gain = target_level - current_level
            normalized = audio * (10 ** (gain / 20))

            # Soft limiting om clipping te voorkomen
            return np.tanh(normalized * 0.95) * 0.95
        else:
            return audio


class AdvancedFocusMusicBot:
    def __init__(self):
        """Geavanceerde Focus Music Bot met meer variatie en betere kwaliteit"""
        self.generator = EnhancedBinauralBeatsGenerator()
        self.track_counter = 0
        self.session_id = int(time.time())  # Unieke sessie ID

        # Uitgebreide track templates met meer variatie
        self.track_templates = [
            # Deep Focus varianten
            {'style': 'deep_focus', 'nature': 'gentle_rain', 'complexity': 'medium', 'duration_range': (32, 42)},
            {'style': 'deep_focus', 'nature': 'soft_ocean', 'complexity': 'simple', 'duration_range': (35, 45)},
            {'style': 'deep_focus', 'nature': None, 'complexity': 'complex', 'duration_range': (28, 38)},

            # Creative Flow varianten
            {'style': 'creative', 'nature': 'forest_breeze', 'complexity': 'medium', 'duration_range': (30, 40)},
            {'style': 'creative', 'nature': 'babbling_brook', 'complexity': 'complex', 'duration_range': (33, 43)},
            {'style': 'creative', 'nature': None, 'complexity': 'medium', 'duration_range': (29, 39)},

            # Meditation varianten
            {'style': 'meditation', 'nature': 'mountain_wind', 'complexity': 'simple', 'duration_range': (40, 55)},
            {'style': 'meditation', 'nature': 'soft_ocean', 'complexity': 'simple', 'duration_range': (35, 50)},

            # Study varianten
            {'style': 'study', 'nature': 'gentle_rain', 'complexity': 'medium', 'duration_range': (25, 35)},
            {'style': 'study', 'nature': 'distant_thunder', 'complexity': 'simple', 'duration_range': (30, 40)},

            # Ambient varianten
            {'style': 'ambient', 'nature': 'mountain_wind', 'complexity': 'complex', 'duration_range': (45, 60)},
            {'style': 'ambient', 'nature': None, 'complexity': 'medium', 'duration_range': (40, 55)},

            # Healing & Relaxed varianten (nieuw!)
            {'style': 'healing', 'nature': 'babbling_brook', 'complexity': 'simple', 'duration_range': (38, 50)},
            {'style': 'relaxed', 'nature': 'forest_breeze', 'complexity': 'medium', 'duration_range': (32, 42)},
            {'style': 'healing', 'nature': None, 'complexity': 'simple', 'duration_range': (35, 45)},
            {'style': 'relaxed', 'nature': 'gentle_rain', 'complexity': 'simple', 'duration_range': (30, 40)}
        ]

        # Mooie track namen
        self.style_names = {
            'deep_focus': ['Deep Focus', 'Concentrated Mind', 'Focus Flow', 'Mental Clarity'],
            'creative': ['Creative Flow', 'Inspiration Wave', 'Artistic Mind', 'Innovation Space'],
            'meditation': ['Peaceful Mind', 'Inner Calm', 'Meditation Space', 'Tranquil Thoughts'],
            'study': ['Study Session', 'Learning Zone', 'Academic Focus', 'Knowledge Flow'],
            'ambient': ['Ambient Space', 'Floating Thoughts', 'Open Mind', 'Ethereal Journey'],
            'healing': ['Healing Waves', 'Restorative Mind', 'Gentle Recovery', 'Peaceful Healing'],
            'relaxed': ['Relaxed State', 'Easy Mind', 'Gentle Flow', 'Soft Focus']
        }

        self.nature_names = {
            'gentle_rain': ['Gentle Rainfall', 'Soft Rain', 'Light Drizzle', 'Rain Whispers'],
            'soft_ocean': ['Ocean Waves', 'Gentle Surf', 'Calm Seas', 'Coastal Breeze'],
            'forest_breeze': ['Forest Sounds', 'Woodland Breeze', 'Nature\'s Breath', 'Tree Whispers'],
            'distant_thunder': ['Distant Thunder', 'Storm Echoes', 'Thunder Dreams', 'Storm Whispers'],
            'mountain_wind': ['Mountain Wind', 'Alpine Breeze', 'Peak Winds', 'Mountain Air'],
            'babbling_brook': ['Babbling Brook', 'Gentle Stream', 'Water Flow', 'Creek Sounds']
        }

    def generate_unique_track_name(self, style: str, nature_type: str = None) -> str:
        """Genereer unieke, mooie track naam"""
        self.track_counter += 1

        # Kies random naam uit opties
        style_options = self.style_names.get(style, [style.title()])
        base_name = random.choice(style_options)

        # Voeg natuur toe indien aanwezig
        if nature_type and nature_type in self.nature_names:
            nature_options = self.nature_names[nature_type]
            nature_name = random.choice(nature_options)
            base_name += f" with {nature_name}"

        # Voeg gamma Hz informatie toe
        base_name += " - 40Hz Gamma"

        # Voeg uniek nummer toe
        return f"{base_name} #{self.track_counter:03d}"

    def make_safe_filename(self, filename: str) -> str:
        """Maak bestandsnaam veilig voor verschillende OS"""
        # Vervang ongeldige karakters
        unsafe_chars = '<>:"/\\|?*'
        safe_filename = filename
        for char in unsafe_chars:
            safe_filename = safe_filename.replace(char, '_')

        # Beperk lengte
        if len(safe_filename) > 100:
            safe_filename = safe_filename[:100]

        return safe_filename

    def print_enhanced_summary(self, playlist_metadata: Dict, output_dir: str):
        """Print uitgebreide samenvatting"""
        print("\n" + "=" * 70)
        print("🎉 PLAYLIST VOLTOOID!")
        print("=" * 70)
        print(f"📁 Locatie: {output_dir}/")
        print(f"🎵 Tracks: {playlist_metadata['total_tracks']}")

        # Statistieken
        styles = {}
        total_duration = 0

        for track in playlist_metadata['tracks']:
            style = track.get('style', 'unknown')
            styles[style] = styles.get(style, 0) + 1
            total_duration += track.get('duration', 0)

        print(f"⏱️  Totale duur: {total_duration // 60}:{total_duration % 60:02d}")
        print(f"🎼 Stijlen: {', '.join([f'{k}({v})' for k, v in styles.items()])}")

        print("\n📋 Bestanden:")
        print(f"   • Audio tracks: {playlist_metadata['total_tracks']} x .wav")
        print(f"   • Metadata: {playlist_metadata['total_tracks']} x .json")
        print(f"   • Playlist info: playlist_info.json")
        print(f"   • Speler script: play_focus_music.py")

        print("\n🚀 Om af te spelen:")
        print(f"   cd {output_dir}")
        print("   python play_focus_music.py")

        print("\n💡 Tips:")
        print("   • Gebruik koptelefoon voor beste binaural effect")
        print("   • 40Hz gamma golven helpen bij focus en concentratie")
        print("   • Zet volume op comfortabel niveau")
        print("=" * 70)

    def create_enhanced_playlist(self, num_tracks: int = 20,
                                 output_dir: str = "enhanced_focus_playlist") -> List[str]:
        """Genereer geavanceerde playlist met meer variatie"""

        print("🎼 Enhanced AI Focus Music Generator gestart!")
        print("=" * 70)
        print(f"Creëren van {num_tracks} rustgevende focus tracks")
        print("Nieuwe features: Zachte overgangen, harmonische lagen, evolving ambient")
        print("=" * 70)

        generated_files = []
        playlist_metadata = {
            'playlist_name': 'Enhanced 40Hz Gamma Focus Collection',
            'session_id': self.session_id,
            'generated_at': datetime.now().isoformat(),
            'total_tracks': num_tracks,
            'generator_version': '2.0_enhanced',
            'tracks': []
        }

        for i in range(num_tracks):
            print(f"\n🎵 Track {i + 1}/{num_tracks}")

            # Kies random template (met meer spreiding)
            template = random.choice(self.track_templates)

            # Random duur binnen bereik
            duration = random.randint(*template['duration_range'])

            # Genereer enhanced track
            audio, metadata = self.generator.generate_focus_masterpiece(
                style=template['style'],
                duration=duration,
                include_nature=template['nature'] is not None,
                nature_type=template['nature'],
                complexity=template['complexity']
            )

            # Genereer unieke naam
            track_name = self.generate_unique_track_name(template['style'], template['nature'])
            safe_filename = self.make_safe_filename(track_name)

            # Sla op
            file_path = self.generator.save_track(audio, metadata, safe_filename, output_dir)
            generated_files.append(file_path)

            # Voeg toe aan playlist metadata
            playlist_metadata['tracks'].append({
                'filename': safe_filename + '.wav',
                'title': track_name,
                **metadata
            })

            # Toon voortgang
            print(f"   ✅ '{track_name[:50]}...' - {duration}s")

        # Sla playlist metadata op
        playlist_file = os.path.join(output_dir, "playlist_info.json")
        with open(playlist_file, 'w') as f:
            json.dump(playlist_metadata, f, indent=2)

        # Creëer eenvoudig afspeelscript
        self.create_simple_player(output_dir)

        self.print_enhanced_summary(playlist_metadata, output_dir)

        return generated_files

    def create_simple_player(self, output_dir: str):
        """Creëer eenvoudig Python afspeelscript"""
        player_script = '''#!/usr/bin/env python3
"""
Eenvoudige Focus Music Player
Gebruik: python play_focus_music.py
"""

import os
import random
import pygame
import json
import time

def init_player():
    """Initialiseer pygame mixer"""
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)

def load_playlist():
    """Laad playlist informatie"""
    try:
        with open('playlist_info.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Geen playlist gevonden!")
        return None

def play_track(filename):
    """Speel een track af"""
    try:
        print(f"🎵 Afspelen: {filename}")
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        # Wacht tot track klaar is
        while pygame.mixer.music.get_busy():
            time.sleep(1)

    except Exception as e:
        print(f"❌ Fout bij afspelen {filename}: {e}")

def main():
    print("🎼 Focus Music Player")
    print("=" * 50)

    init_player()
    playlist = load_playlist()

    if not playlist:
        return

    print(f"📁 Playlist: {playlist['playlist_name']}")
    print(f"🎵 {len(playlist['tracks'])} tracks gevonden")
    print()

    while True:
        print("Opties:")
        print("1. Speel alle tracks af")
        print("2. Speel random track")
        print("3. Toon playlist")
        print("4. Stoppen")

        choice = input("\\nKeuze (1-4): ").strip()

        if choice == "1":
            print("\\n🎶 Afspelen van complete playlist...")
            for track in playlist['tracks']:
                filename = track['filename']
                if os.path.exists(filename):
                    play_track(filename)
                else:
                    print(f"⚠️  Bestand niet gevonden: {filename}")

        elif choice == "2":
            track = random.choice(playlist['tracks'])
            filename = track['filename']
            if os.path.exists(filename):
                print(f"\\n🎲 Random track: {track['title']}")
                play_track(filename)
            else:
                print(f"⚠️  Bestand niet gevonden: {filename}")

        elif choice == "3":
            print("\\n📋 Playlist:")
            for i, track in enumerate(playlist['tracks'], 1):
                duration = track.get('duration', 'N/A')
                style = track.get('style', 'N/A')
                print(f"{i:2d}. {track['title'][:60]} ({duration}s, {style})")

        elif choice == "4":
            print("👋 Veel plezier met je focus sessie!")
            break

        else:
            print("❌ Ongeldige keuze!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n👋 Player gestopt.")
    except Exception as e:
        print(f"❌ Fout: {e}")
'''


# Voeg dit toe aan het einde van je script om het daadwerkelijk uit te voeren:

if __name__ == "__main__":
    print("🎼 Enhanced AI Focus Music Generator")
    print("=" * 60)

    # Vraag gebruiker om input
    try:
        num_tracks = input("Hoeveel tracks wil je genereren? (standaard 5): ").strip()
        if not num_tracks:
            num_tracks = 5
        else:
            num_tracks = int(num_tracks)

        output_folder = input("Output folder naam? (standaard 'focus_music'): ").strip()
        if not output_folder:
            output_folder = "focus_music"

    except ValueError:
        print("❌ Ongeldige invoer, gebruik standaard waarden")
        num_tracks = 5
        output_folder = "focus_music"
    except KeyboardInterrupt:
        print("\n👋 Generator gestopt door gebruiker")
        exit()

    # Maak output directory
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Initialiseer de bot
        music_bot = AdvancedFocusMusicBot()

        # Genereer de playlist
        print(f"\n🚀 Starten met genereren van {num_tracks} tracks...")
        generated_files = music_bot.create_enhanced_playlist(
            num_tracks=num_tracks,
            output_dir=output_folder
        )

        print(f"\n✅ Klaar! {len(generated_files)} tracks gegenereerd in '{output_folder}/'")
        print("🎵 Je kunt nu de tracks afspelen met je favoriete audiospeler")

    except Exception as e:
        print(f"\n❌ Fout tijdens genereren: {e}")
        import traceback

        traceback.print_exc()

    except KeyboardInterrupt:
        print("\n⏹️  Generatie gestopt door gebruiker")


# ===============================
# HULPFUNCTIES DIE ONTBREKEN
# ===============================

# Voeg deze methoden toe aan de EnhancedBinauralBeatsGenerator class:



# Voeg deze methoden toe aan de AdvancedFocusMusicBot class:

def make_safe_filename(self, filename: str) -> str:
    """Maak bestandsnaam veilig voor verschillende OS"""
    # Vervang ongeldige karakters
    unsafe_chars = '<>:"/\\|?*'
    safe_filename = filename
    for char in unsafe_chars:
        safe_filename = safe_filename.replace(char, '_')

    # Beperk lengte
    if len(safe_filename) > 100:
        safe_filename = safe_filename[:100]

    return safe_filename


def print_enhanced_summary(self, playlist_metadata: Dict, output_dir: str):
    """Print uitgebreide samenvatting"""
    print("\n" + "=" * 70)
    print("🎉 PLAYLIST VOLTOOID!")
    print("=" * 70)
    print(f"📁 Locatie: {output_dir}/")
    print(f"🎵 Tracks: {playlist_metadata['total_tracks']}")

    # Statistieken
    styles = {}
    total_duration = 0

    for track in playlist_metadata['tracks']:
        style = track.get('style', 'unknown')
        styles[style] = styles.get(style, 0) + 1
        total_duration += track.get('duration', 0)

    print(f"⏱️  Totale duur: {total_duration // 60}:{total_duration % 60:02d}")
    print(f"🎼 Stijlen: {', '.join([f'{k}({v})' for k, v in styles.items()])}")

    print("\n📋 Bestanden:")
    print(f"   • Audio tracks: {playlist_metadata['total_tracks']} x .wav")
    print(f"   • Metadata: {playlist_metadata['total_tracks']} x .json")
    print(f"   • Playlist info: playlist_info.json")
    print(f"   • Speler script: play_focus_music.py")

    print("\n🚀 Om af te spelen:")
    print(f"   cd {output_dir}")
    print("   python play_focus_music.py")

    print("\n💡 Tips:")
    print("   • Gebruik koptelefoon voor beste binaural effect")
    print("   • 40Hz gamma golven helpen bij focus en concentratie")
    print("   • Zet volume op comfortabel niveau")
    print("=" * 70)