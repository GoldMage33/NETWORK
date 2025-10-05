#!/usr/bin/env python3
"""
Continuous 1kHz-10kHz Live Scanner
Real-time frequency monitoring with pattern detection and live statistics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
import time
import threading
import queue
from collections import deque
import warnings
from scipy.signal import find_peaks, welch, spectrogram
from scipy.fft import fft, fftfreq, fftshift
from scipy.stats import zscore
import json
import os
from collections import Counter
warnings.filterwarnings('ignore')

class ContinuousFrequencyScanner:
    """Real-time frequency scanner for 1kHz-10kHz range."""
    
    def __init__(self):
        self.running = False
        self.data_queue = queue.Queue()
        self.live_data = deque(maxlen=1000)  # Keep last 1000 samples
        self.scan_interval = 0.1  # 100ms between scans
        self.frequency_range = (1000, 10000)  # 1kHz to 10kHz
        self.pattern_history = deque(maxlen=100)  # Pattern detection history
        self.anomaly_threshold = 2.0  # Z-score threshold for anomalies
        self.scan_count = 0
        self.start_time = None
        
        # FFT Analysis parameters
        self.fft_history = deque(maxlen=50)  # Store FFT results for pattern analysis
        self.dominant_frequencies = deque(maxlen=20)  # Track dominant frequency changes
        self.spectral_patterns = []  # Store detected spectral patterns
        self.fft_window_size = 100  # Number of frequency points for FFT
        
        # Live statistics tracking
        self.stats = {
            'total_scans': 0,
            'anomalies_detected': 0,
            'patterns_found': 0,
            'peak_frequency': 0,
            'peak_amplitude': 0,
            'avg_signal_strength': 0,
            'last_update': None,
            'fft_patterns_found': 0,
            'spectral_anomalies': 0,
            'dominant_freq_changes': 0,
            'spectral_stability': 0
        }
        
        # Pattern detection parameters
        self.pattern_memory = []
        self.correlation_threshold = 0.7
        
        print("🔄 Continuous 1kHz-10kHz Scanner Initialized")
        print(f"   • Scan interval: {self.scan_interval*1000:.0f}ms")
        print(f"   • Frequency range: {self.frequency_range[0]}-{self.frequency_range[1]} Hz")
        print(f"   • Buffer size: {self.live_data.maxlen} samples")
    
    def generate_live_frequency_data(self):
        """Generate realistic live frequency data for 1kHz-10kHz range."""
        
        # Create frequency array
        frequencies = np.linspace(self.frequency_range[0], self.frequency_range[1], 100)
        
        # Base signal with time-varying components
        current_time = time.time()
        time_factor = np.sin(current_time * 0.1) * 0.3  # Slow oscillation
        
        audio_amplitudes = []
        radio_amplitudes = []
        
        for freq in frequencies:
            # Audio signal with realistic characteristics
            # Peak around 3-4kHz with time variation
            base_freq = 3500 + time_factor * 500
            audio_base = 0.1 + 0.4 * np.exp(-((freq - base_freq) / 1200) ** 2)
            
            # Add harmonics
            harmonics = 0
            for harmonic in [2000, 4000, 6000, 8000]:
                if abs(freq - harmonic) < 100:
                    harmonics += 0.1 * np.exp(-((freq - harmonic) / 50) ** 2)
            
            # Random noise and time-based variation
            noise = np.random.normal(0, 0.02)
            time_variation = 0.05 * np.sin(current_time * freq * 0.0001)
            
            audio_amp = max(0.01, audio_base + harmonics + noise + time_variation)
            audio_amplitudes.append(audio_amp)
            
            # Radio signal with interference patterns
            radio_base = 0.03 + 0.08 * np.sin(freq * 0.001 + current_time)
            
            # Periodic interference
            interference = 0
            if np.sin(current_time * 2) > 0.5:  # Intermittent interference
                interference_freqs = [1500, 2500, 5000, 7500, 9000]
                for int_freq in interference_freqs:
                    if abs(freq - int_freq) < 150:
                        interference += 0.15 * np.exp(-((freq - int_freq) / 75) ** 2)
            
            radio_noise = np.random.normal(0, 0.01)
            radio_amp = max(0.01, radio_base + interference + radio_noise)
            radio_amplitudes.append(radio_amp)
        
        # Create data structure
        scan_data = {
            'timestamp': datetime.now(),
            'frequencies': frequencies,
            'audio_amplitudes': np.array(audio_amplitudes),
            'radio_amplitudes': np.array(radio_amplitudes),
            'combined_amplitudes': np.array(audio_amplitudes) + np.array(radio_amplitudes),
            'scan_id': self.scan_count
        }
        
        return scan_data
    
    def perform_fft_analysis(self, current_data):
        """Perform comprehensive FFT analysis on current scan data."""
        
        signal = current_data['combined_amplitudes']
        frequencies = current_data['frequencies']
        
        # Perform FFT
        fft_values = fft(signal)
        fft_freqs = fftfreq(len(signal), d=(frequencies[1] - frequencies[0]))
        fft_magnitude = np.abs(fft_values)
        fft_phase = np.angle(fft_values)
        
        # Focus on positive frequencies
        positive_freq_mask = fft_freqs > 0
        fft_freqs_pos = fft_freqs[positive_freq_mask]
        fft_magnitude_pos = fft_magnitude[positive_freq_mask]
        fft_phase_pos = fft_phase[positive_freq_mask]
        
        # Find dominant frequency components
        dominant_indices = np.argsort(fft_magnitude_pos)[-5:]  # Top 5 components
        dominant_freqs = []
        dominant_magnitudes = []
        
        for idx in reversed(dominant_indices):
            if idx < len(fft_freqs_pos):
                freq_hz = abs(fft_freqs_pos[idx])
                magnitude = fft_magnitude_pos[idx]
                if freq_hz > 0.1:  # Avoid DC component
                    # Convert back to actual frequency range
                    actual_freq = freq_hz * (frequencies[-1] - frequencies[0]) + frequencies[0]
                    dominant_freqs.append(actual_freq)
                    dominant_magnitudes.append(magnitude)
        
        # Calculate spectral centroid (frequency center of mass)
        spectral_centroid = np.sum(fft_freqs_pos * fft_magnitude_pos) / np.sum(fft_magnitude_pos) if np.sum(fft_magnitude_pos) > 0 else 0
        
        # Calculate spectral bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((fft_freqs_pos - spectral_centroid) ** 2) * fft_magnitude_pos) / np.sum(fft_magnitude_pos)) if np.sum(fft_magnitude_pos) > 0 else 0
        
        # Calculate spectral rolloff (95% of energy)
        cumulative_magnitude = np.cumsum(fft_magnitude_pos)
        total_magnitude = cumulative_magnitude[-1]
        rolloff_idx = np.where(cumulative_magnitude >= 0.95 * total_magnitude)[0]
        spectral_rolloff = fft_freqs_pos[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
        
        fft_analysis = {
            'timestamp': current_data['timestamp'],
            'fft_magnitude': fft_magnitude_pos,
            'fft_frequencies': fft_freqs_pos,
            'fft_phase': fft_phase_pos,
            'dominant_frequencies': dominant_freqs[:3],  # Top 3
            'dominant_magnitudes': dominant_magnitudes[:3],
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff,
            'total_spectral_energy': np.sum(fft_magnitude_pos ** 2)
        }
        
        # Store in FFT history
        self.fft_history.append(fft_analysis)
        
        return fft_analysis
    
    def detect_fft_patterns(self, current_fft):
        """Detect patterns in FFT data over time."""
        
        fft_patterns = []
        
        if len(self.fft_history) < 5:
            return fft_patterns
        
        # Pattern 1: Stable dominant frequency
        recent_dominant = [fft_data['dominant_frequencies'][0] if fft_data['dominant_frequencies'] else 0 
                          for fft_data in list(self.fft_history)[-5:]]
        
        if len(set([round(f, 0) for f in recent_dominant if f > 0])) <= 2:  # Max 2 different frequencies
            avg_dominant = np.mean([f for f in recent_dominant if f > 0])
            if avg_dominant > 0:
                fft_patterns.append({
                    'type': 'Stable Dominant Frequency',
                    'description': f'Consistent peak at {avg_dominant:.1f} Hz',
                    'confidence': 0.9,
                    'frequency': avg_dominant
                })
        
        # Pattern 2: Spectral centroid drift
        recent_centroids = [fft_data['spectral_centroid'] for fft_data in list(self.fft_history)[-8:]]
        if len(recent_centroids) >= 5:
            centroid_trend = np.polyfit(range(len(recent_centroids)), recent_centroids, 1)[0]
            if abs(centroid_trend) > 0.5:  # Significant drift
                fft_patterns.append({
                    'type': 'Spectral Centroid Drift',
                    'description': f'Frequency center drifting at {centroid_trend:.2f}/scan',
                    'confidence': 0.8,
                    'trend': 'upward' if centroid_trend > 0 else 'downward'
                })
        
        # Pattern 3: Harmonic series detection
        if current_fft['dominant_frequencies']:
            dominant_freq = current_fft['dominant_frequencies'][0]
            harmonics_found = []
            
            for i, freq in enumerate(current_fft['dominant_frequencies'][:3]):
                ratio = freq / dominant_freq if dominant_freq > 0 else 0
                if 1.8 < ratio < 2.2:  # 2nd harmonic
                    harmonics_found.append(2)
                elif 2.8 < ratio < 3.2:  # 3rd harmonic
                    harmonics_found.append(3)
                elif 3.8 < ratio < 4.2:  # 4th harmonic
                    harmonics_found.append(4)
            
            if len(harmonics_found) >= 2:
                fft_patterns.append({
                    'type': 'Harmonic Series',
                    'description': f'Harmonics detected: {harmonics_found} of {dominant_freq:.1f} Hz',
                    'confidence': 0.85,
                    'fundamental': dominant_freq,
                    'harmonics': harmonics_found
                })
        
        # Pattern 4: Spectral energy fluctuation
        recent_energies = [fft_data['total_spectral_energy'] for fft_data in list(self.fft_history)[-10:]]
        if len(recent_energies) >= 5:
            energy_cv = np.std(recent_energies) / np.mean(recent_energies) if np.mean(recent_energies) > 0 else 0
            if energy_cv > 0.3:  # High variability
                fft_patterns.append({
                    'type': 'Energy Fluctuation',
                    'description': f'High spectral energy variability (CV: {energy_cv:.2f})',
                    'confidence': 0.7,
                    'variability': energy_cv
                })
        
        # Pattern 5: Bandwidth changes
        recent_bandwidths = [fft_data['spectral_bandwidth'] for fft_data in list(self.fft_history)[-6:]]
        if len(recent_bandwidths) >= 4:
            bandwidth_trend = np.polyfit(range(len(recent_bandwidths)), recent_bandwidths, 1)[0]
            if abs(bandwidth_trend) > 0.1:
                fft_patterns.append({
                    'type': 'Bandwidth Change',
                    'description': f'Spectral bandwidth {"expanding" if bandwidth_trend > 0 else "narrowing"}',
                    'confidence': 0.75,
                    'rate': bandwidth_trend
                })
        
        return fft_patterns
    
    def detect_spectral_anomalies(self, current_fft):
        """Detect anomalies in spectral domain."""
        
        spectral_anomalies = []
        
        if len(self.fft_history) < 10:
            return spectral_anomalies
        
        # Compare current spectral features with historical data
        historical_centroids = [fft_data['spectral_centroid'] for fft_data in list(self.fft_history)[-10:-1]]
        historical_bandwidths = [fft_data['spectral_bandwidth'] for fft_data in list(self.fft_history)[-10:-1]]
        historical_energies = [fft_data['total_spectral_energy'] for fft_data in list(self.fft_history)[-10:-1]]
        
        # Anomaly 1: Spectral centroid outlier
        if historical_centroids:
            centroid_mean = np.mean(historical_centroids)
            centroid_std = np.std(historical_centroids)
            if centroid_std > 0:
                centroid_zscore = abs(current_fft['spectral_centroid'] - centroid_mean) / centroid_std
                if centroid_zscore > 2.5:
                    spectral_anomalies.append({
                        'type': 'Spectral Centroid Anomaly',
                        'description': f'Unusual frequency center: {current_fft["spectral_centroid"]:.1f} Hz',
                        'z_score': centroid_zscore,
                        'severity': 'High' if centroid_zscore > 3 else 'Medium'
                    })
        
        # Anomaly 2: Bandwidth outlier
        if historical_bandwidths:
            bandwidth_mean = np.mean(historical_bandwidths)
            bandwidth_std = np.std(historical_bandwidths)
            if bandwidth_std > 0:
                bandwidth_zscore = abs(current_fft['spectral_bandwidth'] - bandwidth_mean) / bandwidth_std
                if bandwidth_zscore > 2.5:
                    spectral_anomalies.append({
                        'type': 'Spectral Bandwidth Anomaly',
                        'description': f'Unusual frequency spread: {current_fft["spectral_bandwidth"]:.2f}',
                        'z_score': bandwidth_zscore,
                        'severity': 'High' if bandwidth_zscore > 3 else 'Medium'
                    })
        
        # Anomaly 3: Energy spike
        if historical_energies:
            energy_mean = np.mean(historical_energies)
            energy_std = np.std(historical_energies)
            if energy_std > 0:
                energy_zscore = abs(current_fft['total_spectral_energy'] - energy_mean) / energy_std
                if energy_zscore > 2.5:
                    spectral_anomalies.append({
                        'type': 'Spectral Energy Anomaly',
                        'description': f'Unusual total energy: {current_fft["total_spectral_energy"]:.2f}',
                        'z_score': energy_zscore,
                        'severity': 'High' if energy_zscore > 3 else 'Medium'
                    })
        
        # Anomaly 4: New dominant frequency
        if current_fft['dominant_frequencies']:
            current_dominant = current_fft['dominant_frequencies'][0]
            recent_dominants = []
            for fft_data in list(self.fft_history)[-5:-1]:
                if fft_data['dominant_frequencies']:
                    recent_dominants.append(fft_data['dominant_frequencies'][0])
            
            if recent_dominants:
                # Check if current dominant is significantly different
                freq_differences = [abs(current_dominant - freq) for freq in recent_dominants]
                avg_difference = np.mean(freq_differences)
                if avg_difference > 200:  # 200 Hz difference threshold
                    spectral_anomalies.append({
                        'type': 'Dominant Frequency Shift',
                        'description': f'New dominant frequency: {current_dominant:.1f} Hz',
                        'frequency_shift': avg_difference,
                        'severity': 'Medium'
                    })
        
        return spectral_anomalies
    
    def detect_patterns(self, current_data):
        """Detect patterns in the frequency data."""
        
        patterns_found = []
        
        # Add current scan to pattern memory
        self.pattern_memory.append(current_data['combined_amplitudes'])
        if len(self.pattern_memory) > 50:  # Keep last 50 scans
            self.pattern_memory.pop(0)
        
        if len(self.pattern_memory) >= 5:
            # Pattern 1: Repeating frequency peaks
            current_peaks, _ = find_peaks(current_data['combined_amplitudes'], 
                                        height=np.mean(current_data['combined_amplitudes']) + 
                                               np.std(current_data['combined_amplitudes']))
            
            if len(current_peaks) > 0:
                # Check for consistent peaks across recent scans
                recent_peak_counts = []
                for recent_scan in self.pattern_memory[-5:]:
                    recent_peaks, _ = find_peaks(recent_scan, 
                                               height=np.mean(recent_scan) + np.std(recent_scan))
                    recent_peak_counts.append(len(recent_peaks))
                
                if np.std(recent_peak_counts) < 1.0:  # Consistent peak count
                    patterns_found.append({
                        'type': 'Consistent Peak Pattern',
                        'description': f'Stable {int(np.mean(recent_peak_counts))} peaks detected',
                        'confidence': 0.8,
                        'frequency_range': f"{current_data['frequencies'][0]:.0f}-{current_data['frequencies'][-1]:.0f} Hz"
                    })
            
            # Pattern 2: Amplitude oscillation
            if len(self.pattern_memory) >= 10:
                amplitude_trends = [np.mean(scan) for scan in self.pattern_memory[-10:]]
                # Check for oscillating pattern
                if len(amplitude_trends) >= 6:
                    correlations = []
                    for i in range(len(amplitude_trends) - 5):
                        segment1 = amplitude_trends[i:i+3]
                        segment2 = amplitude_trends[i+3:i+6]
                        if len(segment1) == 3 and len(segment2) == 3:
                            corr = np.corrcoef(segment1, segment2)[0,1]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                    
                    if correlations and np.mean(correlations) > self.correlation_threshold:
                        patterns_found.append({
                            'type': 'Amplitude Oscillation',
                            'description': f'Periodic amplitude variation detected',
                            'confidence': np.mean(correlations),
                            'period': f'~{len(amplitude_trends)} scans'
                        })
            
            # Pattern 3: Frequency drift
            if len(self.pattern_memory) >= 8:
                peak_frequencies = []
                for scan in self.pattern_memory[-8:]:
                    peaks, _ = find_peaks(scan, height=np.mean(scan) + 0.5*np.std(scan))
                    if len(peaks) > 0:
                        # Find dominant peak
                        max_peak_idx = peaks[np.argmax(scan[peaks])]
                        peak_freq = current_data['frequencies'][max_peak_idx]
                        peak_frequencies.append(peak_freq)
                
                if len(peak_frequencies) >= 5:
                    freq_trend = np.polyfit(range(len(peak_frequencies)), peak_frequencies, 1)[0]
                    if abs(freq_trend) > 10:  # Significant drift (>10 Hz per scan)
                        patterns_found.append({
                            'type': 'Frequency Drift',
                            'description': f'Peak frequency drifting at {freq_trend:.1f} Hz/scan',
                            'confidence': 0.7,
                            'direction': 'upward' if freq_trend > 0 else 'downward'
                        })
        
        return patterns_found
    
    def detect_anomalies(self, current_data):
        """Detect anomalies in current scan data."""
        
        anomalies = []
        
        # Z-score based anomaly detection
        combined_amps = current_data['combined_amplitudes']
        z_scores = np.abs(zscore(combined_amps))
        anomalous_indices = np.where(z_scores > self.anomaly_threshold)[0]
        
        for idx in anomalous_indices:
            freq = current_data['frequencies'][idx]
            amp = combined_amps[idx]
            z_score = z_scores[idx]
            
            anomalies.append({
                'type': 'Statistical Anomaly',
                'frequency': freq,
                'amplitude': amp,
                'z_score': z_score,
                'severity': 'High' if z_score > 3.0 else 'Medium'
            })
        
        # Sudden amplitude spike detection
        if len(self.live_data) > 5:
            recent_avg = np.mean([scan['combined_amplitudes'] for scan in list(self.live_data)[-5:]], axis=0)
            current_diff = current_data['combined_amplitudes'] - recent_avg
            spike_indices = np.where(current_diff > 3 * np.std(current_diff))[0]
            
            for idx in spike_indices:
                freq = current_data['frequencies'][idx]
                amp = current_data['combined_amplitudes'][idx]
                
                anomalies.append({
                    'type': 'Amplitude Spike',
                    'frequency': freq,
                    'amplitude': amp,
                    'spike_magnitude': current_diff[idx],
                    'severity': 'High'
                })
        
        return anomalies
    
    def update_statistics(self, current_data, patterns, anomalies, fft_patterns=None, spectral_anomalies=None):
        """Update live statistics."""
        
        self.stats['total_scans'] += 1
        self.stats['patterns_found'] += len(patterns)
        self.stats['anomalies_detected'] += len(anomalies)
        self.stats['last_update'] = datetime.now()
        
        # FFT-specific statistics
        if fft_patterns:
            self.stats['fft_patterns_found'] += len(fft_patterns)
        if spectral_anomalies:
            self.stats['spectral_anomalies'] += len(spectral_anomalies)
        
        # Find peak frequency and amplitude
        max_idx = np.argmax(current_data['combined_amplitudes'])
        self.stats['peak_frequency'] = current_data['frequencies'][max_idx]
        self.stats['peak_amplitude'] = current_data['combined_amplitudes'][max_idx]
        
        # Calculate average signal strength
        self.stats['avg_signal_strength'] = np.mean(current_data['combined_amplitudes'])
        
        # Running averages over recent data
        if len(self.live_data) > 0:
            recent_peak_freqs = []
            recent_avg_strengths = []
            
            for scan in list(self.live_data)[-10:]:  # Last 10 scans
                max_idx = np.argmax(scan['combined_amplitudes'])
                recent_peak_freqs.append(scan['frequencies'][max_idx])
                recent_avg_strengths.append(np.mean(scan['combined_amplitudes']))
            
            self.stats['avg_peak_frequency'] = np.mean(recent_peak_freqs)
            self.stats['running_avg_strength'] = np.mean(recent_avg_strengths)
    
    def print_live_statistics(self, patterns, anomalies, fft_patterns=None, spectral_anomalies=None, current_fft=None):
        """Print live statistics to console."""
        
        # Clear screen (works on most terminals)
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print("🔄 CONTINUOUS 1kHz-10kHz FFT SCANNER - LIVE MONITORING")
        print("=" * 80)
        
        # Runtime information
        if self.start_time:
            runtime = datetime.now() - self.start_time
            print(f"⏱️  Runtime: {str(runtime).split('.')[0]}")
        
        print(f"📊 Scan #{self.stats['total_scans']:,} | Last update: {self.stats['last_update'].strftime('%H:%M:%S.%f')[:-3]}")
        print()
        
        # Current scan statistics
        print("📈 CURRENT SCAN STATISTICS:")
        print(f"   • Peak Frequency: {self.stats['peak_frequency']:.1f} Hz")
        print(f"   • Peak Amplitude: {self.stats['peak_amplitude']:.4f}")
        print(f"   • Average Signal: {self.stats['avg_signal_strength']:.4f}")
        
        if 'running_avg_strength' in self.stats:
            print(f"   • Running Avg (10 scans): {self.stats['running_avg_strength']:.4f}")
            print(f"   • Peak Frequency Trend: {self.stats.get('avg_peak_frequency', 0):.1f} Hz")
        print()
        
        # FFT Analysis Results
        if current_fft:
            print("🌊 CURRENT FFT ANALYSIS:")
            if current_fft['dominant_frequencies']:
                print(f"   • Dominant Frequency: {current_fft['dominant_frequencies'][0]:.1f} Hz")
                print(f"   • Spectral Centroid: {current_fft['spectral_centroid']:.1f} Hz")
                print(f"   • Spectral Bandwidth: {current_fft['spectral_bandwidth']:.2f}")
                print(f"   • Total Spectral Energy: {current_fft['total_spectral_energy']:.2f}")
                
                if len(current_fft['dominant_frequencies']) > 1:
                    print(f"   • Secondary Peaks: {', '.join([f'{f:.1f}' for f in current_fft['dominant_frequencies'][1:3]])} Hz")
            print()
        
        # Cumulative statistics
        print("📊 CUMULATIVE STATISTICS:")
        print(f"   • Total Scans: {self.stats['total_scans']:,}")
        print(f"   • Time-Domain Patterns: {self.stats['patterns_found']:,}")
        print(f"   • FFT Patterns: {self.stats['fft_patterns_found']:,}")
        print(f"   • Time-Domain Anomalies: {self.stats['anomalies_detected']:,}")
        print(f"   • Spectral Anomalies: {self.stats['spectral_anomalies']:,}")
        
        if self.stats['total_scans'] > 0:
            total_patterns = self.stats['patterns_found'] + self.stats['fft_patterns_found']
            total_anomalies = self.stats['anomalies_detected'] + self.stats['spectral_anomalies']
            pattern_rate = (total_patterns / self.stats['total_scans']) * 100
            anomaly_rate = (total_anomalies / self.stats['total_scans']) * 100
            print(f"   • Total Pattern Detection Rate: {pattern_rate:.1f}%")
            print(f"   • Total Anomaly Detection Rate: {anomaly_rate:.1f}%")
        print()
        
        # Current patterns (Time Domain)
        if patterns:
            print("🔍 TIME-DOMAIN PATTERNS THIS SCAN:")
            for i, pattern in enumerate(patterns[:2]):  # Show top 2
                print(f"   {i+1}. {pattern['type']}: {pattern['description']}")
                print(f"      Confidence: {pattern.get('confidence', 0):.2f}")
        else:
            print("🔍 TIME-DOMAIN PATTERNS: None detected")
        
        # FFT patterns
        if fft_patterns:
            print("\n🌊 FFT PATTERNS THIS SCAN:")
            for i, pattern in enumerate(fft_patterns[:2]):  # Show top 2
                print(f"   {i+1}. {pattern['type']}: {pattern['description']}")
                print(f"      Confidence: {pattern.get('confidence', 0):.2f}")
        else:
            print("\n🌊 FFT PATTERNS: None detected")
        print()
        
        # Current anomalies (Time Domain)
        if anomalies:
            print("🚨 TIME-DOMAIN ANOMALIES THIS SCAN:")
            for i, anomaly in enumerate(anomalies[:2]):  # Show top 2
                print(f"   {i+1}. {anomaly['type']} at {anomaly.get('frequency', 'N/A'):.1f} Hz")
                print(f"      Severity: {anomaly['severity']}")
        else:
            print("🚨 TIME-DOMAIN ANOMALIES: None detected")
        
        # Spectral anomalies
        if spectral_anomalies:
            print("\n⚡ SPECTRAL ANOMALIES THIS SCAN:")
            for i, anomaly in enumerate(spectral_anomalies[:2]):  # Show top 2
                print(f"   {i+1}. {anomaly['type']}")
                print(f"      {anomaly['description']} | Severity: {anomaly['severity']}")
        else:
            print("\n⚡ SPECTRAL ANOMALIES: None detected")
        print()
        
        # Buffer status
        buffer_usage = (len(self.live_data) / self.live_data.maxlen) * 100
        print(f"💾 BUFFER: {len(self.live_data)}/{self.live_data.maxlen} ({buffer_usage:.1f}% full)")
        
        # Instructions
        print("-" * 80)
        print("Press Ctrl+C to stop scanning and generate final report")
        print("=" * 80)
    
    def save_scan_data(self, filename_prefix="continuous_scan_data"):
        """Save accumulated scan data to CSV."""
        
        if not self.live_data:
            print("❌ No data to save")
            return None
        
        # Prepare data for CSV
        all_data = []
        
        # Match FFT data with scan data
        fft_dict = {fft_data['timestamp']: fft_data for fft_data in self.fft_history}
        
        for scan in self.live_data:
            # Find corresponding FFT data
            corresponding_fft = fft_dict.get(scan['timestamp'])
            
            for i, freq in enumerate(scan['frequencies']):
                row_data = {
                    'timestamp': scan['timestamp'],
                    'scan_id': scan['scan_id'],
                    'frequency': freq,
                    'audio_amplitude': scan['audio_amplitudes'][i],
                    'radio_amplitude': scan['radio_amplitudes'][i],
                    'combined_amplitude': scan['combined_amplitudes'][i]
                }
                
                # Add FFT data if available
                if corresponding_fft:
                    row_data.update({
                        'spectral_centroid': corresponding_fft['spectral_centroid'],
                        'spectral_bandwidth': corresponding_fft['spectral_bandwidth'],
                        'spectral_rolloff': corresponding_fft['spectral_rolloff'],
                        'total_spectral_energy': corresponding_fft['total_spectral_energy'],
                        'dominant_frequency_1': corresponding_fft['dominant_frequencies'][0] if corresponding_fft['dominant_frequencies'] else 0,
                        'dominant_frequency_2': corresponding_fft['dominant_frequencies'][1] if len(corresponding_fft['dominant_frequencies']) > 1 else 0,
                        'dominant_frequency_3': corresponding_fft['dominant_frequencies'][2] if len(corresponding_fft['dominant_frequencies']) > 2 else 0
                    })
                
                all_data.append(row_data)
        
        # Create DataFrame and save
        df = pd.DataFrame(all_data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        print(f"✅ Scan data saved to: {filename}")
        print(f"   • Total records: {len(df):,}")
        print(f"   • Unique scans: {df['scan_id'].nunique()}")
        print(f"   • Time span: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return filename
    
    def generate_final_report(self):
        """Generate comprehensive final analysis report."""
        
        print("\n" + "=" * 80)
        print("📋 FINAL CONTINUOUS SCAN REPORT")
        print("=" * 80)
        
        if self.start_time:
            total_runtime = datetime.now() - self.start_time
            print(f"⏱️  Total Runtime: {str(total_runtime).split('.')[0]}")
            scan_rate = self.stats['total_scans'] / total_runtime.total_seconds()
            print(f"📊 Average Scan Rate: {scan_rate:.2f} scans/second")
        
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   • Total Scans Completed: {self.stats['total_scans']:,}")
        print(f"   • Time-Domain Patterns Found: {self.stats['patterns_found']:,}")
        print(f"   • FFT Patterns Found: {self.stats['fft_patterns_found']:,}")
        print(f"   • Time-Domain Anomalies: {self.stats['anomalies_detected']:,}")
        print(f"   • Spectral Anomalies: {self.stats['spectral_anomalies']:,}")
        
        total_patterns = self.stats['patterns_found'] + self.stats['fft_patterns_found']
        total_anomalies = self.stats['anomalies_detected'] + self.stats['spectral_anomalies']
        print(f"   • TOTAL PATTERNS: {total_patterns:,}")
        print(f"   • TOTAL ANOMALIES: {total_anomalies:,}")
        
        if self.live_data:
            # Analyze all collected data
            all_combined = []
            all_frequencies = []
            
            for scan in self.live_data:
                all_combined.extend(scan['combined_amplitudes'])
                all_frequencies.extend(scan['frequencies'])
            
            print(f"\n🔍 DATA ANALYSIS:")
            print(f"   • Data Points Collected: {len(all_combined):,}")
            print(f"   • Average Signal Strength: {np.mean(all_combined):.4f}")
            print(f"   • Signal Range: {np.min(all_combined):.4f} - {np.max(all_combined):.4f}")
            print(f"   • Standard Deviation: {np.std(all_combined):.4f}")
            
            # Find most active frequency
            freq_activity = {}
            for scan in self.live_data:
                max_idx = np.argmax(scan['combined_amplitudes'])
                peak_freq = scan['frequencies'][max_idx]
                freq_bin = int(peak_freq // 100) * 100  # 100Hz bins
                freq_activity[freq_bin] = freq_activity.get(freq_bin, 0) + 1
            
            if freq_activity:
                most_active_freq = max(freq_activity, key=freq_activity.get)
                print(f"   • Most Active Frequency Band: {most_active_freq}-{most_active_freq+100} Hz")
                print(f"   • Peak Activity Count: {freq_activity[most_active_freq]} scans")
        
        # Save final data
        saved_file = self.save_scan_data()
        
        print(f"\n✅ Continuous scan analysis complete!")
        return saved_file
    
    def scan_continuously(self, duration_seconds=None):
        """Main continuous scanning loop."""
        
        print(f"\n🚀 Starting continuous frequency scanning...")
        print(f"   • Target: 1kHz-10kHz range")
        print(f"   • Scan interval: {self.scan_interval*1000:.0f}ms")
        if duration_seconds:
            print(f"   • Duration: {duration_seconds} seconds")
        print(f"   • Press Ctrl+C to stop\n")
        
        self.running = True
        self.start_time = datetime.now()
        self.scan_count = 0
        
        try:
            while self.running:
                # Generate new scan data
                current_data = self.generate_live_frequency_data()
                self.scan_count += 1
                current_data['scan_id'] = self.scan_count
                
                # Add to live data buffer
                self.live_data.append(current_data)
                
                # Perform FFT analysis
                current_fft = self.perform_fft_analysis(current_data)
                
                # Detect time-domain patterns and anomalies
                patterns = self.detect_patterns(current_data)
                anomalies = self.detect_anomalies(current_data)
                
                # Detect FFT patterns and spectral anomalies
                fft_patterns = self.detect_fft_patterns(current_fft)
                spectral_anomalies = self.detect_spectral_anomalies(current_fft)
                
                # Update statistics
                self.update_statistics(current_data, patterns, anomalies, fft_patterns, spectral_anomalies)
                
                # Display live statistics
                self.print_live_statistics(patterns, anomalies, fft_patterns, spectral_anomalies, current_fft)
                
                # Check duration limit
                if duration_seconds and (datetime.now() - self.start_time).total_seconds() >= duration_seconds:
                    print(f"\n⏰ Duration limit reached ({duration_seconds}s)")
                    break
                
                # Wait for next scan
                time.sleep(self.scan_interval)
                
        except KeyboardInterrupt:
            print(f"\n\n⏹️  Scanning stopped by user")
        finally:
            self.running = False
            
        # Generate final report
        return self.generate_final_report()


def main():
    """Main function to run the continuous scanner."""
    
    print("🎯 CONTINUOUS 1kHz-10kHz FFT FREQUENCY SCANNER")
    print("Real-time monitoring with FFT analysis, pattern detection and live statistics")
    print("-" * 70)
    
    # Initialize scanner
    scanner = ContinuousFrequencyScanner()
    
    # Ask user for scan duration
    try:
        duration_input = input("Enter scan duration in seconds (or press Enter for unlimited): ").strip()
        duration = int(duration_input) if duration_input else None
    except ValueError:
        duration = None
        print("Invalid input, running unlimited scan")
    
    # Start continuous scanning
    result_file = scanner.scan_continuously(duration_seconds=duration)
    
    if result_file:
        print(f"\n📊 Scan data saved to: {result_file}")
        print("You can now analyze this data with other tools in the project")


if __name__ == "__main__":
    main()
