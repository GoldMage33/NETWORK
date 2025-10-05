#!/usr/bin/env python3
"""
CSV Graph Generator
Creates comprehensive graphs from combined_frequency_analysis.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from scipy.stats import zscore
import threading
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class LiveFrequencyScanner:
    """Real-time 1kHz-10kHz frequency scanner with FFT analysis and peak tracking."""
    
    def __init__(self):
        self.running = False
        self.scan_data = deque(maxlen=100)  # Store last 100 scans
        self.scan_interval = 0.1  # 100ms between scans
        self.frequency_range = (1000, 10000)  # 1kHz to 10kHz
        self.fft_history = deque(maxlen=50)
        self.pattern_count = 0
        self.anomaly_count = 0
        
        # Peak tracking for audio segment (1kHz-10kHz)
        self.audio_peak_history = deque(maxlen=200)  # Track last 200 audio peaks
        self.peak_frequency_bands = {
            'low': (1000, 2500),    # 1-2.5kHz
            'mid': (2500, 5000),    # 2.5-5kHz
            'high': (5000, 10000)   # 5-10kHz
        }
        self.peak_tracking_enabled = True
        
        # Silent communication detection
        self.silent_comm_patterns = []
        self.communication_threshold = 0.15  # Amplitude threshold for comm detection
        self.silent_mode = False
        
        # Enhanced layered network pattern recording
        self.layered_network_patterns = []
        self.network_layer_signatures = {
            'physical_layer': [],      # 1-2kHz range patterns
            'data_link_layer': [],     # 2-3kHz range patterns  
            'network_layer': [],       # 3-4kHz range patterns
            'transport_layer': [],     # 4-5kHz range patterns
            'session_layer': [],       # 5-6kHz range patterns
            'presentation_layer': [],  # 6-7kHz range patterns
            'application_layer': []    # 7-10kHz range patterns
        }
        self.layer_detection_history = deque(maxlen=100)
        
        # Advanced pattern detection
        self.deep_network_patterns = {
            'protocol_signatures': {},     # Protocol-specific patterns
            'timing_patterns': {},         # Temporal behavior patterns
            'encryption_indicators': [],   # Encrypted traffic patterns
            'tunnel_detection': [],        # Tunneling behavior
            'covert_channels': [],         # Hidden communication channels
            'network_topology': {},        # Network structure analysis
            'attack_patterns': [],         # Potential security threats
            'qos_patterns': {},           # Quality of Service indicators
        }
        
        # Enhanced frequency sub-bands for deeper analysis
        self.frequency_subbands = {
            'physical_layer': {
                'carrier_signals': (1000, 1200),    # Carrier frequencies
                'modulation': (1200, 1400),         # Modulation patterns
                'sync_signals': (1400, 1600),       # Synchronization
                'error_correction': (1600, 1800),   # Error correction codes
                'line_coding': (1800, 2000)         # Line coding patterns
            },
            'data_link_layer': {
                'frame_headers': (2000, 2200),      # Frame header patterns
                'mac_addresses': (2200, 2400),      # MAC addressing
                'flow_control': (2400, 2600),       # Flow control signals
                'error_detection': (2600, 2800),    # Error detection
                'llc_patterns': (2800, 3000)        # LLC sublayer
            },
            'network_layer': {
                'ip_headers': (3000, 3200),         # IP header patterns
                'routing_protocols': (3200, 3400),   # Routing information
                'icmp_patterns': (3400, 3600),      # ICMP messages
                'fragmentation': (3600, 3800),      # Packet fragmentation
                'qos_marking': (3800, 4000)         # QoS markings
            },
            'transport_layer': {
                'tcp_handshake': (4000, 4200),      # TCP connection setup
                'udp_streams': (4200, 4400),        # UDP data streams
                'port_scanning': (4400, 4600),      # Port scan patterns
                'congestion_control': (4600, 4800), # Congestion control
                'segment_ordering': (4800, 5000)    # Segment reordering
            },
            'session_layer': {
                'session_establishment': (5000, 5200), # Session setup
                'authentication': (5200, 5400),        # Auth patterns
                'session_management': (5400, 5600),    # Session control
                'checkpointing': (5600, 5800),         # Session checkpoints
                'session_termination': (5800, 6000)    # Session teardown
            },
            'presentation_layer': {
                'encryption_overhead': (6000, 6200),   # Encryption patterns
                'compression': (6200, 6400),           # Data compression
                'format_conversion': (6400, 6600),     # Data format changes
                'character_encoding': (6600, 6800),    # Character sets
                'ssl_tls_patterns': (6800, 7000)       # SSL/TLS handshakes
            },
            'application_layer': {
                'http_patterns': (7000, 7200),         # HTTP protocols
                'dns_queries': (7200, 7400),           # DNS resolution
                'email_protocols': (7400, 7600),       # SMTP/POP3/IMAP
                'file_transfer': (7600, 7800),         # FTP patterns
                'streaming_media': (7800, 8000),       # Media streaming
                'web_services': (8000, 8200),          # Web service calls
                'database_queries': (8200, 8400),      # Database protocols
                'p2p_protocols': (8400, 8600),         # Peer-to-peer
                'vpn_overhead': (8600, 8800),          # VPN encapsulation
                'covert_apps': (8800, 9000),           # Hidden applications
                'malware_c2': (9000, 9200),            # Command & control
                'exfiltration': (9200, 9400),          # Data exfiltration
                'backdoors': (9400, 9600),             # Backdoor communication
                'steganography': (9600, 9800),         # Hidden data
                'anomalous_traffic': (9800, 10000)     # Unusual patterns
            }
        }
        
        # Live plotting setup
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('Live 1kHz-10kHz Scanner with Layered Network Analysis', fontsize=16, fontweight='bold')
        
    def generate_live_scan(self):
        """Generate realistic live frequency scan data with enhanced audio peak tracking."""
        frequencies = np.linspace(self.frequency_range[0], self.frequency_range[1], 100)
        current_time = time.time()
        
        # Generate realistic signals with time variation and peak tracking
        audio_amps = []
        radio_amps = []
        
        for freq in frequencies:
            # Enhanced audio signal with specific peak behavior in 1kHz-10kHz
            base_freq = 3500 + 300 * np.sin(current_time * 0.5)
            audio_base = 0.15 + 0.35 * np.exp(-((freq - base_freq) / 1200) ** 2)
            
            # Add frequency band specific enhancements
            band_enhancement = 0
            if 1000 <= freq <= 2500:  # Low band
                band_enhancement = 0.1 * np.sin(current_time * 2 + freq * 0.001)
            elif 2500 <= freq <= 5000:  # Mid band - primary audio communication
                band_enhancement = 0.2 * np.sin(current_time * 1.5 + freq * 0.0005)
                # Silent communication pattern
                if self.silent_mode and np.sin(current_time * 10) > 0.7:
                    band_enhancement += 0.15 * np.exp(-((freq - 3800) / 200) ** 2)
            elif 5000 <= freq <= 10000:  # High band
                band_enhancement = 0.08 * np.sin(current_time * 3 + freq * 0.0002)
            
            # Add harmonics and noise
            harmonics = 0
            for harmonic in [2000, 4000, 6000, 8000]:
                if abs(freq - harmonic) < 100:
                    harmonics += 0.1 * np.exp(-((freq - harmonic) / 50) ** 2)
            
            audio_noise = np.random.normal(0, 0.02)
            time_mod = 0.05 * np.sin(current_time * freq * 0.0001)
            
            # Silent communication bursts
            silent_comm = 0
            if not self.silent_mode and np.random.random() < 0.05:  # 5% chance
                if 2000 <= freq <= 4000:  # Focus in speech range
                    silent_comm = 0.25 * np.exp(-((freq - 3000) / 400) ** 2)
            
            audio_amp = max(0.01, audio_base + band_enhancement + harmonics + audio_noise + time_mod + silent_comm)
            audio_amps.append(audio_amp)
            
            # Radio signal with interference (unchanged)
            radio_base = 0.05 + 0.08 * np.sin(freq * 0.001 + current_time * 2)
            interference = 0
            
            # Periodic interference
            if np.sin(current_time * 3) > 0.3:
                interference_freqs = [1500, 2500, 5000, 7500, 9000]
                for int_freq in interference_freqs:
                    if abs(freq - int_freq) < 150:
                        interference += 0.2 * np.exp(-((freq - int_freq) / 75) ** 2)
            
            radio_noise = np.random.normal(0, 0.01)
            radio_amp = max(0.01, radio_base + interference + radio_noise)
            radio_amps.append(radio_amp)
        
        scan_data = {
            'timestamp': datetime.now(),
            'frequencies': frequencies,
            'audio_amplitudes': np.array(audio_amps),
            'radio_amplitudes': np.array(radio_amps),
            'combined_amplitudes': np.array(audio_amps) + np.array(radio_amps)
        }
        
        # Track audio peaks in the 1kHz-10kHz range
        if self.peak_tracking_enabled:
            self.track_audio_peaks(scan_data)
        
        return scan_data
    
    def perform_fft_analysis(self, scan_data):
        """Perform FFT analysis on scan data."""
        signal = scan_data['combined_amplitudes']
        freqs = scan_data['frequencies']
        
        # FFT computation
        fft_vals = fft(signal)
        fft_freqs = fftfreq(len(signal), d=(freqs[1] - freqs[0]))
        fft_magnitude = np.abs(fft_vals)
        
        # Focus on positive frequencies
        pos_mask = fft_freqs > 0
        fft_freqs_pos = fft_freqs[pos_mask]
        fft_mag_pos = fft_magnitude[pos_mask]
        
        # Find dominant frequencies
        dominant_indices = np.argsort(fft_mag_pos)[-3:]
        dominant_freqs = []
        dominant_mags = []
        
        for idx in reversed(dominant_indices):
            if idx < len(fft_freqs_pos) and fft_freqs_pos[idx] > 0.1:
                # Convert back to actual frequency
                actual_freq = fft_freqs_pos[idx] * (freqs[-1] - freqs[0]) + freqs[0]
                dominant_freqs.append(actual_freq)
                dominant_mags.append(fft_mag_pos[idx])
        
        # Spectral features
        spectral_centroid = np.sum(fft_freqs_pos * fft_mag_pos) / np.sum(fft_mag_pos) if np.sum(fft_mag_pos) > 0 else 0
        spectral_bandwidth = np.sqrt(np.sum(((fft_freqs_pos - spectral_centroid) ** 2) * fft_mag_pos) / np.sum(fft_mag_pos)) if np.sum(fft_mag_pos) > 0 else 0
        total_energy = np.sum(fft_mag_pos ** 2)
        
        return {
            'fft_frequencies': fft_freqs_pos,
            'fft_magnitude': fft_mag_pos,
            'dominant_frequencies': dominant_freqs[:3],
            'dominant_magnitudes': dominant_mags[:3],
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'total_energy': total_energy,
            'timestamp': scan_data['timestamp']
        }
    
    def track_audio_peaks(self, scan_data):
        """Track peaks specifically in the audio segment (1kHz-10kHz)."""
        audio_amps = scan_data['audio_amplitudes']
        frequencies = scan_data['frequencies']
        
        # Find peaks in audio signal
        peaks, properties = find_peaks(
            audio_amps,
            height=np.mean(audio_amps) + 0.5 * np.std(audio_amps),
            distance=5  # Minimum 5 samples between peaks
        )
        
        peak_data = {
            'timestamp': scan_data['timestamp'],
            'peaks': [],
            'band_distribution': {'low': 0, 'mid': 0, 'high': 0}
        }
        
        for peak_idx in peaks:
            peak_freq = frequencies[peak_idx]
            peak_amp = audio_amps[peak_idx]
            
            # Classify peak by frequency band
            band = 'low'
            if peak_freq >= 5000:
                band = 'high'
            elif peak_freq >= 2500:
                band = 'mid'
            
            peak_info = {
                'frequency': peak_freq,
                'amplitude': peak_amp,
                'band': band,
                'index': peak_idx
            }
            
            peak_data['peaks'].append(peak_info)
            peak_data['band_distribution'][band] += 1
        
        self.audio_peak_history.append(peak_data)
        
        # Detect communication patterns
        self.detect_silent_communication(peak_data)
        
        return peak_data
    
    def detect_silent_communication(self, peak_data):
        """Detect potential silent communication patterns in audio peaks."""
        if len(self.audio_peak_history) < 10:
            return
        
        # Look for patterns in recent peak history
        recent_peaks = list(self.audio_peak_history)[-10:]
        
        # Pattern 1: Consistent mid-band activity (2.5-5kHz)
        mid_band_activity = [p['band_distribution']['mid'] for p in recent_peaks]
        if np.mean(mid_band_activity) > 2 and np.std(mid_band_activity) < 1:
            self.silent_comm_patterns.append({
                'type': 'sustained_mid_band',
                'timestamp': peak_data['timestamp'],
                'confidence': min(0.9, np.mean(mid_band_activity) / 5)
            })
        
        # Pattern 2: Rhythmic peak patterns
        peak_counts = [len(p['peaks']) for p in recent_peaks]
        if len(peak_counts) >= 8:
            # Check for periodic patterns
            fft_peaks = np.abs(fft(peak_counts))
            if np.max(fft_peaks[1:4]) > len(peak_counts) * 0.3:  # Strong low-frequency component
                self.silent_comm_patterns.append({
                    'type': 'rhythmic_pattern',
                    'timestamp': peak_data['timestamp'],
                    'confidence': 0.7
                })
        
        # Pattern 3: Sudden amplitude changes in speech range (2-4kHz)
        speech_range_peaks = []
        for peak_info in peak_data['peaks']:
            if 2000 <= peak_info['frequency'] <= 4000:
                speech_range_peaks.append(peak_info['amplitude'])
        
        if speech_range_peaks and len(self.audio_peak_history) >= 5:
            # Compare with previous speech range activity
            prev_speech_amps = []
            for prev_peak_data in list(self.audio_peak_history)[-5:-1]:
                for peak_info in prev_peak_data['peaks']:
                    if 2000 <= peak_info['frequency'] <= 4000:
                        prev_speech_amps.append(peak_info['amplitude'])
            
            if prev_speech_amps:
                current_avg = np.mean(speech_range_peaks)
                prev_avg = np.mean(prev_speech_amps)
                
                if current_avg > prev_avg * 1.5:  # 50% increase
                    self.silent_comm_patterns.append({
                        'type': 'speech_burst',
                        'timestamp': peak_data['timestamp'],
                        'confidence': min(0.8, (current_avg - prev_avg) / prev_avg)
                    })
        
        # Keep only recent patterns (last 50)
        if len(self.silent_comm_patterns) > 50:
            self.silent_comm_patterns = self.silent_comm_patterns[-50:]
    
    def detect_layered_network_patterns(self, scan_data, fft_data):
        """Detect and record patterns specific to layered network architecture."""
        
        frequencies = scan_data['frequencies']
        audio_amps = scan_data['audio_amplitudes']
        combined_amps = scan_data['combined_amplitudes']
        
        # Define OSI layer frequency mappings for network analysis
        layer_ranges = {
            'physical_layer': (1000, 2000),      # Physical transmission patterns
            'data_link_layer': (2000, 3000),     # Frame synchronization patterns
            'network_layer': (3000, 4000),       # Routing/addressing patterns
            'transport_layer': (4000, 5000),     # Session management patterns
            'session_layer': (5000, 6000),       # Connection management patterns
            'presentation_layer': (6000, 7000),  # Data transformation patterns
            'application_layer': (7000, 10000)   # Application protocol patterns
        }
        
        layer_detection = {
            'timestamp': scan_data['timestamp'],
            'layers_active': [],
            'layer_signatures': {},
            'network_topology_indicators': [],
            'leakage_indicators': []
        }
        
        for layer_name, (low_freq, high_freq) in layer_ranges.items():
            # Find frequencies in this layer's range
            layer_mask = (frequencies >= low_freq) & (frequencies < high_freq)
            layer_freqs = frequencies[layer_mask]
            layer_audio_amps = audio_amps[layer_mask]
            layer_combined_amps = combined_amps[layer_mask]
            
            if len(layer_freqs) == 0:
                continue
            
            # Analyze layer activity
            layer_activity = {
                'frequency_range': (low_freq, high_freq),
                'peak_amplitude': np.max(layer_combined_amps),
                'average_amplitude': np.mean(layer_combined_amps),
                'amplitude_variance': np.var(layer_combined_amps),
                'dominant_frequency': layer_freqs[np.argmax(layer_combined_amps)],
                'activity_score': 0
            }
            
            # Calculate activity score based on amplitude and variance
            normalized_peak = layer_activity['peak_amplitude'] / np.mean(combined_amps)
            normalized_variance = layer_activity['amplitude_variance'] / np.var(combined_amps)
            layer_activity['activity_score'] = normalized_peak * 0.7 + normalized_variance * 0.3
            
            # Detect layer-specific patterns
            layer_patterns = []
            
            # Pattern 1: Consistent high activity (active layer)
            if layer_activity['activity_score'] > 1.5:
                layer_patterns.append('high_activity')
                layer_detection['layers_active'].append(layer_name)
            
            # Pattern 2: Rhythmic patterns (protocol timing)
            if len(self.layer_detection_history) >= 10:
                recent_scores = []
                for hist_entry in list(self.layer_detection_history)[-10:]:
                    if layer_name in hist_entry['layer_signatures']:
                        recent_scores.append(hist_entry['layer_signatures'][layer_name]['activity_score'])
                
                if len(recent_scores) >= 8:
                    # Check for periodic behavior
                    score_fft = np.abs(fft(recent_scores))
                    if np.max(score_fft[1:4]) > len(recent_scores) * 0.4:
                        layer_patterns.append('rhythmic_protocol')
            
            # Pattern 3: Sudden spikes (potential leakage or intrusion)
            if len(self.layer_detection_history) >= 5:
                recent_peaks = []
                for hist_entry in list(self.layer_detection_history)[-5:]:
                    if layer_name in hist_entry['layer_signatures']:
                        recent_peaks.append(hist_entry['layer_signatures'][layer_name]['peak_amplitude'])
                
                if recent_peaks:
                    avg_recent = np.mean(recent_peaks)
                    if layer_activity['peak_amplitude'] > avg_recent * 2.0:
                        layer_patterns.append('amplitude_spike')
                        layer_detection['leakage_indicators'].append({
                            'layer': layer_name,
                            'type': 'amplitude_spike',
                            'severity': min(1.0, (layer_activity['peak_amplitude'] - avg_recent) / avg_recent),
                            'frequency': layer_activity['dominant_frequency']
                        })
            
            # Pattern 4: Cross-layer interference
            if layer_activity['activity_score'] > 1.2:
                # Check if adjacent layers are also active
                adjacent_layers = []
                layer_names = list(layer_ranges.keys())
                current_idx = layer_names.index(layer_name)
                
                if current_idx > 0:
                    adjacent_layers.append(layer_names[current_idx - 1])
                if current_idx < len(layer_names) - 1:
                    adjacent_layers.append(layer_names[current_idx + 1])
                
                # Check recent history for adjacent layer activity
                for adj_layer in adjacent_layers:
                    if len(self.layer_detection_history) >= 3:
                        recent_adj_activity = []
                        for hist_entry in list(self.layer_detection_history)[-3:]:
                            if adj_layer in hist_entry['layer_signatures']:
                                recent_adj_activity.append(hist_entry['layer_signatures'][adj_layer]['activity_score'])
                        
                        if recent_adj_activity and np.mean(recent_adj_activity) > 1.0:
                            layer_patterns.append('cross_layer_interference')
                            layer_detection['network_topology_indicators'].append({
                                'type': 'layer_coupling',
                                'primary_layer': layer_name,
                                'coupled_layer': adj_layer,
                                'coupling_strength': layer_activity['activity_score'] * np.mean(recent_adj_activity)
                            })
            
            layer_activity['patterns'] = layer_patterns
            layer_detection['layer_signatures'][layer_name] = layer_activity
            
            # Store in layer-specific history
            self.network_layer_signatures[layer_name].append({
                'timestamp': scan_data['timestamp'],
                'signature': layer_activity,
                'patterns': layer_patterns
            })
            
            # Keep only recent signatures (last 50 per layer)
            if len(self.network_layer_signatures[layer_name]) > 50:
                self.network_layer_signatures[layer_name] = self.network_layer_signatures[layer_name][-50:]
        
        # Detect network topology patterns
        active_layers = layer_detection['layers_active']
        if len(active_layers) >= 3:
            # Multiple layers active - potential network stack activity
            layer_detection['network_topology_indicators'].append({
                'type': 'multi_layer_activity',
                'active_layers': active_layers,
                'layer_count': len(active_layers),
                'topology_confidence': len(active_layers) / 7.0  # 7 OSI layers
            })
        
        # Detect potential obscured layers
        if len(active_layers) < 2 and len(layer_detection['leakage_indicators']) > 0:
            layer_detection['network_topology_indicators'].append({
                'type': 'obscured_layer_detection',
                'suspected_layers': [leak['layer'] for leak in layer_detection['leakage_indicators']],
                'confidence': np.mean([leak['severity'] for leak in layer_detection['leakage_indicators']])
            })
        
        # Store the complete layer detection
        self.layer_detection_history.append(layer_detection)
        
        # Generate layered network pattern records
        if layer_detection['layers_active'] or layer_detection['leakage_indicators']:
            network_pattern = {
                'timestamp': scan_data['timestamp'],
                'pattern_type': 'layered_network_activity',
                'active_layers': layer_detection['layers_active'],
                'topology_indicators': layer_detection['network_topology_indicators'],
                'leakage_indicators': layer_detection['leakage_indicators'],
                'confidence': self.calculate_network_confidence(layer_detection),
                'fft_correlation': self.correlate_with_fft(layer_detection, fft_data)
            }
            
            self.layered_network_patterns.append(network_pattern)
            
            # Keep only recent patterns (last 100)
            if len(self.layered_network_patterns) > 100:
                self.layered_network_patterns = self.layered_network_patterns[-100:]
        
        return layer_detection
    
    def calculate_network_confidence(self, layer_detection):
        """Calculate confidence score for network pattern detection."""
        confidence = 0.0
        
        # Base confidence from active layers
        confidence += len(layer_detection['layers_active']) * 0.1
        
        # Boost confidence for topology indicators
        for indicator in layer_detection['network_topology_indicators']:
            if indicator['type'] == 'multi_layer_activity':
                confidence += indicator['topology_confidence'] * 0.3
            elif indicator['type'] == 'layer_coupling':
                confidence += indicator['coupling_strength'] * 0.1
            elif indicator['type'] == 'obscured_layer_detection':
                confidence += indicator['confidence'] * 0.2
        
        # Boost confidence for leakage indicators
        for leak in layer_detection['leakage_indicators']:
            confidence += leak['severity'] * 0.15
        
        return min(1.0, confidence)
    
    def correlate_with_fft(self, layer_detection, fft_data):
        """Correlate layer detection with FFT analysis."""
        correlation = {
            'dominant_frequencies_in_layers': [],
            'spectral_energy_distribution': {},
            'harmonic_relationships': []
        }
        
        # Check if dominant FFT frequencies fall in active layers
        for freq in fft_data['dominant_frequencies']:
            for layer_name, signature in layer_detection['layer_signatures'].items():
                layer_range = signature['frequency_range']
                if layer_range[0] <= freq < layer_range[1]:
                    correlation['dominant_frequencies_in_layers'].append({
                        'frequency': freq,
                        'layer': layer_name,
                        'activity_score': signature['activity_score']
                    })
        
        # Calculate spectral energy distribution across layers
        for layer_name, signature in layer_detection['layer_signatures'].items():
            if signature['activity_score'] > 0.5:
                correlation['spectral_energy_distribution'][layer_name] = signature['activity_score']
        
        return correlation
    
    def analyze_deep_network_patterns(self, scan_data, fft_data, layer_detection):
        """Perform deep analysis of network patterns across all layers."""
        
        frequencies = scan_data['frequencies']
        audio_amps = scan_data['audio_amplitudes']
        combined_amps = scan_data['combined_amplitudes']
        timestamp = scan_data['timestamp']
        
        deep_analysis = {
            'timestamp': timestamp,
            'protocol_analysis': {},
            'timing_analysis': {},
            'security_analysis': {},
            'topology_analysis': {},
            'performance_analysis': {}
        }
        
        # 1. Protocol Signature Analysis
        deep_analysis['protocol_analysis'] = self.analyze_protocol_signatures(
            frequencies, audio_amps, combined_amps, layer_detection
        )
        
        # 2. Timing Pattern Analysis
        deep_analysis['timing_analysis'] = self.analyze_timing_patterns(
            timestamp, layer_detection
        )
        
        # 3. Security Pattern Analysis
        deep_analysis['security_analysis'] = self.analyze_security_patterns(
            frequencies, combined_amps, layer_detection
        )
        
        # 4. Network Topology Analysis
        deep_analysis['topology_analysis'] = self.analyze_network_topology(
            layer_detection, fft_data
        )
        
        # 5. Performance Analysis
        deep_analysis['performance_analysis'] = self.analyze_network_performance(
            frequencies, combined_amps, layer_detection
        )
        
        return deep_analysis
    
    def analyze_protocol_signatures(self, frequencies, audio_amps, combined_amps, layer_detection):
        """Analyze protocol-specific signatures in frequency patterns."""
        
        protocol_analysis = {
            'detected_protocols': [],
            'protocol_confidence': {},
            'protocol_behaviors': {},
            'unusual_patterns': []
        }
        
        for layer_name, subbands in self.frequency_subbands.items():
            if layer_name not in layer_detection['layer_signatures']:
                continue
                
            layer_signature = layer_detection['layer_signatures'][layer_name]
            
            for subband_name, (low_freq, high_freq) in subbands.items():
                # Find frequencies in this subband
                subband_mask = (frequencies >= low_freq) & (frequencies < high_freq)
                subband_freqs = frequencies[subband_mask]
                subband_amps = combined_amps[subband_mask]
                
                if len(subband_freqs) == 0:
                    continue
                
                # Analyze subband characteristics
                subband_peak = np.max(subband_amps)
                subband_avg = np.mean(subband_amps)
                subband_var = np.var(subband_amps)
                
                # Protocol pattern detection
                protocol_confidence = 0.0
                detected_behaviors = []
                
                # Specific protocol pattern analysis
                if subband_name == 'tcp_handshake' and subband_peak > subband_avg * 2:
                    protocol_confidence = min(1.0, subband_peak / subband_avg / 2)
                    detected_behaviors.append('tcp_connection_activity')
                    protocol_analysis['detected_protocols'].append('TCP')
                
                elif subband_name == 'dns_queries' and subband_var > np.var(combined_amps) * 1.5:
                    protocol_confidence = min(1.0, subband_var / np.var(combined_amps) / 1.5)
                    detected_behaviors.append('dns_resolution_activity')
                    protocol_analysis['detected_protocols'].append('DNS')
                
                elif subband_name == 'http_patterns' and subband_peak > 0.3:
                    protocol_confidence = min(1.0, subband_peak / 0.3)
                    detected_behaviors.append('http_traffic')
                    protocol_analysis['detected_protocols'].append('HTTP')
                
                elif subband_name == 'ssl_tls_patterns' and subband_avg > np.mean(combined_amps) * 1.3:
                    protocol_confidence = min(1.0, subband_avg / np.mean(combined_amps) / 1.3)
                    detected_behaviors.append('encrypted_traffic')
                    protocol_analysis['detected_protocols'].append('TLS/SSL')
                
                elif subband_name == 'vpn_overhead' and subband_peak > 0.25:
                    protocol_confidence = min(1.0, subband_peak / 0.25)
                    detected_behaviors.append('vpn_tunneling')
                    protocol_analysis['detected_protocols'].append('VPN')
                
                elif subband_name == 'malware_c2' and subband_peak > 0.2:
                    protocol_confidence = min(1.0, subband_peak / 0.2)
                    detected_behaviors.append('suspicious_c2_activity')
                    protocol_analysis['unusual_patterns'].append({
                        'type': 'potential_malware_c2',
                        'frequency_range': (low_freq, high_freq),
                        'confidence': protocol_confidence,
                        'peak_amplitude': subband_peak
                    })
                
                if protocol_confidence > 0.3:
                    protocol_analysis['protocol_confidence'][subband_name] = protocol_confidence
                    protocol_analysis['protocol_behaviors'][subband_name] = detected_behaviors
        
        return protocol_analysis
    
    def analyze_timing_patterns(self, timestamp, layer_detection):
        """Analyze temporal patterns in network layer activity."""
        
        timing_analysis = {
            'periodic_patterns': [],
            'burst_patterns': [],
            'latency_indicators': [],
            'synchronization_patterns': []
        }
        
        if len(self.layer_detection_history) < 10:
            return timing_analysis
        
        # Analyze recent activity patterns
        recent_history = list(self.layer_detection_history)[-10:]
        
        # Check for periodic behavior in each layer
        for layer_name in self.network_layer_signatures.keys():
            layer_activities = []
            timestamps = []
            
            for hist_entry in recent_history:
                if layer_name in hist_entry['layer_signatures']:
                    layer_activities.append(hist_entry['layer_signatures'][layer_name]['activity_score'])
                    timestamps.append(hist_entry['timestamp'])
                else:
                    layer_activities.append(0.0)
                    timestamps.append(hist_entry['timestamp'])
            
            if len(layer_activities) >= 8:
                # FFT analysis for periodicity
                activity_fft = np.abs(fft(layer_activities))
                dominant_period = np.argmax(activity_fft[1:5]) + 1
                
                if activity_fft[dominant_period] > len(layer_activities) * 0.4:
                    timing_analysis['periodic_patterns'].append({
                        'layer': layer_name,
                        'period': dominant_period,
                        'strength': activity_fft[dominant_period] / len(layer_activities),
                        'type': 'periodic_protocol_activity'
                    })
                
                # Burst detection
                activity_mean = np.mean(layer_activities)
                activity_std = np.std(layer_activities)
                
                for i, activity in enumerate(layer_activities):
                    if activity > activity_mean + 2 * activity_std:
                        timing_analysis['burst_patterns'].append({
                            'layer': layer_name,
                            'timestamp': timestamps[i],
                            'intensity': activity,
                            'burst_ratio': activity / activity_mean if activity_mean > 0 else 0
                        })
        
        return timing_analysis
    
    def analyze_security_patterns(self, frequencies, combined_amps, layer_detection):
        """Analyze patterns that may indicate security issues or attacks."""
        
        security_analysis = {
            'threat_indicators': [],
            'anomalous_behaviors': [],
            'covert_channels': [],
            'attack_signatures': []
        }
        
        # 1. Scan for suspicious frequency patterns
        suspicious_ranges = [
            ('malware_c2', (9000, 9200)),
            ('exfiltration', (9200, 9400)),
            ('backdoors', (9400, 9600)),
            ('steganography', (9600, 9800)),
            ('anomalous_traffic', (9800, 10000))
        ]
        
        for pattern_name, (low_freq, high_freq) in suspicious_ranges:
            pattern_mask = (frequencies >= low_freq) & (frequencies < high_freq)
            pattern_amps = combined_amps[pattern_mask]
            
            if len(pattern_amps) > 0:
                pattern_peak = np.max(pattern_amps)
                pattern_avg = np.mean(pattern_amps)
                
                if pattern_peak > 0.15:  # Threshold for suspicious activity
                    threat_level = min(1.0, pattern_peak / 0.15)
                    
                    security_analysis['threat_indicators'].append({
                        'type': pattern_name,
                        'frequency_range': (low_freq, high_freq),
                        'threat_level': threat_level,
                        'peak_amplitude': pattern_peak,
                        'average_amplitude': pattern_avg,
                        'description': self.get_threat_description(pattern_name)
                    })
        
        # 2. Covert channel detection
        self.detect_covert_channels(security_analysis, layer_detection, combined_amps)
        
        # 3. Attack pattern analysis
        self.detect_attack_patterns(security_analysis, layer_detection)
        
        return security_analysis
    
    def detect_covert_channels(self, security_analysis, layer_detection, combined_amps):
        """Detect potential covert communication channels."""
        
        # Look for unusual layer combinations
        active_layers = layer_detection['layers_active']
        
        # Covert channel indicators
        if 'physical_layer' in active_layers and 'application_layer' in active_layers:
            if len(active_layers) == 2:  # Only these two active
                security_analysis['covert_channels'].append({
                    'type': 'layer_skipping_channel',
                    'description': 'Direct physical to application layer communication',
                    'confidence': 0.7,
                    'layers_involved': ['physical_layer', 'application_layer']
                })
        
        # Look for timing-based covert channels
        if len(self.layer_detection_history) >= 5:
            recent_activities = []
            for hist_entry in list(self.layer_detection_history)[-5:]:
                total_activity = sum(
                    sig['activity_score'] for sig in hist_entry['layer_signatures'].values()
                )
                recent_activities.append(total_activity)
            
            # Check for very regular patterns (potential timing channel)
            activity_std = np.std(recent_activities)
            activity_mean = np.mean(recent_activities)
            
            if activity_std < activity_mean * 0.1 and activity_mean > 1.0:
                security_analysis['covert_channels'].append({
                    'type': 'timing_channel',
                    'description': 'Unusually regular timing patterns detected',
                    'confidence': 0.6,
                    'regularity_score': 1.0 - (activity_std / activity_mean)
                })
    
    def detect_attack_patterns(self, security_analysis, layer_detection):
        """Detect patterns that may indicate network attacks."""
        
        # Port scanning detection
        if 'transport_layer' in layer_detection['layer_signatures']:
            transport_sig = layer_detection['layer_signatures']['transport_layer']
            if transport_sig['activity_score'] > 2.0:
                security_analysis['attack_signatures'].append({
                    'type': 'potential_port_scan',
                    'confidence': min(1.0, transport_sig['activity_score'] / 2.0),
                    'description': 'High transport layer activity suggesting port scanning'
                })
        
        # DDoS indicators
        total_activity = sum(
            sig['activity_score'] for sig in layer_detection['layer_signatures'].values()
        )
        
        if total_activity > 10.0:  # Very high overall activity
            security_analysis['attack_signatures'].append({
                'type': 'potential_ddos',
                'confidence': min(1.0, total_activity / 10.0),
                'description': 'Extremely high network activity across multiple layers'
            })
    
    def get_threat_description(self, pattern_name):
        """Get human-readable description of threat patterns."""
        descriptions = {
            'malware_c2': 'Command and control communication patterns',
            'exfiltration': 'Potential data exfiltration activity',
            'backdoors': 'Backdoor communication signatures',
            'steganography': 'Hidden data transmission patterns',
            'anomalous_traffic': 'Unusual network traffic patterns'
        }
        return descriptions.get(pattern_name, 'Unknown threat pattern')
    
    def analyze_network_topology(self, layer_detection, fft_data):
        """Analyze network topology based on layer interactions."""
        
        topology_analysis = {
            'network_type_indicators': [],
            'topology_confidence': 0.0,
            'routing_patterns': [],
            'network_complexity': 0.0
        }
        
        active_layers = layer_detection['layers_active']
        layer_count = len(active_layers)
        
        # Determine likely network topology
        if layer_count >= 5:
            topology_analysis['network_type_indicators'].append('complex_enterprise_network')
            topology_analysis['topology_confidence'] = 0.8
        elif layer_count >= 3:
            topology_analysis['network_type_indicators'].append('standard_tcp_ip_network')
            topology_analysis['topology_confidence'] = 0.6
        elif layer_count == 2:
            topology_analysis['network_type_indicators'].append('simple_point_to_point')
            topology_analysis['topology_confidence'] = 0.4
        
        # Calculate network complexity
        topology_analysis['network_complexity'] = layer_count / 7.0  # Normalized to OSI layers
        
        return topology_analysis
    
    def analyze_network_performance(self, frequencies, combined_amps, layer_detection):
        """Analyze network performance indicators."""
        
        performance_analysis = {
            'throughput_indicators': {},
            'latency_indicators': {},
            'congestion_indicators': [],
            'qos_patterns': []
        }
        
        # Throughput analysis based on amplitude patterns
        for layer_name, layer_sig in layer_detection['layer_signatures'].items():
            if layer_sig['activity_score'] > 0.5:
                performance_analysis['throughput_indicators'][layer_name] = {
                    'relative_throughput': layer_sig['activity_score'],
                    'peak_capacity': layer_sig['peak_amplitude'],
                    'utilization': min(1.0, layer_sig['activity_score'] / 3.0)
                }
        
        # Congestion detection
        if 'transport_layer' in layer_detection['layer_signatures']:
            transport_sig = layer_detection['layer_signatures']['transport_layer']
            if transport_sig['amplitude_variance'] > 0.1:
                performance_analysis['congestion_indicators'].append({
                    'type': 'transport_layer_congestion',
                    'severity': min(1.0, transport_sig['amplitude_variance'] / 0.1),
                    'affected_frequency': transport_sig['dominant_frequency']
                })
        
        return performance_analysis
    
    def store_deep_analysis_results(self, deep_analysis):
        """Store results from deep network analysis."""
        
        timestamp = deep_analysis['timestamp']
        
        # Store protocol signatures
        for protocol in deep_analysis['protocol_analysis']['detected_protocols']:
            if protocol not in self.deep_network_patterns['protocol_signatures']:
                self.deep_network_patterns['protocol_signatures'][protocol] = []
            
            self.deep_network_patterns['protocol_signatures'][protocol].append({
                'timestamp': timestamp,
                'confidence': deep_analysis['protocol_analysis']['protocol_confidence'],
                'behaviors': deep_analysis['protocol_analysis']['protocol_behaviors']
            })
        
        # Store timing patterns
        for timing_pattern in deep_analysis['timing_analysis']['periodic_patterns']:
            layer = timing_pattern['layer']
            if layer not in self.deep_network_patterns['timing_patterns']:
                self.deep_network_patterns['timing_patterns'][layer] = []
            
            self.deep_network_patterns['timing_patterns'][layer].append({
                'timestamp': timestamp,
                'pattern': timing_pattern
            })
        
        # Store security indicators
        for threat in deep_analysis['security_analysis']['threat_indicators']:
            self.deep_network_patterns['attack_patterns'].append({
                'timestamp': timestamp,
                'threat': threat
            })
        
        # Store covert channels
        for covert in deep_analysis['security_analysis']['covert_channels']:
            self.deep_network_patterns['covert_channels'].append({
                'timestamp': timestamp,
                'channel': covert
            })
        
        # Store topology analysis
        topology_key = str(len(deep_analysis['topology_analysis']['network_type_indicators']))
        if topology_key not in self.deep_network_patterns['network_topology']:
            self.deep_network_patterns['network_topology'][topology_key] = []
        
        self.deep_network_patterns['network_topology'][topology_key].append({
            'timestamp': timestamp,
            'analysis': deep_analysis['topology_analysis']
        })
        
        # Store QoS patterns
        for layer, perf_data in deep_analysis['performance_analysis']['throughput_indicators'].items():
            if layer not in self.deep_network_patterns['qos_patterns']:
                self.deep_network_patterns['qos_patterns'][layer] = []
            
            self.deep_network_patterns['qos_patterns'][layer].append({
                'timestamp': timestamp,
                'performance': perf_data
            })
        
        # Keep only recent patterns (last 50 per category)
        for category in self.deep_network_patterns:
            if isinstance(self.deep_network_patterns[category], list):
                if len(self.deep_network_patterns[category]) > 50:
                    self.deep_network_patterns[category] = self.deep_network_patterns[category][-50:]
            elif isinstance(self.deep_network_patterns[category], dict):
                for subcategory in self.deep_network_patterns[category]:
                    if isinstance(self.deep_network_patterns[category][subcategory], list):
                        if len(self.deep_network_patterns[category][subcategory]) > 50:
                            self.deep_network_patterns[category][subcategory] = self.deep_network_patterns[category][subcategory][-50:]
    
    def detect_patterns(self, current_scan, current_fft):
        """Detect patterns in real-time including audio peak tracking."""
        patterns = []
        
        if len(self.scan_data) < 5:
            return patterns
        
        # Pattern 1: Frequency stability
        recent_peaks = []
        for scan in list(self.scan_data)[-5:]:
            peak_idx = np.argmax(scan['combined_amplitudes'])
            recent_peaks.append(scan['frequencies'][peak_idx])
        
        if len(set([round(f, 0) for f in recent_peaks])) <= 2:
            avg_peak = np.mean(recent_peaks)
            patterns.append(f"Stable Peak at {avg_peak:.1f} Hz")
        
        # Pattern 2: FFT harmonic detection
        if len(current_fft['dominant_frequencies']) >= 2:
            fundamental = current_fft['dominant_frequencies'][0]
            for freq in current_fft['dominant_frequencies'][1:]:
                ratio = freq / fundamental if fundamental > 0 else 0
                if 1.8 < ratio < 2.2:
                    patterns.append(f"2nd Harmonic detected: {freq:.1f} Hz")
                elif 2.8 < ratio < 3.2:
                    patterns.append(f"3rd Harmonic detected: {freq:.1f} Hz")
        
        # Pattern 3: Spectral centroid drift
        if len(self.fft_history) >= 8:
            recent_centroids = [fft['spectral_centroid'] for fft in list(self.fft_history)[-8:]]
            centroid_trend = np.polyfit(range(len(recent_centroids)), recent_centroids, 1)[0]
            if abs(centroid_trend) > 1.0:
                direction = "upward" if centroid_trend > 0 else "downward"
                patterns.append(f"Spectral centroid drifting {direction}")
        
        # Pattern 4: Audio peak band distribution
        if len(self.audio_peak_history) >= 5:
            recent_bands = list(self.audio_peak_history)[-5:]
            total_low = sum(p['band_distribution']['low'] for p in recent_bands)
            total_mid = sum(p['band_distribution']['mid'] for p in recent_bands)
            total_high = sum(p['band_distribution']['high'] for p in recent_bands)
            
            if total_mid > total_low + total_high:
                patterns.append(f"Mid-band dominance (2.5-5kHz): {total_mid} peaks")
            elif total_low > 5:
                patterns.append(f"Low-band activity (1-2.5kHz): {total_low} peaks")
            elif total_high > 3:
                patterns.append(f"High-band activity (5-10kHz): {total_high} peaks")
        
        # Pattern 5: Silent communication detection
        recent_comm_patterns = [p for p in self.silent_comm_patterns 
                               if (current_scan['timestamp'] - p['timestamp']).total_seconds() < 5]
        
        for comm_pattern in recent_comm_patterns[-2:]:  # Show last 2
            pattern_type = comm_pattern['type'].replace('_', ' ').title()
            confidence = comm_pattern['confidence']
            patterns.append(f"Silent Comm: {pattern_type} ({confidence:.1%})")
        
        # Pattern 6: Layered network detection
        recent_network_patterns = [p for p in self.layered_network_patterns 
                                  if (current_scan['timestamp'] - p['timestamp']).total_seconds() < 10]
        
        for network_pattern in recent_network_patterns[-2:]:  # Show last 2
            active_layers = len(network_pattern['active_layers'])
            confidence = network_pattern['confidence']
            leakage_count = len(network_pattern['leakage_indicators'])
            
            if leakage_count > 0:
                patterns.append(f"Network Leakage: {leakage_count} indicators ({confidence:.1%})")
            if active_layers > 0:
                patterns.append(f"Layer Activity: {active_layers} layers ({confidence:.1%})")
        
        # Pattern 7: Deep network analysis patterns
        recent_time = current_scan['timestamp']
        
        # Protocol detection patterns
        recent_protocols = []
        for protocol, entries in self.deep_network_patterns['protocol_signatures'].items():
            recent_entries = [e for e in entries if (recent_time - e['timestamp']).total_seconds() < 5]
            if recent_entries:
                recent_protocols.append(protocol)
        
        if recent_protocols:
            protocols_str = ', '.join(recent_protocols[:3])
            patterns.append(f"Protocols: {protocols_str}")
        
        # Security threat patterns
        recent_threats = [t for t in self.deep_network_patterns['attack_patterns'] 
                         if (recent_time - t['timestamp']).total_seconds() < 15]
        
        if recent_threats:
            high_threats = [t for t in recent_threats if t['threat']['threat_level'] > 0.7]
            if high_threats:
                threat_types = [t['threat']['type'] for t in high_threats[:2]]
                patterns.append(f"⚠️ Threats: {', '.join(threat_types)}")
        
        # Covert channel detection
        recent_covert = [c for c in self.deep_network_patterns['covert_channels'] 
                        if (recent_time - c['timestamp']).total_seconds() < 20]
        
        if recent_covert:
            covert_types = [c['channel']['type'] for c in recent_covert[:2]]
            patterns.append(f"🕵️ Covert: {', '.join(covert_types)}")
        
        return patterns
    
    def detect_anomalies(self, current_scan, current_fft):
        """Detect anomalies in real-time."""
        anomalies = []
        
        # Amplitude anomalies
        z_scores = np.abs(zscore(current_scan['combined_amplitudes']))
        anomalous_indices = np.where(z_scores > 2.5)[0]
        
        for idx in anomalous_indices[:3]:  # Top 3
            freq = current_scan['frequencies'][idx]
            amp = current_scan['combined_amplitudes'][idx]
            anomalies.append(f"Amplitude spike at {freq:.1f} Hz: {amp:.3f}")
        
        # Spectral anomalies
        if len(self.fft_history) >= 5:
            recent_energies = [fft['total_energy'] for fft in list(self.fft_history)[-5:]]
            energy_mean = np.mean(recent_energies)
            energy_std = np.std(recent_energies)
            
            if energy_std > 0:
                energy_zscore = abs(current_fft['total_energy'] - energy_mean) / energy_std
                if energy_zscore > 2.0:
                    anomalies.append(f"Spectral energy anomaly: Z-score {energy_zscore:.2f}")
        
        return anomalies
    
    def update_live_plots(self, current_scan, current_fft, patterns, anomalies):
        """Update live plots with current data."""
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        freqs = current_scan['frequencies']
        
        # Plot 1: Current frequency spectrum
        self.axes[0, 0].plot(freqs, current_scan['audio_amplitudes'], 'b-', label='Audio', alpha=0.8)
        self.axes[0, 0].plot(freqs, current_scan['radio_amplitudes'], 'r-', label='Radio', alpha=0.8)
        self.axes[0, 0].plot(freqs, current_scan['combined_amplitudes'], 'g-', label='Combined', linewidth=2)
        self.axes[0, 0].set_xlabel('Frequency (Hz)')
        self.axes[0, 0].set_ylabel('Amplitude')
        self.axes[0, 0].set_title('Live Frequency Spectrum')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: FFT Magnitude
        if len(current_fft['fft_frequencies']) > 0:
            self.axes[0, 1].semilogy(current_fft['fft_frequencies'], current_fft['fft_magnitude'])
            self.axes[0, 1].set_xlabel('FFT Frequency')
            self.axes[0, 1].set_ylabel('Magnitude (log)')
            self.axes[0, 1].set_title('Live FFT Analysis')
            self.axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Spectral features over time
        if len(self.fft_history) > 1:
            timestamps = [fft['timestamp'] for fft in self.fft_history]
            centroids = [fft['spectral_centroid'] for fft in self.fft_history]
            bandwidths = [fft['spectral_bandwidth'] for fft in self.fft_history]
            
            time_indices = range(len(timestamps))
            self.axes[0, 2].plot(time_indices, centroids, 'b-', label='Centroid')
            self.axes[0, 2].plot(time_indices, bandwidths, 'r-', label='Bandwidth')
            self.axes[0, 2].set_xlabel('Time (scans)')
            self.axes[0, 2].set_ylabel('Frequency (Hz)')
            self.axes[0, 2].set_title('Spectral Features Over Time')
            self.axes[0, 2].legend()
            self.axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Enhanced Audio Peak Tracking (1kHz-10kHz)
        if len(self.audio_peak_history) > 1:
            # Track peak frequencies in each band
            low_band_peaks = []
            mid_band_peaks = []
            high_band_peaks = []
            time_indices = []
            
            for i, peak_data in enumerate(self.audio_peak_history):
                time_indices.append(i)
                
                # Count peaks in each band
                low_count = peak_data['band_distribution']['low']
                mid_count = peak_data['band_distribution']['mid']
                high_count = peak_data['band_distribution']['high']
                
                low_band_peaks.append(low_count)
                mid_band_peaks.append(mid_count)
                high_band_peaks.append(high_count)
            
            # Plot stacked area chart for peak distribution
            self.axes[1, 0].fill_between(time_indices, 0, low_band_peaks, 
                                       alpha=0.7, color='blue', label='Low (1-2.5kHz)')
            self.axes[1, 0].fill_between(time_indices, low_band_peaks, 
                                       np.array(low_band_peaks) + np.array(mid_band_peaks),
                                       alpha=0.7, color='green', label='Mid (2.5-5kHz)')
            self.axes[1, 0].fill_between(time_indices, 
                                       np.array(low_band_peaks) + np.array(mid_band_peaks),
                                       np.array(low_band_peaks) + np.array(mid_band_peaks) + np.array(high_band_peaks),
                                       alpha=0.7, color='red', label='High (5-10kHz)')
            
            self.axes[1, 0].set_xlabel('Scan Number')
            self.axes[1, 0].set_ylabel('Peak Count per Band')
            self.axes[1, 0].set_title('Audio Peak Distribution (1kHz-10kHz)')
            self.axes[1, 0].legend(fontsize=8)
            self.axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Dominant FFT frequencies
        if current_fft['dominant_frequencies']:
            freqs_to_plot = current_fft['dominant_frequencies'][:5]
            mags_to_plot = current_fft['dominant_magnitudes'][:5]
            
            self.axes[1, 1].bar(range(len(freqs_to_plot)), mags_to_plot, alpha=0.7)
            self.axes[1, 1].set_xlabel('Dominant Frequency Rank')
            self.axes[1, 1].set_ylabel('FFT Magnitude')
            self.axes[1, 1].set_title('Top Dominant Frequencies')
            
            # Add frequency labels
            for i, freq in enumerate(freqs_to_plot):
                self.axes[1, 1].text(i, mags_to_plot[i], f'{freq:.0f}Hz', 
                                   ha='center', va='bottom', fontsize=8)
            self.axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Enhanced Status and information
        self.axes[1, 2].text(0.05, 0.95, f'Scan Time: {current_scan["timestamp"].strftime("%H:%M:%S")}', 
                           transform=self.axes[1, 2].transAxes, fontsize=10, fontweight='bold')
        self.axes[1, 2].text(0.05, 0.88, f'Patterns: {len(patterns)} | Anomalies: {len(anomalies)}', 
                           transform=self.axes[1, 2].transAxes, fontsize=10)
        
        # Audio peak tracking info
        if self.audio_peak_history:
            latest_peaks = self.audio_peak_history[-1]
            total_audio_peaks = len(latest_peaks['peaks'])
            self.axes[1, 2].text(0.05, 0.81, f'Audio Peaks: {total_audio_peaks}', 
                               transform=self.axes[1, 2].transAxes, fontsize=10)
            
            # Band distribution
            bands = latest_peaks['band_distribution']
            self.axes[1, 2].text(0.05, 0.74, f'Low: {bands["low"]} | Mid: {bands["mid"]} | High: {bands["high"]}', 
                               transform=self.axes[1, 2].transAxes, fontsize=9)
        
        # Communication and network status
        recent_comm = len([p for p in self.silent_comm_patterns 
                          if (current_scan['timestamp'] - p['timestamp']).total_seconds() < 10])
        recent_network = len([p for p in self.layered_network_patterns 
                             if (current_scan['timestamp'] - p['timestamp']).total_seconds() < 10])
        
        comm_color = 'orange' if recent_comm > 0 else 'green'
        network_color = 'red' if recent_network > 0 else 'blue'
        
        self.axes[1, 2].text(0.05, 0.67, f'Comm: {recent_comm} | Network: {recent_network}', 
                           transform=self.axes[1, 2].transAxes, fontsize=10)
        
        # Show active network layers if any
        if self.layer_detection_history:
            latest_detection = self.layer_detection_history[-1]
            active_layers = len(latest_detection['layers_active'])
            leakage_indicators = len(latest_detection['leakage_indicators'])
            
            layer_text = f'Active Layers: {active_layers}'
            if leakage_indicators > 0:
                layer_text += f' | Leakage: {leakage_indicators}'
            
            layer_color = 'red' if leakage_indicators > 0 else 'blue' if active_layers > 0 else 'gray'
            self.axes[1, 2].text(0.05, 0.60, layer_text, 
                               transform=self.axes[1, 2].transAxes, fontsize=9, color=layer_color)
        
        # Display patterns (limited to fit)
        y_pos = 0.50
        for i, pattern in enumerate(patterns[:2]):
            if 'Silent Comm' in pattern:
                color = 'purple'
            elif 'Network' in pattern or 'Layer' in pattern:
                color = 'red'
            else:
                color = 'green'
            self.axes[1, 2].text(0.05, y_pos - i*0.06, f'• {pattern[:32]}{"..." if len(pattern) > 32 else ""}', 
                               transform=self.axes[1, 2].transAxes, fontsize=8, color=color)
        
        # Display anomalies
        y_pos = 0.36
        for i, anomaly in enumerate(anomalies[:2]):
            self.axes[1, 2].text(0.05, y_pos - i*0.06, f'⚠ {anomaly[:32]}{"..." if len(anomaly) > 32 else ""}', 
                               transform=self.axes[1, 2].transAxes, fontsize=8, color='red')
        
        self.axes[1, 2].set_title('Live Analysis Status & Peak Tracking')
        self.axes[1, 2].axis('off')
        
        # Update spectral features text
        if current_fft['spectral_centroid'] > 0:
            self.axes[1, 2].text(0.05, 0.25, f'Spectral Centroid: {current_fft["spectral_centroid"]:.0f} Hz', 
                               transform=self.axes[1, 2].transAxes, fontsize=8)
            self.axes[1, 2].text(0.05, 0.18, f'Spectral Bandwidth: {current_fft["spectral_bandwidth"]:.1f}', 
                               transform=self.axes[1, 2].transAxes, fontsize=8)
            self.axes[1, 2].text(0.05, 0.11, f'Total Energy: {current_fft["total_energy"]:.1f}', 
                               transform=self.axes[1, 2].transAxes, fontsize=8)
        
        # Peak tracking mode indicator
        mode_text = "Silent Mode: ON" if self.silent_mode else "Peak Tracking: ON"
        mode_color = "red" if self.silent_mode else "blue"
        self.axes[1, 2].text(0.05, 0.04, mode_text, 
                           transform=self.axes[1, 2].transAxes, fontsize=9, 
                           color=mode_color, fontweight='bold')
        
        plt.tight_layout()
        plt.pause(0.01)  # Small pause for plot update
    
    def start_live_scanning(self, duration_seconds=30, silent_mode=False):
        """Start live scanning with enhanced peak tracking and silent communication detection."""
        
        self.silent_mode = silent_mode
        mode_desc = "Silent Communication Mode" if silent_mode else "Peak Tracking Mode"
        
        print(f"\n🚀 Starting Enhanced 1kHz-10kHz Scanner with Audio Peak Tracking")
        print(f"   • Mode: {mode_desc}")
        print(f"   • Duration: {duration_seconds} seconds")
        print(f"   • Scan interval: {self.scan_interval*1000:.0f}ms")
        print(f"   • Audio peak tracking: ENABLED")
        print(f"   • Silent communication detection: ENABLED")
        print(f"   • Real-time plots updating...")
        print("   • Press Ctrl+C to stop early\n")
        
        self.running = True
        start_time = time.time()
        scan_count = 0
        
        try:
            while self.running and (time.time() - start_time) < duration_seconds:
                # Generate new scan
                current_scan = self.generate_live_scan()
                self.scan_data.append(current_scan)
                scan_count += 1
                
                # Perform FFT analysis
                current_fft = self.perform_fft_analysis(current_scan)
                self.fft_history.append(current_fft)
                
                # Detect layered network patterns
                layer_detection = self.detect_layered_network_patterns(current_scan, current_fft)
                
                # Perform deep network analysis
                deep_analysis = self.analyze_deep_network_patterns(current_scan, current_fft, layer_detection)
                
                # Store deep analysis results
                self.store_deep_analysis_results(deep_analysis)
                
                # Detect patterns and anomalies
                patterns = self.detect_patterns(current_scan, current_fft)
                anomalies = self.detect_anomalies(current_scan, current_fft)
                
                self.pattern_count += len(patterns)
                self.anomaly_count += len(anomalies)
                
                # Update live plots
                self.update_live_plots(current_scan, current_fft, patterns, anomalies)
                
                # Enhanced console output every 10 scans
                if scan_count % 10 == 0:
                    audio_peaks = len(self.audio_peak_history[-1]['peaks']) if self.audio_peak_history else 0
                    comm_patterns = len([p for p in self.silent_comm_patterns 
                                       if (current_scan['timestamp'] - p['timestamp']).total_seconds() < 5])
                    network_patterns = len([p for p in self.layered_network_patterns 
                                          if (current_scan['timestamp'] - p['timestamp']).total_seconds() < 10])
                    active_layers = len(layer_detection['layers_active']) if 'layer_detection' in locals() else 0
                    
                    # Deep analysis counters
                    protocols = len(self.deep_network_patterns['protocol_signatures'])
                    threats = len(self.deep_network_patterns['attack_patterns'])
                    covert = len(self.deep_network_patterns['covert_channels'])
                    
                    print(f"Scan #{scan_count} | Patterns: {self.pattern_count} | Anomalies: {self.anomaly_count} | Audio: {audio_peaks} | Comm: {comm_patterns} | Net: {network_patterns} | Layers: {active_layers} | Protocols: {protocols} | Threats: {threats} | Covert: {covert}")
                
                time.sleep(self.scan_interval)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Live scanning stopped by user")
        
        self.running = False
        runtime = time.time() - start_time
        
        # Enhanced completion summary
        total_audio_peaks = sum(len(p['peaks']) for p in self.audio_peak_history)
        total_comm_patterns = len(self.silent_comm_patterns)
        total_network_patterns = len(self.layered_network_patterns)
        
        print(f"\n✅ Enhanced live scanning complete!")
        print(f"   • Total scans: {scan_count}")
        print(f"   • Runtime: {runtime:.1f} seconds")
        print(f"   • Scan rate: {scan_count/runtime:.1f} scans/second")
        print(f"   • Total patterns detected: {self.pattern_count}")
        print(f"   • Total anomalies detected: {self.anomaly_count}")
        print(f"   • Total audio peaks tracked: {total_audio_peaks}")
        print(f"   • Silent communication patterns: {total_comm_patterns}")
        print(f"   • Layered network patterns: {total_network_patterns}")
        
        # Peak band distribution summary
        if self.audio_peak_history:
            total_low = sum(p['band_distribution']['low'] for p in self.audio_peak_history)
            total_mid = sum(p['band_distribution']['mid'] for p in self.audio_peak_history)
            total_high = sum(p['band_distribution']['high'] for p in self.audio_peak_history)
            print(f"   • Peak distribution - Low: {total_low}, Mid: {total_mid}, High: {total_high}")
        
        # Layered network analysis summary
        if self.layered_network_patterns:
            leakage_patterns = sum(1 for p in self.layered_network_patterns if p['leakage_indicators'])
            high_confidence_patterns = sum(1 for p in self.layered_network_patterns if p['confidence'] > 0.7)
            print(f"   • Network leakage detections: {leakage_patterns}")
            print(f"   • High confidence network patterns: {high_confidence_patterns}")
            
            # Layer activity summary
            layer_activity_counts = {}
            for pattern in self.layered_network_patterns:
                for layer in pattern['active_layers']:
                    layer_activity_counts[layer] = layer_activity_counts.get(layer, 0) + 1
            
            if layer_activity_counts:
                print(f"   • Most active network layers:")
                sorted_layers = sorted(layer_activity_counts.items(), key=lambda x: x[1], reverse=True)
                for layer, count in sorted_layers[:3]:
                    layer_name = layer.replace('_', ' ').title()
                    print(f"     - {layer_name}: {count} activations")
        
        # Deep network analysis summary
        print(f"\n🔍 DEEP NETWORK ANALYSIS SUMMARY:")
        
        # Protocol analysis
        detected_protocols = list(self.deep_network_patterns['protocol_signatures'].keys())
        if detected_protocols:
            print(f"   • Detected protocols: {', '.join(detected_protocols[:5])}")
        
        # Security analysis
        total_threats = len(self.deep_network_patterns['attack_patterns'])
        high_severity_threats = sum(1 for t in self.deep_network_patterns['attack_patterns']
                                   if t['threat']['threat_level'] > 0.7)
        if total_threats > 0:
            print(f"   • Security threats detected: {total_threats} (High severity: {high_severity_threats})")
        
        # Covert channels
        covert_channels = len(self.deep_network_patterns['covert_channels'])
        if covert_channels > 0:
            covert_types = set(c['channel']['type'] for c in self.deep_network_patterns['covert_channels'])
            print(f"   • Covert channels detected: {covert_channels} ({', '.join(covert_types)})")
        
        # Network topology
        if self.deep_network_patterns['network_topology']:
            topology_indicators = []
            for topo_list in self.deep_network_patterns['network_topology'].values():
                for topo in topo_list:
                    topology_indicators.extend(topo['analysis']['network_type_indicators'])
            
            if topology_indicators:
                unique_topologies = set(topology_indicators)
                print(f"   • Network topologies identified: {', '.join(unique_topologies)}")
        
        # Performance patterns
        qos_layers = list(self.deep_network_patterns['qos_patterns'].keys())
        if qos_layers:
            print(f"   • Performance monitoring on layers: {', '.join(qos_layers[:3])}")
        
        # Keep the final plot open
        plt.ioff()
        plt.show()
        
        return {
            'total_scans': scan_count,
            'runtime': runtime,
            'patterns': self.pattern_count,
            'anomalies': self.anomaly_count,
            'audio_peaks': total_audio_peaks,
            'communication_patterns': total_comm_patterns,
            'scan_data': list(self.scan_data),
            'fft_data': list(self.fft_history),
            'peak_data': list(self.audio_peak_history),
            'comm_data': self.silent_comm_patterns,
            'network_patterns': self.layered_network_patterns,
            'layer_signatures': dict(self.network_layer_signatures),
            'layer_detections': list(self.layer_detection_history),
            'deep_analysis': self.deep_network_patterns
        }


def create_comprehensive_graphs():
    """Generate comprehensive graphs from the CSV data."""
    
    print("=" * 70)
    print("📊 CSV FREQUENCY DATA GRAPH GENERATOR WITH LIVE SCANNING")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load the CSV data
    try:
        print("📁 Loading CSV data...")
        df = pd.read_csv('data/combined_frequency_analysis.csv')
        print(f"✓ Data loaded successfully!")
        print(f"  • Total data points: {len(df):,}")
        print(f"  • Frequency range: {df['frequency'].min():.0f} - {df['frequency'].max():.0f} Hz")
        print(f"  • Columns: {list(df.columns)}")
        print()
    except FileNotFoundError:
        print("❌ CSV file not found: data/combined_frequency_analysis.csv")
        return
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create the main comprehensive graph
    fig = plt.figure(figsize=(20, 16))
    
    # Create a grid layout for multiple subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Main Frequency Spectrum (Linear Scale)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df['frequency']/1000, df['audio_amplitude'], 
            label='Audio Signal', linewidth=2, alpha=0.8, color='#1f77b4')
    ax1.plot(df['frequency']/1000, df['radio_amplitude'], 
            label='Radio Signal', linewidth=2, alpha=0.8, color='#ff7f0e')
    ax1.plot(df['frequency']/1000, df['combined_amplitude'], 
            label='Combined Signal', linewidth=3, alpha=0.9, color='#2ca02c')
    
    ax1.set_xlabel('Frequency (kHz)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title('Complete Frequency Spectrum Analysis', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Logarithmic Scale Plot
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.semilogx(df['frequency'], df['combined_amplitude'], 
                linewidth=2, color='#d62728', alpha=0.8)
    ax2.set_xlabel('Frequency (Hz) - Log Scale', fontsize=10)
    ax2.set_ylabel('Combined Amplitude', fontsize=10)
    ax2.set_title('Log Scale View', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Amplitude Ratio Analysis
    ax3 = fig.add_subplot(gs[1, 0])
    # Clean up infinite and NaN values
    clean_ratio = df['amplitude_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    ax3.plot(df['frequency']/1000000, clean_ratio, 
            linewidth=2, color='#9467bd', alpha=0.8)
    ax3.set_xlabel('Frequency (MHz)', fontsize=10)
    ax3.set_ylabel('Audio/Radio Ratio', fontsize=10)
    ax3.set_title('Amplitude Ratio Analysis', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    # Limit y-axis to reasonable range
    ratio_95th = np.percentile(clean_ratio[clean_ratio > 0], 95) if len(clean_ratio[clean_ratio > 0]) > 0 else 1
    ax3.set_ylim(0, min(ratio_95th, 10))
    
    # 4. Signal Strength Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(df['combined_amplitude'], bins=50, alpha=0.7, color='#17becf', edgecolor='black')
    ax4.set_xlabel('Combined Amplitude', fontsize=10)
    ax4.set_ylabel('Frequency Count', fontsize=10)
    ax4.set_title('Amplitude Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Peak Detection Visualization
    ax5 = fig.add_subplot(gs[1, 2])
    # Find peaks in combined amplitude
    peaks, properties = find_peaks(
        df['combined_amplitude'], 
        height=df['combined_amplitude'].mean() + 0.5 * df['combined_amplitude'].std(),
        distance=max(1, len(df)//100)
    )
    
    ax5.plot(df['frequency']/1000000, df['combined_amplitude'], 
            linewidth=2, color='#bcbd22', alpha=0.8, label='Signal')
    if len(peaks) > 0:
        ax5.plot(df.iloc[peaks]['frequency']/1000000, df.iloc[peaks]['combined_amplitude'], 
                'ro', markersize=8, alpha=0.8, label=f'{len(peaks)} Peaks')
    ax5.set_xlabel('Frequency (MHz)', fontsize=10)
    ax5.set_ylabel('Combined Amplitude', fontsize=10)
    ax5.set_title('Peak Detection', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Frequency Band Analysis
    ax6 = fig.add_subplot(gs[2, :])
    
    # Define frequency bands
    bands = [
        ("Audio", 20, 20000, '#1f77b4'),
        ("LF", 30000, 300000, '#ff7f0e'),
        ("MF", 300000, 3000000, '#2ca02c'),
        ("HF", 3000000, 30000000, '#d62728'),
        ("VHF", 30000000, 300000000, '#9467bd'),
        ("UHF", 300000000, 1000000000, '#8c564b')
    ]
    
    band_centers = []
    band_amplitudes = []
    band_colors = []
    band_labels = []
    
    for band_name, low_freq, high_freq, color in bands:
        band_mask = (df['frequency'] >= low_freq) & (df['frequency'] <= high_freq)
        band_data = df[band_mask]
        
        if len(band_data) > 0:
            center_freq = np.log10((low_freq + high_freq) / 2)  # Log scale for x-axis
            avg_amplitude = band_data['combined_amplitude'].mean()
            max_amplitude = band_data['combined_amplitude'].max()
            
            band_centers.append(center_freq)
            band_amplitudes.append(avg_amplitude)
            band_colors.append(color)
            band_labels.append(f"{band_name}\n({len(band_data)} pts)")
            
            # Also plot max amplitude as a different marker
            ax6.scatter(center_freq, max_amplitude, s=150, c=color, marker='^', 
                       alpha=0.7, edgecolors='black', linewidth=1)
    
    if band_centers:
        # Plot average amplitudes
        ax6.scatter(band_centers, band_amplitudes, s=200, c=band_colors, 
                   alpha=0.8, edgecolors='black', linewidth=2, label='Average')
        
        # Connect with lines
        if len(band_centers) > 1:
            ax6.plot(band_centers, band_amplitudes, '--', alpha=0.5, color='gray', linewidth=2)
        
        # Add labels
        for i, (x, y, label) in enumerate(zip(band_centers, band_amplitudes, band_labels)):
            ax6.annotate(label, (x, y), xytext=(0, 20), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=band_colors[i], alpha=0.3))
    
    ax6.set_xlabel('Frequency Band (Log Scale)', fontsize=12)
    ax6.set_ylabel('Amplitude', fontsize=12)
    ax6.set_title('Frequency Band Analysis (Circles=Average, Triangles=Maximum)', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Comprehensive Network Frequency Analysis from CSV Data', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save the comprehensive graph
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'csv_comprehensive_graph_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Comprehensive graph saved as: {filename}")
    
    # Show the plot
    plt.show()
    
    # Generate detailed statistics
    generate_statistics_report(df, peaks)
    
    # Create additional focused graphs
    create_focused_graphs(df, timestamp)


def generate_statistics_report(df, peaks):
    """Generate detailed statistics from the CSV data."""
    
    print("\n" + "="*50)
    print("📊 DETAILED STATISTICS REPORT")
    print("="*50)
    
    # Basic statistics
    print(f"📈 DATASET OVERVIEW:")
    print(f"   • Total frequency points: {len(df):,}")
    print(f"   • Frequency range: {df['frequency'].min():.0f} - {df['frequency'].max():.0f} Hz")
    print(f"   • Frequency span: {(df['frequency'].max() - df['frequency'].min())/1000000:.2f} MHz")
    print()
    
    # Amplitude statistics
    print(f"🔊 AMPLITUDE ANALYSIS:")
    for col in ['audio_amplitude', 'radio_amplitude', 'combined_amplitude']:
        print(f"   {col.replace('_', ' ').title()}:")
        print(f"     - Mean: {df[col].mean():.6f}")
        print(f"     - Max:  {df[col].max():.6f}")
        print(f"     - Min:  {df[col].min():.6f}")
        print(f"     - Std:  {df[col].std():.6f}")
    print()
    
    # Peak analysis
    if len(peaks) > 0:
        print(f"🎯 PEAK ANALYSIS:")
        print(f"   • Total peaks detected: {len(peaks)}")
        print(f"   • Top 5 peaks by amplitude:")
        
        peak_data = []
        for peak_idx in peaks:
            freq = df.iloc[peak_idx]['frequency']
            amp = df.iloc[peak_idx]['combined_amplitude']
            peak_data.append({'frequency': freq, 'amplitude': amp, 'index': peak_idx})
        
        # Sort by amplitude
        peak_data.sort(key=lambda x: x['amplitude'], reverse=True)
        
        for i, peak in enumerate(peak_data[:5]):
            freq_display = f"{peak['frequency']/1000:.1f} kHz"
            if peak['frequency'] > 1000000:
                freq_display = f"{peak['frequency']/1000000:.2f} MHz"
            print(f"     {i+1}. {freq_display} - Amplitude: {peak['amplitude']:.6f}")
    print()
    
    # Frequency band distribution
    print(f"📡 FREQUENCY BAND DISTRIBUTION:")
    bands = [
        ("Audio (20Hz-20kHz)", 20, 20000),
        ("LF (30kHz-300kHz)", 30000, 300000),
        ("MF (300kHz-3MHz)", 300000, 3000000),
        ("HF (3MHz-30MHz)", 3000000, 30000000),
        ("VHF (30MHz-300MHz)", 30000000, 300000000),
        ("UHF (300MHz-1GHz)", 300000000, 1000000000)
    ]
    
    for band_name, low_freq, high_freq in bands:
        band_mask = (df['frequency'] >= low_freq) & (df['frequency'] <= high_freq)
        band_count = band_mask.sum()
        if band_count > 0:
            band_avg = df[band_mask]['combined_amplitude'].mean()
            print(f"   • {band_name}: {band_count} points (Avg amp: {band_avg:.4f})")


def create_focused_graphs(df, timestamp):
    """Create additional focused graphs for specific analysis."""
    
    print(f"\n🎯 Creating focused analysis graphs...")
    
    # 1. Audio vs Radio Comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(df['audio_amplitude'], df['radio_amplitude'], 
               alpha=0.6, s=20, c=df['frequency'], cmap='viridis')
    plt.xlabel('Audio Amplitude')
    plt.ylabel('Radio Amplitude')
    plt.title('Audio vs Radio Amplitude Correlation')
    plt.colorbar(label='Frequency (Hz)')
    plt.grid(True, alpha=0.3)
    
    # 2. Amplitude over frequency (zoomed)
    plt.subplot(2, 2, 2)
    # Focus on frequencies with significant amplitude
    high_amp_mask = df['combined_amplitude'] > df['combined_amplitude'].quantile(0.75)
    focus_df = df[high_amp_mask]
    
    plt.plot(focus_df['frequency']/1000000, focus_df['combined_amplitude'], 
            'o-', markersize=4, linewidth=1, alpha=0.7)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Combined Amplitude')
    plt.title('High Amplitude Frequencies (Top 25%)')
    plt.grid(True, alpha=0.3)
    
    # 3. Ratio analysis
    plt.subplot(2, 2, 3)
    clean_ratio = df['amplitude_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean_ratio) > 0:
        plt.hist(clean_ratio, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Audio/Radio Ratio')
        plt.ylabel('Count')
        plt.title('Amplitude Ratio Distribution')
        plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 4. Time-series like view (frequency as pseudo-time)
    plt.subplot(2, 2, 4)
    # Sample every Nth point for cleaner visualization
    step = max(1, len(df) // 1000)
    sample_df = df[::step]
    
    plt.plot(range(len(sample_df)), sample_df['combined_amplitude'], 
            linewidth=1, alpha=0.8)
    plt.xlabel('Data Point Index')
    plt.ylabel('Combined Amplitude')
    plt.title('Amplitude Progression Across Dataset')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save focused graphs
    focused_filename = f'csv_focused_analysis_{timestamp}.png'
    plt.savefig(focused_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Focused analysis graphs saved as: {focused_filename}")
    plt.show()


def main():
    """Main function with enhanced options for peak tracking and silent communication."""
    
    print("🎯 ENHANCED FREQUENCY ANALYSIS SYSTEM")
    print("Choose analysis mode:")
    print("1. Static CSV Analysis (comprehensive graphs from existing data)")
    print("2. Live 1kHz-10kHz Scanner with Audio Peak Tracking")
    print("3. Live Scanner with Silent Communication Detection")
    print("4. Both (static analysis first, then live scanning)")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n📊 Starting static CSV analysis...")
            create_comprehensive_graphs()
            
        elif choice == "2":
            print("\n🔄 Starting enhanced live frequency scanner with peak tracking...")
            duration = input("Enter scan duration in seconds (default 30): ").strip()
            duration = int(duration) if duration.isdigit() else 30
            
            scanner = LiveFrequencyScanner()
            results = scanner.start_live_scanning(duration_seconds=duration, silent_mode=False)
            
            # Optionally save live scan results
            save_choice = input("\nSave live scan data to CSV? (y/n): ").strip().lower()
            if save_choice == 'y':
                save_live_scan_data(results)
            
        elif choice == "3":
            print("\n🔄 Starting live scanner with silent communication detection...")
            duration = input("Enter scan duration in seconds (default 30): ").strip()
            duration = int(duration) if duration.isdigit() else 30
            
            scanner = LiveFrequencyScanner()
            results = scanner.start_live_scanning(duration_seconds=duration, silent_mode=True)
            
            # Optionally save live scan results
            save_choice = input("\nSave live scan data to CSV? (y/n): ").strip().lower()
            if save_choice == 'y':
                save_live_scan_data(results)
                
        elif choice == "4":
            print("\n📊 Starting static CSV analysis first...")
            create_comprehensive_graphs()
            
            input("\nPress Enter to continue to live scanning...")
            print("\n🔄 Starting enhanced live frequency scanner...")
            duration = input("Enter scan duration in seconds (default 30): ").strip()
            duration = int(duration) if duration.isdigit() else 30
            
            silent_choice = input("Enable silent communication mode? (y/n): ").strip().lower()
            silent_mode = silent_choice == 'y'
            
            scanner = LiveFrequencyScanner()
            results = scanner.start_live_scanning(duration_seconds=duration, silent_mode=silent_mode)
            
            # Optionally save live scan results
            save_choice = input("\nSave live scan data to CSV? (y/n): ").strip().lower()
            if save_choice == 'y':
                save_live_scan_data(results)
        else:
            print("Invalid choice. Running static analysis by default.")
            create_comprehensive_graphs()
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("Running static analysis as fallback.")
        create_comprehensive_graphs()


def save_live_scan_data(results):
    """Save enhanced live scan results including peak tracking and communication data."""
    
    if not results['scan_data']:
        print("No scan data to save.")
        return
    
    print("💾 Saving enhanced live scan data...")
    
    # Prepare data for CSV
    all_data = []
    scan_data = results['scan_data']
    fft_data = {fft['timestamp']: fft for fft in results['fft_data']}
    peak_data = {p['timestamp']: p for p in results.get('peak_data', [])}
    
    for scan in scan_data:
        corresponding_fft = fft_data.get(scan['timestamp'])
        corresponding_peaks = peak_data.get(scan['timestamp'])
        
        for i, freq in enumerate(scan['frequencies']):
            row = {
                'timestamp': scan['timestamp'],
                'frequency': freq,
                'audio_amplitude': scan['audio_amplitudes'][i],
                'radio_amplitude': scan['radio_amplitudes'][i],
                'combined_amplitude': scan['combined_amplitudes'][i]
            }
            
            # Add FFT data if available
            if corresponding_fft:
                row.update({
                    'spectral_centroid': corresponding_fft['spectral_centroid'],
                    'spectral_bandwidth': corresponding_fft['spectral_bandwidth'],
                    'total_energy': corresponding_fft['total_energy'],
                    'dominant_frequency_1': corresponding_fft['dominant_frequencies'][0] if corresponding_fft['dominant_frequencies'] else 0,
                    'dominant_frequency_2': corresponding_fft['dominant_frequencies'][1] if len(corresponding_fft['dominant_frequencies']) > 1 else 0,
                    'dominant_frequency_3': corresponding_fft['dominant_frequencies'][2] if len(corresponding_fft['dominant_frequencies']) > 2 else 0
                })
            
            # Add peak tracking data if available
            if corresponding_peaks:
                row.update({
                    'audio_peaks_total': len(corresponding_peaks['peaks']),
                    'audio_peaks_low_band': corresponding_peaks['band_distribution']['low'],
                    'audio_peaks_mid_band': corresponding_peaks['band_distribution']['mid'],
                    'audio_peaks_high_band': corresponding_peaks['band_distribution']['high']
                })
                
                # Add individual peak frequencies (up to 5)
                peak_freqs = [p['frequency'] for p in corresponding_peaks['peaks'][:5]]
                for j, peak_freq in enumerate(peak_freqs):
                    row[f'peak_frequency_{j+1}'] = peak_freq
                
                # Fill remaining peak columns with 0
                for j in range(len(peak_freqs), 5):
                    row[f'peak_frequency_{j+1}'] = 0
            
            all_data.append(row)
    
    # Save main scan data to CSV
    df = pd.DataFrame(all_data)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'live_scan_data_{timestamp}.csv'
    df.to_csv(filename, index=False)
    
    # Save communication patterns separately if they exist
    comm_data = results.get('comm_data', [])
    if comm_data:
        comm_df = pd.DataFrame(comm_data)
        comm_filename = f'communication_patterns_{timestamp}.csv'
        comm_df.to_csv(comm_filename, index=False)
        print(f"✅ Communication patterns saved to: {comm_filename}")
        print(f"   • Total patterns: {len(comm_df)}")
    
    # Save layered network patterns separately if they exist
    network_data = results.get('network_patterns', [])
    if network_data:
        network_df = pd.DataFrame(network_data)
        network_filename = f'layered_network_patterns_{timestamp}.csv'
        network_df.to_csv(network_filename, index=False)
        print(f"✅ Layered network patterns saved to: {network_filename}")
        print(f"   • Total network patterns: {len(network_df)}")
        
        # Save detailed layer signatures
        layer_signatures = results.get('layer_signatures', {})
        if layer_signatures:
            # Flatten layer signatures for CSV
            layer_signature_records = []
            for layer_name, signatures in layer_signatures.items():
                for sig in signatures:
                    record = {
                        'layer': layer_name,
                        'timestamp': sig['timestamp'],
                        'activity_score': sig['signature']['activity_score'],
                        'peak_amplitude': sig['signature']['peak_amplitude'],
                        'dominant_frequency': sig['signature']['dominant_frequency'],
                        'patterns': ', '.join(sig['patterns'])
                    }
                    layer_signature_records.append(record)
            
            if layer_signature_records:
                layer_sig_df = pd.DataFrame(layer_signature_records)
                layer_sig_filename = f'network_layer_signatures_{timestamp}.csv'
                layer_sig_df.to_csv(layer_sig_filename, index=False)
                print(f"✅ Network layer signatures saved to: {layer_sig_filename}")
                print(f"   • Total layer signatures: {len(layer_sig_df)}")
    
    # Save deep analysis data
    deep_analysis = results.get('deep_analysis', {})
    if deep_analysis:
        # Save protocol signatures
        protocol_data = []
        for protocol, entries in deep_analysis.get('protocol_signatures', {}).items():
            for entry in entries:
                protocol_data.append({
                    'protocol': protocol,
                    'timestamp': entry['timestamp'],
                    'confidence_data': str(entry['confidence']),
                    'behaviors': str(entry['behaviors'])
                })
        
        if protocol_data:
            protocol_df = pd.DataFrame(protocol_data)
            protocol_filename = f'detected_protocols_{timestamp}.csv'
            protocol_df.to_csv(protocol_filename, index=False)
            print(f"✅ Protocol signatures saved to: {protocol_filename}")
            print(f"   • Total protocol detections: {len(protocol_df)}")
        
        # Save security threats
        threat_data = []
        for entry in deep_analysis.get('attack_patterns', []):
            threat_info = entry['threat']
            threat_data.append({
                'timestamp': entry['timestamp'],
                'threat_type': threat_info['type'],
                'threat_level': threat_info['threat_level'],
                'frequency_range': str(threat_info.get('frequency_range', '')),
                'description': threat_info['description'],
                'peak_amplitude': threat_info.get('peak_amplitude', 0)
            })
        
        if threat_data:
            threat_df = pd.DataFrame(threat_data)
            threat_filename = f'security_threats_{timestamp}.csv'
            threat_df.to_csv(threat_filename, index=False)
            print(f"✅ Security threats saved to: {threat_filename}")
            print(f"   • Total threat detections: {len(threat_df)}")
        
        # Save covert channels
        covert_data = []
        for entry in deep_analysis.get('covert_channels', []):
            channel_info = entry['channel']
            covert_data.append({
                'timestamp': entry['timestamp'],
                'channel_type': channel_info['type'],
                'confidence': channel_info['confidence'],
                'description': channel_info['description'],
                'layers_involved': str(channel_info.get('layers_involved', []))
            })
        
        if covert_data:
            covert_df = pd.DataFrame(covert_data)
            covert_filename = f'covert_channels_{timestamp}.csv'
            covert_df.to_csv(covert_filename, index=False)
            print(f"✅ Covert channels saved to: {covert_filename}")
            print(f"   • Total covert channel detections: {len(covert_df)}")
    
    print(f"✅ Enhanced live scan data saved to: {filename}")
    print(f"   • Total records: {len(df):,}")
    print(f"   • Scans captured: {len(scan_data)}")
    print(f"   • Audio peaks tracked: {results.get('audio_peaks', 0)}")
    print(f"   • Communication patterns: {results.get('communication_patterns', 0)}")
    print(f"   • Network patterns: {len(network_data)}")
    print(f"   • Deep analysis data: {len(deep_analysis)} categories")
    print(f"   • Time span: {df['timestamp'].min()} to {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
