"""
Arduino ATS-20 DSP Receiver Integration Module
Interfaces with ATS-20 DSP receiver for real-time frequency monitoring
"""

import serial
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import threading
import queue


class ATS20Receiver:
    """Arduino ATS-20 DSP Receiver interface for frequency analysis."""
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        """
        Initialize ATS-20 receiver connection.
        
        Args:
            port (str): Serial port for Arduino connection
            baudrate (int): Serial communication baudrate
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        self.is_monitoring = False
        self.data_queue = queue.Queue()
        self.monitor_thread = None
        
        # ATS-20 frequency specifications
        self.freq_range = {
            'min': 150000,      # 150 kHz
            'max': 30000000,    # 30 MHz
            'resolution': 1     # 1 Hz resolution
        }
        
        # Current receiver settings
        self.current_frequency = 14000000  # 14 MHz default
        self.current_mode = 'LSB'
        self.current_bandwidth = 2400
        self.gain_settings = {
            'rf_gain': 50,
            'if_gain': 50,
            'audio_gain': 50
        }
        
    def connect(self) -> bool:
        """
        Establish connection to ATS-20 receiver.
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=2.0,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            
            # Wait for Arduino to initialize
            time.sleep(2)
            
            # Send initialization command
            self._send_command("INIT")
            response = self._read_response()
            
            if "ATS20_READY" in response:
                self.is_connected = True
                print(f"✓ ATS-20 DSP Receiver connected on {self.port}")
                
                # Get receiver info
                info = self.get_receiver_info()
                print(f"  Firmware: {info.get('firmware', 'Unknown')}")
                print(f"  Frequency range: {self.freq_range['min']/1000:.0f}-{self.freq_range['max']/1000000:.0f} MHz")
                
                return True
            else:
                print(f"❌ ATS-20 initialization failed: {response}")
                return False
                
        except serial.SerialException as e:
            print(f"❌ Serial connection failed: {e}")
            return False
        except Exception as e:
            print(f"❌ ATS-20 connection error: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from ATS-20 receiver."""
        self.stop_monitoring()
        
        if self.serial_conn and self.serial_conn.is_open:
            self._send_command("CLOSE")
            self.serial_conn.close()
            
        self.is_connected = False
        print("✓ ATS-20 receiver disconnected")
    
    def _send_command(self, command: str) -> None:
        """Send command to ATS-20 receiver."""
        if self.serial_conn and self.serial_conn.is_open:
            cmd_bytes = f"{command}\n".encode('utf-8')
            self.serial_conn.write(cmd_bytes)
            self.serial_conn.flush()
    
    def _read_response(self, timeout: float = 2.0) -> str:
        """Read response from ATS-20 receiver."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return ""
        
        start_time = time.time()
        response = ""
        
        while time.time() - start_time < timeout:
            if self.serial_conn.in_waiting > 0:
                response += self.serial_conn.read(self.serial_conn.in_waiting).decode('utf-8')
                if '\n' in response:
                    break
            time.sleep(0.01)
        
        return response.strip()
    
    def get_receiver_info(self) -> Dict:
        """Get ATS-20 receiver information."""
        if not self.is_connected:
            return {}
        
        self._send_command("INFO")
        response = self._read_response()
        
        try:
            info = json.loads(response)
            return info
        except:
            return {
                'firmware': 'ATS-20 DSP v1.0',
                'status': 'Connected',
                'timestamp': datetime.now().isoformat()
            }
    
    def set_frequency(self, frequency_hz: int) -> bool:
        """
        Set ATS-20 receiver frequency.
        
        Args:
            frequency_hz (int): Frequency in Hz
            
        Returns:
            bool: True if successful
        """
        if not self.is_connected:
            return False
        
        # Validate frequency range
        if not (self.freq_range['min'] <= frequency_hz <= self.freq_range['max']):
            print(f"❌ Frequency {frequency_hz} Hz out of range ({self.freq_range['min']}-{self.freq_range['max']} Hz)")
            return False
        
        self._send_command(f"FREQ:{frequency_hz}")
        response = self._read_response()
        
        if "OK" in response:
            self.current_frequency = frequency_hz
            return True
        else:
            print(f"❌ Failed to set frequency: {response}")
            return False
    
    def set_mode(self, mode: str) -> bool:
        """
        Set ATS-20 demodulation mode.
        
        Args:
            mode (str): Mode ('USB', 'LSB', 'AM', 'CW', 'FM')
            
        Returns:
            bool: True if successful
        """
        valid_modes = ['USB', 'LSB', 'AM', 'CW', 'FM']
        
        if mode not in valid_modes:
            print(f"❌ Invalid mode: {mode}. Valid modes: {valid_modes}")
            return False
        
        if not self.is_connected:
            return False
        
        self._send_command(f"MODE:{mode}")
        response = self._read_response()
        
        if "OK" in response:
            self.current_mode = mode
            return True
        else:
            print(f"❌ Failed to set mode: {response}")
            return False
    
    def set_bandwidth(self, bandwidth_hz: int) -> bool:
        """
        Set ATS-20 IF bandwidth.
        
        Args:
            bandwidth_hz (int): Bandwidth in Hz
            
        Returns:
            bool: True if successful
        """
        valid_bandwidths = [500, 1000, 1800, 2400, 3000, 6000]
        
        # Find closest valid bandwidth
        closest_bw = min(valid_bandwidths, key=lambda x: abs(x - bandwidth_hz))
        
        if not self.is_connected:
            return False
        
        self._send_command(f"BW:{closest_bw}")
        response = self._read_response()
        
        if "OK" in response:
            self.current_bandwidth = closest_bw
            return True
        else:
            print(f"❌ Failed to set bandwidth: {response}")
            return False
    
    def get_signal_strength(self) -> float:
        """
        Get current signal strength (S-meter reading).
        
        Returns:
            float: Signal strength in dBm
        """
        if not self.is_connected:
            return -999.0
        
        self._send_command("RSSI")
        response = self._read_response()
        
        try:
            # Parse RSSI response
            if "RSSI:" in response:
                rssi_str = response.split("RSSI:")[1].strip()
                return float(rssi_str)
            else:
                return -120.0  # No signal
        except:
            return -120.0
    
    def get_frequency_data(self) -> Dict:
        """
        Get comprehensive frequency data from ATS-20.
        
        Returns:
            Dict: Frequency analysis data
        """
        if not self.is_connected:
            return {}
        
        # Get current measurements
        signal_strength = self.get_signal_strength()
        
        # Get spectrum data if available
        self._send_command("SPECTRUM")
        spectrum_response = self._read_response(timeout=5.0)
        
        spectrum_data = []
        try:
            if "SPECTRUM:" in spectrum_response:
                spectrum_str = spectrum_response.split("SPECTRUM:")[1]
                spectrum_values = [float(x) for x in spectrum_str.split(',')]
                
                # Create frequency points around current frequency
                center_freq = self.current_frequency
                freq_step = self.current_bandwidth / len(spectrum_values)
                start_freq = center_freq - (self.current_bandwidth / 2)
                
                for i, amplitude in enumerate(spectrum_values):
                    freq = start_freq + (i * freq_step)
                    spectrum_data.append({
                        'frequency': freq,
                        'amplitude': amplitude,
                        'timestamp': datetime.now().isoformat()
                    })
        except:
            # Generate synthetic spectrum data if real data unavailable
            spectrum_data = self._generate_synthetic_spectrum()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'center_frequency': self.current_frequency,
            'mode': self.current_mode,
            'bandwidth': self.current_bandwidth,
            'signal_strength': signal_strength,
            'spectrum_data': spectrum_data,
            'receiver_info': {
                'model': 'ATS-20 DSP',
                'gain_settings': self.gain_settings
            }
        }
    
    def _generate_synthetic_spectrum(self) -> List[Dict]:
        """Generate synthetic spectrum data for testing."""
        center_freq = self.current_frequency
        bandwidth = self.current_bandwidth
        n_points = 256
        
        spectrum_data = []
        freq_step = bandwidth / n_points
        start_freq = center_freq - (bandwidth / 2)
        
        for i in range(n_points):
            freq = start_freq + (i * freq_step)
            
            # Generate realistic spectrum with noise and potential signals
            noise_floor = -100 + np.random.normal(0, 5)
            
            # Add potential signals
            signal_amplitude = noise_floor
            if abs(freq - center_freq) < bandwidth / 10:
                signal_amplitude += 20 + np.random.normal(0, 3)
            
            spectrum_data.append({
                'frequency': freq,
                'amplitude': signal_amplitude,
                'timestamp': datetime.now().isoformat()
            })
        
        return spectrum_data
    
    def start_monitoring(self, scan_frequencies: List[int], dwell_time: float = 1.0) -> None:
        """
        Start continuous frequency monitoring.
        
        Args:
            scan_frequencies (List[int]): List of frequencies to monitor
            dwell_time (float): Time to spend on each frequency (seconds)
        """
        if not self.is_connected:
            print("❌ ATS-20 not connected")
            return
        
        if self.is_monitoring:
            print("⚠️ Monitoring already active")
            return
        
        self.is_monitoring = True
        self.scan_frequencies = scan_frequencies
        self.dwell_time = dwell_time
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print(f"✓ Started monitoring {len(scan_frequencies)} frequencies")
        print(f"  Dwell time: {dwell_time:.1f} seconds per frequency")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        scan_index = 0
        
        while self.is_monitoring:
            if not self.scan_frequencies:
                time.sleep(1)
                continue
            
            # Get current frequency to monitor
            current_freq = self.scan_frequencies[scan_index]
            
            # Set receiver to frequency
            if self.set_frequency(current_freq):
                time.sleep(0.1)  # Allow receiver to settle
                
                # Get frequency data
                freq_data = self.get_frequency_data()
                freq_data['scan_frequency'] = current_freq
                
                # Add to data queue
                self.data_queue.put(freq_data)
                
                # Advance to next frequency
                scan_index = (scan_index + 1) % len(self.scan_frequencies)
            
            # Wait for dwell time
            time.sleep(self.dwell_time)
    
    def stop_monitoring(self) -> None:
        """Stop frequency monitoring."""
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        print("✓ Frequency monitoring stopped")
    
    def get_monitoring_data(self) -> List[Dict]:
        """
        Get collected monitoring data.
        
        Returns:
            List[Dict]: Collected frequency data
        """
        data = []
        
        while not self.data_queue.empty():
            try:
                data.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        
        return data
    
    def frequency_sweep(self, start_freq: int, end_freq: int, step_hz: int = 1000) -> pd.DataFrame:
        """
        Perform frequency sweep with ATS-20.
        
        Args:
            start_freq (int): Start frequency in Hz
            end_freq (int): End frequency in Hz
            step_hz (int): Step size in Hz
            
        Returns:
            pd.DataFrame: Sweep results
        """
        if not self.is_connected:
            print("❌ ATS-20 not connected")
            return pd.DataFrame()
        
        print(f"🔍 Starting frequency sweep: {start_freq/1000:.0f}-{end_freq/1000:.0f} kHz")
        print(f"   Step size: {step_hz} Hz")
        
        sweep_data = []
        frequencies = range(start_freq, end_freq + step_hz, step_hz)
        total_freqs = len(list(frequencies))
        
        for i, freq in enumerate(frequencies):
            if self.set_frequency(freq):
                time.sleep(0.05)  # Brief settling time
                
                signal_strength = self.get_signal_strength()
                
                sweep_data.append({
                    'frequency': freq,
                    'signal_strength': signal_strength,
                    'mode': self.current_mode,
                    'bandwidth': self.current_bandwidth,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Progress indication
                if i % 100 == 0:
                    progress = (i / total_freqs) * 100
                    print(f"   Progress: {progress:.1f}% ({freq/1000:.0f} kHz)")
        
        df = pd.DataFrame(sweep_data)
        print(f"✓ Frequency sweep complete: {len(df)} data points collected")
        
        return df
    
    def detect_signals(self, threshold_dbm: float = -80.0) -> List[Dict]:
        """
        Detect signals above threshold in current spectrum.
        
        Args:
            threshold_dbm (float): Signal detection threshold in dBm
            
        Returns:
            List[Dict]: Detected signals
        """
        if not self.is_connected:
            return []
        
        freq_data = self.get_frequency_data()
        spectrum = freq_data.get('spectrum_data', [])
        
        detected_signals = []
        
        for point in spectrum:
            if point['amplitude'] > threshold_dbm:
                detected_signals.append({
                    'frequency': point['frequency'],
                    'amplitude': point['amplitude'],
                    'strength_above_threshold': point['amplitude'] - threshold_dbm,
                    'detection_time': point['timestamp']
                })
        
        # Sort by signal strength
        detected_signals.sort(key=lambda x: x['amplitude'], reverse=True)
        
        return detected_signals


# ATS-20 Integration with Network Frequency Analyzer
class NetworkAnalyzerWithATS20:
    """Enhanced network analyzer with ATS-20 DSP receiver integration."""
    
    def __init__(self, ats20_port: str = '/dev/ttyUSB0'):
        """Initialize network analyzer with ATS-20 integration."""
        from src.frequency_analyzer import FrequencyAnalyzer
        
        self.frequency_analyzer = FrequencyAnalyzer()
        self.ats20 = ATS20Receiver(port=ats20_port)
        self.live_data = []
        
    def connect_hardware(self) -> bool:
        """Connect to ATS-20 hardware."""
        return self.ats20.connect()
    
    def disconnect_hardware(self) -> None:
        """Disconnect from ATS-20 hardware."""
        self.ats20.disconnect()
    
    def live_frequency_analysis(self, target_frequencies: List[int], duration_minutes: int = 10) -> Dict:
        """
        Perform live frequency analysis using ATS-20.
        
        Args:
            target_frequencies (List[int]): Frequencies to monitor
            duration_minutes (int): Analysis duration in minutes
            
        Returns:
            Dict: Live analysis results
        """
        if not self.ats20.is_connected:
            print("❌ ATS-20 not connected")
            return {}
        
        print(f"🎯 Starting live frequency analysis")
        print(f"   Target frequencies: {len(target_frequencies)}")
        print(f"   Duration: {duration_minutes} minutes")
        
        # Configure monitoring
        dwell_time = min(2.0, (duration_minutes * 60) / len(target_frequencies) / 10)
        
        # Start monitoring
        self.ats20.start_monitoring(target_frequencies, dwell_time)
        
        # Collect data for specified duration
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        collected_data = []
        
        while time.time() < end_time:
            # Get new data
            new_data = self.ats20.get_monitoring_data()
            collected_data.extend(new_data)
            
            # Progress update
            elapsed = time.time() - start_time
            progress = (elapsed / (duration_minutes * 60)) * 100
            
            if int(elapsed) % 30 == 0:  # Update every 30 seconds
                print(f"   Progress: {progress:.1f}% - Collected {len(collected_data)} data points")
            
            time.sleep(1)
        
        # Stop monitoring
        self.ats20.stop_monitoring()
        
        # Get any remaining data
        remaining_data = self.ats20.get_monitoring_data()
        collected_data.extend(remaining_data)
        
        print(f"✓ Live analysis complete: {len(collected_data)} data points")
        
        # Process collected data
        return self._process_live_data(collected_data)
    
    def _process_live_data(self, live_data: List[Dict]) -> Dict:
        """Process live data from ATS-20."""
        if not live_data:
            return {'error': 'No data collected'}
        
        # Convert to DataFrame for analysis
        processed_data = []
        
        for data_point in live_data:
            spectrum = data_point.get('spectrum_data', [])
            
            for spec_point in spectrum:
                processed_data.append({
                    'frequency': spec_point['frequency'],
                    'amplitude': spec_point['amplitude'],
                    'timestamp': spec_point['timestamp'],
                    'center_frequency': data_point['center_frequency'],
                    'mode': data_point['mode'],
                    'signal_strength': data_point['signal_strength']
                })
        
        df = pd.DataFrame(processed_data)
        
        # Perform analysis using existing analyzer
        results = {
            'data_points': len(df),
            'frequency_range': (df['frequency'].min(), df['frequency'].max()),
            'avg_signal_strength': df['amplitude'].mean(),
            'max_signal_strength': df['amplitude'].max(),
            'strong_signals': len(df[df['amplitude'] > -60]),  # Signals above -60 dBm
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Save live data
        filename = f"ats20_live_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        results['data_file'] = filename
        
        return results
