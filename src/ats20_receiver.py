"""
Hardware data collection for NETWORK frequency analysis.
Supports RTL-SDR for radio frequencies and audio input for acoustic data.
"""

import numpy as np
import pandas as pd
import time
from typing import Optional, Dict, List
import threading
import queue
import logging
import subprocess
import platform

# Try to import hardware libraries
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None

try:
    from rtlsdr import RtlSdr
    RTLSDR_AVAILABLE = True
except ImportError:
    RTLSDR_AVAILABLE = False
    RtlSdr = None

logger = logging.getLogger(__name__)


class HardwareDataCollector:
    """Collects frequency data from hardware sources."""

    def __init__(self, sample_rate: int = 44100, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = None
        self.sdr = None
        self.is_collecting = False
        self.data_queue = queue.Queue()
        self.collection_thread = None

        # Audio settings
        self.audio_format = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
        self.channels = 1
        self.audio_device_index = None

        # SDR settings
        self.sdr_center_freq = 100e6  # 100 MHz default
        self.sdr_sample_rate = 2.4e6  # 2.4 MS/s
        self.sdr_gain = 'auto'

    def initialize_audio(self, device_index: Optional[int] = None) -> bool:
        """Initialize audio input device."""
        if not PYAUDIO_AVAILABLE:
            logger.warning("PyAudio not available. Install with: pip install pyaudio")
            return False

        try:
            self.audio = pyaudio.PyAudio()

            # Find suitable input device
            if device_index is None:
                device_index = self.audio.get_default_input_device_info()['index']

            self.audio_device_index = device_index

            # Test device
            stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )
            stream.close()

            logger.info(f"Audio device initialized: {self.audio.get_device_info_by_index(device_index)['name']}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize audio device: {e}")
            return False

    def initialize_sdr(self, center_freq: float = 100e6, sample_rate: float = 2.4e6,
                      gain: str = 'auto') -> bool:
        """Initialize RTL-SDR device."""
        if not RTLSDR_AVAILABLE:
            logger.warning("RTL-SDR library not available. Install with: pip install rtlsdr")
            logger.info("For RTL-SDR support, also install librtlsdr system library")
            return False

        try:
            self.sdr = RtlSdr()
            self.sdr.center_freq = center_freq
            self.sdr.sample_rate = sample_rate
            self.sdr.gain = gain

            logger.info(f"RTL-SDR initialized: {center_freq/1e6:.1f} MHz center, {sample_rate/1e6:.1f} MS/s")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize RTL-SDR: {e}")
            return False

    def collect_audio_data(self, duration: float = 1.0) -> Optional[np.ndarray]:
        """Collect audio data for specified duration."""
        if not self.audio:
            logger.error("Audio device not initialized")
            return None

        try:
            stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.audio_device_index,
                frames_per_buffer=self.chunk_size
            )

            frames = []
            num_chunks = int(self.sample_rate / self.chunk_size * duration)

            for _ in range(num_chunks):
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(data)

            stream.stop_stream()
            stream.close()

            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

            return audio_data

        except Exception as e:
            logger.error(f"Audio data collection failed: {e}")
            return None

    def collect_sdr_data(self, duration: float = 1.0) -> Optional[np.ndarray]:
        """Collect SDR data for specified duration."""
        if not self.sdr:
            logger.error("RTL-SDR not initialized")
            return None

        try:
            # Read samples
            num_samples = int(self.sdr.sample_rate * duration)
            samples = self.sdr.read_samples(num_samples)

            return samples

        except Exception as e:
            logger.error(f"SDR data collection failed: {e}")
            return None

    def simulate_audio_data(self, duration: float = 1.0) -> np.ndarray:
        """Generate simulated audio data for testing."""
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples)

        # Generate complex audio signal with multiple frequencies
        signal = (
            0.5 * np.sin(2 * np.pi * 440 * t) +  # A4 note
            0.3 * np.sin(2 * np.pi * 880 * t) +  # A5 note
            0.2 * np.sin(2 * np.pi * 1320 * t) + # E6 note
            0.1 * np.random.normal(0, 0.1, num_samples)  # Noise
        )

        return signal

    def simulate_sdr_data(self, duration: float = 1.0) -> np.ndarray:
        """Generate simulated SDR IQ data for testing."""
        num_samples = int(2.4e6 * duration)  # 2.4 MS/s

        # Generate complex IQ samples
        t = np.linspace(0, duration, num_samples)
        carrier_freq = 2 * np.pi * 1e6  # 1 MHz modulation

        # AM modulated signal
        modulation = 0.5 * (1 + np.sin(2 * np.pi * 1000 * t))  # 1kHz modulation
        signal = modulation * np.exp(1j * carrier_freq * t)

        # Add some noise
        signal += 0.1 * (np.random.normal(0, 1, num_samples) + 1j * np.random.normal(0, 1, num_samples))

        return signal

    def audio_to_frequency_domain(self, audio_data: np.ndarray) -> pd.DataFrame:
        """Convert audio time-domain data to frequency domain."""
        # Apply FFT
        fft = np.fft.fft(audio_data)
        freqs = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)

        # Get positive frequencies only
        pos_mask = freqs > 0
        frequencies = freqs[pos_mask]
        amplitudes = np.abs(fft[pos_mask])

        # Convert to dB
        amplitudes_db = 20 * np.log10(amplitudes + 1e-10)

        data = pd.DataFrame({
            'frequency': frequencies,
            'amplitude': amplitudes_db,
            'data_type': 'audio_hw',
            'timestamp': pd.Timestamp.now()
        })

        return data

    def sdr_to_frequency_domain(self, sdr_data: np.ndarray) -> pd.DataFrame:
        """Convert SDR IQ data to frequency domain."""
        # Apply FFT to IQ samples
        fft = np.fft.fft(sdr_data)
        freqs = np.fft.fftfreq(len(sdr_data), 1/self.sdr_sample_rate)

        # Shift to center frequency
        center_freq = self.sdr.center_freq if self.sdr else self.sdr_center_freq
        freqs += center_freq

        # Get positive frequencies in reasonable range
        mask = (freqs > 0) & (freqs < 3e9)  # Up to 3 GHz
        frequencies = freqs[mask]
        amplitudes = np.abs(fft[mask])

        # Convert to dB
        amplitudes_db = 20 * np.log10(amplitudes + 1e-10)

        data = pd.DataFrame({
            'frequency': frequencies,
            'amplitude': amplitudes_db,
            'data_type': 'radio_hw',
            'timestamp': pd.Timestamp.now()
        })

        return data

    def collect_hardware_data(self, audio_duration: float = 1.0,
                            sdr_duration: float = 1.0) -> Dict[str, pd.DataFrame]:
        """Collect data from both audio and SDR hardware (or simulate if not available)."""
        results = {}

        # Collect audio data
        if self.audio:
            logger.info("Collecting audio data...")
            audio_time_data = self.collect_audio_data(audio_duration)
            if audio_time_data is not None:
                audio_freq_data = self.audio_to_frequency_domain(audio_time_data)
                results['audio'] = audio_freq_data
                logger.info(f"Collected {len(audio_freq_data)} audio frequency points")
        else:
            logger.info("Simulating audio data (hardware not available)...")
            audio_time_data = self.simulate_audio_data(audio_duration)
            audio_freq_data = self.audio_to_frequency_domain(audio_time_data)
            results['audio'] = audio_freq_data
            logger.info(f"Simulated {len(audio_freq_data)} audio frequency points")

        # Collect SDR data
        if self.sdr:
            logger.info("Collecting SDR data...")
            sdr_iq_data = self.collect_sdr_data(sdr_duration)
            if sdr_iq_data is not None:
                sdr_freq_data = self.sdr_to_frequency_domain(sdr_iq_data)
                results['radio'] = sdr_freq_data
                logger.info(f"Collected {len(sdr_freq_data)} radio frequency points")
        else:
            logger.info("Simulating SDR data (hardware not available)...")
            sdr_iq_data = self.simulate_sdr_data(sdr_duration)
            sdr_freq_data = self.sdr_to_frequency_domain(sdr_iq_data)
            results['radio'] = sdr_freq_data
            logger.info(f"Simulated {len(sdr_freq_data)} radio frequency points")

        return results

    def start_continuous_collection(self, callback=None):
        """Start continuous data collection in background thread."""
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._continuous_collect, args=(callback,))
        self.collection_thread.daemon = True
        self.collection_thread.start()

    def stop_continuous_collection(self):
        """Stop continuous data collection."""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2)

    def _continuous_collect(self, callback):
        """Continuous collection loop."""
        while self.is_collecting:
            try:
                data = self.collect_hardware_data(0.5, 0.5)  # 0.5s each
                if data:
                    self.data_queue.put(data)
                    if callback:
                        callback(data)
                time.sleep(0.1)  # Small delay
            except Exception as e:
                logger.error(f"Continuous collection error: {e}")
                time.sleep(1)

    def get_latest_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Get latest collected data from queue."""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

    def cleanup(self):
        """Clean up hardware resources."""
        if self.audio:
            self.audio.terminate()
        if self.sdr:
            self.sdr.close()

    def list_audio_devices(self) -> List[Dict]:
        """List available audio input devices."""
        devices = []
        if PYAUDIO_AVAILABLE and self.audio:
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'],
                        'rate': info['defaultSampleRate']
                    })
        elif platform.system() == 'Darwin':  # macOS
            # Try to list devices using system_profiler
            try:
                result = subprocess.run(['system_profiler', 'SPAudioDataType'],
                                      capture_output=True, text=True, timeout=5)
                if 'Input' in result.stdout:
                    devices.append({
                        'index': 0,
                        'name': 'Default Audio Input',
                        'channels': 1,
                        'rate': 44100
                    })
            except:
                pass
        return devices

    def list_sdr_devices(self) -> List[Dict]:
        """List available RTL-SDR devices."""
        if not RTLSDR_AVAILABLE:
            return []

        try:
            # Try to initialize to test
            test_sdr = RtlSdr()
            test_sdr.close()
            return [{'index': 0, 'name': 'RTL-SDR Device (Simulated)'}]
        except:
            return []


def main():
    """Test hardware data collection."""
    logging.basicConfig(level=logging.INFO)

    collector = HardwareDataCollector()

    print("NETWORK Hardware Data Collection Test")
    print("=" * 40)

    # Test audio
    print("\nTesting audio devices...")
    audio_devices = collector.list_audio_devices()
    if audio_devices:
        print(f"Found {len(audio_devices)} audio input device(s):")
        for dev in audio_devices:
            print(f"  {dev['index']}: {dev['name']}")

        if collector.initialize_audio():
            print("✓ Audio device initialized")
        else:
            print("⚠ Audio device initialization failed (using simulation)")
    else:
        print("No audio input devices found (using simulation)")

    # Test SDR
    print("\nTesting RTL-SDR devices...")
    sdr_devices = collector.list_sdr_devices()
    if sdr_devices:
        print(f"Found {len(sdr_devices)} SDR device(s):")
        for dev in sdr_devices:
            print(f"  {dev['index']}: {dev['name']}")

        if collector.initialize_sdr():
            print("✓ RTL-SDR initialized")
        else:
            print("⚠ RTL-SDR initialization failed (using simulation)")
    else:
        print("No RTL-SDR devices found (using simulation)")

    # Collect sample data
    print("\nCollecting sample data...")
    data = collector.collect_hardware_data(1.0, 1.0)

    for data_type, df in data.items():
        print(f"✓ Collected {len(df)} {data_type} frequency points")
        if len(df) > 0:
            print(f"  Frequency range: {df['frequency'].min()/1e3:.1f} - {df['frequency'].max()/1e3:.1f} kHz")
            print(f"  Max amplitude: {df['amplitude'].max():.1f} dB")

    collector.cleanup()
    print("\nHardware collection test complete")


if __name__ == "__main__":
    main()
