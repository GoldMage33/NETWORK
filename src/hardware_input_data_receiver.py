"""
Hardware data collection for NETWORK frequency analysis.
Supports audio input for acoustic data.
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

logger = logging.getLogger(__name__)


class HardwareDataCollector:
    """Collects frequency data from hardware sources."""

    def __init__(self, sample_rate: int = 44100, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = None
        self.is_collecting = False
        self.data_queue = queue.Queue()
        self.collection_thread = None

        # Audio settings
        self.audio_format = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
        self.channels = 1
        self.audio_device_index = None

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

    def audio_to_radio_domain(self, audio_freq_data: pd.DataFrame) -> pd.DataFrame:
        """Convert audio frequency data to simulate radio frequency characteristics."""
        if audio_freq_data.empty:
            return pd.DataFrame()

        # Create a copy of the audio data
        radio_data = audio_freq_data.copy()

        # Shift frequencies to radio range (20kHz to 1MHz)
        # Audio covers 0Hz-20kHz, we'll map this to radio frequencies
        freq_range = radio_data['frequency'].max() - radio_data['frequency'].min()
        radio_base_freq = 20000  # 20 kHz base frequency

        # Scale and shift frequencies to radio range (20kHz to 1MHz = 980kHz spread)
        # Map audio range (0-20kHz) to radio range (20kHz-1MHz)
        audio_range = radio_data['frequency'].max() - radio_data['frequency'].min()  # 20000 Hz
        radio_range = 1000000 - 20000  # 980000 Hz
        scale_factor = radio_range / audio_range if audio_range > 0 else 1.0
        
        radio_data['frequency'] = radio_base_freq + (radio_data['frequency'] - radio_data['frequency'].min()) * scale_factor

        # Adjust magnitude to simulate radio signal characteristics
        # Radio signals typically have different amplitude distributions
        radio_data['magnitude'] = radio_data['amplitude'] * np.random.uniform(0.1, 2.0, len(radio_data))

        # Add some noise to simulate radio interference patterns
        noise = np.random.normal(0, 0.1, len(radio_data))
        radio_data['magnitude'] = np.maximum(0, radio_data['magnitude'] + noise)

        # Update data type and add radio-specific columns
        radio_data['data_type'] = 'radio_from_audio'
        radio_data['amplitude'] = radio_data['magnitude']

        return radio_data

    def collect_hardware_data(self, audio_duration: float = 1.0) -> Dict[str, pd.DataFrame]:
        """Collect data from audio hardware (use microphone for both audio and radio simulation)."""
        results = {}

        # Always try to initialize audio if PyAudio is available
        audio_initialized = False
        if PYAUDIO_AVAILABLE:
            try:
                self.audio = pyaudio.PyAudio()
                audio_initialized = True
                logger.info("Audio system initialized for microphone input")
            except Exception as e:
                logger.warning(f"Could not initialize audio system: {e}")

        # Collect audio data (always try microphone first)
        if audio_initialized and self.initialize_audio():
            logger.info("Collecting audio data from microphone...")
            audio_time_data = self.collect_audio_data(audio_duration)
            if audio_time_data is not None:
                audio_freq_data = self.audio_to_frequency_domain(audio_time_data)
                results['audio'] = audio_freq_data
                logger.info(f"Collected {len(audio_freq_data)} audio frequency points from microphone")
            else:
                logger.warning("Microphone data collection failed, using simulation")
                audio_time_data = self.simulate_audio_data(audio_duration)
                audio_freq_data = self.audio_to_frequency_domain(audio_time_data)
                results['audio'] = audio_freq_data
                logger.info(f"Simulated {len(audio_freq_data)} audio frequency points")
        else:
            logger.info("Microphone not available, simulating audio data...")
            audio_time_data = self.simulate_audio_data(audio_duration)
            audio_freq_data = self.audio_to_frequency_domain(audio_time_data)
            results['audio'] = audio_freq_data
            logger.info(f"Simulated {len(audio_freq_data)} audio frequency points")

        # Use microphone data to simulate radio characteristics
        logger.info("Using microphone for radio frequency simulation...")
        if 'audio' in results:
            # Use the collected audio data and process it to simulate radio characteristics
            radio_freq_data = self.audio_to_radio_domain(results['audio'])
            results['radio'] = radio_freq_data
            logger.info(f"Generated {len(radio_freq_data)} radio frequency points from microphone")
        else:
            # No audio data available, simulate radio data
            logger.info("No audio data available, simulating radio data...")
            # Create simulated radio data based on audio simulation
            radio_time_data = self.simulate_audio_data(audio_duration)
            radio_freq_data = self.audio_to_frequency_domain(radio_time_data)
            radio_freq_data = self.audio_to_radio_domain(radio_freq_data)
            results['radio'] = radio_freq_data
            logger.info(f"Simulated {len(radio_freq_data)} radio frequency points")

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
    print("\nRTL-SDR support removed - using audio simulation for radio data")

    # Collect sample data
    print("\nCollecting sample data...")
    data = collector.collect_hardware_data(1.0)

    for data_type, df in data.items():
        print(f"✓ Collected {len(df)} {data_type} frequency points")
        if len(df) > 0:
            print(f"  Frequency range: {df['frequency'].min()/1e3:.1f} - {df['frequency'].max()/1e3:.1f} kHz")
            print(f"  Max amplitude: {df['amplitude'].max():.1f} dB")

    collector.cleanup()
    print("\nHardware collection test complete")


if __name__ == "__main__":
    main()
