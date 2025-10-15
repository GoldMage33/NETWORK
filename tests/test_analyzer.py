"""
Unit tests for the Network Frequency Analysis Tool.
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.frequency_analyzer import FrequencyAnalyzer
from src.data_loader import DataLoader
from src.signal_processor import SignalProcessor
from src.layer_detector import LayerDetector


class TestFrequencyAnalyzer(unittest.TestCase):
    """Test cases for FrequencyAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FrequencyAnalyzer(resolution_hz=1.0)

    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.resolution_hz, 1.0)
        self.assertIsNone(self.analyzer.audio_data)
        self.assertIsNone(self.analyzer.radio_data)
        self.assertIsNone(self.analyzer.combined_data)

    def test_load_sample_data(self):
        """Test loading sample data when files don't exist."""
        # This should generate sample data
        self.analyzer.load_audio_frequencies('nonexistent_audio.csv')
        self.analyzer.load_radio_frequencies('nonexistent_radio.csv')

        self.assertIsNotNone(self.analyzer.audio_data)
        self.assertIsNotNone(self.analyzer.radio_data)
        self.assertTrue(len(self.analyzer.audio_data) > 0)
        self.assertTrue(len(self.analyzer.radio_data) > 0)

    def test_combine_frequency_data(self):
        """Test frequency data combination."""
        # Load sample data
        self.analyzer.load_audio_frequencies('test_audio.csv')
        self.analyzer.load_radio_frequencies('test_radio.csv')

        # Combine data
        combined = self.analyzer.combine_frequency_data()

        self.assertIsNotNone(combined)
        self.assertTrue('frequency' in combined.columns)
        self.assertTrue('audio_amplitude' in combined.columns)
        self.assertTrue('radio_amplitude' in combined.columns)
        self.assertTrue('combined_amplitude' in combined.columns)

    def test_detect_layer_anomalies(self):
        """Test layer anomaly detection."""
        # Load sample data
        self.analyzer.load_audio_frequencies('test_audio.csv')
        self.analyzer.load_radio_frequencies('test_radio.csv')

        # Run analysis
        results = self.analyzer.detect_layer_anomalies()

        # Check result structure
        self.assertIn('leakage_points', results)
        self.assertIn('obscured_layers', results)
        self.assertIn('frequency_correlations', results)
        self.assertIn('network_topology', results)
        self.assertIn('anomaly_score', results)

        # Check data types
        self.assertIsInstance(results['leakage_points'], list)
        self.assertIsInstance(results['obscured_layers'], list)
        self.assertIsInstance(results['frequency_correlations'], dict)
        self.assertIsInstance(results['network_topology'], dict)
        self.assertIsInstance(results['anomaly_score'], (int, float))

        # Check anomaly score range
        self.assertTrue(0 <= results['anomaly_score'] <= 1)


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = DataLoader()

    def test_generate_sample_audio_data(self):
        """Test audio sample data generation."""
        data = self.loader._generate_sample_data('audio', n_samples=100)

        self.assertEqual(len(data), 100)
        self.assertTrue('frequency' in data.columns)
        self.assertTrue('amplitude' in data.columns)
        self.assertTrue('data_type' in data.columns)
        self.assertEqual(data['data_type'].iloc[0], 'audio')

        # Check frequency range for audio (allow small floating point tolerance)
        self.assertTrue(data['frequency'].min() >= 20)
        self.assertTrue(data['frequency'].max() <= 20001)  # Allow small tolerance for floating point

    def test_generate_sample_radio_data(self):
        """Test radio sample data generation."""
        data = self.loader._generate_sample_data('radio', n_samples=100)

        self.assertEqual(len(data), 100)
        self.assertTrue('frequency' in data.columns)
        self.assertTrue('amplitude' in data.columns)
        self.assertTrue('data_type' in data.columns)
        self.assertEqual(data['data_type'].iloc[0], 'radio')

        # Check frequency range for radio (allow small floating point tolerance)
        self.assertTrue(data['frequency'].min() >= 3000)
        self.assertTrue(data['frequency'].max() <= 3e9 + 1)  # Allow small tolerance for floating point


class TestSignalProcessor(unittest.TestCase):
    """Test cases for SignalProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = SignalProcessor(resolution_hz=1.0)

        # Create sample data
        frequencies = np.linspace(0, 1000, 1000)
        audio_amplitudes = np.sin(2 * np.pi * frequencies / 100) + 0.1 * np.random.randn(1000)
        radio_amplitudes = np.cos(2 * np.pi * frequencies / 200) + 0.1 * np.random.randn(1000)

        self.audio_data = pd.DataFrame({
            'frequency': frequencies,
            'amplitude': np.abs(audio_amplitudes),
            'data_type': 'audio'
        })

        self.radio_data = pd.DataFrame({
            'frequency': frequencies,
            'amplitude': np.abs(radio_amplitudes),
            'data_type': 'radio'
        })

    def test_combine_frequencies(self):
        """Test frequency combination."""
        combined = self.processor.combine_frequencies(self.audio_data, self.radio_data)

        self.assertTrue('frequency' in combined.columns)
        self.assertTrue('audio_amplitude' in combined.columns)
        self.assertTrue('radio_amplitude' in combined.columns)
        self.assertTrue('combined_amplitude' in combined.columns)
        self.assertTrue('amplitude_ratio' in combined.columns)

    def test_preprocess_signals(self):
        """Test signal preprocessing."""
        combined = self.processor.combine_frequencies(self.audio_data, self.radio_data)
        processed = self.processor.preprocess_signals(combined)

        # Check for smoothed signals
        self.assertTrue('audio_smooth' in processed.columns)
        self.assertTrue('radio_smooth' in processed.columns)

        # Check for derivatives
        self.assertTrue('audio_derivative' in processed.columns)
        self.assertTrue('radio_derivative' in processed.columns)

        # Check for spectral features
        self.assertTrue('spectral_centroid' in processed.columns)
        self.assertTrue('spectral_bandwidth' in processed.columns)

        # Check for peak detection
        self.assertTrue('is_peak' in processed.columns)


class TestLayerDetector(unittest.TestCase):
    """Test cases for LayerDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = LayerDetector()

        # Create sample processed data
        n_samples = 500
        frequencies = np.linspace(0, 1000, n_samples)

        # Create synthetic data with some anomalies
        normal_amplitude = 0.1 + 0.05 * np.random.randn(n_samples)
        anomaly_indices = [100, 200, 300]

        for idx in anomaly_indices:
            normal_amplitude[idx] += 1.0  # Add strong anomalies

        self.test_data = pd.DataFrame({
            'frequency': frequencies,
            'combined_amplitude': normal_amplitude,
            'audio_amplitude': normal_amplitude * 0.6,
            'radio_amplitude': normal_amplitude * 0.4,
            'spectral_centroid': frequencies + 0.1 * np.random.randn(n_samples),
            'spectral_bandwidth': 50 + 10 * np.random.randn(n_samples),
            'is_peak': [i in anomaly_indices for i in range(n_samples)]
        })

    def test_detect_leakage(self):
        """Test leakage detection."""
        leakage_points = self.detector.detect_leakage(self.test_data)

        self.assertIsInstance(leakage_points, list)

        # Should detect some leakage points
        self.assertTrue(len(leakage_points) > 0)

        # Check structure of leakage points
        if leakage_points:
            point = leakage_points[0]
            self.assertIn('frequency', point)
            self.assertIn('strength', point)
            self.assertIn('detection_methods', point)

    def test_calculate_anomaly_score(self):
        """Test anomaly score calculation."""
        score = self.detector.calculate_anomaly_score(self.test_data)

        self.assertIsInstance(score, float)
        self.assertTrue(0 <= score <= 1)

    def test_infer_topology(self):
        """Test network topology inference."""
        topology = self.detector.infer_topology(self.test_data)

        self.assertIsInstance(topology, dict)
        self.assertIn('layer_count', topology)
        self.assertIn('connectivity', topology)
        self.assertIn('complexity_score', topology)


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)
