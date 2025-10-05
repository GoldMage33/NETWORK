"""
Signal processing functions for frequency analysis and layer detection.
"""

import numpy as np
import pandas as pd
from scipy import signal, fft
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


class SignalProcessor:
    """Advanced signal processing for frequency analysis."""
    
    def __init__(self, resolution_hz: float = 1.0):
        """
        Initialize signal processor.
        
        Args:
            resolution_hz (float): Frequency resolution in Hz
        """
        self.resolution_hz = resolution_hz
        self.scaler = StandardScaler()
        
    def combine_frequencies(self, audio_data: pd.DataFrame, radio_data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine audio and radio frequency data for joint analysis.
        
        Args:
            audio_data (pd.DataFrame): Audio frequency data
            radio_data (pd.DataFrame): Radio frequency data
            
        Returns:
            pd.DataFrame: Combined frequency dataset
        """
        # Use a more reasonable frequency grid to avoid memory issues
        combined_freq_min = min(audio_data['frequency'].min(), radio_data['frequency'].min())
        combined_freq_max = max(audio_data['frequency'].max(), radio_data['frequency'].max())
        
        # Limit the number of points to avoid memory issues
        max_points = 10000  # Reasonable limit
        freq_range = combined_freq_max - combined_freq_min
        actual_resolution = max(self.resolution_hz, freq_range / max_points)
        
        n_points = min(max_points, int(freq_range / actual_resolution) + 1)
        freq_grid = np.linspace(combined_freq_min, combined_freq_max, n_points)
        
        # Interpolate both datasets to common grid
        audio_interp = np.interp(freq_grid, audio_data['frequency'], audio_data['amplitude'])
        radio_interp = np.interp(freq_grid, radio_data['frequency'], radio_data['amplitude'])
        
        # Create combined dataset
        combined_data = pd.DataFrame({
            'frequency': freq_grid,
            'audio_amplitude': audio_interp,
            'radio_amplitude': radio_interp,
            'combined_amplitude': audio_interp + radio_interp,
            'amplitude_ratio': np.divide(audio_interp, radio_interp, 
                                       out=np.zeros_like(audio_interp), 
                                       where=radio_interp!=0)
        })
        
        return combined_data
        
    def preprocess_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess combined frequency data for analysis.
        
        Args:
            data (pd.DataFrame): Combined frequency data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        processed_data = data.copy()
        
        # Apply smoothing to reduce noise
        processed_data['audio_smooth'] = gaussian_filter1d(
            processed_data['audio_amplitude'], sigma=2
        )
        processed_data['radio_smooth'] = gaussian_filter1d(
            processed_data['radio_amplitude'], sigma=2
        )
        
        # Calculate derivatives for edge detection
        processed_data['audio_derivative'] = np.gradient(processed_data['audio_smooth'])
        processed_data['radio_derivative'] = np.gradient(processed_data['radio_smooth'])
        
        # Normalize amplitudes
        numeric_cols = ['audio_amplitude', 'radio_amplitude', 'combined_amplitude']
        processed_data[numeric_cols] = self.scaler.fit_transform(processed_data[numeric_cols])
        
        # Calculate spectral features
        processed_data = self._calculate_spectral_features(processed_data)
        
        # Detect frequency peaks
        processed_data = self._detect_frequency_peaks(processed_data)
        
        return processed_data
        
    def _calculate_spectral_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate spectral features for each frequency band.
        
        Args:
            data (pd.DataFrame): Frequency data
            
        Returns:
            pd.DataFrame: Data with spectral features
        """
        try:
            # Simplified spectral centroid (weighted frequency mean)
            weights = data['combined_amplitude'] + 1e-8  # Avoid division by zero
            data['spectral_centroid'] = np.average(data['frequency'], weights=weights)
            
            # Simplified spectral bandwidth (standard deviation)
            data['spectral_bandwidth'] = np.std(data['frequency'])
            
            # Simplified spectral rolloff (90th percentile frequency)
            data['spectral_rolloff'] = np.percentile(data['frequency'], 90)
            
            # Zero crossing rate (simplified)
            combined_amp = data['combined_amplitude'].values
            zero_crossings = np.sum(np.diff(np.sign(combined_amp)) != 0)
            data['zero_crossing_rate'] = zero_crossings / len(combined_amp)
            
        except Exception as e:
            print(f"Warning: Spectral feature calculation failed: {e}")
            # Fall back to simple values
            data['spectral_centroid'] = data['frequency'].mean()
            data['spectral_bandwidth'] = data['frequency'].std()
            data['spectral_rolloff'] = data['frequency'].quantile(0.9)
            data['zero_crossing_rate'] = 0.1
            
        return data
        
    def _rolling_spectral_centroid(self, frequencies: np.ndarray, amplitudes: np.ndarray, 
                                 window_size: int = 50) -> np.ndarray:
        """Calculate rolling spectral centroid."""
        centroids = np.zeros_like(frequencies)
        half_window = window_size // 2
        
        for i in range(len(frequencies)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(frequencies), i + half_window + 1)
            
            freq_window = frequencies[start_idx:end_idx]
            amp_window = amplitudes[start_idx:end_idx]
            
            if np.sum(amp_window) > 0:
                centroids[i] = np.sum(freq_window * amp_window) / np.sum(amp_window)
            else:
                centroids[i] = frequencies[i]
                
        return centroids
        
    def _rolling_spectral_bandwidth(self, frequencies: np.ndarray, amplitudes: np.ndarray,
                                  window_size: int = 50) -> np.ndarray:
        """Calculate rolling spectral bandwidth."""
        bandwidths = np.zeros_like(frequencies)
        half_window = window_size // 2
        
        centroids = self._rolling_spectral_centroid(frequencies, amplitudes, window_size)
        
        for i in range(len(frequencies)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(frequencies), i + half_window + 1)
            
            freq_window = frequencies[start_idx:end_idx]
            amp_window = amplitudes[start_idx:end_idx]
            
            if np.sum(amp_window) > 0:
                bandwidths[i] = np.sqrt(
                    np.sum(((freq_window - centroids[i]) ** 2) * amp_window) / np.sum(amp_window)
                )
                
        return bandwidths
        
    def _rolling_spectral_rolloff(self, frequencies: np.ndarray, amplitudes: np.ndarray,
                                window_size: int = 50, rolloff_percent: float = 0.85) -> np.ndarray:
        """Calculate rolling spectral rolloff."""
        rolloffs = np.zeros_like(frequencies)
        half_window = window_size // 2
        
        for i in range(len(frequencies)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(frequencies), i + half_window + 1)
            
            freq_window = frequencies[start_idx:end_idx]
            amp_window = amplitudes[start_idx:end_idx]
            
            if len(amp_window) == 0:
                rolloffs[i] = frequencies[i]
                continue
                
            cumsum = np.cumsum(amp_window)
            if len(cumsum) > 0 and cumsum[-1] > 0:
                rolloff_idx = np.where(cumsum >= rolloff_percent * cumsum[-1])[0]
                if len(rolloff_idx) > 0:
                    rolloffs[i] = freq_window[rolloff_idx[0]]
                else:
                    rolloffs[i] = freq_window[-1] if len(freq_window) > 0 else frequencies[i]
            else:
                rolloffs[i] = frequencies[i]
                
        return rolloffs
        
    def _rolling_zero_crossing_rate(self, amplitudes: np.ndarray, window_size: int = 50) -> np.ndarray:
        """Calculate rolling zero crossing rate."""
        zcr = np.zeros_like(amplitudes)
        half_window = window_size // 2
        
        for i in range(len(amplitudes)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(amplitudes), i + half_window + 1)
            
            amp_window = amplitudes[start_idx:end_idx]
            
            # Count zero crossings
            crossings = np.sum(np.diff(np.sign(amp_window)) != 0)
            zcr[i] = crossings / len(amp_window)
            
        return zcr
        
    def _detect_frequency_peaks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect peaks in frequency spectrum.
        
        Args:
            data (pd.DataFrame): Frequency data
            
        Returns:
            pd.DataFrame: Data with peak indicators
        """
        try:
            # Calculate minimum distance (ensure it's at least 1)
            min_distance = max(1, int(10 / self.resolution_hz))
            
            # Find peaks in combined amplitude
            peaks, properties = signal.find_peaks(
                data['combined_amplitude'],
                height=0.1,  # Minimum peak height
                distance=min_distance,  # Minimum distance between peaks
                prominence=0.05  # Minimum prominence
            )
            
            # Create peak indicators
            data['is_peak'] = False
            if len(peaks) > 0:
                data.iloc[peaks, data.columns.get_loc('is_peak')] = True
            
            # Peak properties
            data['peak_height'] = 0.0
            data['peak_prominence'] = 0.0
            
            if len(peaks) > 0 and 'peak_heights' in properties:
                data.iloc[peaks, data.columns.get_loc('peak_height')] = properties['peak_heights']
                if 'prominences' in properties:
                    data.iloc[peaks, data.columns.get_loc('peak_prominence')] = properties['prominences']
                    
        except Exception as e:
            print(f"Warning: Peak detection failed: {e}")
            # Fall back to simple peak detection
            data['is_peak'] = False
            data['peak_height'] = 0.0
            data['peak_prominence'] = 0.0
            
            # Simple peak detection: points above 90th percentile
            threshold = data['combined_amplitude'].quantile(0.9)
            data['is_peak'] = data['combined_amplitude'] > threshold
            
        return data
        
    def apply_filtering(self, data: pd.DataFrame, filter_type: str = 'bandpass',
                       low_freq: float = None, high_freq: float = None) -> pd.DataFrame:
        """
        Apply frequency filtering to the data.
        
        Args:
            data (pd.DataFrame): Frequency data
            filter_type (str): Type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop')
            low_freq (float): Low frequency cutoff
            high_freq (float): High frequency cutoff
            
        Returns:
            pd.DataFrame: Filtered data
        """
        filtered_data = data.copy()
        
        if filter_type == 'lowpass' and high_freq:
            mask = filtered_data['frequency'] <= high_freq
        elif filter_type == 'highpass' and low_freq:
            mask = filtered_data['frequency'] >= low_freq
        elif filter_type == 'bandpass' and low_freq and high_freq:
            mask = (filtered_data['frequency'] >= low_freq) & (filtered_data['frequency'] <= high_freq)
        elif filter_type == 'bandstop' and low_freq and high_freq:
            mask = (filtered_data['frequency'] < low_freq) | (filtered_data['frequency'] > high_freq)
        else:
            return filtered_data  # No filtering applied
            
        # Apply mask
        for col in ['audio_amplitude', 'radio_amplitude', 'combined_amplitude']:
            if col in filtered_data.columns:
                filtered_data.loc[~mask, col] = 0
                
        return filtered_data
        
    def calculate_coherence(self, signal1: np.ndarray, signal2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate coherence between two signals.
        
        Args:
            signal1 (np.ndarray): First signal
            signal2 (np.ndarray): Second signal
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Frequencies and coherence values
        """
        fs = 1.0 / self.resolution_hz  # Sampling frequency
        
        # Calculate coherence
        freqs, coherence = signal.coherence(signal1, signal2, fs=fs, nperseg=min(256, len(signal1)//4))
        
        return freqs, coherence
