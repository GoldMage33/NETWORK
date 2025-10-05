"""
Data loading utilities for frequency analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class DataLoader:
    """Handles loading and validation of frequency data from various sources."""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.txt', '.json']
        
    def load_frequency_data(self, file_path: str, data_type: str = 'generic') -> pd.DataFrame:
        """
        Load frequency data from file.
        
        Args:
            file_path (str): Path to data file
            data_type (str): Type of data ('audio', 'radio', 'generic')
            
        Returns:
            pd.DataFrame: Loaded frequency data
        """
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                data = pd.read_json(file_path)
            elif file_path.endswith('.txt'):
                data = pd.read_csv(file_path, delimiter='\t')
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
                
            # Validate and standardize columns
            data = self._validate_frequency_data(data, data_type)
            
            return data
            
        except FileNotFoundError:
            # Generate sample data if file doesn't exist
            print(f"File not found: {file_path}. Generating sample data.")
            return self._generate_sample_data(data_type)
            
    def _validate_frequency_data(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Validate and standardize frequency data format.
        
        Args:
            data (pd.DataFrame): Raw data
            data_type (str): Type of data
            
        Returns:
            pd.DataFrame: Validated data
        """
        required_columns = ['frequency', 'amplitude']
        
        # Check if required columns exist or try to infer them
        if not all(col in data.columns for col in required_columns):
            # Try to infer column names
            freq_col = None
            amp_col = None
            
            for col in data.columns:
                if 'freq' in col.lower() or 'hz' in col.lower():
                    freq_col = col
                elif 'amp' in col.lower() or 'power' in col.lower() or 'magnitude' in col.lower():
                    amp_col = col
                    
            if freq_col and amp_col:
                data = data.rename(columns={freq_col: 'frequency', amp_col: 'amplitude'})
            else:
                # Assume first two columns are frequency and amplitude
                if len(data.columns) >= 2:
                    data.columns = ['frequency', 'amplitude'] + list(data.columns[2:])
                else:
                    raise ValueError("Cannot identify frequency and amplitude columns")
                    
        # Add metadata columns
        data['data_type'] = data_type
        data['timestamp'] = pd.Timestamp.now()
        
        # Validate data types
        data['frequency'] = pd.to_numeric(data['frequency'], errors='coerce')
        data['amplitude'] = pd.to_numeric(data['amplitude'], errors='coerce')
        
        # Remove invalid rows
        data = data.dropna(subset=['frequency', 'amplitude'])
        
        return data
        
    def _generate_sample_data(self, data_type: str, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate sample frequency data for testing.
        
        Args:
            data_type (str): Type of data to generate
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Sample frequency data
        """
        if data_type == 'audio':
            # Audio frequency range: 20 Hz to 20 kHz
            frequencies = np.logspace(np.log10(20), np.log10(20000), n_samples)
            # Typical audio spectrum with some peaks
            amplitudes = (
                np.random.exponential(0.1, n_samples) + 
                0.5 * np.sin(frequencies / 1000) +
                0.3 * np.random.normal(0, 0.1, n_samples)
            )
            
        elif data_type == 'radio':
            # Radio frequency range: 3 kHz to 300 GHz (focus on common bands)
            frequencies = np.logspace(np.log10(3000), np.log10(3e9), n_samples)
            # Radio spectrum with characteristic patterns
            amplitudes = (
                np.random.exponential(0.05, n_samples) +
                0.8 * np.exp(-((frequencies - 100e6) / 50e6) ** 2) +  # FM band peak
                0.4 * np.random.normal(0, 0.05, n_samples)
            )
            
        else:  # generic
            frequencies = np.linspace(1, 10000, n_samples)
            amplitudes = np.random.exponential(0.1, n_samples)
            
        # Ensure positive amplitudes
        amplitudes = np.abs(amplitudes)
        
        data = pd.DataFrame({
            'frequency': frequencies,
            'amplitude': amplitudes,
            'data_type': data_type,
            'timestamp': pd.Timestamp.now()
        })
        
        return data
        
    def save_frequency_data(self, data: pd.DataFrame, file_path: str) -> None:
        """
        Save frequency data to file.
        
        Args:
            data (pd.DataFrame): Frequency data to save
            file_path (str): Output file path
        """
        if file_path.endswith('.csv'):
            data.to_csv(file_path, index=False)
        elif file_path.endswith('.json'):
            data.to_json(file_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported output format: {file_path}")
            
        print(f"Data saved to {file_path}")
        
    def get_data_info(self, data: pd.DataFrame) -> Dict:
        """
        Get summary information about frequency data.
        
        Args:
            data (pd.DataFrame): Frequency data
            
        Returns:
            Dict: Summary information
        """
        return {
            'samples': len(data),
            'frequency_range': (data['frequency'].min(), data['frequency'].max()),
            'amplitude_range': (data['amplitude'].min(), data['amplitude'].max()),
            'data_types': data['data_type'].unique().tolist(),
            'memory_usage': data.memory_usage(deep=True).sum()
        }
