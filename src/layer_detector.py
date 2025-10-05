"""
Layer detection algorithms for identifying network anomalies and topology.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from scipy.spatial.distance import pdist, squareform


class LayerDetector:
    """Detects leakage and obscured layers in network frequency data."""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
    def detect_leakage(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect frequency leakage points that indicate network layer issues.
        
        Args:
            data (pd.DataFrame): Processed frequency data
            
        Returns:
            List[Dict]: Detected leakage points with metadata
        """
        leakage_points = []
        
        # Method 1: Statistical outlier detection
        outliers_statistical = self._detect_statistical_outliers(data)
        
        # Method 2: Isolation Forest for anomaly detection
        outliers_isolation = self._detect_isolation_outliers(data)
        
        # Method 3: Frequency domain analysis
        outliers_frequency = self._detect_frequency_anomalies(data)
        
        # Method 4: Cross-correlation analysis
        outliers_correlation = self._detect_correlation_anomalies(data)
        
        # Combine all methods
        all_outliers = set(outliers_statistical + outliers_isolation + 
                          outliers_frequency + outliers_correlation)
        
        for idx in all_outliers:
            if idx < len(data):
                point_data = data.iloc[idx]
                leakage_point = {
                    'index': idx,
                    'frequency': point_data['frequency'],
                    'strength': abs(point_data['combined_amplitude']),
                    'audio_amplitude': point_data.get('audio_amplitude', 0),
                    'radio_amplitude': point_data.get('radio_amplitude', 0),
                    'detection_methods': []
                }
                
                # Record which methods detected this point
                if idx in outliers_statistical:
                    leakage_point['detection_methods'].append('statistical')
                if idx in outliers_isolation:
                    leakage_point['detection_methods'].append('isolation_forest')
                if idx in outliers_frequency:
                    leakage_point['detection_methods'].append('frequency_domain')
                if idx in outliers_correlation:
                    leakage_point['detection_methods'].append('correlation')
                    
                leakage_points.append(leakage_point)
        
        # Sort by strength (descending)
        leakage_points.sort(key=lambda x: x['strength'], reverse=True)
        
        return leakage_points
        
    def detect_obscured_layers(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect obscured network layers using clustering and pattern analysis.
        
        Args:
            data (pd.DataFrame): Processed frequency data
            
        Returns:
            List[Dict]: Detected obscured layers
        """
        obscured_layers = []
        
        # Prepare features for clustering
        features = self._prepare_layer_features(data)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=10).fit(features)
        
        # Analyze clusters
        unique_labels = set(clustering.labels_)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
                
            cluster_mask = clustering.labels_ == label
            cluster_data = data[cluster_mask]
            
            if len(cluster_data) < 5:  # Skip small clusters
                continue
                
            # Analyze cluster characteristics
            layer_info = self._analyze_layer_cluster(cluster_data, label)
            
            # Check if this represents an obscured layer
            if self._is_obscured_layer(layer_info, data):
                obscured_layers.append(layer_info)
                
        return obscured_layers
        
    def analyze_correlations(self, data: pd.DataFrame) -> Dict:
        """
        Analyze correlations between frequency bands and signals.
        
        Args:
            data (pd.DataFrame): Processed frequency data
            
        Returns:
            Dict: Correlation analysis results
        """
        correlations = {}
        
        # Audio-Radio correlation
        if 'audio_amplitude' in data.columns and 'radio_amplitude' in data.columns:
            correlations['audio_radio'] = np.corrcoef(
                data['audio_amplitude'], data['radio_amplitude']
            )[0, 1]
            
        # Frequency band correlations
        correlations['frequency_bands'] = self._analyze_frequency_band_correlations(data)
        
        # Spectral feature correlations
        spectral_features = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']
        available_features = [f for f in spectral_features if f in data.columns]
        
        if len(available_features) > 1:
            feature_data = data[available_features].dropna()
            if len(feature_data) > 0:
                correlations['spectral_features'] = feature_data.corr().to_dict()
                
        # Peak correlations
        if 'is_peak' in data.columns:
            correlations['peak_analysis'] = self._analyze_peak_correlations(data)
            
        return correlations
        
    def infer_topology(self, data: pd.DataFrame) -> Dict:
        """
        Infer network topology from frequency analysis.
        
        Args:
            data (pd.DataFrame): Processed frequency data
            
        Returns:
            Dict: Network topology information
        """
        topology = {}
        
        # Estimate number of layers using clustering
        features = self._prepare_layer_features(data)
        
        # Try different numbers of clusters
        silhouette_scores = []
        k_range = range(2, min(11, len(features) // 10))
        
        for k in k_range:
            if k >= len(features):
                break
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            
            # Calculate silhouette score approximation
            if len(set(cluster_labels)) > 1:
                score = self._calculate_cluster_quality(features, cluster_labels)
                silhouette_scores.append((k, score))
                
        if silhouette_scores:
            best_k = max(silhouette_scores, key=lambda x: x[1])[0]
            topology['layer_count'] = best_k
        else:
            topology['layer_count'] = 1
            
        # Analyze connectivity
        topology['connectivity'] = self._analyze_network_connectivity(data)
        
        # Identify dominant frequencies
        topology['dominant_frequencies'] = self._identify_dominant_frequencies(data)
        
        # Network complexity score
        topology['complexity_score'] = self._calculate_network_complexity(data)
        
        return topology
        
    def calculate_anomaly_score(self, data: pd.DataFrame) -> float:
        """
        Calculate overall anomaly score for the network.
        
        Args:
            data (pd.DataFrame): Processed frequency data
            
        Returns:
            float: Anomaly score (0-1, higher means more anomalous)
        """
        scores = []
        
        # Statistical deviation score
        if 'combined_amplitude' in data.columns:
            z_scores = np.abs(zscore(data['combined_amplitude']))
            stat_score = np.mean(z_scores > 2)  # Fraction beyond 2 std devs
            scores.append(stat_score)
            
        # Peak irregularity score
        if 'is_peak' in data.columns:
            peak_count = data['is_peak'].sum()
            expected_peaks = len(data) * 0.02  # Expect ~2% peaks
            peak_score = min(1.0, abs(peak_count - expected_peaks) / expected_peaks)
            scores.append(peak_score)
            
        # Spectral irregularity score
        spectral_cols = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']
        available_spectral = [col for col in spectral_cols if col in data.columns]
        
        if available_spectral:
            spectral_data = data[available_spectral].dropna()
            if len(spectral_data) > 0:
                spectral_score = np.mean([
                    np.std(spectral_data[col]) / (np.mean(spectral_data[col]) + 1e-8)
                    for col in available_spectral
                ])
                scores.append(min(1.0, spectral_score))
                
        # Frequency distribution score
        freq_score = self._calculate_frequency_distribution_score(data)
        scores.append(freq_score)
        
        # Combine scores
        if scores:
            return np.mean(scores)
        else:
            return 0.0
            
    def _detect_statistical_outliers(self, data: pd.DataFrame, threshold: float = 3.0) -> List[int]:
        """Detect outliers using statistical methods."""
        outliers = []
        
        if 'combined_amplitude' in data.columns:
            z_scores = np.abs(zscore(data['combined_amplitude']))
            outliers.extend(np.where(z_scores > threshold)[0].tolist())
            
        return outliers
        
    def _detect_isolation_outliers(self, data: pd.DataFrame) -> List[int]:
        """Detect outliers using Isolation Forest."""
        features = self._prepare_anomaly_features(data)
        
        if len(features) == 0:
            return []
            
        outlier_labels = self.isolation_forest.fit_predict(features)
        outliers = np.where(outlier_labels == -1)[0].tolist()
        
        return outliers
        
    def _detect_frequency_anomalies(self, data: pd.DataFrame) -> List[int]:
        """Detect anomalies in frequency domain."""
        outliers = []
        
        # Large derivative changes
        if 'audio_derivative' in data.columns:
            deriv_threshold = np.percentile(np.abs(data['audio_derivative']), 95)
            outliers.extend(np.where(np.abs(data['audio_derivative']) > deriv_threshold)[0].tolist())
            
        if 'radio_derivative' in data.columns:
            deriv_threshold = np.percentile(np.abs(data['radio_derivative']), 95)
            outliers.extend(np.where(np.abs(data['radio_derivative']) > deriv_threshold)[0].tolist())
            
        return outliers
        
    def _detect_correlation_anomalies(self, data: pd.DataFrame) -> List[int]:
        """Detect anomalies using correlation analysis."""
        outliers = []
        
        if 'audio_amplitude' in data.columns and 'radio_amplitude' in data.columns:
            # Rolling correlation
            window_size = min(50, len(data) // 10)
            if window_size > 5:
                rolling_corr = data['audio_amplitude'].rolling(window_size).corr(
                    data['radio_amplitude']
                ).fillna(0)
                
                # Find points with unusual correlation
                corr_threshold = np.percentile(np.abs(rolling_corr), 90)
                outliers.extend(np.where(np.abs(rolling_corr) > corr_threshold)[0].tolist())
                
        return outliers
        
    def _prepare_layer_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for layer detection clustering."""
        feature_cols = []
        
        # Basic amplitude features
        if 'audio_amplitude' in data.columns:
            feature_cols.append('audio_amplitude')
        if 'radio_amplitude' in data.columns:
            feature_cols.append('radio_amplitude')
        if 'combined_amplitude' in data.columns:
            feature_cols.append('combined_amplitude')
            
        # Spectral features
        spectral_cols = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'zero_crossing_rate']
        for col in spectral_cols:
            if col in data.columns:
                feature_cols.append(col)
                
        # Derivative features
        if 'audio_derivative' in data.columns:
            feature_cols.append('audio_derivative')
        if 'radio_derivative' in data.columns:
            feature_cols.append('radio_derivative')
            
        if not feature_cols:
            return np.array([])
            
        features = data[feature_cols].fillna(0).values
        
        # Normalize features
        if len(features) > 0:
            features = self.scaler.fit_transform(features)
            
        return features
        
    def _prepare_anomaly_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for anomaly detection."""
        return self._prepare_layer_features(data)
        
    def _analyze_layer_cluster(self, cluster_data: pd.DataFrame, label: int) -> Dict:
        """Analyze characteristics of a layer cluster."""
        freq_range = (cluster_data['frequency'].min(), cluster_data['frequency'].max())
        
        layer_info = {
            'id': label,
            'frequency_range': freq_range,
            'size': len(cluster_data),
            'avg_amplitude': cluster_data['combined_amplitude'].mean(),
            'amplitude_std': cluster_data['combined_amplitude'].std(),
            'peak_count': cluster_data.get('is_peak', pd.Series([False]*len(cluster_data))).sum()
        }
        
        return layer_info
        
    def _is_obscured_layer(self, layer_info: Dict, full_data: pd.DataFrame) -> bool:
        """Determine if a layer cluster represents an obscured layer."""
        # Criteria for obscured layer:
        # 1. Low average amplitude compared to overall data
        # 2. High variability in amplitude
        # 3. Few or no peaks
        
        overall_avg = full_data['combined_amplitude'].mean()
        overall_std = full_data['combined_amplitude'].std()
        
        is_low_amplitude = layer_info['avg_amplitude'] < overall_avg - overall_std
        is_high_variability = layer_info['amplitude_std'] > overall_std * 1.5
        is_few_peaks = layer_info['peak_count'] / layer_info['size'] < 0.01
        
        return is_low_amplitude and (is_high_variability or is_few_peaks)
        
    def _analyze_frequency_band_correlations(self, data: pd.DataFrame) -> Dict:
        """Analyze correlations between different frequency bands."""
        correlations = {}
        
        # Define frequency bands
        freq_min, freq_max = data['frequency'].min(), data['frequency'].max()
        n_bands = 5
        band_edges = np.linspace(freq_min, freq_max, n_bands + 1)
        
        band_amplitudes = []
        for i in range(n_bands):
            band_mask = (data['frequency'] >= band_edges[i]) & (data['frequency'] < band_edges[i+1])
            band_data = data[band_mask]
            if len(band_data) > 0:
                band_amplitudes.append(band_data['combined_amplitude'].mean())
            else:
                band_amplitudes.append(0)
                
        # Calculate correlation matrix between bands
        if len(band_amplitudes) > 1:
            correlations['band_correlations'] = np.corrcoef(band_amplitudes).tolist()
            correlations['band_ranges'] = [(band_edges[i], band_edges[i+1]) for i in range(n_bands)]
            
        return correlations
        
    def _analyze_peak_correlations(self, data: pd.DataFrame) -> Dict:
        """Analyze correlations related to frequency peaks."""
        peak_data = data[data['is_peak'] == True]
        
        if len(peak_data) == 0:
            return {'peak_count': 0}
            
        analysis = {
            'peak_count': len(peak_data),
            'avg_peak_frequency': peak_data['frequency'].mean(),
            'peak_frequency_std': peak_data['frequency'].std(),
            'avg_peak_amplitude': peak_data['combined_amplitude'].mean()
        }
        
        # Peak spacing analysis
        if len(peak_data) > 1:
            peak_frequencies = sorted(peak_data['frequency'].values)
            spacings = np.diff(peak_frequencies)
            analysis['avg_peak_spacing'] = np.mean(spacings)
            analysis['peak_spacing_std'] = np.std(spacings)
            
        return analysis
        
    def _analyze_network_connectivity(self, data: pd.DataFrame) -> float:
        """Analyze network connectivity based on frequency patterns."""
        if 'audio_amplitude' not in data.columns or 'radio_amplitude' not in data.columns:
            return 0.0
            
        # Calculate cross-correlation
        correlation = np.corrcoef(data['audio_amplitude'], data['radio_amplitude'])[0, 1]
        
        # Normalize to 0-1 range
        connectivity = (correlation + 1) / 2
        
        return connectivity
        
    def _identify_dominant_frequencies(self, data: pd.DataFrame, n_dominant: int = 5) -> List[float]:
        """Identify dominant frequencies in the spectrum."""
        if 'combined_amplitude' not in data.columns:
            return []
            
        # Find peaks and sort by amplitude
        peak_data = data[data.get('is_peak', False) == True]
        
        if len(peak_data) == 0:
            # If no peaks detected, use highest amplitude points
            top_indices = data['combined_amplitude'].nlargest(n_dominant).index
            return data.loc[top_indices, 'frequency'].tolist()
        else:
            # Use detected peaks
            dominant_peaks = peak_data.nlargest(n_dominant, 'combined_amplitude')
            return dominant_peaks['frequency'].tolist()
            
    def _calculate_network_complexity(self, data: pd.DataFrame) -> float:
        """Calculate network complexity score."""
        complexity_factors = []
        
        # Frequency diversity
        if 'frequency' in data.columns:
            freq_range = data['frequency'].max() - data['frequency'].min()
            complexity_factors.append(min(1.0, freq_range / 10000))  # Normalize to reasonable range
            
        # Amplitude variability
        if 'combined_amplitude' in data.columns:
            amp_cv = data['combined_amplitude'].std() / (data['combined_amplitude'].mean() + 1e-8)
            complexity_factors.append(min(1.0, amp_cv))
            
        # Peak density
        if 'is_peak' in data.columns:
            peak_density = data['is_peak'].sum() / len(data)
            complexity_factors.append(min(1.0, peak_density * 50))  # Scale appropriately
            
        # Spectral complexity
        spectral_cols = ['spectral_centroid', 'spectral_bandwidth']
        available_spectral = [col for col in spectral_cols if col in data.columns]
        
        if available_spectral:
            spectral_complexity = np.mean([
                data[col].std() / (data[col].mean() + 1e-8)
                for col in available_spectral
            ])
            complexity_factors.append(min(1.0, spectral_complexity))
            
        return np.mean(complexity_factors) if complexity_factors else 0.0
        
    def _calculate_cluster_quality(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate approximate cluster quality score."""
        if len(set(labels)) <= 1:
            return 0.0
            
        # Calculate within-cluster sum of squares
        wcss = 0
        for label in set(labels):
            cluster_points = features[labels == label]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                wcss += np.sum((cluster_points - centroid) ** 2)
                
        # Calculate between-cluster sum of squares
        overall_centroid = np.mean(features, axis=0)
        bcss = 0
        for label in set(labels):
            cluster_points = features[labels == label]
            if len(cluster_points) > 0:
                cluster_centroid = np.mean(cluster_points, axis=0)
                bcss += len(cluster_points) * np.sum((cluster_centroid - overall_centroid) ** 2)
                
        # Return ratio (higher is better)
        return bcss / (wcss + 1e-8)
        
    def _calculate_frequency_distribution_score(self, data: pd.DataFrame) -> float:
        """Calculate score based on frequency distribution irregularity."""
        if 'combined_amplitude' not in data.columns:
            return 0.0
            
        # Divide frequency range into bins and calculate distribution
        n_bins = 20
        hist, _ = np.histogram(data['combined_amplitude'], bins=n_bins)
        
        # Calculate entropy (higher entropy = more uniform = lower anomaly score)
        hist = hist + 1e-8  # Avoid log(0)
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log(hist))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(n_bins)
        normalized_entropy = entropy / max_entropy
        
        # Return anomaly score (1 - normalized_entropy)
        return 1.0 - normalized_entropy
