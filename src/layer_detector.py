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
        
    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data before processing to prevent sklearn errors."""
        if data.empty:
            return data
            
        # Remove NaN and infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        
        if data.empty:
            return data
            
        # Ensure numeric columns are finite
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Remove non-finite values
            data = data[np.isfinite(data[col])]
            
            # Ensure minimum variance (avoid all-same values)
            if data[col].std() == 0 and len(data) > 1:
                # Add small noise to prevent division by zero
                noise = np.random.normal(0, 1e-8, len(data))
                data[col] = data[col] + noise
                
        return data
        
    def detect_leakage(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect frequency leakage points that indicate network layer issues.

        Enhanced criteria:
        - Multi-threshold statistical detection
        - Frequency-specific thresholds
        - Temporal consistency requirements
        - Signal-to-noise ratio validation
        - Cross-method validation

        Args:
            data (pd.DataFrame): Processed frequency data

        Returns:
            List[Dict]: Detected leakage points with metadata
        """
        leakage_points = []

        # Enhanced detection methods with refined criteria
        outliers_statistical = self._detect_enhanced_statistical_outliers(data)
        outliers_isolation = self._detect_isolation_outliers(data)
        outliers_frequency = self._detect_enhanced_frequency_anomalies(data)
        outliers_correlation = self._detect_enhanced_correlation_anomalies(data)
        outliers_snr = self._detect_signal_noise_anomalies(data)

        # Combine all methods with confidence scoring
        candidate_points = self._combine_detection_methods(
            data, outliers_statistical, outliers_isolation,
            outliers_frequency, outliers_correlation, outliers_snr
        )

        # Apply temporal consistency filter
        consistent_leakage = self._filter_temporal_consistency(data, candidate_points)

        # Apply signal-to-noise validation
        validated_leakage = self._validate_signal_noise_ratio(data, consistent_leakage)

        for point in validated_leakage:
            idx = point['index']
            if idx < len(data):
                point_data = data.iloc[idx]
                leakage_point = {
                    'index': idx,
                    'frequency': point_data['frequency'],
                    'strength': abs(point_data['combined_amplitude']),
                    'audio_amplitude': point_data.get('audio_amplitude', 0),
                    'radio_amplitude': point_data.get('radio_amplitude', 0),
                    'detection_methods': point['methods'],
                    'confidence_score': point['confidence'],
                    'snr_ratio': point.get('snr_ratio', 0),
                    'frequency_range': self._classify_frequency_range(point_data['frequency'])
                }

                leakage_points.append(leakage_point)

        # Sort by confidence score and strength
        leakage_points.sort(key=lambda x: (x['confidence_score'], x['strength']), reverse=True)

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
            
    def _detect_enhanced_statistical_outliers(self, data: pd.DataFrame) -> List[Dict]:
        """Enhanced statistical outlier detection with multi-threshold approach."""
        outliers = []

        if 'combined_amplitude' in data.columns:
            amplitudes = data['combined_amplitude'].values

            # Multi-threshold detection
            thresholds = [2.5, 3.0, 3.5, 4.0]  # Progressive thresholds
            weights = [0.3, 0.4, 0.2, 0.1]     # Confidence weights

            for threshold, weight in zip(thresholds, weights):
                z_scores = np.abs(zscore(amplitudes))
                outlier_indices = np.where(z_scores > threshold)[0]

                for idx in outlier_indices:
                    outliers.append({
                        'index': idx,
                        'confidence': weight,
                        'method': 'statistical',
                        'threshold': threshold,
                        'z_score': z_scores[idx]
                    })

        return outliers

    def _detect_enhanced_frequency_anomalies(self, data: pd.DataFrame) -> List[Dict]:
        """Enhanced frequency domain analysis with adaptive thresholds."""
        outliers = []

        # Frequency-specific thresholds
        freq_ranges = [
            (0, 1000, 2.0),      # Low frequency: more sensitive
            (1000, 10000, 2.5),   # Audio range: moderate
            (10000, 50000, 3.0),  # Radio range: standard
            (50000, float('inf'), 3.5)  # High frequency: less sensitive
        ]

        for freq_min, freq_max, threshold in freq_ranges:
            range_mask = (data['frequency'] >= freq_min) & (data['frequency'] < freq_max)
            range_data = data[range_mask]

            if len(range_data) == 0:
                continue

            # Derivative-based detection
            if 'audio_derivative' in range_data.columns:
                deriv_values = range_data['audio_derivative'].abs()
                deriv_threshold = np.percentile(deriv_values, 95)
                outlier_mask = deriv_values > deriv_threshold

                for local_idx in np.where(outlier_mask)[0]:
                    global_idx = range_data.index[local_idx]
                    outliers.append({
                        'index': global_idx,
                        'confidence': 0.6,
                        'method': 'frequency_domain',
                        'frequency_range': (freq_min, freq_max),
                        'derivative_value': deriv_values.iloc[local_idx]
                    })

            if 'radio_derivative' in range_data.columns:
                deriv_values = range_data['radio_derivative'].abs()
                deriv_threshold = np.percentile(deriv_values, 95)
                outlier_mask = deriv_values > deriv_threshold

                for local_idx in np.where(outlier_mask)[0]:
                    global_idx = range_data.index[local_idx]
                    outliers.append({
                        'index': global_idx,
                        'confidence': 0.6,
                        'method': 'frequency_domain',
                        'frequency_range': (freq_min, freq_max),
                        'derivative_value': deriv_values.iloc[local_idx]
                    })

        return outliers

    def _detect_enhanced_correlation_anomalies(self, data: pd.DataFrame) -> List[Dict]:
        """Enhanced correlation analysis with temporal patterns."""
        outliers = []

        if 'audio_amplitude' in data.columns and 'radio_amplitude' in data.columns:
            # Multiple window sizes for correlation analysis
            window_sizes = [20, 50, 100]
            weights = [0.4, 0.4, 0.2]

            for window_size, weight in zip(window_sizes, weights):
                if window_size >= len(data):
                    continue

                # Rolling correlation with different methods
                rolling_corr = data['audio_amplitude'].rolling(window_size).corr(
                    data['radio_amplitude']
                ).fillna(0)

                # Detect unusual correlations (too high or too low)
                corr_threshold_high = np.percentile(np.abs(rolling_corr), 90)
                corr_threshold_low = np.percentile(np.abs(rolling_corr), 10)

                high_corr_mask = np.abs(rolling_corr) > corr_threshold_high
                low_corr_mask = np.abs(rolling_corr) < corr_threshold_low

                for mask, threshold_type in [(high_corr_mask, 'high'), (low_corr_mask, 'low')]:
                    outlier_indices = np.where(mask)[0]
                    for idx in outlier_indices:
                        outliers.append({
                            'index': idx,
                            'confidence': weight,
                            'method': 'correlation',
                            'correlation_value': rolling_corr.iloc[idx],
                            'window_size': window_size,
                            'threshold_type': threshold_type
                        })

        return outliers

    def _detect_signal_noise_anomalies(self, data: pd.DataFrame) -> List[Dict]:
        """Detect anomalies based on signal-to-noise ratio."""
        outliers = []

        if 'combined_amplitude' in data.columns:
            amplitudes = data['combined_amplitude'].values

            # Calculate local noise estimate using rolling standard deviation
            window_sizes = [10, 25, 50]
            weights = [0.3, 0.4, 0.3]

            for window_size, weight in zip(window_sizes, weights):
                if window_size >= len(amplitudes):
                    continue

                # Rolling noise estimate
                rolling_noise = pd.Series(amplitudes).rolling(window_size).std().fillna(0)

                # Calculate SNR
                snr_values = amplitudes / (rolling_noise + 1e-8)

                # Detect points with unusually high SNR (potential leakage)
                snr_threshold = np.percentile(snr_values, 95)
                outlier_mask = snr_values > snr_threshold

                outlier_indices = np.where(outlier_mask)[0]
                for idx in outlier_indices:
                    outliers.append({
                        'index': idx,
                        'confidence': weight,
                        'method': 'signal_noise',
                        'snr_ratio': snr_values[idx],
                        'noise_estimate': rolling_noise.iloc[idx],
                        'window_size': window_size
                    })

        return outliers
        
    def _detect_isolation_outliers(self, data: pd.DataFrame) -> List[int]:
        """Detect outliers using Isolation Forest."""
        # Validate and clean data first
        clean_data = self._validate_and_clean_data(data)
        
        features = self._prepare_anomaly_features(clean_data)
        
        if len(features) == 0 or len(features) < 2:
            return []
            
        try:
            outlier_labels = self.isolation_forest.fit_predict(features)
            outliers = np.where(outlier_labels == -1)[0].tolist()
        except Exception as e:
            # If sklearn fails, return empty list
            print(f"Warning: Isolation Forest failed: {e}")
            outliers = []
        
        return outliers
        
    def _combine_detection_methods(self, data: pd.DataFrame, *method_results) -> List[Dict]:
        """Combine results from multiple detection methods with confidence scoring."""
        point_confidence = {}

        # Collect all candidate points
        for method_outliers in method_results:
            for outlier in method_outliers:
                # Handle both old format (int) and new format (dict)
                if isinstance(outlier, int):
                    # Convert old format to new format
                    idx = outlier
                    outlier_info = {
                        'index': idx,
                        'confidence': 0.5,  # Default confidence for old methods
                        'method': 'legacy',
                        'method_details': []
                    }
                else:
                    # New format
                    idx = outlier['index']
                    outlier_info = outlier

                if idx not in point_confidence:
                    point_confidence[idx] = {
                        'index': idx,
                        'confidence': 0.0,
                        'methods': [],
                        'method_details': []
                    }

                point_confidence[idx]['confidence'] += outlier_info['confidence']
                point_confidence[idx]['methods'].append(outlier_info['method'])
                point_confidence[idx]['method_details'].append(outlier_info)

        # Filter points that appear in multiple methods (higher confidence)
        candidates = []
        for idx, info in point_confidence.items():
            # Require at least 2 detection methods or high single-method confidence
            if len(info['methods']) >= 2 or info['confidence'] >= 0.7:
                # Boost confidence for multi-method detection
                if len(info['methods']) >= 2:
                    info['confidence'] *= 1.2
                info['confidence'] = min(1.0, info['confidence'])
                candidates.append(info)

        return candidates

    def _filter_temporal_consistency(self, data: pd.DataFrame, candidates: List[Dict]) -> List[Dict]:
        """Filter candidates based on temporal consistency (persistence over time)."""
        consistent_candidates = []

        for candidate in candidates:
            idx = candidate['index']

            # Check neighboring points for similar anomalies
            window_size = min(10, len(data) // 20)  # Adaptive window size
            start_idx = max(0, idx - window_size)
            end_idx = min(len(data), idx + window_size + 1)

            window_data = data.iloc[start_idx:end_idx]

            # Count similar anomalies in the temporal window
            if 'combined_amplitude' in window_data.columns:
                center_amplitude = abs(data.iloc[idx]['combined_amplitude'])
                window_amplitudes = window_data['combined_amplitude'].abs()

                # Find points within 50% of center amplitude
                similar_points = np.sum(
                    (window_amplitudes >= center_amplitude * 0.5) &
                    (window_amplitudes <= center_amplitude * 1.5)
                )

                # Require at least 3 similar points in temporal window
                if similar_points >= 3:
                    candidate['temporal_consistency'] = similar_points / len(window_data)
                    consistent_candidates.append(candidate)
                else:
                    # Reduce confidence for isolated anomalies
                    candidate['confidence'] *= 0.7

        return consistent_candidates

    def _validate_signal_noise_ratio(self, data: pd.DataFrame, candidates: List[Dict]) -> List[Dict]:
        """Validate candidates using signal-to-noise ratio analysis."""
        validated_candidates = []

        for candidate in candidates:
            idx = candidate['index']

            # Calculate local SNR
            window_size = 20
            start_idx = max(0, idx - window_size // 2)
            end_idx = min(len(data), idx + window_size // 2 + 1)

            window_data = data.iloc[start_idx:end_idx]

            if 'combined_amplitude' in window_data.columns and len(window_data) > 1:
                signal_amplitude = abs(data.iloc[idx]['combined_amplitude'])
                noise_amplitude = window_data['combined_amplitude'].std()

                if noise_amplitude > 0:
                    snr = signal_amplitude / noise_amplitude

                    # Require minimum SNR for leakage detection
                    min_snr_threshold = 3.0
                    if snr >= min_snr_threshold:
                        candidate['snr_ratio'] = snr
                        candidate['noise_level'] = noise_amplitude
                        validated_candidates.append(candidate)
                    else:
                        # Reduce confidence for low SNR points
                        candidate['confidence'] *= 0.8

        return validated_candidates

    def _classify_frequency_range(self, frequency: float) -> str:
        """Classify frequency into standard ranges."""
        if frequency < 1000:
            return 'ELF/VLF'
        elif frequency < 10000:
            return 'Audio'
        elif frequency < 30000:
            return 'VHF Low'
        elif frequency < 50000:
            return 'VHF High'
        elif frequency < 100000:
            return 'UHF Low'
        else:
            return 'UHF High'
        
    def _prepare_layer_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for layer detection clustering."""
        # Validate and clean data first
        data = self._validate_and_clean_data(data)
        
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
        
        # Normalize features with error handling
        if len(features) > 0:
            try:
                features = self.scaler.fit_transform(features)
            except Exception as e:
                print(f"Warning: Feature scaling failed: {e}")
                # Return unscaled features if scaling fails
                pass
            
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
