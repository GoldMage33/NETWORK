"""
Visualization utilities for frequency analysis and network layer detection.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class Visualizer:
    """Handles visualization of frequency analysis and layer detection results."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize visualizer with plotting style.
        
        Args:
            style (str): Matplotlib style to use
        """
        plt.style.use('default')  # Use default style as seaborn-v0_8 may not be available
        sns.set_palette("husl")
        self.figure_size = (12, 8)
        
    def plot_frequency_spectrum(self, data: pd.DataFrame, output_path: str = None) -> None:
        """
        Plot frequency spectrum analysis.
        
        Args:
            data (pd.DataFrame): Combined frequency data
            output_path (str, optional): Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Main spectrum plot
        axes[0, 0].plot(data['frequency'], data['audio_amplitude'], 
                       label='Audio', alpha=0.7, linewidth=1)
        axes[0, 0].plot(data['frequency'], data['radio_amplitude'], 
                       label='Radio', alpha=0.7, linewidth=1)
        axes[0, 0].plot(data['frequency'], data['combined_amplitude'], 
                       label='Combined', alpha=0.8, linewidth=2)
        axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Frequency Spectrum Analysis')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Log scale plot
        axes[0, 1].semilogx(data['frequency'], data['combined_amplitude'])
        axes[0, 1].set_xlabel('Frequency (Hz) - Log Scale')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].set_title('Frequency Spectrum (Log Scale)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Amplitude ratio plot
        if 'amplitude_ratio' in data.columns:
            axes[1, 0].plot(data['frequency'], data['amplitude_ratio'])
            axes[1, 0].set_xlabel('Frequency (Hz)')
            axes[1, 0].set_ylabel('Audio/Radio Ratio')
            axes[1, 0].set_title('Audio to Radio Amplitude Ratio')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Peak detection visualization
        if 'is_peak' in data.columns:
            peak_data = data[data['is_peak'] == True]
            axes[1, 1].plot(data['frequency'], data['combined_amplitude'], 
                           'b-', alpha=0.6, label='Spectrum')
            if len(peak_data) > 0:
                axes[1, 1].scatter(peak_data['frequency'], peak_data['combined_amplitude'],
                                  color='red', s=50, alpha=0.8, label='Peaks')
            axes[1, 1].set_xlabel('Frequency (Hz)')
            axes[1, 1].set_ylabel('Amplitude')
            axes[1, 1].set_title('Peak Detection')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Frequency spectrum plot saved to {output_path}")
        else:
            plt.show()
            
    def plot_layer_analysis(self, results: Dict, output_path: str = None) -> None:
        """
        Plot layer analysis results.
        
        Args:
            results (Dict): Results from layer detection
            output_path (str, optional): Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Leakage points
        leakage_points = results.get('leakage_points', [])
        if leakage_points:
            frequencies = [point['frequency'] for point in leakage_points[:20]]  # Top 20
            strengths = [point['strength'] for point in leakage_points[:20]]
            
            axes[0, 0].bar(range(len(frequencies)), strengths, alpha=0.7)
            axes[0, 0].set_xlabel('Leakage Point Index')
            axes[0, 0].set_ylabel('Strength')
            axes[0, 0].set_title(f'Top {len(frequencies)} Leakage Points')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add frequency labels on x-axis
            if len(frequencies) <= 10:
                axes[0, 0].set_xticks(range(len(frequencies)))
                axes[0, 0].set_xticklabels([f'{f:.0f}Hz' for f in frequencies], rotation=45)
        else:
            axes[0, 0].text(0.5, 0.5, 'No leakage points detected', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Leakage Points')
        
        # Obscured layers
        obscured_layers = results.get('obscured_layers', [])
        if obscured_layers:
            layer_ids = [layer['id'] for layer in obscured_layers]
            layer_sizes = [layer['size'] for layer in obscured_layers]
            
            axes[0, 1].bar(layer_ids, layer_sizes, alpha=0.7)
            axes[0, 1].set_xlabel('Layer ID')
            axes[0, 1].set_ylabel('Layer Size')
            axes[0, 1].set_title('Obscured Layers')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No obscured layers detected', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Obscured Layers')
        
        # Frequency correlations heatmap
        correlations = results.get('frequency_correlations', {})
        if 'spectral_features' in correlations and correlations['spectral_features']:
            corr_data = pd.DataFrame(correlations['spectral_features'])
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[1, 0], cbar_kws={'label': 'Correlation'})
            axes[1, 0].set_title('Spectral Feature Correlations')
        else:
            axes[1, 0].text(0.5, 0.5, 'No correlation data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Frequency Correlations')
        
        # Network topology visualization
        topology = results.get('network_topology', {})
        if topology:
            # Create a simple topology visualization
            layer_count = topology.get('layer_count', 1)
            connectivity = topology.get('connectivity', 0)
            complexity = topology.get('complexity_score', 0)
            
            metrics = ['Layers', 'Connectivity', 'Complexity']
            values = [layer_count/10, connectivity, complexity]  # Normalize layers for visualization
            
            axes[1, 1].bar(metrics, values, alpha=0.7)
            axes[1, 1].set_ylabel('Score/Count (normalized)')
            axes[1, 1].set_title('Network Topology Metrics')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[1, 1].text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
        else:
            axes[1, 1].text(0.5, 0.5, 'No topology data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Network Topology')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Layer analysis plot saved to {output_path}")
        else:
            plt.show()
            
    def plot_anomaly_heatmap(self, leakage_points: List[Dict], output_path: str = None) -> None:
        """
        Plot heatmap of anomaly detection results.
        
        Args:
            leakage_points (List[Dict]): Detected leakage points
            output_path (str, optional): Path to save plot
        """
        if not leakage_points:
            plt.figure(figsize=self.figure_size)
            plt.text(0.5, 0.5, 'No anomalies detected', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=16)
            plt.title('Anomaly Heatmap')
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Anomaly heatmap saved to {output_path}")
            else:
                plt.show()
            return
        
        # Prepare data for heatmap
        frequencies = [point['frequency'] for point in leakage_points]
        strengths = [point['strength'] for point in leakage_points]
        methods = []
        
        for point in leakage_points:
            method_str = ', '.join(point.get('detection_methods', ['unknown']))
            methods.append(method_str)
        
        # Create frequency bins for heatmap
        n_bins = min(50, len(leakage_points))
        freq_min, freq_max = min(frequencies), max(frequencies)
        
        if freq_max > freq_min:
            freq_bins = np.linspace(freq_min, freq_max, n_bins)
            
            # Bin the data
            binned_strengths = np.zeros(n_bins - 1)
            
            for freq, strength in zip(frequencies, strengths):
                bin_idx = np.digitize(freq, freq_bins) - 1
                if 0 <= bin_idx < len(binned_strengths):
                    binned_strengths[bin_idx] = max(binned_strengths[bin_idx], strength)
            
            # Create heatmap
            plt.figure(figsize=(15, 6))
            
            # Reshape for heatmap (single row)
            heatmap_data = binned_strengths.reshape(1, -1)
            
            im = plt.imshow(heatmap_data, cmap='hot', aspect='auto', 
                           extent=[freq_min, freq_max, 0, 1])
            
            plt.colorbar(im, label='Anomaly Strength')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Anomaly Level')
            plt.title('Frequency Anomaly Heatmap')
            
            # Add frequency ticks
            n_ticks = 10
            freq_ticks = np.linspace(freq_min, freq_max, n_ticks)
            plt.xticks(freq_ticks, [f'{f:.0f}' for f in freq_ticks])
            
        else:
            # Single frequency case
            plt.figure(figsize=self.figure_size)
            plt.scatter(frequencies, strengths, c=strengths, cmap='hot', s=100, alpha=0.7)
            plt.colorbar(label='Strength')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Strength')
            plt.title('Anomaly Points')
            plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Anomaly heatmap saved to {output_path}")
        else:
            plt.show()
            
    def create_interactive_plot(self, data: pd.DataFrame, results: Dict = None) -> None:
        """
        Create interactive plotly visualization.
        
        Args:
            data (pd.DataFrame): Combined frequency data
            results (Dict, optional): Analysis results
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Frequency Spectrum', 'Amplitude Comparison', 
                          'Peak Analysis', 'Anomaly Detection'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Main spectrum plot
        fig.add_trace(
            go.Scatter(x=data['frequency'], y=data['audio_amplitude'], 
                      name='Audio', mode='lines', opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data['frequency'], y=data['radio_amplitude'], 
                      name='Radio', mode='lines', opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data['frequency'], y=data['combined_amplitude'], 
                      name='Combined', mode='lines', line=dict(width=3)),
            row=1, col=1
        )
        
        # Amplitude comparison
        if 'amplitude_ratio' in data.columns:
            fig.add_trace(
                go.Scatter(x=data['frequency'], y=data['amplitude_ratio'], 
                          name='Audio/Radio Ratio', mode='lines'),
                row=1, col=2
            )
        
        # Peak analysis
        if 'is_peak' in data.columns:
            peak_data = data[data['is_peak'] == True]
            fig.add_trace(
                go.Scatter(x=data['frequency'], y=data['combined_amplitude'], 
                          name='Spectrum', mode='lines', opacity=0.6),
                row=2, col=1
            )
            if len(peak_data) > 0:
                fig.add_trace(
                    go.Scatter(x=peak_data['frequency'], y=peak_data['combined_amplitude'], 
                              name='Peaks', mode='markers', 
                              marker=dict(size=8, color='red')),
                    row=2, col=1
                )
        
        # Anomaly detection
        if results and 'leakage_points' in results:
            leakage_points = results['leakage_points'][:20]  # Top 20
            if leakage_points:
                frequencies = [point['frequency'] for point in leakage_points]
                strengths = [point['strength'] for point in leakage_points]
                
                fig.add_trace(
                    go.Scatter(x=frequencies, y=strengths, 
                              name='Anomalies', mode='markers',
                              marker=dict(size=10, color='orange', symbol='diamond')),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Interactive Frequency Analysis Dashboard",
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
        fig.update_yaxes(title_text="Ratio", row=1, col=2)
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=2, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)
        fig.update_yaxes(title_text="Anomaly Strength", row=2, col=2)
        
        fig.show()
        
    def generate_summary_report_figure(self, results: Dict, output_path: str = None) -> None:
        """
        Generate comprehensive summary report figure.
        
        Args:
            results (Dict): Complete analysis results
            output_path (str, optional): Path to save figure
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Row 1: Overview metrics
        anomaly_score = results.get('anomaly_score', 0)
        leakage_count = len(results.get('leakage_points', []))
        obscured_count = len(results.get('obscured_layers', []))
        
        overview_metrics = ['Anomaly Score', 'Leakage Points', 'Obscured Layers']
        overview_values = [anomaly_score, leakage_count/10, obscured_count/5]  # Normalize for display
        
        bars = axes[0, 0].bar(overview_metrics, overview_values, 
                             color=['red', 'orange', 'blue'], alpha=0.7)
        axes[0, 0].set_title('Analysis Overview', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Score/Count (normalized)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, [anomaly_score, leakage_count, obscured_count]):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.2f}' if isinstance(value, float) else str(value),
                           ha='center', va='bottom')
        
        # Network topology pie chart
        topology = results.get('network_topology', {})
        if topology and topology.get('layer_count', 0) > 0:
            layer_count = topology['layer_count']
            connectivity = topology.get('connectivity', 0)
            complexity = topology.get('complexity_score', 0)
            
            labels = ['Detected Layers', 'Connectivity', 'Complexity', 'Other']
            sizes = [layer_count, connectivity*10, complexity*10, 
                    max(0, 20 - layer_count - connectivity*10 - complexity*10)]
            
            axes[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Network Topology Distribution', fontsize=14, fontweight='bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'No topology data', ha='center', va='center',
                           transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_title('Network Topology', fontsize=14, fontweight='bold')
        
        # Row 2: Detection methods breakdown
        detection_methods = {}
        for point in results.get('leakage_points', []):
            for method in point.get('detection_methods', []):
                detection_methods[method] = detection_methods.get(method, 0) + 1
        
        if detection_methods:
            methods = list(detection_methods.keys())
            counts = list(detection_methods.values())
            
            axes[1, 0].bar(methods, counts, alpha=0.7, color='green')
            axes[1, 0].set_title('Detection Methods Used', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('Detection Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No detection data', ha='center', va='center',
                           transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_title('Detection Methods', fontsize=14, fontweight='bold')
        
        # Frequency distribution of anomalies
        leakage_points = results.get('leakage_points', [])
        if leakage_points:
            frequencies = [point['frequency'] for point in leakage_points]
            axes[1, 1].hist(frequencies, bins=20, alpha=0.7, color='red', edgecolor='black')
            axes[1, 1].set_title('Anomaly Frequency Distribution', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Frequency (Hz)')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No anomalies detected', ha='center', va='center',
                           transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_title('Anomaly Distribution', fontsize=14, fontweight='bold')
        
        # Row 3: Correlation matrix and recommendations
        correlations = results.get('frequency_correlations', {})
        if 'spectral_features' in correlations and correlations['spectral_features']:
            corr_df = pd.DataFrame(correlations['spectral_features'])
            sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0, 
                       ax=axes[2, 0], cbar_kws={'label': 'Correlation'})
            axes[2, 0].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        else:
            axes[2, 0].text(0.5, 0.5, 'No correlation data', ha='center', va='center',
                           transform=axes[2, 0].transAxes, fontsize=12)
            axes[2, 0].set_title('Feature Correlations', fontsize=14, fontweight='bold')
        
        # Recommendations text
        recommendations = self._generate_recommendations(results)
        axes[2, 1].text(0.05, 0.95, recommendations, transform=axes[2, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[2, 1].set_title('Recommendations', fontsize=14, fontweight='bold')
        axes[2, 1].axis('off')
        
        plt.suptitle('Network Frequency Analysis - Complete Report', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Summary report figure saved to {output_path}")
        else:
            plt.show()
            
    def _generate_recommendations(self, results: Dict) -> str:
        """Generate text recommendations based on analysis results."""
        recommendations = []
        
        anomaly_score = results.get('anomaly_score', 0)
        leakage_count = len(results.get('leakage_points', []))
        obscured_count = len(results.get('obscured_layers', []))
        
        recommendations.append("RECOMMENDATIONS:")
        recommendations.append("================")
        
        if anomaly_score > 0.7:
            recommendations.append("⚠ HIGH ANOMALY SCORE")
            recommendations.append("  • Investigate network integrity")
            recommendations.append("  • Check for interference sources")
        elif anomaly_score > 0.4:
            recommendations.append("⚡ MODERATE ANOMALY SCORE")
            recommendations.append("  • Monitor network performance")
        else:
            recommendations.append("✓ Low anomaly score - network appears stable")
        
        recommendations.append("")
        
        if leakage_count > 10:
            recommendations.append("⚠ MULTIPLE LEAKAGE POINTS")
            recommendations.append("  • Check physical connections")
            recommendations.append("  • Examine signal isolation")
        elif leakage_count > 0:
            recommendations.append("⚡ Some leakage detected")
            recommendations.append("  • Monitor specific frequencies")
        
        recommendations.append("")
        
        if obscured_count > 3:
            recommendations.append("⚠ MULTIPLE OBSCURED LAYERS")
            recommendations.append("  • Review network topology")
            recommendations.append("  • Check layer visibility")
        elif obscured_count > 0:
            recommendations.append("⚡ Some layers obscured")
            recommendations.append("  • Investigate layer access")
        
        recommendations.append("")
        recommendations.append("Next steps:")
        recommendations.append("• Run targeted frequency sweeps")
        recommendations.append("• Implement continuous monitoring")
        recommendations.append("• Document baseline measurements")
        
        return "\n".join(recommendations)
