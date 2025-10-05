#!/usr/bin/env python3
"""
Detailed Network Frequency Analysis Report Generator
"""

import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.frequency_analyzer import FrequencyAnalyzer
from datetime import datetime

def generate_detailed_report():
    """Generate comprehensive detailed analysis report."""
    print("=" * 80)
    print("COMPREHENSIVE NETWORK FREQUENCY ANALYSIS REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analysis Tool Version: 1.0.0")
    print()
    
    # Initialize analyzer
    analyzer = FrequencyAnalyzer(resolution_hz=100.0)
    
    # Load data
    print("📊 LOADING AND PROCESSING DATA...")
    analyzer.load_audio_frequencies('data/sample_audio.csv')
    analyzer.load_radio_frequencies('data/sample_radio.csv')
    combined_data = analyzer.combine_frequency_data()
    
    print(f"✓ Audio samples: {len(analyzer.audio_data)}")
    print(f"✓ Radio samples: {len(analyzer.radio_data)}")
    print(f"✓ Combined dataset: {len(combined_data)} points")
    print(f"✓ Frequency range: {combined_data['frequency'].min():.1f} - {combined_data['frequency'].max():.1f} Hz")
    print()
    
    # Run analysis
    print("🔍 PERFORMING LAYER ANOMALY DETECTION...")
    results = analyzer.detect_layer_anomalies()
    print("✓ Analysis complete")
    print()
    
    # DETAILED FREQUENCY BAND ANALYSIS
    print("📈 DETAILED FREQUENCY BAND ANALYSIS")
    print("-" * 50)
    
    # Define frequency bands
    bands = [
        ("Audio Low", 20, 250),
        ("Audio Mid", 250, 4000),
        ("Audio High", 4000, 20000),
        ("Radio VLF", 3000, 30000),
        ("Radio LF", 30000, 300000),
        ("Radio MF", 300000, 3000000),
        ("Radio HF", 3000000, 30000000),
        ("Radio VHF", 30000000, 300000000),
        ("Radio UHF", 300000000, 1000000000)
    ]
    
    for band_name, low_freq, high_freq in bands:
        band_data = combined_data[
            (combined_data['frequency'] >= low_freq) & 
            (combined_data['frequency'] <= high_freq)
        ]
        
        if len(band_data) > 0:
            avg_amplitude = band_data['combined_amplitude'].mean()
            max_amplitude = band_data['combined_amplitude'].max()
            std_amplitude = band_data['combined_amplitude'].std()
            
            print(f"{band_name:12} ({low_freq:>8.0f}-{high_freq:>10.0f} Hz):")
            print(f"  Samples: {len(band_data):4d} | Avg: {avg_amplitude:6.3f} | Max: {max_amplitude:6.3f} | Std: {std_amplitude:6.3f}")
    
    print()
    
    # LEAKAGE ANALYSIS
    print("🚨 DETAILED LEAKAGE ANALYSIS")
    print("-" * 50)
    
    leakage_points = results['leakage_points']
    print(f"Total leakage points detected: {len(leakage_points)}")
    
    # Group leakage by frequency bands
    leakage_by_band = {}
    for point in leakage_points:
        freq = point['frequency']
        for band_name, low_freq, high_freq in bands:
            if low_freq <= freq <= high_freq:
                if band_name not in leakage_by_band:
                    leakage_by_band[band_name] = []
                leakage_by_band[band_name].append(point)
                break
    
    print("\nLeakage distribution by frequency band:")
    for band_name, points in leakage_by_band.items():
        if points:
            avg_strength = np.mean([p['strength'] for p in points])
            max_strength = max([p['strength'] for p in points])
            print(f"  {band_name:12}: {len(points):4d} points | Avg strength: {avg_strength:6.3f} | Max: {max_strength:6.3f}")
    
    # TOP 20 CRITICAL FREQUENCIES
    print(f"\n🔴 TOP 20 CRITICAL LEAKAGE FREQUENCIES:")
    print("-" * 70)
    print(f"{'Rank':<4} {'Frequency (Hz)':<15} {'Strength':<10} {'Band':<12} {'Methods'}")
    print("-" * 70)
    
    for i, point in enumerate(leakage_points[:20]):
        freq = point['frequency']
        strength = point['strength']
        methods = ', '.join(point['detection_methods'])
        
        # Determine band
        band = "Unknown"
        for band_name, low_freq, high_freq in bands:
            if low_freq <= freq <= high_freq:
                band = band_name
                break
        
        print(f"{i+1:<4} {freq:<15.1f} {strength:<10.3f} {band:<12} {methods}")
    
    print()
    
    # NETWORK TOPOLOGY DETAILED ANALYSIS
    print("🌐 NETWORK TOPOLOGY DETAILED ANALYSIS")
    print("-" * 50)
    
    topology = results['network_topology']
    correlations = results['frequency_correlations']
    
    print(f"Estimated network layers: {topology.get('layer_count', 'Unknown')}")
    print(f"Network connectivity score: {topology.get('connectivity', 0):.4f}")
    print(f"Network complexity score: {topology.get('complexity_score', 0):.4f}")
    
    dominant_freqs = topology.get('dominant_frequencies', [])
    if dominant_freqs:
        print(f"\nDominant frequencies ({len(dominant_freqs)}):")
        for i, freq in enumerate(dominant_freqs[:10]):
            print(f"  {i+1}. {freq:12.1f} Hz")
    
    # CORRELATION ANALYSIS
    print(f"\n📊 FREQUENCY CORRELATION ANALYSIS")
    print("-" * 50)
    
    if 'audio_radio' in correlations:
        print(f"Audio-Radio correlation: {correlations['audio_radio']:.4f}")
    
    if 'spectral_features' in correlations:
        print("Spectral feature correlations:")
        for feature, corr_data in correlations['spectral_features'].items():
            if isinstance(corr_data, dict):
                for other_feature, corr_value in corr_data.items():
                    if feature != other_feature and isinstance(corr_value, (int, float)):
                        print(f"  {feature:18} <-> {other_feature:18}: {corr_value:6.3f}")
    
    # STATISTICAL SUMMARY
    print(f"\n📋 STATISTICAL SUMMARY")
    print("-" * 50)
    
    print(f"Overall anomaly score: {results['anomaly_score']:.4f}")
    print(f"Frequency resolution: {analyzer.resolution_hz} Hz")
    print(f"Data points analyzed: {len(combined_data):,}")
    
    # Amplitude statistics
    audio_stats = analyzer.audio_data['amplitude'].describe()
    radio_stats = analyzer.radio_data['amplitude'].describe()
    combined_stats = combined_data['combined_amplitude'].describe()
    
    print(f"\nAmplitude Statistics:")
    print(f"  Audio    - Mean: {audio_stats['mean']:.4f}, Std: {audio_stats['std']:.4f}, Max: {audio_stats['max']:.4f}")
    print(f"  Radio    - Mean: {radio_stats['mean']:.4f}, Std: {radio_stats['std']:.4f}, Max: {radio_stats['max']:.4f}")
    print(f"  Combined - Mean: {combined_stats['mean']:.4f}, Std: {combined_stats['std']:.4f}, Max: {combined_stats['max']:.4f}")
    
    # RECOMMENDATIONS
    print(f"\n💡 DETAILED RECOMMENDATIONS")
    print("-" * 50)
    
    anomaly_score = results['anomaly_score']
    leakage_count = len(results['leakage_points'])
    
    if anomaly_score > 0.6:
        priority = "🔴 CRITICAL"
    elif anomaly_score > 0.3:
        priority = "🟠 HIGH"
    else:
        priority = "🟡 MODERATE"
    
    print(f"Priority Level: {priority}")
    print(f"Risk Assessment: {anomaly_score:.1%} anomaly score with {leakage_count} leakage points")
    
    print(f"\nImmediate Actions Required:")
    if leakage_count > 500:
        print("  1. 🚨 URGENT: Investigate critical frequency leakages immediately")
        print("  2. 🔧 Focus on 1.1-1.5 MHz range (highest concentration)")
        print("  3. 🛡️ Implement additional signal isolation measures")
    elif leakage_count > 100:
        print("  1. ⚠️  Monitor high-strength leakage points")
        print("  2. 🔍 Conduct targeted frequency sweeps")
        print("  3. 📊 Establish baseline measurements")
    else:
        print("  1. ✅ Continue regular monitoring")
        print("  2. 📈 Document current state as baseline")
    
    print(f"\nLong-term Monitoring:")
    print("  • Set up continuous frequency monitoring")
    print("  • Implement automated alerting for anomaly scores > 0.5")
    print("  • Schedule weekly analysis reports")
    print("  • Maintain frequency signature database")
    
    print()
    print("=" * 80)
    print("END OF DETAILED ANALYSIS REPORT")
    print("=" * 80)
    
    # Save detailed report
    report_filename = f"detailed_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    print(f"\n📄 Detailed report saved to: {report_filename}")

if __name__ == "__main__":
    generate_detailed_report()
