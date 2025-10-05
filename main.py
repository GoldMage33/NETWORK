#!/usr/bin/env python3
"""
Main example script demonstrating the Network Frequency Analysis Tool.

This script shows how to use the FrequencyAnalyzer to detect leakage
and obscured layers in network frequency data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.frequency_analyzer import FrequencyAnalyzer


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("NETWORK FREQUENCY ANALYSIS TOOL")
    print("=" * 60)
    print()
    
    # Initialize analyzer with moderate resolution to avoid memory issues
    print("Initializing frequency analyzer with optimized resolution...")
    analyzer = FrequencyAnalyzer(resolution_hz=100.0)  # Use 100Hz resolution for demo
    print("✓ Analyzer initialized")
    print()
    
    # Load frequency data (will generate sample data if files don't exist)
    print("Loading frequency data...")
    try:
        analyzer.load_audio_frequencies('data/sample_audio.csv')
        analyzer.load_radio_frequencies('data/sample_radio.csv')
        print("✓ Frequency data loaded successfully")
    except Exception as e:
        print(f"Note: Using generated sample data ({e})")
    print()
    
    # Combine frequency data
    print("Combining frequency datasets...")
    combined_data = analyzer.combine_frequency_data()
    print(f"✓ Combined dataset created with {len(combined_data)} samples")
    print(f"  Frequency range: {combined_data['frequency'].min():.1f} - {combined_data['frequency'].max():.1f} Hz")
    print()
    
    # Detect layer anomalies
    print("Detecting network layer anomalies...")
    results = analyzer.detect_layer_anomalies()
    print("✓ Analysis complete")
    print()
    
    # Display results summary
    print("ANALYSIS RESULTS:")
    print("-" * 40)
    print(f"Anomaly Score: {results['anomaly_score']:.3f}")
    print(f"Leakage Points: {len(results['leakage_points'])}")
    print(f"Obscured Layers: {len(results['obscured_layers'])}")
    
    # Show top leakage points
    if results['leakage_points']:
        print("\nTop 5 Leakage Points:")
        for i, point in enumerate(results['leakage_points'][:5]):
            print(f"  {i+1}. {point['frequency']:.1f} Hz - Strength: {point['strength']:.3f}")
            print(f"     Detection methods: {', '.join(point['detection_methods'])}")
    
    # Show obscured layers
    if results['obscured_layers']:
        print(f"\nObscured Layers ({len(results['obscured_layers'])}):")
        for layer in results['obscured_layers']:
            freq_range = layer['frequency_range']
            print(f"  Layer {layer['id']}: {freq_range[0]:.1f}-{freq_range[1]:.1f} Hz")
            print(f"    Size: {layer['size']}, Avg Amplitude: {layer['avg_amplitude']:.3f}")
    
    # Network topology
    topology = results['network_topology']
    print(f"\nNetwork Topology:")
    print(f"  Estimated layers: {topology.get('layer_count', 'Unknown')}")
    print(f"  Connectivity score: {topology.get('connectivity', 0):.3f}")
    print(f"  Complexity score: {topology.get('complexity_score', 0):.3f}")
    
    # Show dominant frequencies
    dominant_freqs = topology.get('dominant_frequencies', [])
    if dominant_freqs:
        print(f"  Dominant frequencies: {', '.join([f'{f:.1f}Hz' for f in dominant_freqs[:5]])}")
    print()
    
    # Generate comprehensive report
    print("Generating analysis report...")
    analyzer.generate_report(results)
    print("✓ Report generated")
    print()
    
    # Offer to save data
    save_option = input("Save combined frequency data to CSV? (y/n): ").lower().strip()
    if save_option == 'y':
        output_file = 'data/combined_frequency_analysis.csv'
        analyzer.export_data(output_file, 'combined')
        print(f"✓ Data saved to {output_file}")
    
    print()
    print("Analysis complete! Check the generated plots and report for detailed insights.")
    print("=" * 60)


if __name__ == "__main__":
    main()
