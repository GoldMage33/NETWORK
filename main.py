#!/usr/bin/env python3
"""
Main example script demonstrating the Network Frequency Analysis Tool.
"""

import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.frequency_analyzer import FrequencyAnalyzer
from src.data_loader import DataLoader


def main(use_hardware=False, hardware_duration=1.0):
    """Main demonstration function."""
    print("NETWORK FREQUENCY ANALYSIS TOOL")
    print("=" * 40)

    # Initialize analyzer
    analyzer = FrequencyAnalyzer(resolution_hz=100.0)
    print("✓ Analyzer initialized")

    if use_hardware:
        # Load hardware data
        print("Loading hardware frequency data...")
        loader = DataLoader()
        hw_data = loader.load_hardware_data(hardware_duration, use_hardware=True)
        
        if hw_data:
            # Set analyzer data directly
            if 'audio' in hw_data:
                analyzer.audio_data = hw_data['audio']
                print(f"✓ Loaded {len(hw_data['audio'])} hardware audio frequency points")
            if 'radio' in hw_data:
                analyzer.radio_data = hw_data['radio']
                print(f"✓ Loaded {len(hw_data['radio'])} hardware radio frequency points")
        else:
            print("⚠ Hardware data collection failed, falling back to sample data")
            use_hardware = False
    
    if not use_hardware:
        # Load sample frequency data
        print("Loading frequency data...")
        analyzer.load_audio_frequencies('data/sample_audio.csv')
        analyzer.load_radio_frequencies('data/sample_radio.csv')
        print("✓ Data loaded")

    # Combine and analyze
    combined_data = analyzer.combine_frequency_data()
    print(f"✓ Combined {len(combined_data)} data points")

    results = analyzer.detect_layer_anomalies()
    print("✓ Analysis complete")

    # Display summary
    print("\nRESULTS:")
    print(f"Anomaly Score: {results['anomaly_score']:.3f}")
    print(f"Leakage Points: {len(results['leakage_points'])}")
    print(f"Obscured Layers: {len(results['obscured_layers'])}")

    # Show top leakage points
    if results['leakage_points']:
        print("\nTop Leakage Points:")
        for i, point in enumerate(results['leakage_points'][:3]):
            print(f"  {point['frequency']:.1f} Hz - Strength: {point['strength']:.3f}")

    # Network topology
    topology = results['network_topology']
    print(f"\nTopology: {topology.get('layer_count', 'Unknown')} layers")

    # Generate report
    analyzer.generate_report(results)
    print("✓ Report generated")

    # Automatically save combined data
    analyzer.export_data('data/combined_frequency_analysis.csv', 'combined')
    print("✓ Data automatically saved")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NETWORK Frequency Analysis Tool')
    parser.add_argument('--hardware', action='store_true', 
                       help='Use hardware data collection instead of sample data')
    parser.add_argument('--duration', type=float, default=1.0,
                       help='Hardware data collection duration in seconds')
    
    args = parser.parse_args()
    main(use_hardware=args.hardware, hardware_duration=args.duration)
