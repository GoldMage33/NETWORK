#!/usr/bin/env python3
"""
Enhanced Network Frequency Analysis with Arduino ATS-20 DSP Receiver Integration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.frequency_analyzer import FrequencyAnalyzer
from datetime import datetime
import time

# Mock ATS-20 integration for demonstration (real hardware interface available in ats20_receiver.py)
class MockATS20Receiver:
    """Mock ATS-20 receiver for demonstration without hardware."""
    
    def __init__(self, port='/dev/ttyUSB0'):
        self.port = port
        self.is_connected = False
        self.current_frequency = 14000000  # 14 MHz
        self.freq_range = {'min': 150000, 'max': 30000000}
        
    def connect(self):
        """Simulate ATS-20 connection."""
        print(f"🔗 Simulating ATS-20 DSP Receiver connection on {self.port}")
        print("   Note: This is a simulation. For real hardware, install pyserial:")
        print("   pip install pyserial")
        self.is_connected = True
        return True
        
    def disconnect(self):
        """Simulate disconnection."""
        self.is_connected = False
        print("✓ ATS-20 receiver disconnected (simulated)")
        
    def set_frequency(self, freq_hz):
        """Simulate frequency setting."""
        if self.freq_range['min'] <= freq_hz <= self.freq_range['max']:
            self.current_frequency = freq_hz
            return True
        return False
        
    def get_signal_strength(self):
        """Simulate signal strength reading."""
        import random
        return -120 + random.uniform(0, 60)  # -120 to -60 dBm
        
    def frequency_sweep(self, start_freq, end_freq, step_hz=1000):
        """Simulate frequency sweep."""
        import pandas as pd
        import numpy as np
        
        print(f"🔍 ATS-20 Frequency Sweep: {start_freq/1000:.0f}-{end_freq/1000:.0f} kHz")
        
        sweep_data = []
        frequencies = range(start_freq, end_freq + step_hz, step_hz)
        
        for i, freq in enumerate(frequencies):
            # Simulate realistic signal readings
            noise_floor = -110 + np.random.normal(0, 5)
            
            # Add some "signals" for demo
            signal_strength = noise_floor
            if freq in [1203375, 1110809, 1295941]:  # Our known leakage frequencies
                signal_strength += 30 + np.random.normal(0, 3)
                
            sweep_data.append({
                'frequency': freq,
                'signal_strength': signal_strength,
                'timestamp': datetime.now().isoformat()
            })
            
            if i % 50 == 0:
                progress = (i / len(list(frequencies))) * 100
                print(f"   ATS-20 Progress: {progress:.1f}%")
        
        return pd.DataFrame(sweep_data)


def main_with_ats20():
    """Enhanced main function with ATS-20 DSP receiver integration."""
    print("=" * 80)
    print("🎯 NETWORK FREQUENCY ANALYSIS WITH ATS-20 DSP RECEIVER")
    print("=" * 80)
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize components
    print("🔧 INITIALIZING ANALYSIS COMPONENTS...")
    analyzer = FrequencyAnalyzer(resolution_hz=100.0)
    ats20 = MockATS20Receiver('/dev/ttyUSB0')  # Use MockATS20Receiver for demo
    print("✓ Frequency analyzer initialized")
    
    # Connect to ATS-20
    print("\n📡 CONNECTING TO ATS-20 DSP RECEIVER...")
    if ats20.connect():
        print("✓ ATS-20 DSP Receiver connected")
        print(f"  Frequency range: {ats20.freq_range['min']/1000:.0f}-{ats20.freq_range['max']/1000:.0f} kHz")
    else:
        print("❌ ATS-20 connection failed - continuing with software analysis only")
    
    # Load baseline frequency data
    print("\n📊 LOADING BASELINE FREQUENCY DATA...")
    analyzer.load_audio_frequencies('data/sample_audio.csv')
    analyzer.load_radio_frequencies('data/sample_radio.csv')
    combined_data = analyzer.combine_frequency_data()
    print(f"✓ Baseline data loaded: {len(combined_data)} samples")
    
    # Perform initial software analysis
    print("\n🔍 PERFORMING INITIAL SOFTWARE ANALYSIS...")
    results = analyzer.detect_layer_anomalies()
    print("✓ Initial analysis complete")
    print(f"  Anomaly score: {results['anomaly_score']:.3f}")
    print(f"  Leakage points detected: {len(results['leakage_points'])}")
    
    # Get top frequencies for hardware verification
    top_frequencies = []
    for point in results['leakage_points'][:10]:  # Top 10 frequencies
        freq_hz = int(point['frequency'])
        if ats20.freq_range['min'] <= freq_hz <= ats20.freq_range['max']:
            top_frequencies.append(freq_hz)
    
    print(f"  Frequencies for ATS-20 verification: {len(top_frequencies)}")
    
    # Hardware verification with ATS-20
    if ats20.is_connected and top_frequencies:
        print("\n🎯 HARDWARE VERIFICATION WITH ATS-20...")
        
        hardware_results = []
        
        for i, freq in enumerate(top_frequencies[:5]):  # Verify top 5
            print(f"   Verifying {freq/1000:.1f} kHz ({i+1}/{min(5, len(top_frequencies))})")
            
            if ats20.set_frequency(freq):
                time.sleep(0.1)  # Allow receiver to settle
                signal_strength = ats20.get_signal_strength()
                
                hardware_results.append({
                    'frequency': freq,
                    'software_strength': next(p['strength'] for p in results['leakage_points'] if int(p['frequency']) == freq),
                    'hardware_signal': signal_strength,
                    'verified': signal_strength > -80  # Signal detected threshold
                })
                
                status = "✓ CONFIRMED" if signal_strength > -80 else "❌ NOT DETECTED"
                print(f"     Software: {hardware_results[-1]['software_strength']:.3f} | Hardware: {signal_strength:.1f} dBm | {status}")
        
        # Hardware verification summary
        verified_count = sum(1 for r in hardware_results if r['verified'])
        print(f"\n📊 HARDWARE VERIFICATION SUMMARY:")
        print(f"   Frequencies tested: {len(hardware_results)}")
        print(f"   Hardware confirmed: {verified_count}")
        print(f"   Verification rate: {(verified_count/len(hardware_results)*100):.1f}%")
    
    # Focused sweep on MF band with ATS-20
    if ats20.is_connected:
        print(f"\n🔍 ATS-20 FOCUSED SWEEP ON MF BAND (0.3-3 MHz)...")
        
        # Define MF band sweep parameters
        mf_start = 300000   # 300 kHz
        mf_end = 3000000    # 3 MHz
        mf_step = 5000      # 5 kHz steps
        
        sweep_data = ats20.frequency_sweep(mf_start, mf_end, mf_step)
        
        if not sweep_data.empty:
            print(f"✓ MF band sweep complete: {len(sweep_data)} data points")
            
            # Analyze sweep results
            strong_signals = sweep_data[sweep_data['signal_strength'] > -80]
            very_strong = sweep_data[sweep_data['signal_strength'] > -60]
            
            print(f"   Strong signals (>-80 dBm): {len(strong_signals)}")
            print(f"   Very strong signals (>-60 dBm): {len(very_strong)}")
            
            if len(strong_signals) > 0:
                print(f"\n🚨 TOP ATS-20 DETECTED SIGNALS IN MF BAND:")
                top_signals = strong_signals.nlargest(5, 'signal_strength')
                
                for i, (_, signal) in enumerate(top_signals.iterrows()):
                    freq_khz = signal['frequency'] / 1000
                    strength = signal['signal_strength']
                    print(f"   {i+1}. {freq_khz:8.1f} kHz | {strength:6.1f} dBm")
            
            # Save sweep data
            sweep_filename = f"ats20_mf_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            sweep_data.to_csv(sweep_filename, index=False)
            print(f"   Sweep data saved to: {sweep_filename}")
    
    # Combined analysis report
    print(f"\n📋 COMBINED ANALYSIS REPORT")
    print("-" * 60)
    print(f"Analysis Method: Software + Hardware (ATS-20)")
    print(f"Software Analysis:")
    print(f"  Anomaly Score: {results['anomaly_score']:.3f}")
    print(f"  Leakage Points: {len(results['leakage_points'])}")
    print(f"  Network Layers: {results['network_topology'].get('layer_count', 'Unknown')}")
    
    if ats20.is_connected:
        print(f"Hardware Verification:")
        print(f"  ATS-20 Model: DSP Receiver (150 kHz - 30 MHz)")
        print(f"  Connection Status: Connected")
        if 'hardware_results' in locals():
            print(f"  Verified Frequencies: {verified_count}/{len(hardware_results)}")
    
    # Generate comprehensive report
    print(f"\n📄 GENERATING COMPREHENSIVE REPORT...")
    analyzer.generate_report(results)
    
    # Recommendations
    print(f"\n💡 ENHANCED RECOMMENDATIONS WITH ATS-20 INTEGRATION:")
    print("-" * 60)
    print("1. 🎯 TARGETED MONITORING:")
    print("   • Use ATS-20 for continuous monitoring of verified frequencies")
    print("   • Set up automated sweeps every 15 minutes")
    print("   • Focus on MF band (0.3-3 MHz) for real-time detection")
    
    print("\n2. 🔧 HARDWARE DEPLOYMENT:")
    print("   • Deploy ATS-20 at network perimeter points")
    print("   • Configure alert thresholds: >-80 dBm for investigation")
    print("   • Implement logging for all signals >-90 dBm")
    
    print("\n3. 📊 DATA INTEGRATION:")
    print("   • Correlate software analysis with hardware measurements")
    print("   • Build baseline profiles using ATS-20 data")
    print("   • Create automated reporting from hardware monitoring")
    
    # Cleanup
    if ats20.is_connected:
        ats20.disconnect()
    
    print()
    print("=" * 80)
    print("🎯 ENHANCED NETWORK ANALYSIS WITH ATS-20 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main_with_ats20()
