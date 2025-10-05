#!/usr/bin/env python3
"""
Focused Radio MF Band Analysis (0.3-3 MHz)
Detailed investigation of critical frequency leakage points
"""

import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.frequency_analyzer import FrequencyAnalyzer
from datetime import datetime

def analyze_radio_mf_band():
    """Focused analysis of Radio MF band (0.3-3 MHz)."""
    print("=" * 80)
    print("🎯 FOCUSED RADIO MF BAND ANALYSIS (0.3-3 MHz)")
    print("=" * 80)
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Target Band: Radio Medium Frequency (0.3-3.0 MHz)")
    print()
    
    # Initialize analyzer with high resolution for MF band
    print("🔧 Initializing high-resolution analyzer for MF band...")
    analyzer = FrequencyAnalyzer(resolution_hz=10.0)  # Higher resolution for detailed analysis
    
    # Load data
    analyzer.load_audio_frequencies('data/sample_audio.csv')
    analyzer.load_radio_frequencies('data/sample_radio.csv')
    combined_data = analyzer.combine_frequency_data()
    
    # Filter for Radio MF band (300 kHz to 3 MHz)
    mf_band_min = 300000  # 300 kHz
    mf_band_max = 3000000  # 3 MHz
    
    mf_data = combined_data[
        (combined_data['frequency'] >= mf_band_min) & 
        (combined_data['frequency'] <= mf_band_max)
    ].copy()
    
    print(f"✓ MF Band data isolated: {len(mf_data)} samples")
    print(f"✓ Frequency range: {mf_data['frequency'].min():.0f} - {mf_data['frequency'].max():.0f} Hz")
    print(f"✓ Frequency span: {(mf_data['frequency'].max() - mf_data['frequency'].min())/1000:.1f} kHz")
    print()
    
    # Run focused analysis on MF band
    print("🔍 PERFORMING DETAILED MF BAND ANOMALY DETECTION...")
    
    # Create a temporary analyzer for MF band analysis
    mf_analyzer = FrequencyAnalyzer(resolution_hz=10.0)
    mf_analyzer.combined_data = mf_data
    
    # Run detection on MF band only
    mf_results = mf_analyzer.layer_detector.detect_leakage(mf_data)
    
    print(f"✓ MF Band analysis complete")
    print(f"✓ Leakage points in MF band: {len(mf_results)}")
    print()
    
    # DETAILED MF BAND BREAKDOWN
    print("📊 DETAILED MF BAND FREQUENCY BREAKDOWN")
    print("-" * 60)
    
    # Sub-divide MF band into smaller segments
    mf_segments = [
        ("Lower MF", 300000, 800000),    # 300-800 kHz
        ("Mid MF", 800000, 1500000),     # 800 kHz - 1.5 MHz  
        ("Upper MF", 1500000, 3000000)   # 1.5-3.0 MHz
    ]
    
    segment_analysis = []
    
    for segment_name, low_freq, high_freq in mf_segments:
        segment_data = mf_data[
            (mf_data['frequency'] >= low_freq) & 
            (mf_data['frequency'] <= high_freq)
        ]
        
        if len(segment_data) > 0:
            # Calculate segment statistics
            avg_amplitude = segment_data['combined_amplitude'].mean()
            max_amplitude = segment_data['combined_amplitude'].max()
            std_amplitude = segment_data['combined_amplitude'].std()
            
            # Count leakage points in this segment
            segment_leakage = [p for p in mf_results 
                             if low_freq <= p['frequency'] <= high_freq]
            
            segment_info = {
                'name': segment_name,
                'freq_range': f"{low_freq/1000:.0f}-{high_freq/1000:.0f} kHz",
                'samples': len(segment_data),
                'avg_amp': avg_amplitude,
                'max_amp': max_amplitude,
                'std_amp': std_amplitude,
                'leakage_count': len(segment_leakage),
                'max_leakage_strength': max([p['strength'] for p in segment_leakage]) if segment_leakage else 0
            }
            
            segment_analysis.append(segment_info)
            
            print(f"{segment_name:10} ({segment_info['freq_range']:12}):")
            print(f"  Samples: {segment_info['samples']:4d} | Avg Amp: {segment_info['avg_amp']:6.3f} | Max Amp: {segment_info['max_amp']:6.3f}")
            print(f"  Leakage: {segment_info['leakage_count']:4d} | Max Leak: {segment_info['max_leakage_strength']:6.3f} | Std: {segment_info['std_amp']:6.3f}")
            print()
    
    # CRITICAL FREQUENCY HOTSPOTS
    print("🚨 CRITICAL MF BAND LEAKAGE HOTSPOTS")
    print("-" * 60)
    
    # Sort MF leakage by strength
    mf_results_sorted = sorted(mf_results, key=lambda x: x['strength'], reverse=True)
    
    print(f"{'Rank':<4} {'Frequency':<12} {'Strength':<10} {'Band Segment':<12} {'Risk Level'}")
    print("-" * 60)
    
    for i, point in enumerate(mf_results_sorted[:15]):  # Top 15 in MF band
        freq = point['frequency']
        strength = point['strength']
        
        # Determine which MF segment
        segment = "Unknown"
        for seg_info in segment_analysis:
            # Parse frequency range more carefully
            range_parts = seg_info['freq_range'].replace(' kHz', '').split('-')
            seg_low = float(range_parts[0]) * 1000
            seg_high = float(range_parts[1]) * 1000
            if seg_low <= freq <= seg_high:
                segment = seg_info['name']
                break
        
        # Risk level based on strength
        if strength > 10.0:
            risk = "🔴 EXTREME"
        elif strength > 8.0:
            risk = "🟠 CRITICAL"
        elif strength > 5.0:
            risk = "🟡 HIGH"
        else:
            risk = "🟢 MODERATE"
            
        print(f"{i+1:<4} {freq/1000:8.1f} kHz {strength:<10.3f} {segment:<12} {risk}")
    
    print()
    
    # AMPLITUDE PATTERN ANALYSIS
    print("📈 MF BAND AMPLITUDE PATTERN ANALYSIS")
    print("-" * 60)
    
    # Analyze amplitude patterns in MF band
    audio_mf = mf_data['audio_amplitude']
    radio_mf = mf_data['radio_amplitude']
    combined_mf = mf_data['combined_amplitude']
    
    print(f"Audio Signal in MF Band:")
    print(f"  Mean: {audio_mf.mean():8.4f} | Std: {audio_mf.std():8.4f} | Max: {audio_mf.max():8.4f}")
    print(f"  Range: {audio_mf.max() - audio_mf.min():8.4f} | Median: {audio_mf.median():8.4f}")
    
    print(f"\nRadio Signal in MF Band:")
    print(f"  Mean: {radio_mf.mean():8.4f} | Std: {radio_mf.std():8.4f} | Max: {radio_mf.max():8.4f}")
    print(f"  Range: {radio_mf.max() - radio_mf.min():8.4f} | Median: {radio_mf.median():8.4f}")
    
    print(f"\nCombined Signal in MF Band:")
    print(f"  Mean: {combined_mf.mean():8.4f} | Std: {combined_mf.std():8.4f} | Max: {combined_mf.max():8.4f}")
    print(f"  Range: {combined_mf.max() - combined_mf.min():8.4f} | Median: {combined_mf.median():8.4f}")
    
    # Signal correlation in MF band
    mf_correlation = np.corrcoef(audio_mf, radio_mf)[0, 1]
    print(f"\nAudio-Radio Correlation in MF Band: {mf_correlation:.4f}")
    
    print()
    
    # FREQUENCY SPACING ANALYSIS
    print("🔍 MF BAND FREQUENCY SPACING ANALYSIS")
    print("-" * 60)
    
    if len(mf_results_sorted) > 1:
        # Analyze spacing between leakage points
        leak_frequencies = [p['frequency'] for p in mf_results_sorted]
        leak_frequencies.sort()
        
        spacings = np.diff(leak_frequencies)
        
        print(f"Leakage Point Spacing Analysis:")
        print(f"  Average spacing: {np.mean(spacings)/1000:8.1f} kHz")
        print(f"  Minimum spacing: {np.min(spacings)/1000:8.1f} kHz")
        print(f"  Maximum spacing: {np.max(spacings)/1000:8.1f} kHz")
        print(f"  Spacing std dev: {np.std(spacings)/1000:8.1f} kHz")
        
        # Look for regular patterns
        common_spacings = {}
        for spacing in spacings:
            spacing_khz = round(spacing / 1000, 1)  # Round to 0.1 kHz
            common_spacings[spacing_khz] = common_spacings.get(spacing_khz, 0) + 1
        
        # Find most common spacings
        sorted_spacings = sorted(common_spacings.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_spacings:
            print(f"\nMost Common Frequency Spacings:")
            for spacing_khz, count in sorted_spacings[:5]:
                print(f"  {spacing_khz:6.1f} kHz: {count:2d} occurrences")
    
    print()
    
    # SECURITY ASSESSMENT FOR MF BAND
    print("🛡️ MF BAND SECURITY ASSESSMENT")
    print("-" * 60)
    
    total_mf_leakage = len(mf_results)
    critical_mf_leakage = len([p for p in mf_results if p['strength'] > 8.0])
    extreme_mf_leakage = len([p for p in mf_results if p['strength'] > 10.0])
    
    print(f"MF Band Threat Assessment:")
    print(f"  Total leakage points: {total_mf_leakage}")
    print(f"  Critical threats (>8.0): {critical_mf_leakage}")
    print(f"  Extreme threats (>10.0): {extreme_mf_leakage}")
    
    # Calculate threat density
    mf_span_mhz = (mf_band_max - mf_band_min) / 1000000
    threat_density = total_mf_leakage / mf_span_mhz
    
    print(f"  Threat density: {threat_density:.1f} leaks per MHz")
    
    # Risk classification
    if extreme_mf_leakage > 5:
        risk_level = "🔴 CRITICAL - IMMEDIATE ACTION REQUIRED"
    elif critical_mf_leakage > 10:
        risk_level = "🟠 HIGH - URGENT ATTENTION NEEDED"
    elif total_mf_leakage > 20:
        risk_level = "🟡 MODERATE - MONITORING REQUIRED"
    else:
        risk_level = "🟢 LOW - ROUTINE MAINTENANCE"
    
    print(f"\nOverall MF Band Risk Level: {risk_level}")
    
    # TARGETED RECOMMENDATIONS
    print("\n💡 MF BAND SPECIFIC RECOMMENDATIONS")
    print("-" * 60)
    
    print("Immediate Actions for MF Band (0.3-3 MHz):")
    
    # Recommendations based on segment analysis
    worst_segment = max(segment_analysis, key=lambda x: x['max_leakage_strength'])
    
    print(f"1. 🎯 PRIORITY TARGET: {worst_segment['name']} segment")
    print(f"   Frequency range: {worst_segment['freq_range']}")
    print(f"   Maximum leak strength: {worst_segment['max_leakage_strength']:.3f}")
    print(f"   Leakage count: {worst_segment['leakage_count']}")
    
    print(f"\n2. 🔧 TECHNICAL ACTIONS:")
    print(f"   • Install bandpass filters for {worst_segment['freq_range']}")
    print(f"   • Enhance RF shielding in MF frequency range")
    print(f"   • Implement real-time MF band monitoring")
    
    print(f"\n3. 🚨 EMERGENCY MEASURES:")
    if extreme_mf_leakage > 0:
        print(f"   • Isolate {extreme_mf_leakage} extreme threat frequencies immediately")
        top_threat = mf_results_sorted[0]
        print(f"   • Primary target: {top_threat['frequency']/1000:.1f} kHz (strength: {top_threat['strength']:.3f})")
    
    print(f"\n4. 📊 MONITORING PROTOCOL:")
    print(f"   • Continuous monitoring of 0.8-1.6 MHz (highest concentration)")
    print(f"   • Alert threshold: Signal strength > 8.0")
    print(f"   • Sweep interval: Every 5 minutes for MF band")
    
    # Save MF band specific data
    mf_filename = f"mf_band_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    mf_data.to_csv(mf_filename, index=False)
    
    print(f"\n📁 MF Band data exported to: {mf_filename}")
    
    print()
    print("=" * 80)
    print("🎯 MF BAND FOCUSED ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    analyze_radio_mf_band()
