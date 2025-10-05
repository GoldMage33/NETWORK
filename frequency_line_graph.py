#!/usr/bin/env python3
"""
Combined Frequency Chart Line Graph Generator
Creates comprehensive line graph visualizations of frequency analysis data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns

def create_combined_frequency_line_graph():
    """Generate comprehensive line graph of combined frequency analysis."""
    
    print("=" * 60)
    print("📊 COMBINED FREQUENCY LINE GRAPH GENERATOR")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load the combined frequency data
    try:
        print("📈 Loading combined frequency analysis data...")
        df = pd.read_csv('data/combined_frequency_analysis.csv')
        print(f"✓ Data loaded: {len(df)} frequency points")
        print(f"  Frequency range: {df['frequency'].min():.0f} - {df['frequency'].max():.0f} Hz")
        print()
    except FileNotFoundError:
        print("❌ Combined frequency analysis file not found!")
        print("   Please run the main analysis first: python main.py")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create comprehensive multi-panel line graph
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Combined Network Frequency Analysis - Line Graphs', fontsize=16, fontweight='bold')
    
    # Panel 1: Main Combined Frequency Spectrum (Linear Scale)
    axes[0, 0].plot(df['frequency']/1000, df['audio_amplitude'], 
                   label='Audio Signal', linewidth=1.5, alpha=0.8, color='blue')
    axes[0, 0].plot(df['frequency']/1000, df['radio_amplitude'], 
                   label='Radio Signal', linewidth=1.5, alpha=0.8, color='red')
    axes[0, 0].plot(df['frequency']/1000, df['combined_amplitude'], 
                   label='Combined Signal', linewidth=2, alpha=0.9, color='purple')
    
    axes[0, 0].set_xlabel('Frequency (kHz)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Combined Frequency Spectrum (Linear Scale)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(df['frequency'].min()/1000, df['frequency'].max()/1000)
    
    # Panel 2: Log Scale Frequency Spectrum
    axes[0, 1].semilogx(df['frequency'], df['combined_amplitude'], 
                       linewidth=2, color='green', alpha=0.8, label='Combined Signal')
    axes[0, 1].semilogx(df['frequency'], df['audio_amplitude'], 
                       linewidth=1, color='blue', alpha=0.6, label='Audio Signal')
    axes[0, 1].semilogx(df['frequency'], df['radio_amplitude'], 
                       linewidth=1, color='red', alpha=0.6, label='Radio Signal')
    
    axes[0, 1].set_xlabel('Frequency (Hz) - Log Scale')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Frequency Spectrum (Logarithmic Scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Panel 3: Amplitude Ratio Analysis
    if 'amplitude_ratio' in df.columns:
        # Clean up infinite and NaN values for plotting
        clean_ratio = df['amplitude_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        axes[1, 0].plot(df['frequency']/1000000, clean_ratio, 
                       linewidth=1.5, color='orange', alpha=0.8)
        axes[1, 0].set_xlabel('Frequency (MHz)')
        axes[1, 0].set_ylabel('Audio/Radio Ratio')
        axes[1, 0].set_title('Audio to Radio Amplitude Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, np.percentile(clean_ratio[clean_ratio > 0], 95))  # Limit to 95th percentile
    
    # Panel 4: Frequency Band Analysis
    # Define frequency bands for analysis
    bands = [
        ("Audio Low", 20, 250, 'blue'),
        ("Audio Mid", 250, 4000, 'cyan'),
        ("Audio High", 4000, 20000, 'lightblue'),
        ("Radio LF", 30000, 300000, 'orange'),
        ("Radio MF", 300000, 3000000, 'red'),
        ("Radio HF", 3000000, 30000000, 'darkred'),
        ("Radio VHF", 30000000, 300000000, 'purple'),
        ("Radio UHF", 300000000, 1000000000, 'magenta')
    ]
    
    # Calculate average amplitude for each band
    band_data = []
    for band_name, low_freq, high_freq, color in bands:
        band_mask = (df['frequency'] >= low_freq) & (df['frequency'] <= high_freq)
        band_df = df[band_mask]
        
        if len(band_df) > 0:
            avg_amplitude = band_df['combined_amplitude'].mean()
            max_amplitude = band_df['combined_amplitude'].max()
            freq_center = (low_freq + high_freq) / 2
            
            band_data.append({
                'band': band_name,
                'center_freq': freq_center,
                'avg_amplitude': avg_amplitude,
                'max_amplitude': max_amplitude,
                'color': color,
                'sample_count': len(band_df)
            })
    
    if band_data:
        band_df = pd.DataFrame(band_data)
        
        # Plot band analysis as connected line
        axes[1, 1].plot(range(len(band_df)), band_df['avg_amplitude'], 
                       'o-', linewidth=2, markersize=8, color='darkgreen', 
                       label='Average Amplitude')
        axes[1, 1].plot(range(len(band_df)), band_df['max_amplitude'], 
                       's-', linewidth=2, markersize=6, color='red', alpha=0.7,
                       label='Maximum Amplitude')
        
        axes[1, 1].set_xlabel('Frequency Band')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].set_title('Amplitude by Frequency Band')
        axes[1, 1].set_xticks(range(len(band_df)))
        axes[1, 1].set_xticklabels(band_df['band'], rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comprehensive line graph
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'combined_frequency_line_graph_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Comprehensive line graph saved as: {filename}")
    
    # Show the plot
    plt.show()
    
    # Create focused MF band line graph
    print("\n📊 Creating focused MF band line graph...")
    create_mf_band_line_graph(df, timestamp)
    
    # Generate statistics summary
    print("\n📋 FREQUENCY DATA STATISTICS:")
    print("-" * 40)
    print(f"Total data points: {len(df):,}")
    print(f"Frequency span: {(df['frequency'].max() - df['frequency'].min())/1000000:.1f} MHz")
    print(f"Average combined amplitude: {df['combined_amplitude'].mean():.4f}")
    print(f"Maximum combined amplitude: {df['combined_amplitude'].max():.4f}")
    print(f"Standard deviation: {df['combined_amplitude'].std():.4f}")
    
    # Identify peaks in the line graph
    identify_frequency_peaks(df)


def create_mf_band_line_graph(full_df, timestamp):
    """Create focused line graph for MF band (0.3-3 MHz)."""
    
    # Filter for MF band
    mf_mask = (full_df['frequency'] >= 300000) & (full_df['frequency'] <= 3000000)
    mf_df = full_df[mf_mask].copy()
    
    if len(mf_df) == 0:
        print("❌ No MF band data available")
        return
    
    # Create MF band focused line graph
    plt.figure(figsize=(14, 8))
    
    # Main plot with dual y-axis
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Left y-axis - Amplitude
    color1 = 'tab:blue'
    ax1.set_xlabel('Frequency (kHz)', fontsize=12)
    ax1.set_ylabel('Amplitude', color=color1, fontsize=12)
    
    line1 = ax1.plot(mf_df['frequency']/1000, mf_df['audio_amplitude'], 
                    color='blue', linewidth=2, alpha=0.8, label='Audio Signal')
    line2 = ax1.plot(mf_df['frequency']/1000, mf_df['radio_amplitude'], 
                    color='red', linewidth=2, alpha=0.8, label='Radio Signal')
    line3 = ax1.plot(mf_df['frequency']/1000, mf_df['combined_amplitude'], 
                    color='purple', linewidth=3, alpha=0.9, label='Combined Signal')
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Right y-axis - Ratio (if available)
    if 'amplitude_ratio' in mf_df.columns:
        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Audio/Radio Ratio', color=color2, fontsize=12)
        
        clean_ratio = mf_df['amplitude_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
        line4 = ax2.plot(mf_df['frequency']/1000, clean_ratio, 
                        color='orange', linewidth=2, alpha=0.7, linestyle='--', 
                        label='Amplitude Ratio')
        ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title and formatting
    plt.title('MF Band (0.3-3 MHz) Detailed Line Graph Analysis', fontsize=14, fontweight='bold')
    
    # Combined legend
    lines1 = line1 + line2 + line3
    labels1 = [l.get_label() for l in lines1]
    
    if 'amplitude_ratio' in mf_df.columns:
        lines1 += line4
        labels1 += [line4[0].get_label()]
    
    ax1.legend(lines1, labels1, loc='upper right')
    
    # Add frequency markers for known bands
    mf_bands = [
        (535, 1605, 'AM Broadcast', 'lightgray'),
        (1800, 2000, 'Amateur 160m', 'lightgreen'),
        (2300, 2495, 'Shortwave', 'lightyellow')
    ]
    
    for low, high, name, color in mf_bands:
        if low >= mf_df['frequency'].min()/1000 and high <= mf_df['frequency'].max()/1000:
            ax1.axvspan(low, high, alpha=0.2, color=color, label=f'{name} Band')
    
    plt.tight_layout()
    
    # Save MF band graph
    mf_filename = f'mf_band_line_graph_{timestamp}.png'
    plt.savefig(mf_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ MF band line graph saved as: {mf_filename}")
    plt.show()
    
    # MF band statistics
    print(f"📊 MF BAND STATISTICS:")
    print(f"   Data points in MF band: {len(mf_df)}")
    print(f"   Frequency range: {mf_df['frequency'].min()/1000:.1f} - {mf_df['frequency'].max()/1000:.1f} kHz")
    print(f"   Average amplitude: {mf_df['combined_amplitude'].mean():.4f}")
    print(f"   Peak amplitude: {mf_df['combined_amplitude'].max():.4f}")


def identify_frequency_peaks(df):
    """Identify and report frequency peaks in the line graph."""
    
    from scipy.signal import find_peaks
    
    print(f"\n🔍 FREQUENCY PEAK ANALYSIS:")
    print("-" * 40)
    
    # Find peaks in combined amplitude
    peaks, properties = find_peaks(
        df['combined_amplitude'], 
        height=df['combined_amplitude'].mean() + df['combined_amplitude'].std(),
        distance=len(df)//100,  # Minimum distance between peaks
        prominence=0.01
    )
    
    if len(peaks) > 0:
        peak_data = []
        for i, peak_idx in enumerate(peaks):
            freq_hz = df.iloc[peak_idx]['frequency']
            amplitude = df.iloc[peak_idx]['combined_amplitude']
            
            peak_data.append({
                'rank': i + 1,
                'frequency_hz': freq_hz,
                'frequency_khz': freq_hz / 1000,
                'frequency_mhz': freq_hz / 1000000,
                'amplitude': amplitude
            })
        
        # Sort by amplitude (highest first)
        peak_data.sort(key=lambda x: x['amplitude'], reverse=True)
        
        print(f"📈 Top {min(10, len(peak_data))} Amplitude Peaks:")
        print(f"{'Rank':<4} {'Frequency':<12} {'Amplitude':<10} {'Band'}")
        print("-" * 40)
        
        for peak in peak_data[:10]:
            freq_display = f"{peak['frequency_khz']:.1f} kHz"
            if peak['frequency_hz'] > 1000000:
                freq_display = f"{peak['frequency_mhz']:.2f} MHz"
            
            # Determine frequency band
            freq = peak['frequency_hz']
            if freq < 20000:
                band = "Audio"
            elif freq < 300000:
                band = "LF"
            elif freq < 3000000:
                band = "MF"
            elif freq < 30000000:
                band = "HF"
            elif freq < 300000000:
                band = "VHF"
            else:
                band = "UHF"
            
            print(f"{peak['rank']:<4} {freq_display:<12} {peak['amplitude']:<10.4f} {band}")
    else:
        print("❌ No significant peaks detected in frequency spectrum")


if __name__ == "__main__":
    create_combined_frequency_line_graph()
