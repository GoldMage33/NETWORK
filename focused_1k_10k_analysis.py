#!/usr/bin/env python3
"""
Focused 1kHz-10kHz Frequency Analysis Program
Detailed analysis and visualization of the 1kHz to 10kHz frequency range
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from scipy.signal import find_peaks, welch, spectrogram
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

def analyze_1k_10k_range():
    """Main analysis function for 1kHz-10kHz frequency range."""
    
    print("=" * 80)
    print("🎯 FOCUSED 1kHz-10kHz FREQUENCY ANALYSIS PROGRAM")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load the CSV data
    try:
        print("📁 Loading frequency data...")
        df = pd.read_csv('data/combined_frequency_analysis.csv')
        print(f"✓ Full dataset loaded: {len(df):,} points")
        
        # Filter for 1kHz-10kHz range
        freq_mask = (df['frequency'] >= 1000) & (df['frequency'] <= 10000)
        focus_df = df[freq_mask].copy()
        
        print(f"🎯 Filtered to 1kHz-10kHz range: {len(focus_df)} points")
        
        if len(focus_df) == 0:
            print("❌ No data points found in 1kHz-10kHz range!")
            print("💡 Generating synthetic data for demonstration...")
            focus_df = generate_synthetic_1k_10k_data()
        
        print(f"   • Frequency range: {focus_df['frequency'].min():.1f} - {focus_df['frequency'].max():.1f} Hz")
        print()
        
    except FileNotFoundError:
        print("❌ CSV file not found. Generating synthetic 1kHz-10kHz data...")
        focus_df = generate_synthetic_1k_10k_data()
    
    # Create comprehensive analysis
    create_focused_graphs(focus_df)
    analyze_frequency_characteristics(focus_df)
    detect_patterns_and_anomalies(focus_df)
    
    return focus_df


def generate_synthetic_1k_10k_data():
    """Generate realistic synthetic data for 1kHz-10kHz range."""
    
    print("🔧 Generating synthetic 1kHz-10kHz frequency data...")
    
    # Create frequency array with high resolution
    frequencies = np.linspace(1000, 10000, 1000)  # 1000 points for high resolution
    
    # Generate realistic audio frequency response
    audio_amplitudes = []
    radio_amplitudes = []
    
    for freq in frequencies:
        # Audio characteristics in 1-10kHz range
        # Peak around 3-4kHz (human voice range)
        audio_base = 0.1 + 0.3 * np.exp(-((freq - 3500) / 1500) ** 2)
        audio_noise = np.random.normal(0, 0.02)
        
        # Add harmonics at 2kHz, 4kHz, 6kHz, 8kHz
        harmonics = 0
        for harmonic in [2000, 4000, 6000, 8000]:
            if abs(freq - harmonic) < 100:
                harmonics += 0.15 * np.exp(-((freq - harmonic) / 50) ** 2)
        
        audio_amp = max(0.01, audio_base + harmonics + audio_noise)
        audio_amplitudes.append(audio_amp)
        
        # Radio interference patterns
        # Stronger at certain frequencies (potential interference)
        radio_base = 0.05 + 0.1 * np.sin(freq * 0.001)
        
        # Add specific interference frequencies
        interference = 0
        interference_freqs = [1500, 2500, 3500, 5000, 7500, 9000]
        for int_freq in interference_freqs:
            if abs(freq - int_freq) < 200:
                interference += 0.2 * np.exp(-((freq - int_freq) / 100) ** 2)
        
        radio_noise = np.random.normal(0, 0.01)
        radio_amp = max(0.01, radio_base + interference + radio_noise)
        radio_amplitudes.append(radio_amp)
    
    # Create DataFrame
    synthetic_df = pd.DataFrame({
        'frequency': frequencies,
        'audio_amplitude': audio_amplitudes,
        'radio_amplitude': radio_amplitudes,
        'combined_amplitude': [a + r for a, r in zip(audio_amplitudes, radio_amplitudes)],
        'amplitude_ratio': [a / r if r > 0 else 0 for a, r in zip(audio_amplitudes, radio_amplitudes)]
    })
    
    print(f"✓ Generated {len(synthetic_df)} synthetic data points")
    return synthetic_df


def create_focused_graphs(df):
    """Create comprehensive graphs for 1kHz-10kHz analysis."""
    
    print("📊 Creating focused 1kHz-10kHz graphs...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    
    # 1. Main frequency spectrum (high resolution)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['frequency'], df['audio_amplitude'], 
            label='Audio Signal', linewidth=2, alpha=0.8, color='#1f77b4')
    ax1.plot(df['frequency'], df['radio_amplitude'], 
            label='Radio Signal', linewidth=2, alpha=0.8, color='#ff7f0e')
    ax1.plot(df['frequency'], df['combined_amplitude'], 
            label='Combined Signal', linewidth=3, alpha=0.9, color='#2ca02c')
    
    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title('1kHz-10kHz Frequency Spectrum Analysis', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1000, 10000)
    
    # Add frequency markers for important audio bands
    audio_bands = [
        (1000, 2000, 'Low Mid', 'lightblue'),
        (2000, 4000, 'Mid Range', 'lightgreen'),
        (4000, 6000, 'Upper Mid', 'lightyellow'),
        (6000, 8000, 'Presence', 'lightcoral'),
        (8000, 10000, 'Brilliance', 'lightpink')
    ]
    
    for low, high, name, color in audio_bands:
        ax1.axvspan(low, high, alpha=0.1, color=color)
        ax1.text((low + high) / 2, ax1.get_ylim()[1] * 0.9, name, 
                ha='center', va='center', fontsize=9, rotation=0)
    
    # 2. Peak detection and analysis
    ax2 = fig.add_subplot(gs[1, 0])
    peaks, properties = find_peaks(df['combined_amplitude'], 
                                  height=df['combined_amplitude'].mean(),
                                  distance=max(1, len(df)//50))
    
    ax2.plot(df['frequency'], df['combined_amplitude'], 
            linewidth=2, color='#17becf', alpha=0.8, label='Combined Signal')
    if len(peaks) > 0:
        ax2.plot(df.iloc[peaks]['frequency'], df.iloc[peaks]['combined_amplitude'], 
                'ro', markersize=8, alpha=0.8, label=f'{len(peaks)} Peaks')
        
        # Annotate top 3 peaks
        peak_amps = df.iloc[peaks]['combined_amplitude'].values
        top_peaks = np.argsort(peak_amps)[-3:]
        for i, peak_idx in enumerate(peaks[top_peaks]):
            freq = df.iloc[peak_idx]['frequency']
            amp = df.iloc[peak_idx]['combined_amplitude']
            ax2.annotate(f'{freq:.0f}Hz\n{amp:.3f}', 
                        (freq, amp), xytext=(10, 10), 
                        textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax2.set_xlabel('Frequency (Hz)', fontsize=10)
    ax2.set_ylabel('Combined Amplitude', fontsize=10)
    ax2.set_title('Peak Detection Analysis', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Amplitude ratio analysis
    ax3 = fig.add_subplot(gs[1, 1])
    clean_ratio = df['amplitude_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    ax3.plot(df['frequency'], clean_ratio, 
            linewidth=2, color='#9467bd', alpha=0.8)
    ax3.set_xlabel('Frequency (Hz)', fontsize=10)
    ax3.set_ylabel('Audio/Radio Ratio', fontsize=10)
    ax3.set_title('Audio to Radio Ratio', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Signal correlation analysis
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.scatter(df['audio_amplitude'], df['radio_amplitude'], 
               alpha=0.6, s=30, c=df['frequency'], cmap='viridis')
    ax4.set_xlabel('Audio Amplitude', fontsize=10)
    ax4.set_ylabel('Radio Amplitude', fontsize=10)
    ax4.set_title('Audio vs Radio Correlation', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Frequency (Hz)', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Spectral density analysis
    ax5 = fig.add_subplot(gs[2, 0])
    freqs, psd = welch(df['combined_amplitude'], fs=len(df)/(df['frequency'].max()-df['frequency'].min()))
    ax5.semilogy(freqs, psd, linewidth=2, color='#8c564b')
    ax5.set_xlabel('Normalized Frequency', fontsize=10)
    ax5.set_ylabel('Power Spectral Density', fontsize=10)
    ax5.set_title('Power Spectral Density', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Amplitude distribution
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(df['audio_amplitude'], bins=30, alpha=0.7, label='Audio', color='#1f77b4')
    ax6.hist(df['radio_amplitude'], bins=30, alpha=0.7, label='Radio', color='#ff7f0e')
    ax6.hist(df['combined_amplitude'], bins=30, alpha=0.7, label='Combined', color='#2ca02c')
    ax6.set_xlabel('Amplitude', fontsize=10)
    ax6.set_ylabel('Frequency Count', fontsize=10)
    ax6.set_title('Amplitude Distribution', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 7. Running average analysis
    ax7 = fig.add_subplot(gs[2, 2])
    window_size = max(1, len(df) // 20)
    running_avg = df['combined_amplitude'].rolling(window=window_size, center=True).mean()
    ax7.plot(df['frequency'], df['combined_amplitude'], alpha=0.3, color='gray', label='Raw')
    ax7.plot(df['frequency'], running_avg, linewidth=3, color='red', label=f'Running Avg ({window_size})')
    ax7.set_xlabel('Frequency (Hz)', fontsize=10)
    ax7.set_ylabel('Amplitude', fontsize=10)
    ax7.set_title('Smoothed Trend Analysis', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # 8. Frequency band energy analysis
    ax8 = fig.add_subplot(gs[3, :])
    
    # Divide 1-10kHz into sub-bands
    sub_bands = [
        ('1-2kHz', 1000, 2000),
        ('2-3kHz', 2000, 3000),
        ('3-4kHz', 3000, 4000),
        ('4-5kHz', 4000, 5000),
        ('5-6kHz', 5000, 6000),
        ('6-7kHz', 6000, 7000),
        ('7-8kHz', 7000, 8000),
        ('8-9kHz', 8000, 9000),
        ('9-10kHz', 9000, 10000)
    ]
    
    band_names = []
    audio_energies = []
    radio_energies = []
    combined_energies = []
    
    for band_name, low_freq, high_freq in sub_bands:
        band_mask = (df['frequency'] >= low_freq) & (df['frequency'] <= high_freq)
        band_data = df[band_mask]
        
        if len(band_data) > 0:
            # Calculate energy (sum of squared amplitudes)
            audio_energy = np.sum(band_data['audio_amplitude'] ** 2)
            radio_energy = np.sum(band_data['radio_amplitude'] ** 2)
            combined_energy = np.sum(band_data['combined_amplitude'] ** 2)
            
            band_names.append(band_name)
            audio_energies.append(audio_energy)
            radio_energies.append(radio_energy)
            combined_energies.append(combined_energy)
    
    x_pos = np.arange(len(band_names))
    width = 0.25
    
    ax8.bar(x_pos - width, audio_energies, width, label='Audio Energy', alpha=0.8, color='#1f77b4')
    ax8.bar(x_pos, radio_energies, width, label='Radio Energy', alpha=0.8, color='#ff7f0e')
    ax8.bar(x_pos + width, combined_energies, width, label='Combined Energy', alpha=0.8, color='#2ca02c')
    
    ax8.set_xlabel('Frequency Band', fontsize=12)
    ax8.set_ylabel('Energy (Amplitude²·Hz)', fontsize=12)
    ax8.set_title('Energy Distribution Across 1kHz Sub-bands', fontsize=14, fontweight='bold')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(band_names, rotation=45, ha='right')
    ax8.legend(fontsize=11)
    ax8.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle('Comprehensive 1kHz-10kHz Frequency Analysis', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Save the graph
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'focused_1k_10k_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Focused analysis graph saved as: {filename}")
    
    plt.show()
    
    return peaks


def analyze_frequency_characteristics(df):
    """Analyze specific characteristics of the 1kHz-10kHz range."""
    
    print("\n" + "="*60)
    print("🔍 DETAILED 1kHz-10kHz CHARACTERISTICS ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   • Frequency points: {len(df)}")
    print(f"   • Resolution: {(df['frequency'].max() - df['frequency'].min()) / len(df):.2f} Hz per point")
    print(f"   • Audio amplitude range: {df['audio_amplitude'].min():.4f} - {df['audio_amplitude'].max():.4f}")
    print(f"   • Radio amplitude range: {df['radio_amplitude'].min():.4f} - {df['radio_amplitude'].max():.4f}")
    print(f"   • Combined amplitude range: {df['combined_amplitude'].min():.4f} - {df['combined_amplitude'].max():.4f}")
    
    # Find dominant frequencies
    print(f"\n🎯 DOMINANT FREQUENCIES:")
    top_indices = df.nlargest(5, 'combined_amplitude').index
    for i, idx in enumerate(top_indices):
        freq = df.loc[idx, 'frequency']
        amp = df.loc[idx, 'combined_amplitude']
        audio_amp = df.loc[idx, 'audio_amplitude']
        radio_amp = df.loc[idx, 'radio_amplitude']
        print(f"   {i+1}. {freq:.1f} Hz - Combined: {amp:.4f} (Audio: {audio_amp:.4f}, Radio: {radio_amp:.4f})")
    
    # Signal-to-noise analysis
    print(f"\n📡 SIGNAL QUALITY ANALYSIS:")
    mean_combined = df['combined_amplitude'].mean()
    std_combined = df['combined_amplitude'].std()
    snr = mean_combined / std_combined if std_combined > 0 else float('inf')
    print(f"   • Signal-to-Noise Ratio: {snr:.2f}")
    print(f"   • Signal stability (1/CV): {1/(std_combined/mean_combined) if mean_combined > 0 else 0:.2f}")
    
    # Audio vs Radio dominance
    audio_dominant = (df['audio_amplitude'] > df['radio_amplitude']).sum()
    radio_dominant = (df['radio_amplitude'] > df['audio_amplitude']).sum()
    print(f"   • Audio dominant frequencies: {audio_dominant} ({audio_dominant/len(df)*100:.1f}%)")
    print(f"   • Radio dominant frequencies: {radio_dominant} ({radio_dominant/len(df)*100:.1f}%)")


def detect_patterns_and_anomalies(df):
    """Detect patterns and anomalies in the 1kHz-10kHz range."""
    
    print(f"\n🔍 PATTERN AND ANOMALY DETECTION:")
    print("-" * 40)
    
    # Detect sudden amplitude changes
    amplitude_diff = df['combined_amplitude'].diff().abs()
    high_variation_threshold = amplitude_diff.quantile(0.95)
    high_variation_points = df[amplitude_diff > high_variation_threshold]
    
    if len(high_variation_points) > 0:
        print(f"⚠️  High variation points detected: {len(high_variation_points)}")
        print("   Top 3 variation points:")
        for i, (idx, row) in enumerate(high_variation_points.head(3).iterrows()):
            print(f"      {i+1}. {row['frequency']:.1f} Hz - Amplitude: {row['combined_amplitude']:.4f}")
    
    # Detect periodic patterns
    from scipy.fft import fft, fftfreq
    
    # FFT analysis to find periodic components
    fft_vals = fft(df['combined_amplitude'].values)
    fft_freqs = fftfreq(len(df), d=(df['frequency'].iloc[1] - df['frequency'].iloc[0]))
    
    # Find dominant periodic components
    fft_magnitudes = np.abs(fft_vals)
    dominant_indices = np.argsort(fft_magnitudes)[-5:]  # Top 5
    
    print(f"\n🌊 PERIODIC PATTERN ANALYSIS:")
    print("   Dominant periodic components:")
    for i, idx in enumerate(reversed(dominant_indices)):
        if idx > 0 and idx < len(fft_freqs)//2:  # Avoid DC and Nyquist
            period_freq = abs(fft_freqs[idx])
            magnitude = fft_magnitudes[idx]
            if period_freq > 0:
                period = 1 / period_freq
                print(f"      {i+1}. Period: {period:.1f} Hz, Strength: {magnitude:.2f}")
    
    # Anomaly detection using statistical methods
    Q1 = df['combined_amplitude'].quantile(0.25)
    Q3 = df['combined_amplitude'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    anomalies = df[(df['combined_amplitude'] < lower_bound) | 
                   (df['combined_amplitude'] > upper_bound)]
    
    print(f"\n🚨 STATISTICAL ANOMALIES:")
    print(f"   • Anomalous points: {len(anomalies)} ({len(anomalies)/len(df)*100:.1f}%)")
    if len(anomalies) > 0:
        print("   • Anomaly frequencies:")
        for i, (idx, row) in enumerate(anomalies.head(5).iterrows()):
            anomaly_type = "High" if row['combined_amplitude'] > upper_bound else "Low"
            print(f"      {i+1}. {row['frequency']:.1f} Hz - {anomaly_type} amplitude: {row['combined_amplitude']:.4f}")


def create_additional_plots(df):
    """Create additional specialized plots for the 1kHz-10kHz range."""
    
    print(f"\n📊 Creating additional specialized plots...")
    
    # Create a second figure for additional analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Waterfall plot simulation
    axes[0, 0].imshow([df['combined_amplitude'].values], 
                     aspect='auto', cmap='viridis', 
                     extent=[df['frequency'].min(), df['frequency'].max(), 0, 1])
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('Time (normalized)')
    axes[0, 0].set_title('Frequency Waterfall View')
    
    # 2. Phase relationship (simulated)
    phase_diff = np.cumsum(df['amplitude_ratio'].fillna(0)) % (2 * np.pi)
    axes[0, 1].plot(df['frequency'], phase_diff, linewidth=2, color='purple')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Phase Difference (radians)')
    axes[0, 1].set_title('Audio-Radio Phase Relationship')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Harmonic analysis
    fundamental_freq = 1000  # 1kHz fundamental
    harmonics = []
    harmonic_amps = []
    
    for harmonic in range(2, 11):  # 2nd to 10th harmonic
        harm_freq = fundamental_freq * harmonic
        if harm_freq <= 10000:
            # Find closest frequency in data
            closest_idx = (df['frequency'] - harm_freq).abs().idxmin()
            harmonics.append(harmonic)
            harmonic_amps.append(df.loc[closest_idx, 'combined_amplitude'])
    
    if harmonics:
        axes[1, 0].bar(harmonics, harmonic_amps, alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Harmonic Number')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].set_title('Harmonic Series Analysis (1kHz fundamental)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Frequency stability analysis
    stability_window = max(1, len(df) // 10)
    stability_metric = df['combined_amplitude'].rolling(window=stability_window).std()
    axes[1, 1].plot(df['frequency'], stability_metric, linewidth=2, color='red')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Local Stability (std dev)')
    axes[1, 1].set_title('Frequency Stability Analysis')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save additional plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    additional_filename = f'additional_1k_10k_plots_{timestamp}.png'
    plt.savefig(additional_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Additional plots saved as: {additional_filename}")
    
    plt.show()


if __name__ == "__main__":
    print("🚀 Starting focused 1kHz-10kHz analysis program...")
    df = analyze_1k_10k_range()
    create_additional_plots(df)
    print("\n✅ Focused 1kHz-10kHz analysis complete!")
