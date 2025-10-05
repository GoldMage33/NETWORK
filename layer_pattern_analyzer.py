#!/usr/bin/env python3
"""
Layer Pattern Analyzer - Scans layer attributes and their peaks to find patterns 
that emerge into higher frequencies around 10kHz.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_layer_frequency_patterns(csv_file):
    """Analyze frequency patterns across network layers to identify high-frequency emergence."""
    
    print("🔍 LAYER FREQUENCY PATTERN ANALYSIS")
    print("=" * 60)
    
    # Load the data
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} layer signature records")
        print(f"📅 Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"🌐 Network layers: {df['layer'].unique()}")
        print()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Convert timestamp to datetime for analysis
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. FREQUENCY PROGRESSION ANALYSIS
    print("📈 FREQUENCY PROGRESSION ANALYSIS")
    print("-" * 40)
    
    # Group by layer and analyze frequency patterns
    layer_freq_stats = {}
    for layer in df['layer'].unique():
        layer_data = df[df['layer'] == layer]
        freq_stats = {
            'mean_freq': layer_data['dominant_frequency'].mean(),
            'max_freq': layer_data['dominant_frequency'].max(),
            'min_freq': layer_data['dominant_frequency'].min(),
            'freq_range': layer_data['dominant_frequency'].max() - layer_data['dominant_frequency'].min(),
            'std_freq': layer_data['dominant_frequency'].std(),
            'high_freq_count': len(layer_data[layer_data['dominant_frequency'] >= 9000]),
            'near_10khz_count': len(layer_data[layer_data['dominant_frequency'] >= 9500]),
            'max_activity': layer_data['activity_score'].max(),
            'avg_activity': layer_data['activity_score'].mean()
        }
        layer_freq_stats[layer] = freq_stats
        
        print(f"🔸 {layer.replace('_', ' ').title()}:")
        print(f"   • Frequency Range: {freq_stats['min_freq']:.0f} - {freq_stats['max_freq']:.0f} Hz")
        print(f"   • Mean Frequency: {freq_stats['mean_freq']:.0f} Hz")
        print(f"   • High Freq (≥9kHz): {freq_stats['high_freq_count']} occurrences")
        print(f"   • Near 10kHz (≥9.5kHz): {freq_stats['near_10khz_count']} occurrences")
        print(f"   • Max Activity Score: {freq_stats['max_activity']:.2f}")
        print()
    
    # 2. HIGH FREQUENCY EMERGENCE PATTERNS
    print("🚀 HIGH FREQUENCY EMERGENCE PATTERNS (≥9kHz)")
    print("-" * 50)
    
    high_freq_data = df[df['dominant_frequency'] >= 9000]
    if len(high_freq_data) > 0:
        print(f"Total high-frequency events: {len(high_freq_data)}")
        
        # Analyze by layer
        for layer in high_freq_data['layer'].unique():
            layer_high_freq = high_freq_data[high_freq_data['layer'] == layer]
            print(f"\n🔹 {layer.replace('_', ' ').title()} - High Frequency Analysis:")
            print(f"   • Events: {len(layer_high_freq)}")
            print(f"   • Frequency Range: {layer_high_freq['dominant_frequency'].min():.0f} - {layer_high_freq['dominant_frequency'].max():.0f} Hz")
            print(f"   • Peak Activity: {layer_high_freq['activity_score'].max():.2f}")
            print(f"   • Peak Amplitude: {layer_high_freq['peak_amplitude'].max():.3f}")
            
            # Look for patterns
            cross_layer_events = len(layer_high_freq[layer_high_freq['patterns'].str.contains('cross_layer_interference', na=False)])
            high_activity_events = len(layer_high_freq[layer_high_freq['patterns'].str.contains('high_activity', na=False)])
            amplitude_spikes = len(layer_high_freq[layer_high_freq['patterns'].str.contains('amplitude_spike', na=False)])
            
            print(f"   • Cross-layer interference: {cross_layer_events} events")
            print(f"   • High activity patterns: {high_activity_events} events")
            print(f"   • Amplitude spikes: {amplitude_spikes} events")
    
    # 3. TEMPORAL PATTERN ANALYSIS
    print("\n⏰ TEMPORAL PROGRESSION TO HIGH FREQUENCIES")
    print("-" * 45)
    
    # Sort by timestamp and analyze frequency progression over time
    df_sorted = df.sort_values('timestamp')
    
    # Look for frequency escalation patterns
    for layer in df['layer'].unique():
        layer_data = df_sorted[df_sorted['layer'] == layer].copy()
        if len(layer_data) < 5:
            continue
            
        # Calculate frequency trend
        layer_data['time_index'] = range(len(layer_data))
        freq_trend = np.polyfit(layer_data['time_index'], layer_data['dominant_frequency'], 1)[0]
        
        # Find frequency jumps
        layer_data['freq_diff'] = layer_data['dominant_frequency'].diff()
        large_jumps = layer_data[abs(layer_data['freq_diff']) > 1000]
        
        print(f"🔸 {layer.replace('_', ' ').title()}:")
        print(f"   • Frequency Trend: {freq_trend:+.1f} Hz per scan")
        print(f"   • Large Frequency Jumps (>1kHz): {len(large_jumps)}")
        
        if len(large_jumps) > 0:
            max_jump = large_jumps.loc[abs(large_jumps['freq_diff']).idxmax()]
            print(f"   • Largest Jump: {max_jump['freq_diff']:+.0f} Hz at {max_jump['timestamp']}")
            print(f"     From {max_jump['dominant_frequency'] - max_jump['freq_diff']:.0f} to {max_jump['dominant_frequency']:.0f} Hz")
    
    # 4. CORRELATION ANALYSIS
    print(f"\n🔗 CORRELATION ANALYSIS")
    print("-" * 25)
    
    # Analyze correlations between frequency, activity, and amplitude
    numeric_cols = ['dominant_frequency', 'activity_score', 'peak_amplitude']
    correlation_matrix = df[numeric_cols].corr()
    
    print("Correlation Matrix:")
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i < j:
                corr_val = correlation_matrix.loc[col1, col2]
                print(f"   • {col1} vs {col2}: {corr_val:.3f}")
    
    # 5. PATTERN EMERGENCE HYPOTHESIS
    print(f"\n🧠 PATTERN EMERGENCE HYPOTHESIS")
    print("-" * 35)
    
    # Identify the most likely emergence pattern
    app_layer_high_freq = df[(df['layer'] == 'application_layer') & (df['dominant_frequency'] >= 9000)]
    
    if len(app_layer_high_freq) > 0:
        print("🎯 APPLICATION LAYER EMERGENCE DETECTED:")
        print(f"   • {len(app_layer_high_freq)} high-frequency events in application layer")
        print(f"   • Peak frequency: {app_layer_high_freq['dominant_frequency'].max():.0f} Hz")
        print(f"   • Average activity during high-freq: {app_layer_high_freq['activity_score'].mean():.2f}")
        
        # Check for concurrent activity in other layers
        for _, event in app_layer_high_freq.iterrows():
            timestamp = event['timestamp']
            concurrent = df[(df['timestamp'] == timestamp) & (df['layer'] != 'application_layer')]
            
            if len(concurrent) > 0:
                active_layers = concurrent[concurrent['activity_score'] > 1.0]['layer'].tolist()
                if active_layers:
                    print(f"   • Concurrent activity at {timestamp}: {', '.join(active_layers)}")
                    break
    
    # 6. VISUALIZATION
    create_frequency_analysis_plots(df, csv_file)
    
    return layer_freq_stats

def create_frequency_analysis_plots(df, csv_file):
    """Create comprehensive visualization of frequency patterns."""
    
    print(f"\n📊 GENERATING FREQUENCY ANALYSIS PLOTS")
    print("-" * 40)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Network Layer Frequency Pattern Analysis - Path to 10kHz', fontsize=16, fontweight='bold')
    
    # 1. Frequency Distribution by Layer
    ax1 = axes[0, 0]
    layer_order = ['physical_layer', 'data_link_layer', 'network_layer', 'transport_layer', 
                   'session_layer', 'presentation_layer', 'application_layer']
    
    df_plot = df[df['layer'].isin(layer_order)]
    sns.boxplot(data=df_plot, x='layer', y='dominant_frequency', ax=ax1)
    ax1.set_title('Frequency Distribution by Layer')
    ax1.set_xlabel('Network Layer')
    ax1.set_ylabel('Dominant Frequency (Hz)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.axhline(y=10000, color='red', linestyle='--', alpha=0.7, label='10kHz Target')
    ax1.legend()
    
    # 2. High Frequency Events (≥9kHz)
    ax2 = axes[0, 1]
    high_freq_data = df[df['dominant_frequency'] >= 9000]
    if len(high_freq_data) > 0:
        layer_counts = high_freq_data['layer'].value_counts()
        layer_counts.plot(kind='bar', ax=ax2, color='orange')
        ax2.set_title('High Frequency Events (≥9kHz) by Layer')
        ax2.set_xlabel('Network Layer')
        ax2.set_ylabel('Event Count')
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. Activity Score vs Frequency
    ax3 = axes[0, 2]
    scatter_colors = {'physical_layer': 'blue', 'data_link_layer': 'green', 'network_layer': 'orange',
                     'transport_layer': 'red', 'session_layer': 'purple', 'presentation_layer': 'brown',
                     'application_layer': 'pink'}
    
    for layer in df['layer'].unique():
        layer_data = df[df['layer'] == layer]
        ax3.scatter(layer_data['activity_score'], layer_data['dominant_frequency'], 
                   alpha=0.6, label=layer.replace('_', ' ').title(), 
                   color=scatter_colors.get(layer, 'gray'), s=30)
    
    ax3.set_xlabel('Activity Score')
    ax3.set_ylabel('Dominant Frequency (Hz)')
    ax3.set_title('Activity vs Frequency Correlation')
    ax3.axhline(y=10000, color='red', linestyle='--', alpha=0.7)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 4. Frequency Evolution Over Time
    ax4 = axes[1, 0]
    
    # Convert timestamp to numeric for plotting
    df_sorted = df.sort_values('timestamp')
    df_sorted['time_numeric'] = pd.to_datetime(df_sorted['timestamp']).astype(np.int64) // 10**9
    
    # Plot frequency evolution for application layer (most likely to reach 10kHz)
    app_layer_data = df_sorted[df_sorted['layer'] == 'application_layer']
    if len(app_layer_data) > 1:
        ax4.plot(app_layer_data['time_numeric'], app_layer_data['dominant_frequency'], 
                'o-', label='Application Layer', color='red', alpha=0.8)
    
    # Plot other layers with high frequencies
    for layer in ['transport_layer', 'network_layer']:
        layer_data = df_sorted[df_sorted['layer'] == layer]
        if len(layer_data) > 1:
            ax4.plot(layer_data['time_numeric'], layer_data['dominant_frequency'], 
                    'o-', alpha=0.6, label=layer.replace('_', ' ').title(), markersize=4)
    
    ax4.set_xlabel('Time (Unix Timestamp)')
    ax4.set_ylabel('Dominant Frequency (Hz)')
    ax4.set_title('Frequency Evolution Over Time')
    ax4.axhline(y=10000, color='red', linestyle='--', alpha=0.7, label='10kHz Target')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Peak Amplitude Distribution
    ax5 = axes[1, 1]
    high_freq_data = df[df['dominant_frequency'] >= 9000]
    if len(high_freq_data) > 0:
        ax5.hist(high_freq_data['peak_amplitude'], bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax5.set_xlabel('Peak Amplitude')
        ax5.set_ylabel('Frequency Count')
        ax5.set_title('Amplitude Distribution in High-Freq Range')
        ax5.axvline(x=high_freq_data['peak_amplitude'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {high_freq_data["peak_amplitude"].mean():.3f}')
        ax5.legend()
    
    # 6. Cross-Layer Interference Patterns
    ax6 = axes[1, 2]
    cross_layer_data = df[df['patterns'].str.contains('cross_layer_interference', na=False)]
    
    if len(cross_layer_data) > 0:
        # Create a heatmap of cross-layer interference by frequency bands
        freq_bands = pd.cut(cross_layer_data['dominant_frequency'], 
                           bins=[0, 2000, 4000, 6000, 8000, 10000, 12000], 
                           labels=['0-2k', '2-4k', '4-6k', '6-8k', '8-10k', '10k+'])
        
        heatmap_data = pd.crosstab(cross_layer_data['layer'], freq_bands)
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax6)
        ax6.set_title('Cross-Layer Interference by Frequency Band')
        ax6.set_xlabel('Frequency Band (Hz)')
        ax6.set_ylabel('Network Layer')
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'layer_frequency_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Analysis plot saved as: {filename}")
    
    plt.show()

def identify_10khz_emergence_pathway(df):
    """Identify the specific pathway patterns that lead to 10kHz frequencies."""
    
    print(f"\n🎯 10kHZ EMERGENCE PATHWAY ANALYSIS")
    print("=" * 45)
    
    # Find events closest to 10kHz
    near_10khz = df[df['dominant_frequency'] >= 9500].copy()
    
    if len(near_10khz) == 0:
        print("❌ No events found near 10kHz (≥9.5kHz)")
        return
    
    print(f"📍 Found {len(near_10khz)} events near 10kHz (≥9.5kHz)")
    
    # Group by layer and analyze
    pathway_analysis = {}
    
    for layer in near_10khz['layer'].unique():
        layer_events = near_10khz[near_10khz['layer'] == layer]
        
        pathway_analysis[layer] = {
            'event_count': len(layer_events),
            'max_frequency': layer_events['dominant_frequency'].max(),
            'avg_activity': layer_events['activity_score'].mean(),
            'peak_amplitude': layer_events['peak_amplitude'].max(),
            'pattern_types': layer_events['patterns'].str.split(',').explode().str.strip().value_counts().to_dict()
        }
        
        print(f"\n🔸 {layer.replace('_', ' ').title()}:")
        print(f"   • Events near 10kHz: {len(layer_events)}")
        print(f"   • Highest frequency: {layer_events['dominant_frequency'].max():.0f} Hz")
        print(f"   • Average activity: {layer_events['activity_score'].mean():.2f}")
        print(f"   • Peak amplitude: {layer_events['peak_amplitude'].max():.3f}")
        
        # Show dominant patterns
        patterns = layer_events['patterns'].str.split(',').explode().str.strip().value_counts()
        if len(patterns) > 0:
            print(f"   • Dominant patterns:")
            for pattern, count in patterns.head(3).items():
                if pd.notna(pattern) and pattern != '':
                    print(f"     - {pattern}: {count} occurrences")
    
    # Identify the most promising pathway
    max_freq_layer = None
    max_freq_value = 0
    
    for layer, stats in pathway_analysis.items():
        if stats['max_frequency'] > max_freq_value:
            max_freq_value = stats['max_frequency']
            max_freq_layer = layer
    
    if max_freq_layer:
        print(f"\n🚀 PRIMARY 10kHz EMERGENCE PATHWAY:")
        print(f"   🎯 Layer: {max_freq_layer.replace('_', ' ').title()}")
        print(f"   📊 Peak Frequency: {max_freq_value:.0f} Hz")
        print(f"   📈 Activity Level: {pathway_analysis[max_freq_layer]['avg_activity']:.2f}")
        print(f"   🔊 Peak Amplitude: {pathway_analysis[max_freq_layer]['peak_amplitude']:.3f}")
        
        # Distance to 10kHz
        distance_to_10khz = 10000 - max_freq_value
        print(f"   📏 Distance to 10kHz: {distance_to_10khz:.0f} Hz ({(distance_to_10khz/10000)*100:.1f}%)")
        
        if distance_to_10khz < 500:
            print(f"   ✅ VERY CLOSE TO 10kHz TARGET!")
        elif distance_to_10khz < 1000:
            print(f"   🟡 APPROACHING 10kHz TARGET")
        else:
            print(f"   🔴 SIGNIFICANT GAP TO 10kHz")
    
    return pathway_analysis

def analyze_advanced_layer_patterns(df):
    """Discover advanced patterns in network layer behavior."""
    
    print(f"\n🔍 ADVANCED LAYER PATTERN DISCOVERY")
    print("=" * 45)
    
    # 1. HARMONIC FREQUENCY PATTERNS
    print("🎵 HARMONIC FREQUENCY ANALYSIS")
    print("-" * 35)
    
    harmonic_patterns = {}
    for layer in df['layer'].unique():
        layer_data = df[df['layer'] == layer]
        frequencies = layer_data['dominant_frequency'].values
        
        # Find harmonic relationships
        harmonics = []
        for i, freq1 in enumerate(frequencies):
            for j, freq2 in enumerate(frequencies[i+1:], i+1):
                if freq1 > 0 and freq2 > 0:
                    ratio = freq2 / freq1 if freq1 < freq2 else freq1 / freq2
                    # Check for harmonic relationships (2:1, 3:2, 4:3, etc.)
                    harmonic_ratios = [2.0, 1.5, 1.33, 1.25, 3.0, 4.0]
                    for h_ratio in harmonic_ratios:
                        if abs(ratio - h_ratio) < 0.1:
                            harmonics.append({
                                'freq1': freq1, 'freq2': freq2, 'ratio': ratio,
                                'harmonic_type': f"{h_ratio:.1f}:1"
                            })
        
        if harmonics:
            harmonic_patterns[layer] = harmonics
            print(f"🔸 {layer.replace('_', ' ').title()}:")
            print(f"   • Harmonic relationships found: {len(harmonics)}")
            for h in harmonics[:3]:  # Show top 3
                print(f"     - {h['freq1']:.0f}Hz : {h['freq2']:.0f}Hz = {h['harmonic_type']}")
    
    # 2. PHASE SYNCHRONIZATION PATTERNS
    print(f"\n🌊 PHASE SYNCHRONIZATION ANALYSIS")
    print("-" * 40)
    
    df_sorted = df.sort_values('timestamp')
    sync_patterns = {}
    
    # Group by timestamp to find simultaneous events
    time_groups = df_sorted.groupby('timestamp')
    sync_events = []
    
    for timestamp, group in time_groups:
        if len(group) > 1:  # Multiple layers active at same time
            layers = group['layer'].tolist()
            frequencies = group['dominant_frequency'].tolist()
            activities = group['activity_score'].tolist()
            
            # Check for frequency coherence
            freq_std = np.std(frequencies)
            freq_mean = np.mean(frequencies)
            coherence = 1 - (freq_std / freq_mean) if freq_mean > 0 else 0
            
            sync_events.append({
                'timestamp': timestamp,
                'layers': layers,
                'frequencies': frequencies,
                'activities': activities,
                'coherence': coherence,
                'layer_count': len(layers)
            })
    
    # Find high synchronization events
    high_sync = [e for e in sync_events if e['coherence'] > 0.8 and e['layer_count'] >= 3]
    if high_sync:
        print(f"High synchronization events found: {len(high_sync)}")
        for event in high_sync[:3]:
            print(f"   • {event['timestamp']}: {event['layer_count']} layers")
            print(f"     Coherence: {event['coherence']:.3f}, Avg freq: {np.mean(event['frequencies']):.0f}Hz")
    
    # 3. CROSS-LAYER CASCADE PATTERNS
    print(f"\n⚡ CASCADE PATTERN ANALYSIS")
    print("-" * 30)
    
    cascade_patterns = []
    
    # Look for activity cascading through layers
    layer_order = ['physical_layer', 'data_link_layer', 'network_layer', 'transport_layer', 
                   'session_layer', 'presentation_layer', 'application_layer']
    
    for i in range(len(df_sorted) - 6):  # Need at least 7 consecutive events
        window = df_sorted.iloc[i:i+7]
        
        # Check if we have one event from each layer in sequence
        if len(window['layer'].unique()) == 7:
            layer_sequence = window['layer'].tolist()
            time_span = (pd.to_datetime(window['timestamp'].iloc[-1]) - 
                        pd.to_datetime(window['timestamp'].iloc[0])).total_seconds()
            
            # Check if it follows OSI layer order (roughly)
            order_score = 0
            for j, layer in enumerate(layer_sequence):
                expected_position = layer_order.index(layer) if layer in layer_order else -1
                if expected_position >= 0:
                    order_score += max(0, 7 - abs(j - expected_position))
            
            if order_score > 20 and time_span < 5:  # Good order within 5 seconds
                cascade_patterns.append({
                    'start_time': window['timestamp'].iloc[0],
                    'sequence': layer_sequence,
                    'time_span': time_span,
                    'order_score': order_score,
                    'freq_progression': window['dominant_frequency'].tolist()
                })
    
    if cascade_patterns:
        print(f"Cascade patterns detected: {len(cascade_patterns)}")
        for pattern in cascade_patterns[:2]:
            print(f"   • {pattern['start_time']}: {pattern['time_span']:.1f}s span")
            print(f"     Order score: {pattern['order_score']}, Freq range: {min(pattern['freq_progression']):.0f}-{max(pattern['freq_progression']):.0f}Hz")
    
    # 4. FREQUENCY RESONANCE ANALYSIS
    print(f"\n🎼 FREQUENCY RESONANCE PATTERNS")
    print("-" * 35)
    
    resonance_patterns = {}
    
    # Find frequencies that appear across multiple layers
    all_frequencies = df['dominant_frequency'].round(-1)  # Round to nearest 10Hz
    freq_counts = all_frequencies.value_counts()
    resonant_freqs = freq_counts[freq_counts >= 3].index.tolist()  # Appear 3+ times
    
    for freq in resonant_freqs[:5]:  # Top 5 resonant frequencies
        layers_with_freq = df[df['dominant_frequency'].round(-1) == freq]['layer'].unique()
        resonance_patterns[freq] = {
            'layers': layers_with_freq,
            'occurrences': freq_counts[freq],
            'layer_count': len(layers_with_freq)
        }
        
        print(f"🔸 {freq:.0f}Hz resonance:")
        print(f"   • Appears in {len(layers_with_freq)} layers, {freq_counts[freq]} times")
        print(f"   • Layers: {', '.join([l.replace('_', ' ').title() for l in layers_with_freq])}")
    
    # 5. AMPLITUDE MODULATION PATTERNS
    print(f"\n📊 AMPLITUDE MODULATION ANALYSIS")
    print("-" * 35)
    
    modulation_patterns = {}
    
    for layer in df['layer'].unique():
        layer_data = df[df['layer'] == layer].sort_values('timestamp')
        if len(layer_data) < 10:
            continue
        
        amplitudes = layer_data['peak_amplitude'].values
        
        # Find amplitude oscillation patterns
        # Use FFT to find dominant periods in amplitude modulation
        if len(amplitudes) > 20:
            fft_amps = np.fft.fft(amplitudes - np.mean(amplitudes))
            fft_freqs = np.fft.fftfreq(len(amplitudes))
            
            # Find dominant modulation frequency
            dominant_idx = np.argmax(np.abs(fft_amps[1:len(fft_amps)//2])) + 1
            modulation_freq = abs(fft_freqs[dominant_idx])
            modulation_strength = np.abs(fft_amps[dominant_idx]) / np.abs(fft_amps[0])
            
            if modulation_strength > 0.1:  # Significant modulation
                modulation_patterns[layer] = {
                    'modulation_frequency': modulation_freq,
                    'modulation_strength': modulation_strength,
                    'amplitude_range': amplitudes.max() - amplitudes.min()
                }
                
                print(f"🔸 {layer.replace('_', ' ').title()}:")
                print(f"   • Modulation frequency: {modulation_freq:.4f}")
                print(f"   • Modulation strength: {modulation_strength:.3f}")
                print(f"   • Amplitude range: {amplitudes.min():.3f} - {amplitudes.max():.3f}")
    
    # 6. PATTERN INTERFERENCE DETECTION
    print(f"\n🌐 PATTERN INTERFERENCE ANALYSIS")
    print("-" * 35)
    
    interference_events = df[df['patterns'].str.contains('cross_layer_interference', na=False)]
    
    if len(interference_events) > 0:
        print(f"Cross-layer interference events: {len(interference_events)}")
        
        # Analyze interference by layer combination
        interference_combos = {}
        for timestamp in interference_events['timestamp'].unique():
            concurrent_layers = df[df['timestamp'] == timestamp]['layer'].tolist()
            if len(concurrent_layers) > 1:
                combo_key = '+'.join(sorted(concurrent_layers))
                if combo_key not in interference_combos:
                    interference_combos[combo_key] = 0
                interference_combos[combo_key] += 1
        
        # Show most common interference patterns
        sorted_combos = sorted(interference_combos.items(), key=lambda x: x[1], reverse=True)
        for combo, count in sorted_combos[:3]:
            layers = combo.split('+')
            print(f"   • {' + '.join([l.replace('_', ' ').title() for l in layers])}: {count} events")
    
    # 7. EMERGENT BEHAVIOR DETECTION
    print(f"\n🚀 EMERGENT BEHAVIOR PATTERNS")
    print("-" * 35)
    
    # Look for sudden changes in behavior
    emergent_events = []
    
    for layer in df['layer'].unique():
        layer_data = df[df['layer'] == layer].sort_values('timestamp')
        if len(layer_data) < 5:
            continue
        
        # Calculate rate of change for key metrics
        freq_changes = np.diff(layer_data['dominant_frequency'])
        activity_changes = np.diff(layer_data['activity_score'])
        amplitude_changes = np.diff(layer_data['peak_amplitude'])
        
        # Find sudden spikes (> 2 standard deviations)
        freq_threshold = np.mean(freq_changes) + 2 * np.std(freq_changes)
        activity_threshold = np.mean(activity_changes) + 2 * np.std(activity_changes)
        
        freq_spikes = np.where(freq_changes > freq_threshold)[0]
        activity_spikes = np.where(activity_changes > activity_threshold)[0]
        
        for spike_idx in freq_spikes:
            if spike_idx < len(layer_data) - 1:
                emergent_events.append({
                    'layer': layer,
                    'type': 'frequency_spike',
                    'timestamp': layer_data.iloc[spike_idx + 1]['timestamp'],
                    'magnitude': freq_changes[spike_idx],
                    'before': layer_data.iloc[spike_idx]['dominant_frequency'],
                    'after': layer_data.iloc[spike_idx + 1]['dominant_frequency']
                })
    
    if emergent_events:
        print(f"Emergent behavior events detected: {len(emergent_events)}")
        for event in emergent_events[:3]:
            print(f"   • {event['layer'].replace('_', ' ').title()}: {event['type']}")
            print(f"     {event['before']:.0f} → {event['after']:.0f}Hz (+{event['magnitude']:.0f}Hz)")
    
    return {
        'harmonic_patterns': harmonic_patterns,
        'sync_patterns': high_sync,
        'cascade_patterns': cascade_patterns,
        'resonance_patterns': resonance_patterns,
        'modulation_patterns': modulation_patterns,
        'interference_combos': interference_combos if 'interference_combos' in locals() else {},
        'emergent_events': emergent_events
    }

def analyze_pattern_networks(df):
    """Analyze how patterns form networks and relationships."""
    
    print(f"\n🕸️ PATTERN NETWORK ANALYSIS")
    print("=" * 35)
    
    # 1. Pattern Co-occurrence Network
    pattern_cooccurrence = {}
    
    # Extract all patterns and create co-occurrence matrix
    all_patterns = []
    for patterns_str in df['patterns'].dropna():
        if patterns_str and patterns_str != '':
            patterns = [p.strip() for p in patterns_str.split(',')]
            all_patterns.extend(patterns)
    
    unique_patterns = list(set(all_patterns))
    print(f"Unique pattern types discovered: {len(unique_patterns)}")
    for pattern in unique_patterns:
        if pattern:
            print(f"   • {pattern}")
    
    # 2. Layer Interaction Strength
    print(f"\nLayer Interaction Matrix:")
    interaction_matrix = np.zeros((7, 7))
    layer_names = ['physical_layer', 'data_link_layer', 'network_layer', 'transport_layer', 
                   'session_layer', 'presentation_layer', 'application_layer']
    
    # Calculate interaction strength based on temporal proximity and activity correlation
    for i, layer1 in enumerate(layer_names):
        for j, layer2 in enumerate(layer_names):
            if i != j:
                layer1_data = df[df['layer'] == layer1]
                layer2_data = df[df['layer'] == layer2]
                
                # Find temporal correlations
                interaction_count = 0
                for timestamp in layer1_data['timestamp']:
                    # Look for layer2 activity within 1 second
                    concurrent = layer2_data[
                        abs((pd.to_datetime(layer2_data['timestamp']) - 
                            pd.to_datetime(timestamp)).dt.total_seconds()) <= 1.0
                    ]
                    interaction_count += len(concurrent)
                
                interaction_matrix[i][j] = interaction_count
    
    # Show strongest interactions
    max_interactions = []
    for i in range(7):
        for j in range(7):
            if i != j and interaction_matrix[i][j] > 0:
                max_interactions.append((layer_names[i], layer_names[j], interaction_matrix[i][j]))
    
    max_interactions.sort(key=lambda x: x[2], reverse=True)
    print("Strongest layer interactions:")
    for layer1, layer2, strength in max_interactions[:5]:
        print(f"   • {layer1.replace('_', ' ').title()} ↔ {layer2.replace('_', ' ').title()}: {strength:.0f}")
    
    return {
        'unique_patterns': unique_patterns,
        'interaction_matrix': interaction_matrix,
        'strongest_interactions': max_interactions[:5]
    }

def main():
    """Main analysis function."""
    
    # Use the most recent network layer signatures file
    csv_file = '/Users/dominikkomorek/NETWORK/network_layer_signatures_20251006_001645.csv'
    
    print("🔬 ADVANCED NETWORK LAYER PATTERN DISCOVERY")
    print("=" * 60)
    print(f"📁 Analyzing: {csv_file}")
    print(f"🎯 Target: Discover sophisticated patterns in network layers")
    print()
    
    try:
        # Load and analyze the data
        df = pd.read_csv(csv_file)
        
        # Perform basic analysis first
        layer_stats = analyze_layer_frequency_patterns(csv_file)
        
        # Identify specific 10kHz emergence pathways
        pathway_analysis = identify_10khz_emergence_pathway(df)
        
        # NEW: Advanced pattern discovery
        advanced_patterns = analyze_advanced_layer_patterns(df)
        
        # NEW: Pattern network analysis
        network_patterns = analyze_pattern_networks(df)
        
        # Generate summary report
        print(f"\n📋 COMPREHENSIVE EXECUTIVE SUMMARY")
        print("=" * 40)
        
        total_events = len(df)
        high_freq_events = len(df[df['dominant_frequency'] >= 9000])
        near_10khz_events = len(df[df['dominant_frequency'] >= 9500])
        
        print(f"• Total layer signature events: {total_events}")
        print(f"• High frequency events (≥9kHz): {high_freq_events} ({(high_freq_events/total_events)*100:.1f}%)")
        print(f"• Near 10kHz events (≥9.5kHz): {near_10khz_events} ({(near_10khz_events/total_events)*100:.1f}%)")
        
        if near_10khz_events > 0:
            closest_freq = df['dominant_frequency'].max()
            print(f"• Closest to 10kHz: {closest_freq:.0f} Hz")
            print(f"• Gap to 10kHz: {10000 - closest_freq:.0f} Hz")
        
        # Most active layer in high frequencies
        if high_freq_events > 0:
            high_freq_data = df[df['dominant_frequency'] >= 9000]
            most_active_layer = high_freq_data['layer'].value_counts().index[0]
            print(f"• Most active layer in high frequencies: {most_active_layer.replace('_', ' ').title()}")
        
        # Advanced pattern summary
        print(f"\n🔍 ADVANCED PATTERN DISCOVERIES:")
        print(f"• Harmonic patterns found in {len(advanced_patterns['harmonic_patterns'])} layers")
        print(f"• High synchronization events: {len(advanced_patterns['sync_patterns'])}")
        print(f"• Cascade patterns detected: {len(advanced_patterns['cascade_patterns'])}")
        print(f"• Resonant frequencies: {len(advanced_patterns['resonance_patterns'])}")
        print(f"• Modulation patterns: {len(advanced_patterns['modulation_patterns'])}")
        print(f"• Emergent behavior events: {len(advanced_patterns['emergent_events'])}")
        print(f"• Unique pattern types: {len(network_patterns['unique_patterns'])}")
        
        print(f"\n🎯 CONCLUSION: Advanced pattern analysis complete. Multiple sophisticated patterns discovered.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
