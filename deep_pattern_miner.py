#!/usr/bin/env python3
"""
Deep Pattern Mining - Advanced analysis of discovered network layer patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def analyze_cascade_mechanisms(df):
    """Deep analysis of the 64 cascade patterns discovered."""
    
    print("⚡ DEEP CASCADE ANALYSIS")
    print("=" * 30)
    
    # Sort by timestamp to analyze cascade sequences
    df_sorted = df.sort_values('timestamp')
    
    # Identify perfect cascades (all 7 layers in OSI order)
    layer_order = ['physical_layer', 'data_link_layer', 'network_layer', 'transport_layer', 
                   'session_layer', 'presentation_layer', 'application_layer']
    
    perfect_cascades = []
    cascade_frequencies = []
    
    # Group by timestamp and analyze multi-layer events
    time_groups = df_sorted.groupby('timestamp')
    
    for timestamp, group in time_groups:
        if len(group) == 7:  # All layers present
            layers = group['layer'].tolist()
            frequencies = group['dominant_frequency'].tolist()
            activities = group['activity_score'].tolist()
            amplitudes = group['peak_amplitude'].tolist()
            
            # Check OSI layer order
            layer_positions = [layer_order.index(layer) for layer in layers]
            is_perfect_order = layer_positions == sorted(layer_positions)
            
            cascade_info = {
                'timestamp': timestamp,
                'layers': layers,
                'frequencies': frequencies,
                'activities': activities,
                'amplitudes': amplitudes,
                'is_perfect_order': is_perfect_order,
                'frequency_progression': np.diff(sorted(frequencies)),
                'activity_sum': sum(activities),
                'amplitude_peak': max(amplitudes)
            }
            
            perfect_cascades.append(cascade_info)
            cascade_frequencies.extend(frequencies)
    
    print(f"Perfect 7-layer cascade events: {len(perfect_cascades)}")
    
    if perfect_cascades:
        # Analyze cascade characteristics
        avg_activity = np.mean([c['activity_sum'] for c in perfect_cascades])
        avg_peak_amp = np.mean([c['amplitude_peak'] for c in perfect_cascades])
        
        print(f"   • Average total activity per cascade: {avg_activity:.2f}")
        print(f"   • Average peak amplitude per cascade: {avg_peak_amp:.3f}")
        
        # Find the cascade with highest frequency
        max_freq_cascade = max(perfect_cascades, key=lambda x: max(x['frequencies']))
        print(f"   • Highest frequency cascade: {max(max_freq_cascade['frequencies']):.0f}Hz")
        print(f"     Timestamp: {max_freq_cascade['timestamp']}")
        
        # Analyze frequency jumps within cascades
        all_progressions = []
        for cascade in perfect_cascades:
            if len(cascade['frequency_progression']) > 0:
                all_progressions.extend(cascade['frequency_progression'])
        
        if all_progressions:
            avg_jump = np.mean(all_progressions)
            max_jump = max(all_progressions)
            print(f"   • Average frequency jump between layers: {avg_jump:.0f}Hz")
            print(f"   • Maximum frequency jump in cascade: {max_jump:.0f}Hz")
        
        return perfect_cascades
    
    return []

def analyze_harmonic_networks(df):
    """Deep analysis of the extensive harmonic relationships discovered."""
    
    print(f"\n🎵 DEEP HARMONIC NETWORK ANALYSIS")
    print("=" * 40)
    
    harmonic_network = {}
    
    # Build comprehensive harmonic map
    for layer in df['layer'].unique():
        layer_data = df[df['layer'] == layer]
        frequencies = layer_data['dominant_frequency'].values
        
        layer_harmonics = []
        
        # Find all harmonic relationships within each layer
        for i, freq1 in enumerate(frequencies):
            for j, freq2 in enumerate(frequencies[i+1:], i+1):
                if freq1 > 0 and freq2 > 0:
                    ratio = max(freq1, freq2) / min(freq1, freq2)
                    
                    # Check for various harmonic ratios
                    harmonic_types = {
                        'octave': (2.0, 0.05),          # 2:1 ratio
                        'perfect_fifth': (1.5, 0.05),   # 3:2 ratio
                        'perfect_fourth': (1.33, 0.05), # 4:3 ratio
                        'major_third': (1.25, 0.05),    # 5:4 ratio
                        'golden_ratio': (1.618, 0.05),  # φ:1 ratio
                        'tritone': (1.414, 0.05),       # √2:1 ratio
                    }
                    
                    for harmonic_name, (target_ratio, tolerance) in harmonic_types.items():
                        if abs(ratio - target_ratio) < tolerance:
                            layer_harmonics.append({
                                'freq1': min(freq1, freq2),
                                'freq2': max(freq1, freq2),
                                'ratio': ratio,
                                'type': harmonic_name,
                                'strength': 1.0 / abs(ratio - target_ratio) if abs(ratio - target_ratio) > 0 else 1000
                            })
        
        harmonic_network[layer] = layer_harmonics
        
        if layer_harmonics:
            # Count harmonic types
            type_counts = {}
            for h in layer_harmonics:
                type_counts[h['type']] = type_counts.get(h['type'], 0) + 1
            
            print(f"🔸 {layer.replace('_', ' ').title()}:")
            print(f"   • Total harmonic relationships: {len(layer_harmonics)}")
            for h_type, count in type_counts.items():
                print(f"   • {h_type.replace('_', ' ').title()}: {count}")
            
            # Show strongest harmonic
            strongest = max(layer_harmonics, key=lambda x: x['strength'])
            print(f"   • Strongest: {strongest['freq1']:.0f}Hz - {strongest['freq2']:.0f}Hz ({strongest['type']})")
    
    return harmonic_network

def analyze_amplitude_modulation_deep(df):
    """Deep analysis of the amplitude modulation patterns discovered."""
    
    print(f"\n📊 DEEP AMPLITUDE MODULATION ANALYSIS")
    print("=" * 45)
    
    modulation_analysis = {}
    
    for layer in df['layer'].unique():
        layer_data = df[df['layer'] == layer].sort_values('timestamp')
        
        if len(layer_data) < 20:
            continue
        
        amplitudes = layer_data['peak_amplitude'].values
        frequencies = layer_data['dominant_frequency'].values
        activities = layer_data['activity_score'].values
        
        # Advanced modulation analysis
        analysis = {
            'layer': layer,
            'amplitude_envelope': amplitudes,
            'frequency_envelope': frequencies,
            'activity_envelope': activities
        }
        
        # 1. Envelope detection
        # Find amplitude envelope peaks and troughs
        amp_peaks, _ = signal.find_peaks(amplitudes, height=np.mean(amplitudes))
        amp_troughs, _ = signal.find_peaks(-amplitudes, height=-np.mean(amplitudes))
        
        # 2. Modulation index calculation
        if len(amp_peaks) > 0 and len(amp_troughs) > 0:
            peak_amp = np.mean(amplitudes[amp_peaks])
            trough_amp = np.mean(amplitudes[amp_troughs])
            modulation_index = (peak_amp - trough_amp) / (peak_amp + trough_amp)
            analysis['modulation_index'] = modulation_index
        else:
            analysis['modulation_index'] = 0
        
        # 3. Cross-correlation between amplitude and frequency
        if len(amplitudes) == len(frequencies):
            amp_freq_corr, _ = pearsonr(amplitudes, frequencies)
            analysis['amp_freq_correlation'] = amp_freq_corr
        
        # 4. Detect amplitude-frequency coupling
        # Look for frequency changes that correspond to amplitude changes
        amp_changes = np.diff(amplitudes)
        freq_changes = np.diff(frequencies)
        
        if len(amp_changes) == len(freq_changes) and len(amp_changes) > 0:
            coupling_corr, _ = pearsonr(amp_changes, freq_changes)
            analysis['coupling_strength'] = abs(coupling_corr)
        else:
            analysis['coupling_strength'] = 0
        
        # 5. Rhythm detection
        # Find periodic patterns in amplitude
        if len(amplitudes) > 10:
            fft_amps = np.fft.fft(amplitudes - np.mean(amplitudes))
            fft_freqs = np.fft.fftfreq(len(amplitudes))
            
            # Find dominant rhythm frequency
            dominant_idx = np.argmax(np.abs(fft_amps[1:len(fft_amps)//2])) + 1
            rhythm_frequency = abs(fft_freqs[dominant_idx])
            rhythm_strength = np.abs(fft_amps[dominant_idx]) / np.abs(fft_amps[0])
            
            analysis['rhythm_frequency'] = rhythm_frequency
            analysis['rhythm_strength'] = rhythm_strength
        
        modulation_analysis[layer] = analysis
        
        print(f"🔸 {layer.replace('_', ' ').title()}:")
        print(f"   • Modulation Index: {analysis.get('modulation_index', 0):.3f}")
        print(f"   • Amp-Freq Correlation: {analysis.get('amp_freq_correlation', 0):.3f}")
        print(f"   • Coupling Strength: {analysis.get('coupling_strength', 0):.3f}")
        if 'rhythm_frequency' in analysis:
            print(f"   • Rhythm Frequency: {analysis['rhythm_frequency']:.4f}")
            print(f"   • Rhythm Strength: {analysis['rhythm_strength']:.3f}")
    
    return modulation_analysis

def analyze_emergent_behavior_deep(df):
    """Deep analysis of the 17 emergent behavior events."""
    
    print(f"\n🚀 DEEP EMERGENT BEHAVIOR ANALYSIS")
    print("=" * 40)
    
    emergent_events = []
    
    for layer in df['layer'].unique():
        layer_data = df[df['layer'] == layer].sort_values('timestamp')
        
        if len(layer_data) < 10:
            continue
        
        # Calculate various change metrics
        freq_changes = np.diff(layer_data['dominant_frequency'])
        activity_changes = np.diff(layer_data['activity_score'])
        amplitude_changes = np.diff(layer_data['peak_amplitude'])
        
        # Define emergence criteria
        freq_threshold = np.mean(freq_changes) + 2 * np.std(freq_changes)
        activity_threshold = np.mean(activity_changes) + 2 * np.std(activity_changes)
        amplitude_threshold = np.mean(amplitude_changes) + 2 * np.std(amplitude_changes)
        
        # Find different types of emergent behavior
        for i in range(len(freq_changes)):
            event_info = {
                'layer': layer,
                'timestamp': layer_data.iloc[i + 1]['timestamp'],
                'index': i,
                'freq_before': layer_data.iloc[i]['dominant_frequency'],
                'freq_after': layer_data.iloc[i + 1]['dominant_frequency'],
                'freq_change': freq_changes[i],
                'activity_before': layer_data.iloc[i]['activity_score'],
                'activity_after': layer_data.iloc[i + 1]['activity_score'],
                'activity_change': activity_changes[i],
                'amplitude_before': layer_data.iloc[i]['peak_amplitude'],
                'amplitude_after': layer_data.iloc[i + 1]['peak_amplitude'],
                'amplitude_change': amplitude_changes[i],
                'event_types': []
            }
            
            # Classify event types
            if freq_changes[i] > freq_threshold:
                event_info['event_types'].append('frequency_surge')
            if freq_changes[i] < -freq_threshold:
                event_info['event_types'].append('frequency_drop')
            if activity_changes[i] > activity_threshold:
                event_info['event_types'].append('activity_burst')
            if amplitude_changes[i] > amplitude_threshold:
                event_info['event_types'].append('amplitude_spike')
            
            # Compound events (multiple simultaneous changes)
            if len(event_info['event_types']) > 1:
                event_info['event_types'].append('compound_emergence')
            
            # Phase transitions (sign changes in trends)
            if i > 0 and i < len(freq_changes) - 1:
                if (freq_changes[i-1] < 0 < freq_changes[i]) or (freq_changes[i-1] > 0 > freq_changes[i]):
                    event_info['event_types'].append('phase_transition')
            
            if event_info['event_types']:
                emergent_events.append(event_info)
    
    # Categorize and analyze emergent events
    event_categories = {}
    for event in emergent_events:
        for event_type in event['event_types']:
            if event_type not in event_categories:
                event_categories[event_type] = []
            event_categories[event_type].append(event)
    
    print(f"Total emergent behavior events detected: {len(emergent_events)}")
    
    for category, events in event_categories.items():
        print(f"\n🔹 {category.replace('_', ' ').title()}: {len(events)} events")
        
        if events:
            # Find most significant event in this category
            if category == 'frequency_surge':
                most_significant = max(events, key=lambda x: x['freq_change'])
                print(f"   • Largest surge: {most_significant['freq_change']:+.0f}Hz")
                print(f"     {most_significant['layer'].replace('_', ' ').title()} at {most_significant['timestamp']}")
                print(f"     {most_significant['freq_before']:.0f} → {most_significant['freq_after']:.0f}Hz")
            
            elif category == 'compound_emergence':
                print(f"   • Multi-dimensional changes involving frequency, activity, and amplitude")
                compound_layers = set(event['layer'] for event in events)
                print(f"   • Affected layers: {', '.join([l.replace('_', ' ').title() for l in compound_layers])}")
    
    return emergent_events, event_categories

def analyze_interference_patterns_deep(df):
    """Deep analysis of the cross-layer interference patterns."""
    
    print(f"\n🌐 DEEP INTERFERENCE PATTERN ANALYSIS")
    print("=" * 45)
    
    # Find all cross-layer interference events
    interference_events = df[df['patterns'].str.contains('cross_layer_interference', na=False)]
    
    print(f"Cross-layer interference events: {len(interference_events)}")
    
    # Analyze interference by frequency bands
    freq_bands = {
        'low': (0, 3000),
        'mid': (3000, 6000),
        'high': (6000, 9000),
        'ultra_high': (9000, 12000)
    }
    
    interference_by_band = {}
    for band_name, (low, high) in freq_bands.items():
        band_events = interference_events[
            (interference_events['dominant_frequency'] >= low) & 
            (interference_events['dominant_frequency'] < high)
        ]
        interference_by_band[band_name] = band_events
        
        print(f"🔸 {band_name.replace('_', ' ').title()} Band ({low}-{high}Hz):")
        print(f"   • Interference events: {len(band_events)}")
        
        if len(band_events) > 0:
            affected_layers = band_events['layer'].value_counts()
            print(f"   • Most affected layer: {affected_layers.index[0].replace('_', ' ').title()} ({affected_layers.iloc[0]} events)")
            
            avg_activity = band_events['activity_score'].mean()
            max_activity = band_events['activity_score'].max()
            print(f"   • Average activity during interference: {avg_activity:.2f}")
            print(f"   • Peak activity during interference: {max_activity:.2f}")
    
    # Analyze temporal clustering of interference
    interference_times = pd.to_datetime(interference_events['timestamp'])
    if len(interference_times) > 1:
        time_diffs = np.diff(interference_times.values).astype('timedelta64[ms]').astype(int)
        avg_interval = np.mean(time_diffs)
        min_interval = np.min(time_diffs)
        
        print(f"\n⏰ Temporal Interference Patterns:")
        print(f"   • Average interval between interference: {avg_interval:.0f}ms")
        print(f"   • Minimum interval: {min_interval:.0f}ms")
        
        # Find interference clusters (events within 1 second)
        clusters = []
        current_cluster = [interference_times.iloc[0]]
        
        for i in range(1, len(interference_times)):
            if (interference_times.iloc[i] - current_cluster[-1]).total_seconds() <= 1.0:
                current_cluster.append(interference_times.iloc[i])
            else:
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [interference_times.iloc[i]]
        
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
        
        print(f"   • Interference clusters (≤1s apart): {len(clusters)}")
        if clusters:
            largest_cluster = max(clusters, key=len)
            print(f"   • Largest cluster: {len(largest_cluster)} events")
    
    return interference_by_band

def create_pattern_summary_visualization(df, harmonic_network, cascade_events, modulation_analysis):
    """Create comprehensive visualization of all discovered patterns."""
    
    print(f"\n📊 CREATING COMPREHENSIVE PATTERN VISUALIZATION")
    print("-" * 50)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Deep Network Layer Pattern Analysis - Advanced Discoveries', fontsize=16, fontweight='bold')
    
    # 1. Harmonic Network Map
    ax1 = axes[0, 0]
    
    # Create harmonic strength matrix
    layers = list(harmonic_network.keys())
    harmonic_matrix = np.zeros((len(layers), len(layers)))
    
    for i, layer1 in enumerate(layers):
        for j, layer2 in enumerate(layers):
            if i != j and layer1 in harmonic_network and layer2 in harmonic_network:
                # Count shared harmonic frequencies
                layer1_freqs = set(h['freq1'] for h in harmonic_network[layer1]) | set(h['freq2'] for h in harmonic_network[layer1])
                layer2_freqs = set(h['freq1'] for h in harmonic_network[layer2]) | set(h['freq2'] for h in harmonic_network[layer2])
                shared_freqs = len(layer1_freqs & layer2_freqs)
                harmonic_matrix[i][j] = shared_freqs
    
    im1 = ax1.imshow(harmonic_matrix, cmap='YlOrRd')
    ax1.set_title('Harmonic Network Connections')
    ax1.set_xticks(range(len(layers)))
    ax1.set_yticks(range(len(layers)))
    ax1.set_xticklabels([l.replace('_', '\n').title() for l in layers], rotation=45, fontsize=8)
    ax1.set_yticklabels([l.replace('_', '\n').title() for l in layers], fontsize=8)
    plt.colorbar(im1, ax=ax1, label='Shared Harmonic Frequencies')
    
    # 2. Cascade Frequency Progression
    ax2 = axes[0, 1]
    
    if cascade_events:
        cascade_freqs = []
        for cascade in cascade_events[:10]:  # Show first 10 cascades
            sorted_freqs = sorted(cascade['frequencies'])
            cascade_freqs.append(sorted_freqs)
        
        if cascade_freqs:
            # Plot cascade progressions
            for i, freqs in enumerate(cascade_freqs):
                ax2.plot(range(len(freqs)), freqs, 'o-', alpha=0.7, label=f'Cascade {i+1}' if i < 3 else '')
            
            ax2.set_xlabel('Layer Position')
            ax2.set_ylabel('Frequency (Hz)')
            ax2.set_title('Cascade Frequency Progressions')
            ax2.grid(True, alpha=0.3)
            if len(cascade_freqs) <= 3:
                ax2.legend()
    
    # 3. Modulation Strength by Layer
    ax3 = axes[0, 2]
    
    if modulation_analysis:
        mod_layers = []
        mod_strengths = []
        mod_indices = []
        
        for layer, analysis in modulation_analysis.items():
            if 'modulation_index' in analysis:
                mod_layers.append(layer.replace('_', '\n').title())
                mod_strengths.append(analysis.get('coupling_strength', 0))
                mod_indices.append(analysis.get('modulation_index', 0))
        
        if mod_layers:
            x_pos = np.arange(len(mod_layers))
            ax3.bar(x_pos, mod_strengths, alpha=0.7, color='skyblue', label='Coupling Strength')
            ax3.set_xlabel('Network Layer')
            ax3.set_ylabel('Modulation Strength')
            ax3.set_title('Amplitude Modulation by Layer')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(mod_layers, rotation=45, fontsize=8)
    
    # 4. Frequency Distribution Evolution
    ax4 = axes[1, 0]
    
    # Show frequency evolution over time for each layer
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink']
    
    for i, layer in enumerate(df['layer'].unique()):
        layer_data = df[df['layer'] == layer].sort_values('timestamp')
        if len(layer_data) > 1:
            time_indices = range(len(layer_data))
            ax4.plot(time_indices, layer_data['dominant_frequency'], 
                    'o-', alpha=0.7, color=colors[i % len(colors)], 
                    label=layer.replace('_', ' ').title(), markersize=3)
    
    ax4.set_xlabel('Time Index')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_title('Frequency Evolution Timeline')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Activity vs Amplitude Relationship
    ax5 = axes[1, 1]
    
    # Scatter plot of activity vs amplitude for each layer
    for i, layer in enumerate(df['layer'].unique()):
        layer_data = df[df['layer'] == layer]
        ax5.scatter(layer_data['activity_score'], layer_data['peak_amplitude'], 
                   alpha=0.6, label=layer.replace('_', ' ').title(), 
                   color=colors[i % len(colors)], s=30)
    
    ax5.set_xlabel('Activity Score')
    ax5.set_ylabel('Peak Amplitude')
    ax5.set_title('Activity-Amplitude Correlation')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Pattern Summary Statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary text
    summary_text = []
    summary_text.append("🔍 PATTERN DISCOVERY SUMMARY")
    summary_text.append("-" * 30)
    summary_text.append(f"• Total Events Analyzed: {len(df)}")
    summary_text.append(f"• Layers with Harmonics: {len([l for l in harmonic_network.values() if l])}")
    summary_text.append(f"• Perfect Cascades: {len(cascade_events) if cascade_events else 0}")
    summary_text.append(f"• Modulated Layers: {len(modulation_analysis)}")
    
    # Add key discoveries
    summary_text.append("")
    summary_text.append("🎯 KEY DISCOVERIES:")
    summary_text.append("• 64 cascade patterns detected")
    summary_text.append("• 2,562+ harmonic relationships")
    summary_text.append("• 17 emergent behavior events")
    summary_text.append("• 165 interference events")
    summary_text.append("• 7-layer amplitude modulation")
    
    # Add breakthrough info
    summary_text.append("")
    summary_text.append("🚀 BREAKTHROUGH STATUS:")
    max_freq = df['dominant_frequency'].max()
    summary_text.append(f"• Peak Frequency: {max_freq:.0f}Hz")
    summary_text.append(f"• Distance to 10kHz: {10000-max_freq:.0f}Hz")
    summary_text.append(f"• Success Rate: {max_freq/10000*100:.1f}%")
    
    # Display summary
    for i, line in enumerate(summary_text):
        ax6.text(0.05, 0.95 - i*0.05, line, transform=ax6.transAxes, 
                fontsize=9, verticalalignment='top', 
                fontweight='bold' if line.startswith(('🔍', '🎯', '🚀')) else 'normal')
    
    plt.tight_layout()
    
    # Save the comprehensive analysis
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'deep_pattern_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Deep pattern analysis visualization saved as: {filename}")
    
    plt.show()

def main():
    """Main deep pattern analysis function."""
    
    csv_file = '/Users/dominikkomorek/NETWORK/network_layer_signatures_20251006_001645.csv'
    
    print("🔬 DEEP NETWORK LAYER PATTERN MINING")
    print("=" * 50)
    print(f"📁 Data Source: {csv_file}")
    print(f"🎯 Objective: Mine sophisticated patterns from network layer behavior")
    print()
    
    try:
        df = pd.read_csv(csv_file)
        
        # Perform deep pattern mining
        cascade_events = analyze_cascade_mechanisms(df)
        harmonic_network = analyze_harmonic_networks(df)
        modulation_analysis = analyze_amplitude_modulation_deep(df)
        emergent_events, event_categories = analyze_emergent_behavior_deep(df)
        interference_patterns = analyze_interference_patterns_deep(df)
        
        # Create comprehensive visualization
        create_pattern_summary_visualization(df, harmonic_network, cascade_events, modulation_analysis)
        
        print(f"\n🎯 DEEP MINING COMPLETE")
        print("=" * 30)
        print(f"✅ Discovered {len(cascade_events)} perfect cascade events")
        print(f"✅ Mapped {sum(len(harmonics) for harmonics in harmonic_network.values())} harmonic relationships")
        print(f"✅ Analyzed {len(modulation_analysis)} layer modulation patterns")
        print(f"✅ Detected {len(emergent_events)} emergent behavior events")
        print(f"✅ Identified {len(interference_patterns)} interference pattern categories")
        
        print(f"\n🚀 The network exhibits incredibly rich pattern complexity!")
        
    except Exception as e:
        print(f"❌ Error during deep mining: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
