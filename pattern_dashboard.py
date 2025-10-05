#!/usr/bin/env python3
"""
Pattern Categorization Dashboard - Visual interface for categorized patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('default')
sns.set_palette("husl")

def create_categorization_dashboard():
    """Create comprehensive visualization dashboard for pattern categories."""
    
    print("📊 Creating Pattern Categorization Dashboard...")
    
    # Load categorization data
    try:
        with open('pattern_categorization_20251006_003509.json', 'r') as f:
            categorization_data = json.load(f)
    except:
        print("❌ Could not load categorization data")
        return
    
    # Load original data
    df = pd.read_csv('network_layer_signatures_20251006_001645.csv')
    
    # Create comprehensive dashboard
    fig = plt.figure(figsize=(20, 24))
    
    # Dashboard title
    fig.suptitle('Network Pattern Categorization Dashboard\nComprehensive Analysis of 350 Layer Events', 
                fontsize=24, fontweight='bold', y=0.98)
    
    # 1. Category Overview (Top Left)
    ax1 = plt.subplot(4, 3, 1)
    categories = list(categorization_data['taxonomy']['primary_categories'].keys())
    subcategory_counts = [data['subcategory_count'] for data in 
                         categorization_data['taxonomy']['primary_categories'].values()]
    priorities = [data['research_priority'] for data in 
                 categorization_data['taxonomy']['primary_categories'].values()]
    
    colors = ['red' if p == 'high' else 'orange' if p == 'medium' else 'green' for p in priorities]
    bars = ax1.bar(range(len(categories)), subcategory_counts, color=colors, alpha=0.7)
    ax1.set_title('Pattern Categories Overview', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Pattern Categories')
    ax1.set_ylabel('Subcategory Count')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels([cat.replace('_', '\n').title() for cat in categories], rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, subcategory_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # 2. Frequency Distribution by Layer (Top Center)
    ax2 = plt.subplot(4, 3, 2)
    layer_freq_data = []
    layers = df['layer'].unique()
    for layer in layers:
        layer_data = df[df['layer'] == layer]['dominant_frequency']
        layer_freq_data.append(layer_data)
    
    bp = ax2.boxplot(layer_freq_data, labels=[l.replace('_', '\n') for l in layers], patch_artist=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_title('Frequency Distribution by Layer', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=10000, color='red', linestyle='--', alpha=0.7, label='10kHz Target')
    ax2.legend()
    
    # 3. Priority Pattern Breakdown (Top Right)
    ax3 = plt.subplot(4, 3, 3)
    high_priority = categorization_data['taxonomy']['priority_patterns']['high_priority']
    medium_priority = categorization_data['taxonomy']['priority_patterns']['medium_priority']
    research_priority = categorization_data['taxonomy']['priority_patterns']['research_priority']
    
    priority_data = [len(high_priority), len(medium_priority), len(research_priority)]
    priority_labels = ['High Priority', 'Medium Priority', 'Research Priority']
    colors = ['red', 'orange', 'blue']
    
    wedges, texts, autotexts = ax3.pie(priority_data, labels=priority_labels, colors=colors, 
                                      autopct='%1.0f%%', startangle=90)
    ax3.set_title('Research Priority Distribution', fontsize=14, fontweight='bold')
    
    # 4. Cascade Pattern Timeline (Second Row Left)
    ax4 = plt.subplot(4, 3, 4)
    cascade_data = categorization_data['categories']['cascade_patterns']
    
    if 'perfect_cascades' in cascade_data and cascade_data['perfect_cascades']:
        cascade_times = [event['timestamp'] for event in cascade_data['perfect_cascades']]
        cascade_activities = [event['total_activity'] for event in cascade_data['perfect_cascades']]
        
        ax4.scatter(cascade_times, cascade_activities, color='red', s=100, alpha=0.7, label='Perfect Cascades')
        ax4.set_title('Perfect Cascade Events Timeline', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Timestamp')
        ax4.set_ylabel('Total Activity')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No Perfect Cascade Data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Perfect Cascade Events Timeline', fontsize=14, fontweight='bold')
    
    # 5. Harmonic Relationship Network (Second Row Center)
    ax5 = plt.subplot(4, 3, 5)
    harmonic_data = categorization_data['categories']['harmonic_patterns']
    
    if 'by_type' in harmonic_data:
        harmonic_types = list(harmonic_data['by_type'].keys())
        harmonic_counts = [data['count'] for data in harmonic_data['by_type'].values()]
        
        bars = ax5.barh(harmonic_types, harmonic_counts, color='purple', alpha=0.7)
        ax5.set_title('Harmonic Relationship Types', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Count')
        
        # Add value labels
        for bar, count in zip(bars, harmonic_counts):
            ax5.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    str(count), ha='left', va='center', fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'No Harmonic Data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Harmonic Relationship Types', fontsize=14, fontweight='bold')
    
    # 6. Emergence Pattern Heatmap (Second Row Right)
    ax6 = plt.subplot(4, 3, 6)
    
    # Create emergence pattern matrix
    emergence_data = categorization_data['categories']['emergence_patterns']
    emergence_types = ['frequency_surges', 'amplitude_spikes', 'activity_bursts', 
                      'phase_transitions', 'compound_events', 'breakthrough_events']
    emergence_counts = [len(emergence_data.get(et, [])) for et in emergence_types]
    
    # Create heatmap data
    emergence_matrix = np.array(emergence_counts).reshape(2, 3)
    emergence_labels = [et.replace('_', ' ').title() for et in emergence_types]
    emergence_labels_matrix = np.array(emergence_labels).reshape(2, 3)
    
    im = ax6.imshow(emergence_matrix, cmap='Reds', aspect='auto')
    ax6.set_title('Emergence Pattern Intensity', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(3):
            text = ax6.text(j, i, f'{emergence_labels_matrix[i, j]}\n({emergence_matrix[i, j]})',
                           ha="center", va="center", color="white", fontweight='bold')
    
    ax6.set_xticks([])
    ax6.set_yticks([])
    
    # 7. Behavioral Pattern Radar (Third Row Left)
    ax7 = plt.subplot(4, 3, 7, projection='polar')
    
    behavioral_data = categorization_data['categories']['behavioral_patterns']
    if 'layer_personalities' in behavioral_data:
        # Create radar chart for layer personalities
        layers = list(behavioral_data['layer_personalities'].keys())
        stability_scores = [data.get('stability_index', 0) for data in behavioral_data['layer_personalities'].values()]
        
        angles = np.linspace(0, 2*np.pi, len(layers), endpoint=False)
        stability_scores_plot = stability_scores + [stability_scores[0]]  # Complete the circle
        angles_plot = np.concatenate((angles, [angles[0]]))
        
        ax7.plot(angles_plot, stability_scores_plot, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax7.fill(angles_plot, stability_scores_plot, alpha=0.25, color='blue')
        ax7.set_xticks(angles)
        ax7.set_xticklabels([l.replace('_', '\n') for l in layers])
        ax7.set_title('Layer Stability Index', fontsize=14, fontweight='bold', pad=20)
    
    # 8. Frequency Progress Timeline (Third Row Center)
    ax8 = plt.subplot(4, 3, 8)
    
    # Calculate frequency progress over time
    df_sorted = df.sort_values('timestamp')
    max_frequencies = []
    timestamps = []
    
    for timestamp in df_sorted['timestamp'].unique():
        time_data = df_sorted[df_sorted['timestamp'] == timestamp]
        max_freq = time_data['dominant_frequency'].max()
        max_frequencies.append(max_freq)
        timestamps.append(timestamp)
    
    ax8.plot(timestamps, max_frequencies, color='green', linewidth=2, marker='o', markersize=4)
    ax8.axhline(y=10000, color='red', linestyle='--', alpha=0.7, label='10kHz Target')
    ax8.axhline(y=9909, color='orange', linestyle='--', alpha=0.7, label='Current Achievement')
    ax8.set_title('Frequency Achievement Progress', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Timestamp')
    ax8.set_ylabel('Maximum Frequency (Hz)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Research Direction Roadmap (Third Row Right)
    ax9 = plt.subplot(4, 3, 9)
    
    research_directions = categorization_data['taxonomy']['research_directions']
    direction_names = list(research_directions.keys())
    
    # Create a simple roadmap visualization
    y_positions = range(len(direction_names))
    priorities = [2, 3, 1]  # Assign priorities for visualization
    
    bars = ax9.barh(y_positions, priorities, color=['red', 'blue', 'orange'], alpha=0.7)
    ax9.set_yticks(y_positions)
    ax9.set_yticklabels([name.replace('_', ' ').title() for name in direction_names])
    ax9.set_xlabel('Research Priority Level')
    ax9.set_title('Future Research Roadmap', fontsize=14, fontweight='bold')
    
    for i, (bar, name) in enumerate(zip(bars, direction_names)):
        objective = research_directions[name]['objective'][:30] + "..."
        ax9.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                objective, ha='left', va='center', fontsize=9)
    
    # 10. Pattern Correlation Matrix (Fourth Row - Full Width)
    ax10 = plt.subplot(4, 1, 4)
    
    # Create correlation matrix for different pattern aspects
    correlation_data = []
    pattern_names = []
    
    for layer in df['layer'].unique():
        layer_data = df[df['layer'] == layer]
        if len(layer_data) > 1:
            correlation_data.append([
                layer_data['dominant_frequency'].mean(),
                layer_data['activity_score'].mean(),
                layer_data['peak_amplitude'].mean(),
                layer_data['dominant_frequency'].std(),
                len(layer_data)
            ])
            pattern_names.append(layer.replace('_', ' ').title())
    
    if correlation_data:
        correlation_matrix = np.corrcoef(np.array(correlation_data).T)
        
        feature_names = ['Avg Frequency', 'Avg Activity', 'Avg Amplitude', 'Freq Variability', 'Event Count']
        
        im = ax10.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax10.set_xticks(range(len(feature_names)))
        ax10.set_yticks(range(len(feature_names)))
        ax10.set_xticklabels(feature_names)
        ax10.set_yticklabels(feature_names)
        ax10.set_title('Pattern Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Add correlation values
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                text = ax10.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                               ha="center", va="center", 
                               color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax10)
        cbar.set_label('Correlation Coefficient')
    
    # Adjust layout and save
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'pattern_categorization_dashboard_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    print(f"✅ Dashboard saved: {filename}")
    
    # Show summary statistics
    print(f"\n📊 CATEGORIZATION DASHBOARD SUMMARY")
    print("=" * 40)
    print(f"📈 Total Events Analyzed: {len(df)}")
    print(f"🏷️  Pattern Categories: {len(categorization_data['taxonomy']['primary_categories'])}")
    print(f"🔍 Subcategories: {sum(data['subcategory_count'] for data in categorization_data['taxonomy']['primary_categories'].values())}")
    print(f"🎯 High Priority Areas: {len(categorization_data['taxonomy']['priority_patterns']['high_priority'])}")
    print(f"🚀 Research Directions: {len(categorization_data['taxonomy']['research_directions'])}")
    
    current_max = df['dominant_frequency'].max()
    print(f"📊 Current Achievement: {current_max:.0f} Hz ({current_max/10000*100:.1f}% of 10kHz target)")
    
    plt.show()
    return filename

if __name__ == "__main__":
    create_categorization_dashboard()
