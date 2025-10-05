#!/usr/bin/env python3
"""
Network Pattern Categorization System - Organize discovered patterns for future analysis
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class NetworkPatternCategorizer:
    """Comprehensive categorization system for network layer patterns."""
    
    def __init__(self):
        self.pattern_categories = {
            'frequency_patterns': {},
            'harmonic_patterns': {},
            'cascade_patterns': {},
            'modulation_patterns': {},
            'emergence_patterns': {},
            'interference_patterns': {},
            'temporal_patterns': {},
            'behavioral_patterns': {},
            'threshold_patterns': {},
            'meta_patterns': {}
        }
        
        self.classification_rules = self._define_classification_rules()
        
    def _define_classification_rules(self):
        """Define rules for pattern classification."""
        return {
            'frequency_bands': {
                'ultra_low': (0, 1000),
                'low': (1000, 3000),
                'mid_low': (3000, 5000),
                'mid': (5000, 7000),
                'mid_high': (7000, 9000),
                'high': (9000, 9500),
                'ultra_high': (9500, 10000),
                'breakthrough': (10000, 12000)
            },
            'activity_levels': {
                'dormant': (0, 0.5),
                'low': (0.5, 1.0),
                'moderate': (1.0, 1.5),
                'high': (1.5, 2.0),
                'extreme': (2.0, 3.0)
            },
            'amplitude_ranges': {
                'weak': (0, 0.2),
                'moderate': (0.2, 0.4),
                'strong': (0.4, 0.6),
                'very_strong': (0.6, 0.8),
                'extreme': (0.8, 1.0)
            },
            'harmonic_types': {
                'octave': 2.0,
                'perfect_fifth': 1.5,
                'perfect_fourth': 1.33,
                'major_third': 1.25,
                'golden_ratio': 1.618,
                'tritone': 1.414
            },
            'layer_roles': {
                'foundation': ['physical_layer'],
                'coordination': ['data_link_layer', 'network_layer'],
                'transport': ['transport_layer'],
                'session': ['session_layer'],
                'presentation': ['presentation_layer'],
                'breakthrough': ['application_layer']
            }
        }
    
    def categorize_frequency_patterns(self, df):
        """Categorize patterns based on frequency characteristics."""
        
        freq_patterns = {
            'by_band': {},
            'progressions': {},
            'jumps': {},
            'stability': {},
            'resonances': {}
        }
        
        # 1. Categorize by frequency bands
        for band_name, (low, high) in self.classification_rules['frequency_bands'].items():
            band_data = df[(df['dominant_frequency'] >= low) & (df['dominant_frequency'] < high)]
            
            if len(band_data) > 0:
                freq_patterns['by_band'][band_name] = {
                    'count': len(band_data),
                    'layers': band_data['layer'].unique().tolist(),
                    'avg_activity': band_data['activity_score'].mean(),
                    'avg_amplitude': band_data['peak_amplitude'].mean(),
                    'frequency_range': (band_data['dominant_frequency'].min(), 
                                      band_data['dominant_frequency'].max()),
                    'dominant_patterns': band_data['patterns'].value_counts().head(3).to_dict()
                }
        
        # 2. Analyze frequency progressions
        for layer in df['layer'].unique():
            layer_data = df[df['layer'] == layer].sort_values('timestamp')
            if len(layer_data) > 5:
                frequencies = layer_data['dominant_frequency'].values
                
                # Calculate progression characteristics
                freq_diff = np.diff(frequencies)
                progression_trend = np.polyfit(range(len(frequencies)), frequencies, 1)[0]
                
                freq_patterns['progressions'][layer] = {
                    'trend': 'ascending' if progression_trend > 0 else 'descending',
                    'trend_strength': abs(progression_trend),
                    'volatility': np.std(freq_diff),
                    'max_jump': max(freq_diff) if len(freq_diff) > 0 else 0,
                    'min_jump': min(freq_diff) if len(freq_diff) > 0 else 0,
                    'jump_count': len([x for x in freq_diff if abs(x) > 500])
                }
        
        # 3. Identify major frequency jumps
        all_jumps = []
        for layer in df['layer'].unique():
            layer_data = df[df['layer'] == layer].sort_values('timestamp')
            if len(layer_data) > 1:
                freq_diff = np.diff(layer_data['dominant_frequency'])
                for i, jump in enumerate(freq_diff):
                    if abs(jump) > 500:  # Significant jump threshold
                        all_jumps.append({
                            'layer': layer,
                            'magnitude': jump,
                            'direction': 'up' if jump > 0 else 'down',
                            'timestamp': layer_data.iloc[i+1]['timestamp'],
                            'from_freq': layer_data.iloc[i]['dominant_frequency'],
                            'to_freq': layer_data.iloc[i+1]['dominant_frequency']
                        })
        
        freq_patterns['jumps'] = sorted(all_jumps, key=lambda x: abs(x['magnitude']), reverse=True)
        
        self.pattern_categories['frequency_patterns'] = freq_patterns
        return freq_patterns
    
    def categorize_harmonic_patterns(self, df):
        """Categorize harmonic relationships and musical patterns."""
        
        harmonic_patterns = {
            'by_type': {},
            'by_layer': {},
            'networks': {},
            'resonances': {}
        }
        
        # Analyze harmonic relationships
        for layer in df['layer'].unique():
            layer_data = df[df['layer'] == layer]
            frequencies = layer_data['dominant_frequency'].values
            
            layer_harmonics = []
            
            # Find harmonic relationships
            for i, freq1 in enumerate(frequencies):
                for j, freq2 in enumerate(frequencies[i+1:], i+1):
                    if freq1 > 0 and freq2 > 0:
                        ratio = max(freq1, freq2) / min(freq1, freq2)
                        
                        # Check against known harmonic ratios
                        for harmonic_name, target_ratio in self.classification_rules['harmonic_types'].items():
                            if abs(ratio - target_ratio) < 0.1:
                                layer_harmonics.append({
                                    'type': harmonic_name,
                                    'ratio': ratio,
                                    'freq_low': min(freq1, freq2),
                                    'freq_high': max(freq1, freq2),
                                    'strength': 1.0 / abs(ratio - target_ratio) if abs(ratio - target_ratio) > 0 else 100
                                })
            
            if layer_harmonics:
                # Categorize by harmonic type
                harmonic_types = defaultdict(list)
                for h in layer_harmonics:
                    harmonic_types[h['type']].append(h)
                
                harmonic_patterns['by_layer'][layer] = {
                    'total_harmonics': len(layer_harmonics),
                    'types': dict(harmonic_types),
                    'strongest': max(layer_harmonics, key=lambda x: x['strength']),
                    'frequency_span': (min(h['freq_low'] for h in layer_harmonics),
                                     max(h['freq_high'] for h in layer_harmonics))
                }
        
        # Categorize by harmonic type across all layers
        for harmonic_type in self.classification_rules['harmonic_types']:
            type_harmonics = []
            for layer_data in harmonic_patterns['by_layer'].values():
                if harmonic_type in layer_data['types']:
                    type_harmonics.extend(layer_data['types'][harmonic_type])
            
            if type_harmonics:
                harmonic_patterns['by_type'][harmonic_type] = {
                    'count': len(type_harmonics),
                    'avg_strength': np.mean([h['strength'] for h in type_harmonics]),
                    'frequency_range': (min(h['freq_low'] for h in type_harmonics),
                                      max(h['freq_high'] for h in type_harmonics)),
                    'layers_involved': len([layer for layer, data in harmonic_patterns['by_layer'].items() 
                                          if harmonic_type in data['types']])
                }
        
        self.pattern_categories['harmonic_patterns'] = harmonic_patterns
        return harmonic_patterns
    
    def categorize_cascade_patterns(self, df):
        """Categorize cascade and synchronization patterns."""
        
        cascade_patterns = {
            'perfect_cascades': [],
            'partial_cascades': [],
            'frequency_ladders': {},
            'synchronization_events': []
        }
        
        # Group by timestamp to find simultaneous events
        time_groups = df.groupby('timestamp')
        
        layer_order = ['physical_layer', 'data_link_layer', 'network_layer', 'transport_layer', 
                      'session_layer', 'presentation_layer', 'application_layer']
        
        for timestamp, group in time_groups:
            layers = group['layer'].tolist()
            frequencies = group['dominant_frequency'].tolist()
            activities = group['activity_score'].tolist()
            
            cascade_info = {
                'timestamp': timestamp,
                'layer_count': len(layers),
                'layers': layers,
                'frequencies': frequencies,
                'activities': activities,
                'total_activity': sum(activities),
                'frequency_span': max(frequencies) - min(frequencies),
                'avg_frequency': np.mean(frequencies)
            }
            
            # Classify cascade type
            if len(layers) == 7:  # All layers present
                cascade_patterns['perfect_cascades'].append(cascade_info)
            elif len(layers) >= 4:  # Significant cascade
                cascade_patterns['partial_cascades'].append(cascade_info)
            
            # Synchronization events (high activity across multiple layers)
            if len(layers) >= 3 and np.mean(activities) > 1.5:
                cascade_patterns['synchronization_events'].append(cascade_info)
        
        # Analyze frequency ladders (progression through OSI layers)
        for cascade in cascade_patterns['perfect_cascades']:
            layer_freq_map = dict(zip(cascade['layers'], cascade['frequencies']))
            ordered_freqs = [layer_freq_map.get(layer, 0) for layer in layer_order]
            
            cascade['frequency_ladder'] = ordered_freqs
            cascade['ladder_progression'] = np.diff(ordered_freqs)
            cascade['is_ascending'] = all(diff >= 0 for diff in cascade['ladder_progression'] if diff != 0)
        
        self.pattern_categories['cascade_patterns'] = cascade_patterns
        return cascade_patterns
    
    def categorize_emergence_patterns(self, df):
        """Categorize emergent and breakthrough patterns."""
        
        emergence_patterns = {
            'frequency_surges': [],
            'amplitude_spikes': [],
            'activity_bursts': [],
            'phase_transitions': [],
            'compound_events': [],
            'breakthrough_events': []
        }
        
        # Analyze emergence patterns for each layer
        for layer in df['layer'].unique():
            layer_data = df[df['layer'] == layer].sort_values('timestamp')
            
            if len(layer_data) < 5:
                continue
            
            # Calculate change metrics
            freq_changes = np.diff(layer_data['dominant_frequency'])
            activity_changes = np.diff(layer_data['activity_score'])
            amplitude_changes = np.diff(layer_data['peak_amplitude'])
            
            # Define thresholds for emergence
            freq_threshold = np.mean(freq_changes) + 2 * np.std(freq_changes)
            activity_threshold = np.mean(activity_changes) + 2 * np.std(activity_changes)
            amplitude_threshold = np.mean(amplitude_changes) + 2 * np.std(amplitude_changes)
            
            # Identify emergence events
            for i in range(len(freq_changes)):
                event = {
                    'layer': layer,
                    'timestamp': layer_data.iloc[i+1]['timestamp'],
                    'freq_before': layer_data.iloc[i]['dominant_frequency'],
                    'freq_after': layer_data.iloc[i+1]['dominant_frequency'],
                    'freq_change': freq_changes[i],
                    'activity_change': activity_changes[i],
                    'amplitude_change': amplitude_changes[i]
                }
                
                # Categorize event type
                event_types = []
                
                if freq_changes[i] > freq_threshold:
                    event_types.append('frequency_surge')
                    emergence_patterns['frequency_surges'].append(event)
                
                if activity_changes[i] > activity_threshold:
                    event_types.append('activity_burst')
                    emergence_patterns['activity_bursts'].append(event)
                
                if amplitude_changes[i] > amplitude_threshold:
                    event_types.append('amplitude_spike')
                    emergence_patterns['amplitude_spikes'].append(event)
                
                # Phase transitions (direction changes)
                if i > 0 and i < len(freq_changes) - 1:
                    prev_trend = freq_changes[i-1]
                    curr_trend = freq_changes[i]
                    if (prev_trend < 0 < curr_trend) or (prev_trend > 0 > curr_trend):
                        event_types.append('phase_transition')
                        emergence_patterns['phase_transitions'].append(event)
                
                # Compound events (multiple simultaneous changes)
                if len(event_types) > 1:
                    event['event_types'] = event_types
                    emergence_patterns['compound_events'].append(event)
                
                # Breakthrough events (high frequency achievements)
                if event['freq_after'] >= 9000:
                    event['breakthrough_level'] = 'high_frequency'
                    emergence_patterns['breakthrough_events'].append(event)
                elif event['freq_after'] >= 9500:
                    event['breakthrough_level'] = 'near_target'
                    emergence_patterns['breakthrough_events'].append(event)
        
        self.pattern_categories['emergence_patterns'] = emergence_patterns
        return emergence_patterns
    
    def categorize_behavioral_patterns(self, df):
        """Categorize behavioral and personality patterns of layers."""
        
        behavioral_patterns = {
            'layer_personalities': {},
            'interaction_patterns': {},
            'stability_patterns': {},
            'adaptation_patterns': {}
        }
        
        # Analyze layer personalities
        for layer in df['layer'].unique():
            layer_data = df[df['layer'] == layer]
            
            # Calculate behavioral metrics
            freq_stability = 1.0 / (1.0 + np.std(layer_data['dominant_frequency']))
            activity_consistency = 1.0 / (1.0 + np.std(layer_data['activity_score']))
            amplitude_control = 1.0 / (1.0 + np.std(layer_data['peak_amplitude']))
            
            # Determine behavioral characteristics
            personality = {
                'stability_index': freq_stability,
                'consistency_index': activity_consistency,
                'control_index': amplitude_control,
                'frequency_range': (layer_data['dominant_frequency'].min(), 
                                  layer_data['dominant_frequency'].max()),
                'activity_range': (layer_data['activity_score'].min(), 
                                 layer_data['activity_score'].max()),
                'behavioral_type': 'unknown'
            }
            
            # Classify behavioral type
            if freq_stability > 0.8 and activity_consistency > 0.8:
                personality['behavioral_type'] = 'stable_consistent'
            elif freq_stability < 0.3 or activity_consistency < 0.3:
                personality['behavioral_type'] = 'dynamic_variable'
            elif layer_data['dominant_frequency'].max() >= 9000:
                personality['behavioral_type'] = 'breakthrough_capable'
            else:
                personality['behavioral_type'] = 'moderate_adaptive'
            
            # Pattern analysis
            pattern_counts = layer_data['patterns'].value_counts()
            personality['dominant_patterns'] = pattern_counts.head(3).to_dict()
            personality['pattern_diversity'] = len(pattern_counts)
            
            behavioral_patterns['layer_personalities'][layer] = personality
        
        self.pattern_categories['behavioral_patterns'] = behavioral_patterns
        return behavioral_patterns
    
    def create_pattern_taxonomy(self):
        """Create a comprehensive taxonomy of all discovered patterns."""
        
        taxonomy = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_categories': len(self.pattern_categories),
                'classification_version': '1.0'
            },
            'primary_categories': {},
            'cross_references': {},
            'priority_patterns': {},
            'research_directions': {}
        }
        
        # Primary category summaries
        for category, patterns in self.pattern_categories.items():
            if patterns:
                taxonomy['primary_categories'][category] = {
                    'subcategory_count': len(patterns),
                    'description': self._get_category_description(category),
                    'key_findings': self._extract_key_findings(category, patterns),
                    'research_priority': self._assess_research_priority(category, patterns)
                }
        
        # Cross-references between categories
        taxonomy['cross_references'] = {
            'frequency_harmonic': 'Frequency patterns show harmonic relationships',
            'cascade_emergence': 'Cascade events trigger emergence patterns',
            'behavioral_modulation': 'Layer personalities correlate with modulation patterns',
            'interference_breakthrough': 'Interference patterns enable breakthrough events'
        }
        
        # Priority patterns for future research
        taxonomy['priority_patterns'] = {
            'high_priority': [
                'perfect_cascade_events',
                'frequency_surge_mechanisms',
                'harmonic_resonance_networks',
                'breakthrough_threshold_patterns'
            ],
            'medium_priority': [
                'cross_layer_interference',
                'amplitude_modulation_rhythms',
                'phase_transition_mechanisms',
                'behavioral_adaptation_patterns'
            ],
            'research_priority': [
                'meta_pattern_emergence',
                'temporal_correlation_networks',
                'stability_breakthrough_balance',
                'multi_dimensional_synchronization'
            ]
        }
        
        # Future research directions
        taxonomy['research_directions'] = {
            '10khz_completion': {
                'objective': 'Close final 91Hz gap to 10kHz target',
                'key_patterns': ['perfect_cascades', 'frequency_surges', 'harmonic_resonance'],
                'approach': 'Coordinate all breakthrough mechanisms simultaneously'
            },
            'pattern_intelligence': {
                'objective': 'Understand emergent intelligence in pattern networks',
                'key_patterns': ['behavioral_personalities', 'adaptation_patterns', 'meta_patterns'],
                'approach': 'Study pattern evolution and self-organization'
            },
            'harmonic_engineering': {
                'objective': 'Exploit musical harmonic relationships for optimization',
                'key_patterns': ['harmonic_networks', 'resonance_amplification'],
                'approach': 'Apply acoustic engineering principles to network patterns'
            }
        }
        
        return taxonomy
    
    def _get_category_description(self, category):
        """Get description for each pattern category."""
        descriptions = {
            'frequency_patterns': 'Patterns based on frequency characteristics and behaviors',
            'harmonic_patterns': 'Musical-scale harmonic relationships and resonances',
            'cascade_patterns': 'Multi-layer synchronization and cascade events',
            'modulation_patterns': 'Amplitude and frequency modulation behaviors',
            'emergence_patterns': 'Spontaneous and breakthrough pattern events',
            'interference_patterns': 'Cross-layer interference and interaction patterns',
            'temporal_patterns': 'Time-based pattern evolution and sequences',
            'behavioral_patterns': 'Layer personality and behavioral characteristics',
            'threshold_patterns': 'Critical threshold and transition patterns',
            'meta_patterns': 'Higher-order patterns and pattern-of-patterns'
        }
        return descriptions.get(category, 'Unknown pattern category')
    
    def _extract_key_findings(self, category, patterns):
        """Extract key findings for each category."""
        
        key_findings = []
        
        if category == 'frequency_patterns' and 'by_band' in patterns:
            breakthrough_band = patterns['by_band'].get('breakthrough', {})
            if breakthrough_band:
                key_findings.append(f"Breakthrough band contains {breakthrough_band['count']} events")
        
        if category == 'harmonic_patterns' and 'by_type' in patterns:
            harmonic_types = len(patterns['by_type'])
            key_findings.append(f"Discovered {harmonic_types} distinct harmonic types")
        
        if category == 'cascade_patterns':
            perfect_cascades = len(patterns.get('perfect_cascades', []))
            key_findings.append(f"Identified {perfect_cascades} perfect 7-layer cascades")
        
        if category == 'emergence_patterns':
            breakthrough_events = len(patterns.get('breakthrough_events', []))
            key_findings.append(f"Detected {breakthrough_events} breakthrough events")
        
        return key_findings
    
    def _assess_research_priority(self, category, patterns):
        """Assess research priority for each category."""
        
        high_priority_categories = ['frequency_patterns', 'cascade_patterns', 'emergence_patterns']
        medium_priority_categories = ['harmonic_patterns', 'behavioral_patterns']
        
        if category in high_priority_categories:
            return 'high'
        elif category in medium_priority_categories:
            return 'medium'
        else:
            return 'low'
    
    def save_categorization(self, filename=None):
        """Save the complete pattern categorization."""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'pattern_categorization_{timestamp}.json'
        
        # Create taxonomy
        taxonomy = self.create_pattern_taxonomy()
        
        # Combine all data
        categorization_data = {
            'taxonomy': taxonomy,
            'categories': self.pattern_categories,
            'classification_rules': self.classification_rules
        }
        
        # Convert numpy types to JSON serializable
        categorization_json = json.loads(json.dumps(categorization_data, default=str))
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(categorization_json, f, indent=2)
        
        return filename

def main():
    """Main categorization function."""
    
    print("🗂️ NETWORK PATTERN CATEGORIZATION SYSTEM")
    print("=" * 50)
    print("📁 Organizing discovered patterns for future analysis")
    print()
    
    # Load the network layer signatures data
    csv_file = '/Users/dominikkomorek/NETWORK/network_layer_signatures_20251006_001645.csv'
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} layer signature events")
        
        # Initialize categorizer
        categorizer = NetworkPatternCategorizer()
        
        # Perform comprehensive categorization
        print("\n🔍 CATEGORIZING PATTERNS...")
        print("-" * 30)
        
        freq_patterns = categorizer.categorize_frequency_patterns(df)
        print(f"✓ Frequency patterns: {len(freq_patterns)} subcategories")
        
        harmonic_patterns = categorizer.categorize_harmonic_patterns(df)
        print(f"✓ Harmonic patterns: {len(harmonic_patterns)} subcategories")
        
        cascade_patterns = categorizer.categorize_cascade_patterns(df)
        print(f"✓ Cascade patterns: {len(cascade_patterns)} subcategories")
        
        emergence_patterns = categorizer.categorize_emergence_patterns(df)
        print(f"✓ Emergence patterns: {len(emergence_patterns)} subcategories")
        
        behavioral_patterns = categorizer.categorize_behavioral_patterns(df)
        print(f"✓ Behavioral patterns: {len(behavioral_patterns)} subcategories")
        
        # Create and save taxonomy
        print(f"\n📊 CREATING PATTERN TAXONOMY...")
        print("-" * 35)
        
        taxonomy = categorizer.create_pattern_taxonomy()
        
        print(f"✓ Primary categories: {len(taxonomy['primary_categories'])}")
        print(f"✓ Cross-references: {len(taxonomy['cross_references'])}")
        print(f"✓ Priority patterns: {len(taxonomy['priority_patterns'])}")
        print(f"✓ Research directions: {len(taxonomy['research_directions'])}")
        
        # Save categorization
        filename = categorizer.save_categorization()
        print(f"\n💾 CATEGORIZATION SAVED")
        print(f"✅ File: {filename}")
        
        # Display summary
        print(f"\n📋 CATEGORIZATION SUMMARY")
        print("=" * 30)
        
        for category, data in taxonomy['primary_categories'].items():
            priority = data['research_priority']
            priority_icon = "🔴" if priority == 'high' else "🟡" if priority == 'medium' else "🟢"
            print(f"{priority_icon} {category.replace('_', ' ').title()}: {data['subcategory_count']} subcategories")
            for finding in data['key_findings'][:2]:  # Show top 2 findings
                print(f"   • {finding}")
        
        print(f"\n🎯 HIGH PRIORITY RESEARCH AREAS:")
        for area in taxonomy['priority_patterns']['high_priority']:
            print(f"   🔴 {area.replace('_', ' ').title()}")
        
        print(f"\n🚀 FUTURE RESEARCH DIRECTIONS:")
        for direction, info in taxonomy['research_directions'].items():
            print(f"   • {direction.replace('_', ' ').title()}: {info['objective']}")
        
        print(f"\n✅ Pattern categorization complete! Ready for future analysis.")
        
    except Exception as e:
        print(f"❌ Error during categorization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
