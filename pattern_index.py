#!/usr/bin/env python3
"""
Pattern Index System - Quick lookup and analysis tool for categorized patterns
"""

import pandas as pd
import json
from datetime import datetime
from collections import defaultdict

class PatternIndex:
    """Comprehensive index system for quick pattern lookup and analysis."""
    
    def __init__(self, categorization_file='pattern_categorization_20251006_003509.json'):
        """Initialize with categorization data."""
        try:
            with open(categorization_file, 'r') as f:
                self.data = json.load(f)
            print(f"✅ Loaded pattern categorization from {categorization_file}")
        except FileNotFoundError:
            print(f"❌ Could not find {categorization_file}")
            self.data = {}
    
    def create_quick_reference_guide(self):
        """Create a quick reference guide for all patterns."""
        
        guide = {
            'pattern_lookup': {},
            'layer_profiles': {},
            'frequency_bands': {},
            'breakthrough_strategies': {},
            'research_priorities': {}
        }
        
        # Pattern Lookup Dictionary
        categories = self.data.get('categories', {})
        
        for category_name, category_data in categories.items():
            guide['pattern_lookup'][category_name] = {
                'description': self._get_category_description(category_name),
                'subcategories': list(category_data.keys()) if isinstance(category_data, dict) else [],
                'key_metrics': self._extract_key_metrics(category_name, category_data),
                'analysis_methods': self._get_analysis_methods(category_name)
            }
        
        # Layer Profiles
        behavioral_data = categories.get('behavioral_patterns', {}).get('layer_personalities', {})
        for layer, personality in behavioral_data.items():
            guide['layer_profiles'][layer] = {
                'behavioral_type': personality.get('behavioral_type', 'unknown'),
                'stability_index': personality.get('stability_index', 0),
                'consistency_index': personality.get('consistency_index', 0),
                'frequency_range': personality.get('frequency_range', [0, 0]),
                'dominant_patterns': personality.get('dominant_patterns', {}),
                'breakthrough_capability': 'High' if personality.get('behavioral_type') == 'breakthrough_capable' else 'Medium'
            }
        
        # Frequency Bands Analysis
        freq_data = categories.get('frequency_patterns', {}).get('by_band', {})
        for band_name, band_info in freq_data.items():
            guide['frequency_bands'][band_name] = {
                'event_count': band_info.get('count', 0),
                'active_layers': band_info.get('layers', []),
                'avg_activity': band_info.get('avg_activity', 0),
                'frequency_range': band_info.get('frequency_range', [0, 0]),
                'breakthrough_potential': self._assess_breakthrough_potential(band_name, band_info)
            }
        
        # Breakthrough Strategies
        taxonomy = self.data.get('taxonomy', {})
        research_directions = taxonomy.get('research_directions', {})
        
        for strategy_name, strategy_info in research_directions.items():
            guide['breakthrough_strategies'][strategy_name] = {
                'objective': strategy_info.get('objective', ''),
                'key_patterns': strategy_info.get('key_patterns', []),
                'approach': strategy_info.get('approach', ''),
                'priority_level': self._assess_strategy_priority(strategy_name),
                'implementation_complexity': self._assess_complexity(strategy_name)
            }
        
        # Research Priorities
        priority_patterns = taxonomy.get('priority_patterns', {})
        for priority_level, patterns in priority_patterns.items():
            guide['research_priorities'][priority_level] = {
                'pattern_count': len(patterns),
                'patterns': patterns,
                'urgency': self._assess_urgency(priority_level),
                'expected_impact': self._assess_impact(priority_level)
            }
        
        return guide
    
    def generate_analysis_commands(self):
        """Generate analysis commands for different pattern investigations."""
        
        commands = {
            'frequency_analysis': [
                "df[df['dominant_frequency'] >= 9000].groupby('layer').size()",
                "df.groupby('layer')['dominant_frequency'].describe()",
                "df[df['dominant_frequency'] >= 9500]['patterns'].value_counts()"
            ],
            'cascade_analysis': [
                "df.groupby('timestamp').size().value_counts()",
                "df[df.groupby('timestamp').transform('size') == 7]",
                "df.groupby(['timestamp', 'layer']).first().reset_index()"
            ],
            'harmonic_analysis': [
                "df.groupby('layer')['dominant_frequency'].apply(lambda x: x.corr(x.shift()))",
                "df['dominant_frequency'].apply(lambda x: x/df['dominant_frequency'].median())",
                "df[df['peak_amplitude'] > df['peak_amplitude'].quantile(0.9)]"
            ],
            'emergence_analysis': [
                "df['freq_change'] = df.groupby('layer')['dominant_frequency'].diff()",
                "df[df['freq_change'] > df['freq_change'].quantile(0.95)]",
                "df.groupby('layer')['activity_score'].rolling(3).mean()"
            ],
            'behavioral_analysis': [
                "df.groupby('layer')['dominant_frequency'].std()",
                "df.groupby('layer')['activity_score'].std()",
                "df.groupby('layer')['patterns'].nunique()"
            ]
        }
        
        return commands
    
    def create_pattern_search_index(self):
        """Create searchable index of all patterns."""
        
        search_index = {
            'by_frequency': defaultdict(list),
            'by_layer': defaultdict(list),
            'by_activity': defaultdict(list),
            'by_pattern_type': defaultdict(list),
            'by_breakthrough_level': defaultdict(list)
        }
        
        # Load original data for indexing
        try:
            df = pd.read_csv('network_layer_signatures_20251006_001645.csv')
            
            for idx, row in df.iterrows():
                event_id = f"event_{idx}"
                
                # Index by frequency bands
                freq = row['dominant_frequency']
                if freq >= 10000:
                    search_index['by_frequency']['breakthrough'].append(event_id)
                elif freq >= 9500:
                    search_index['by_frequency']['near_target'].append(event_id)
                elif freq >= 9000:
                    search_index['by_frequency']['high'].append(event_id)
                elif freq >= 7000:
                    search_index['by_frequency']['mid_high'].append(event_id)
                else:
                    search_index['by_frequency']['lower'].append(event_id)
                
                # Index by layer
                search_index['by_layer'][row['layer']].append(event_id)
                
                # Index by activity level
                activity = row['activity_score']
                if activity >= 2.0:
                    search_index['by_activity']['extreme'].append(event_id)
                elif activity >= 1.5:
                    search_index['by_activity']['high'].append(event_id)
                elif activity >= 1.0:
                    search_index['by_activity']['moderate'].append(event_id)
                else:
                    search_index['by_activity']['low'].append(event_id)
                
                # Index by pattern type
                pattern_type = row.get('patterns', 'unknown')
                search_index['by_pattern_type'][pattern_type].append(event_id)
                
                # Index by breakthrough level
                if freq >= 9000 and activity >= 1.5:
                    search_index['by_breakthrough_level']['high_potential'].append(event_id)
                elif freq >= 8000 and activity >= 1.0:
                    search_index['by_breakthrough_level']['medium_potential'].append(event_id)
                else:
                    search_index['by_breakthrough_level']['developing'].append(event_id)
        
        except Exception as e:
            print(f"⚠️ Could not create search index: {e}")
        
        return dict(search_index)
    
    def _get_category_description(self, category):
        """Get description for category."""
        descriptions = {
            'frequency_patterns': 'Frequency-based network behaviors and characteristics',
            'harmonic_patterns': 'Musical-scale harmonic relationships in network frequencies',
            'cascade_patterns': 'Multi-layer synchronization and coordination events',
            'modulation_patterns': 'Amplitude and frequency modulation behaviors',
            'emergence_patterns': 'Spontaneous pattern emergence and breakthrough events',
            'interference_patterns': 'Cross-layer interference and interaction patterns',
            'behavioral_patterns': 'Layer personality and behavioral characteristics'
        }
        return descriptions.get(category, 'Pattern category')
    
    def _extract_key_metrics(self, category, data):
        """Extract key metrics for each category."""
        metrics = []
        
        if category == 'frequency_patterns' and isinstance(data, dict):
            if 'by_band' in data:
                metrics.append(f"Frequency bands: {len(data['by_band'])}")
            if 'jumps' in data:
                metrics.append(f"Major jumps: {len(data['jumps'])}")
        
        elif category == 'cascade_patterns' and isinstance(data, dict):
            if 'perfect_cascades' in data:
                metrics.append(f"Perfect cascades: {len(data['perfect_cascades'])}")
            if 'synchronization_events' in data:
                metrics.append(f"Sync events: {len(data['synchronization_events'])}")
        
        elif category == 'emergence_patterns' and isinstance(data, dict):
            total_events = sum(len(events) for events in data.values() if isinstance(events, list))
            metrics.append(f"Total emergence events: {total_events}")
        
        return metrics
    
    def _get_analysis_methods(self, category):
        """Get recommended analysis methods for each category."""
        methods = {
            'frequency_patterns': ['Spectral analysis', 'Trend analysis', 'Band comparison'],
            'harmonic_patterns': ['Harmonic ratio analysis', 'Resonance detection', 'Musical theory application'],
            'cascade_patterns': ['Synchronization analysis', 'Cross-correlation', 'Timeline analysis'],
            'emergence_patterns': ['Change point detection', 'Anomaly detection', 'Threshold analysis'],
            'behavioral_patterns': ['Clustering analysis', 'Personality profiling', 'Stability metrics']
        }
        return methods.get(category, ['General statistical analysis'])
    
    def _assess_breakthrough_potential(self, band_name, band_info):
        """Assess breakthrough potential for frequency bands."""
        if band_name in ['breakthrough', 'ultra_high']:
            return 'Very High'
        elif band_name in ['high', 'mid_high']:
            return 'High'
        elif band_name in ['mid']:
            return 'Medium'
        else:
            return 'Low'
    
    def _assess_strategy_priority(self, strategy_name):
        """Assess priority level for strategies."""
        high_priority = ['10khz_completion']
        medium_priority = ['pattern_intelligence']
        
        if strategy_name in high_priority:
            return 'Critical'
        elif strategy_name in medium_priority:
            return 'High'
        else:
            return 'Medium'
    
    def _assess_complexity(self, strategy_name):
        """Assess implementation complexity."""
        complex_strategies = ['pattern_intelligence', 'harmonic_engineering']
        
        if strategy_name in complex_strategies:
            return 'High'
        else:
            return 'Medium'
    
    def _assess_urgency(self, priority_level):
        """Assess urgency for priority levels."""
        urgency_map = {
            'high_priority': 'Immediate',
            'medium_priority': 'Short-term',
            'research_priority': 'Long-term'
        }
        return urgency_map.get(priority_level, 'Medium')
    
    def _assess_impact(self, priority_level):
        """Assess expected impact."""
        impact_map = {
            'high_priority': 'Direct breakthrough capability',
            'medium_priority': 'Significant optimization',
            'research_priority': 'Fundamental understanding'
        }
        return impact_map.get(priority_level, 'Moderate improvement')
    
    def save_pattern_index(self, filename=None):
        """Save the complete pattern index."""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'pattern_index_{timestamp}.json'
        
        # Create comprehensive index
        index_data = {
            'quick_reference': self.create_quick_reference_guide(),
            'analysis_commands': self.generate_analysis_commands(),
            'search_index': self.create_pattern_search_index(),
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0',
                'total_categories': len(self.data.get('categories', {}))
            }
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(index_data, f, indent=2, default=str)
        
        return filename

def main():
    """Main pattern indexing function."""
    
    print("📇 NETWORK PATTERN INDEX SYSTEM")
    print("=" * 40)
    print("🔍 Creating comprehensive pattern lookup system")
    print()
    
    # Initialize index
    pattern_index = PatternIndex()
    
    if not pattern_index.data:
        print("❌ No categorization data available")
        return
    
    # Create and save index
    print("📋 Creating pattern index components...")
    
    quick_ref = pattern_index.create_quick_reference_guide()
    print(f"✓ Quick reference guide: {len(quick_ref)} sections")
    
    commands = pattern_index.generate_analysis_commands()
    print(f"✓ Analysis commands: {len(commands)} categories")
    
    search_index = pattern_index.create_pattern_search_index()
    print(f"✓ Search index: {len(search_index)} dimensions")
    
    # Save complete index
    filename = pattern_index.save_pattern_index()
    print(f"\n💾 Pattern index saved: {filename}")
    
    # Display summary
    print(f"\n📊 PATTERN INDEX SUMMARY")
    print("=" * 30)
    
    print(f"🏷️  Pattern Categories: {len(quick_ref.get('pattern_lookup', {}))}")
    print(f"👤 Layer Profiles: {len(quick_ref.get('layer_profiles', {}))}")
    print(f"📊 Frequency Bands: {len(quick_ref.get('frequency_bands', {}))}")
    print(f"🚀 Breakthrough Strategies: {len(quick_ref.get('breakthrough_strategies', {}))}")
    print(f"🎯 Research Priorities: {len(quick_ref.get('research_priorities', {}))}")
    
    print(f"\n🔍 SEARCH CAPABILITIES:")
    for dimension, indexes in search_index.items():
        total_entries = sum(len(entries) for entries in indexes.values())
        print(f"   • {dimension.replace('_', ' ').title()}: {len(indexes)} categories, {total_entries} entries")
    
    print(f"\n⚡ QUICK ACCESS EXAMPLES:")
    print("   • High breakthrough potential events")
    print("   • Perfect cascade sequences")
    print("   • Harmonic relationship networks")
    print("   • Layer behavioral personalities")
    print("   • Frequency surge mechanisms")
    
    print(f"\n✅ Pattern index system ready for analysis!")

if __name__ == "__main__":
    main()
