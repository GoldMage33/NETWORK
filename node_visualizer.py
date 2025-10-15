#!/usr/bin/env python3
"""
NETWORK Node Visualization
Create visual representations of discovered nodes and their relationships
"""

import json
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import numpy as np
from typing import Dict, Any


class NodeVisualizer:
    """Visualize network nodes and relationships."""

    def __init__(self, node_report_file: str = 'node_discovery_report.json'):
        self.node_report_file = node_report_file
        self.report_data = self.load_report()
        self.G = nx.Graph()

    def load_report(self) -> Dict[str, Any]:
        """Load node discovery report."""
        try:
            with open(self.node_report_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading report: {e}")
            return {}

    def build_network_graph(self):
        """Build NetworkX graph from node data."""
        # Add nodes
        for category, nodes in self.report_data.get('node_categories', {}).items():
            for node in nodes:
                # Extract coordinates
                coords = node.get('coordinates', {})
                x, y = coords.get('x', 0), coords.get('y', 0)

                # Node attributes
                node_attrs = {
                    'category': category,
                    'risk_level': node.get('risk_level', 'unknown'),
                    'type': node.get('type', 'unknown'),
                    'pos': (x, y)
                }

                # Add key attributes
                for key, value in node.get('key_attributes', {}).items():
                    if isinstance(value, (str, int, float)):
                        node_attrs[key] = value

                self.G.add_node(node['id'], **node_attrs)

        # Add relationships (simplified - connect high-risk nodes)
        high_risk_nodes = [n for n, attrs in self.G.nodes(data=True)
                          if attrs.get('risk_level') in ['critical', 'high']]

        # Create connections between high-risk nodes
        for i, node1 in enumerate(high_risk_nodes):
            for node2 in high_risk_nodes[i+1:]:
                # Calculate distance
                pos1 = self.G.nodes[node1]['pos']
                pos2 = self.G.nodes[node2]['pos']
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

                # Connect if close enough
                if distance < 200:
                    self.G.add_edge(node1, node2, weight=1/distance, relationship='risk_connection')

    def get_node_colors(self) -> Dict[str, str]:
        """Get color mapping for node categories."""
        return {
            'network_devices': '#FF6B6B',      # Red
            'frequency_nodes': '#4ECDC4',      # Teal
            'communication_nodes': '#45B7D1',  # Blue
            'monitoring_nodes': '#96CEB4',     # Green
            'data_nodes': '#FFEAA7',           # Yellow
            'topology_nodes': '#DDA0DD'        # Plum
        }

    def get_risk_colors(self) -> Dict[str, str]:
        """Get color mapping for risk levels."""
        return {
            'critical': '#DC143C',   # Crimson
            'high': '#FF4500',      # Orange Red
            'medium': '#FFA500',    # Orange
            'low': '#32CD32',       # Lime Green
            'unknown': '#808080'    # Gray
        }

    def create_category_visualization(self):
        """Create visualization colored by node category."""
        plt.figure(figsize=(16, 12))

        # Get positions
        pos = nx.get_node_attributes(self.G, 'pos')

        # Color by category
        category_colors = self.get_node_colors()
        node_colors = []
        for node in self.G.nodes():
            category = self.G.nodes[node].get('category', 'unknown')
            node_colors.append(category_colors.get(category, '#808080'))

        # Draw nodes
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors,
                              node_size=300, alpha=0.8)

        # Draw edges
        nx.draw_networkx_edges(self.G, pos, alpha=0.3, edge_color='gray')

        # Draw labels (only for important nodes)
        important_nodes = {}
        for node, attrs in self.G.nodes(data=True):
            if attrs.get('risk_level') in ['critical', 'high'] or 'topology' in node:
                # Shorten labels
                if 'device_' in node:
                    label = attrs.get('hostname', node).replace('.home', '')
                elif 'freq_' in node:
                    label = f"F{node.split('_')[1][:4]}"
                elif 'comm_' in node:
                    label = attrs.get('tag_phrase', node).replace(' ', '')
                else:
                    label = node[:8]
                important_nodes[node] = label

        nx.draw_networkx_labels(self.G, pos, important_nodes, font_size=8)

        # Create legend
        legend_elements = []
        for category, color in category_colors.items():
            count = sum(1 for n in self.G.nodes()
                       if self.G.nodes[n].get('category') == category)
            if count > 0:
                plt.scatter([], [], c=color, label=f'{category.replace("_", " ").title()} ({count})',
                           s=100, alpha=0.8)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('NETWORK Node Topology - By Category', fontsize=16, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('node_topology_category.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_risk_visualization(self):
        """Create visualization colored by risk level."""
        plt.figure(figsize=(16, 12))

        pos = nx.get_node_attributes(self.G, 'pos')

        # Color by risk
        risk_colors = self.get_risk_colors()
        node_colors = []
        for node in self.G.nodes():
            risk = self.G.nodes[node].get('risk_level', 'unknown')
            node_colors.append(risk_colors.get(risk, '#808080'))

        # Draw nodes with size based on risk
        node_sizes = []
        for node in self.G.nodes():
            risk = self.G.nodes[node].get('risk_level', 'unknown')
            if risk == 'critical':
                size = 500
            elif risk == 'high':
                size = 400
            elif risk == 'medium':
                size = 300
            elif risk == 'low':
                size = 200
            else:
                size = 250
            node_sizes.append(size)

        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors,
                              node_size=node_sizes, alpha=0.8)

        nx.draw_networkx_edges(self.G, pos, alpha=0.3, edge_color='gray')

        # Labels for critical/high risk nodes
        critical_labels = {}
        for node, attrs in self.G.nodes(data=True):
            if attrs.get('risk_level') in ['critical', 'high']:
                if 'device_' in node:
                    label = attrs.get('hostname', node).replace('.home', '')
                elif 'freq_' in node:
                    label = f"F{node.split('_')[1][:4]}"
                elif 'comm_' in node:
                    label = attrs.get('tag_phrase', node).replace(' ', '')
                else:
                    label = node[:8]
                critical_labels[node] = label

        nx.draw_networkx_labels(self.G, pos, critical_labels, font_size=8, font_weight='bold')

        # Legend
        legend_elements = []
        for risk, color in risk_colors.items():
            count = sum(1 for n in self.G.nodes()
                       if self.G.nodes[n].get('risk_level') == risk)
            if count > 0:
                plt.scatter([], [], c=color, label=f'{risk.title()} Risk ({count})',
                           s=100, alpha=0.8)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('NETWORK Node Topology - By Risk Level', fontsize=16, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('node_topology_risk.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_topology_map(self):
        """Create a simplified topology map."""
        plt.figure(figsize=(14, 10))

        pos = nx.get_node_attributes(self.G, 'pos')

        # Only show network devices and topology nodes
        device_nodes = [n for n in self.G.nodes()
                       if self.G.nodes[n].get('category') == 'network_devices']
        topology_nodes = [n for n in self.G.nodes()
                         if self.G.nodes[n].get('category') == 'topology_nodes']

        # Draw topology core
        if topology_nodes:
            nx.draw_networkx_nodes(self.G, pos, nodelist=topology_nodes,
                                  node_color='purple', node_size=600,
                                  node_shape='s', label='Topology Core')

        # Draw devices with risk-based colors
        risk_colors = self.get_risk_colors()
        for risk_level in ['critical', 'high', 'medium', 'low']:
            risk_nodes = [n for n in device_nodes
                         if self.G.nodes[n].get('risk_level') == risk_level]
            if risk_nodes:
                nx.draw_networkx_nodes(self.G, pos, nodelist=risk_nodes,
                                      node_color=risk_colors[risk_level],
                                      node_size=400, alpha=0.8,
                                      label=f'{risk_level.title()} Risk Devices')

        # Draw connections to topology core
        edges = []
        for device in device_nodes:
            for topo in topology_nodes:
                edges.append((device, topo))

        if edges:
            nx.draw_networkx_edges(self.G, pos, edgelist=edges,
                                  edge_color='blue', alpha=0.5, width=2)

        # Labels
        labels = {}
        for node in device_nodes + topology_nodes:
            if 'device_' in node:
                labels[node] = self.G.nodes[node].get('hostname', node).replace('.home', '')
            elif 'topology' in node:
                labels[node] = 'Network Core'

        nx.draw_networkx_labels(self.G, pos, labels, font_size=10, font_weight='bold')

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('NETWORK Device Topology Map', fontsize=16, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('network_topology_map.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_visualizations(self):
        """Generate all visualizations."""
        print("🎨 Generating node topology visualizations...")

        self.build_network_graph()

        print("  • Creating category-based visualization...")
        self.create_category_visualization()

        print("  • Creating risk-based visualization...")
        self.create_risk_visualization()

        print("  • Creating topology map...")
        self.create_topology_map()

        print("✅ Visualizations saved:")
        print("  - node_topology_category.png")
        print("  - node_topology_risk.png")
        print("  - network_topology_map.png")

    def print_node_summary(self):
        """Print a summary of discovered nodes."""
        print("\n📊 NETWORK Node Discovery Summary")
        print("=" * 50)

        summary = self.report_data.get('discovery_summary', {})

        print(f"Total Nodes Discovered: {summary.get('total_nodes_discovered', 0)}")
        print(f"Total Relationships: {summary.get('total_relationships', 0)}")

        print("\nNode Categories:")
        for category, count in summary.get('node_types_found', {}).items():
            print(f"  • {category.replace('_', ' ').title()}: {count}")

        print("\nRisk Distribution:")
        for risk, count in summary.get('risk_distribution', {}).items():
            risk_icon = {'critical': '🚨', 'high': '⚠️', 'medium': '🟡', 'low': '✅', 'unknown': '❓'}.get(risk, '❓')
            print(f"  {risk_icon} {risk.title()}: {count}")

        print("\nCritical Findings:")
        for finding in self.report_data.get('critical_findings', []):
            print(f"  • {finding}")

        perf = self.report_data.get('performance_insights', {})
        print("\nPerformance Insights:")
        print(f"  • Average Response Time: {perf.get('average_response_time', 'N/A')}ms")
        print(f"  • Response Time Range: {perf.get('response_time_range', 'N/A')}")
        print(f"  • Fastest Device: {perf.get('fastest_device', 'N/A')}")
        print(f"  • Slowest Device: {perf.get('slowest_device', 'N/A')}")

        connectivity = self.report_data.get('connectivity_analysis', {})
        print("\nConnectivity Analysis:")
        print(f"  • Average Connections per Node: {connectivity.get('average_connections_per_node', 0):.2f}")
        print("  • Most Connected Nodes:")
        for node_info in connectivity.get('most_connected_nodes', [])[:3]:
            print(f"    - {node_info['node_id']}: {node_info['connections']} connections")


def main():
    """Main function."""
    visualizer = NodeVisualizer()

    # Generate visualizations
    visualizer.generate_visualizations()

    # Print summary
    visualizer.print_node_summary()


if __name__ == "__main__":
    main()
