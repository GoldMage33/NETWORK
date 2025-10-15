#!/usr/bin/env python3
"""
NETWORK Node Discovery System
Finds and analyzes all types of nodes in the NETWORK system
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Set
from collections import defaultdict


class NodeDiscoverySystem:
    """Comprehensive node discovery and analysis system."""

    def __init__(self, metadata_file: str = 'network_metadata.json'):
        self.metadata_file = metadata_file
        self.metadata = self.load_metadata()
        self.nodes = {
            'network_devices': [],
            'frequency_nodes': [],
            'communication_nodes': [],
            'monitoring_nodes': [],
            'data_nodes': [],
            'topology_nodes': []
        }
        self.node_relationships = defaultdict(list)
        self.node_metrics = {}

    def load_metadata(self) -> Dict[str, Any]:
        """Load network metadata."""
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return {}

    def discover_network_device_nodes(self) -> List[Dict[str, Any]]:
        """Discover physical network device nodes."""
        devices = self.metadata.get('network_analysis', {}).get('network_devices', {}).get('devices', [])

        device_nodes = []
        for device in devices:
            node = {
                'node_id': f"device_{device['ip'].replace('.', '_')}",
                'node_type': 'network_device',
                'ip_address': device['ip'],
                'hostname': device['name'],
                'device_type': device['type'],
                'response_time_ms': device.get('response_time_ms', 0),
                'status': 'active' if device.get('response_time_ms', 999) < 100 else 'slow',
                'coordinates': self.calculate_device_coordinates(device),
                'connections': [],
                'risk_level': self.assess_device_risk(device)
            }
            device_nodes.append(node)

        self.nodes['network_devices'] = device_nodes
        return device_nodes

    def calculate_device_coordinates(self, device: Dict[str, Any]) -> Dict[str, float]:
        """Calculate virtual coordinates for device visualization."""
        ip_parts = device['ip'].split('.')
        base_x = int(ip_parts[2]) * 50
        base_y = int(ip_parts[3]) * 30

        # Add some variation based on response time
        response_factor = device.get('response_time_ms', 100) / 100
        x = base_x + (response_factor * 20)
        y = base_y + (response_factor * 15)

        return {'x': x, 'y': y}

    def assess_device_risk(self, device: Dict[str, Any]) -> str:
        """Assess risk level for a device."""
        response_time = device.get('response_time_ms', 0)

        if device['ip'] == '192.168.1.1':  # Router
            return 'critical'
        elif response_time > 200:
            return 'high'
        elif response_time > 100:
            return 'medium'
        else:
            return 'low'

    def discover_frequency_nodes(self) -> List[Dict[str, Any]]:
        """Discover frequency leakage nodes."""
        leakage_points = self.metadata.get('network_analysis', {}).get('frequency_analysis', {}).get('top_leakage_points', [])

        frequency_nodes = []
        for point in leakage_points:
            node = {
                'node_id': f"freq_{int(point['frequency'])}",
                'node_type': 'frequency_leakage',
                'frequency_hz': point['frequency'],
                'strength_db': point['strength'],
                'rank': point['rank'],
                'coordinates': self.calculate_frequency_coordinates(point),
                'risk_level': self.assess_frequency_risk(point),
                'global_correlation': self.get_global_correlation(point['frequency'])
            }
            frequency_nodes.append(node)

        self.nodes['frequency_nodes'] = frequency_nodes
        return frequency_nodes

    def calculate_frequency_coordinates(self, point: Dict[str, Any]) -> Dict[str, float]:
        """Calculate coordinates for frequency nodes."""
        freq = point['frequency'] / 1000000  # Convert to MHz
        strength = point['strength']

        # Position based on frequency range and strength
        x = (freq / 2) * 100  # Spread across X axis
        y = 200 + (strength * 10)  # Higher strength = higher Y

        return {'x': x, 'y': y}

    def assess_frequency_risk(self, point: Dict[str, Any]) -> str:
        """Assess risk level for frequency leakage."""
        strength = point['strength']
        freq = point['frequency'] / 1000000  # MHz

        # Cellular frequencies are highest risk
        if 700 <= freq <= 2600:
            return 'critical'
        elif strength > 10:
            return 'high'
        elif strength > 5:
            return 'medium'
        else:
            return 'low'

    def get_global_correlation(self, frequency: float) -> str:
        """Get global correlation for frequency."""
        freq_mhz = frequency / 1000000

        if 700 <= freq_mhz <= 2600:
            return 'cellular_network'
        elif 2400 <= freq_mhz <= 2500:
            return 'wifi_2_4ghz'
        elif 5150 <= freq_mhz <= 5850:
            return 'wifi_5ghz'
        elif 2400 <= freq_mhz <= 2480:
            return 'bluetooth'
        else:
            return 'unknown'

    def discover_communication_nodes(self) -> List[Dict[str, Any]]:
        """Discover communication/tag detection nodes."""
        tag_data = self.metadata.get('network_analysis', {}).get('tag_detection', {})

        comm_nodes = []

        # Tag detection nodes
        for tag in tag_data.get('supported_tags', []):
            ranges = tag_data.get('detection_ranges', {}).get(tag, {})

            node = {
                'node_id': f"comm_{tag.replace(' ', '_')}",
                'node_type': 'communication_tag',
                'tag_phrase': tag,
                'frequency_range': ranges.get('frequency_range', 'unknown'),
                'min_strength': ranges.get('min_strength', 0),
                'confidence_max': ranges.get('confidence_max', 0),
                'coordinates': self.calculate_communication_coordinates(tag),
                'risk_level': 'high' if tag == 'neuralink' else 'medium'
            }
            comm_nodes.append(node)

        # Recent detections
        for detection in tag_data.get('recent_detections', []):
            node = {
                'node_id': f"detect_{detection['phrase'].replace(' ', '_')}_{int(detection['frequency'])}",
                'node_type': 'detection_event',
                'phrase': detection['phrase'],
                'frequency': detection['frequency'],
                'strength': detection['strength'],
                'confidence': detection['confidence'],
                'coordinates': self.calculate_detection_coordinates(detection),
                'risk_level': 'critical' if detection['confidence'] > 0.7 else 'high'
            }
            comm_nodes.append(node)

        self.nodes['communication_nodes'] = comm_nodes
        return comm_nodes

    def calculate_communication_coordinates(self, tag: str) -> Dict[str, float]:
        """Calculate coordinates for communication nodes."""
        if tag == 'tag it':
            return {'x': 300, 'y': 400}
        elif tag == 'neuralink':
            return {'x': 500, 'y': 450}
        else:
            return {'x': 400, 'y': 425}

    def calculate_detection_coordinates(self, detection: Dict[str, Any]) -> Dict[str, float]:
        """Calculate coordinates for detection events."""
        freq = detection['frequency'] / 1000  # Convert to kHz
        confidence = detection['confidence']

        x = 200 + (freq / 10)
        y = 300 + (confidence * 200)

        return {'x': x, 'y': y}

    def discover_monitoring_nodes(self) -> List[Dict[str, Any]]:
        """Discover monitoring instance nodes."""
        monitoring_sessions = self.metadata.get('monitoring_sessions', {})

        monitoring_nodes = []

        # Log file nodes
        for log_name, log_data in monitoring_sessions.get('log_files', {}).items():
            node = {
                'node_id': f"monitor_{log_name.replace('.log', '').replace('_', '_')}",
                'node_type': 'monitoring_instance',
                'log_file': log_name,
                'size_bytes': log_data.get('size', 0),
                'last_modified': log_data.get('last_modified', ''),
                'coordinates': self.calculate_monitoring_coordinates(log_name),
                'status': 'active' if 'bg_monitor' in log_name else 'completed'
            }
            monitoring_nodes.append(node)

        # Add summary node
        summary_node = {
            'node_id': 'monitoring_summary',
            'node_type': 'monitoring_coordinator',
            'total_sessions': monitoring_sessions.get('total_sessions', 0),
            'active_instances': monitoring_sessions.get('active_instances', 0),
            'total_runtime': monitoring_sessions.get('total_runtime_seconds', 0),
            'coordinates': {'x': 600, 'y': 100},
            'status': 'coordinator'
        }
        monitoring_nodes.append(summary_node)

        self.nodes['monitoring_nodes'] = monitoring_nodes
        return monitoring_nodes

    def calculate_monitoring_coordinates(self, log_name: str) -> Dict[str, float]:
        """Calculate coordinates for monitoring nodes."""
        if 'bg_monitor' in log_name:
            instance_num = int(log_name.split('_')[-1]) if log_name.split('_')[-1].isdigit() else 1
            return {'x': 700 + (instance_num * 50), 'y': 150 + (instance_num * 30)}
        elif 'console' in log_name:
            return {'x': 650, 'y': 200}
        else:
            return {'x': 600, 'y': 250}

    def discover_data_nodes(self) -> List[Dict[str, Any]]:
        """Discover data source nodes."""
        data_sources = self.metadata.get('data_sources', {})

        data_nodes = []

        for source_name, source_info in data_sources.items():
            node = {
                'node_id': f"data_{source_name.replace('_', '_')}",
                'node_type': 'data_source',
                'source_name': source_name,
                'description': source_info,
                'coordinates': self.calculate_data_coordinates(source_name),
                'data_type': self.classify_data_type(source_name)
            }
            data_nodes.append(node)

        self.nodes['data_nodes'] = data_nodes
        return data_nodes

    def calculate_data_coordinates(self, source_name: str) -> Dict[str, float]:
        """Calculate coordinates for data nodes."""
        if 'audio' in source_name:
            return {'x': 100, 'y': 500}
        elif 'radio' in source_name:
            return {'x': 200, 'y': 500}
        elif 'combined' in source_name:
            return {'x': 150, 'y': 450}
        elif 'log' in source_name:
            return {'x': 300, 'y': 550}
        else:
            return {'x': 250, 'y': 475}

    def classify_data_type(self, source_name: str) -> str:
        """Classify data source type."""
        if 'audio' in source_name:
            return 'frequency_data'
        elif 'radio' in source_name:
            return 'frequency_data'
        elif 'combined' in source_name:
            return 'analysis_data'
        elif 'log' in source_name:
            return 'monitoring_data'
        elif 'report' in source_name:
            return 'report_data'
        else:
            return 'unknown'

    def discover_topology_nodes(self) -> List[Dict[str, Any]]:
        """Discover network topology nodes."""
        analysis = self.metadata.get('network_analysis', {}).get('frequency_analysis', {}).get('analysis_results', {})

        topology_nodes = []

        # Main topology node
        main_node = {
            'node_id': 'topology_main',
            'node_type': 'topology_core',
            'anomaly_score': analysis.get('anomaly_score', 0),
            'leakage_points': analysis.get('leakage_points_count', 0),
            'topology_layers': analysis.get('topology_layers', 0),
            'connectivity_score': analysis.get('connectivity_score', 0),
            'coordinates': {'x': 400, 'y': 50},
            'status': 'isolated' if analysis.get('connectivity_score', 0) == 0 else 'connected'
        }
        topology_nodes.append(main_node)

        # Layer nodes
        layers = analysis.get('topology_layers', 0)
        for i in range(layers):
            layer_node = {
                'node_id': f'topology_layer_{i+1}',
                'node_type': 'topology_layer',
                'layer_number': i + 1,
                'coordinates': {'x': 350 + (i * 50), 'y': 100 + (i * 30)},
                'status': 'active'
            }
            topology_nodes.append(layer_node)

        self.nodes['topology_nodes'] = topology_nodes
        return topology_nodes

    def build_node_relationships(self):
        """Build relationships between nodes."""
        # Device to frequency relationships
        for device in self.nodes['network_devices']:
            for freq_node in self.nodes['frequency_nodes']:
                if freq_node['risk_level'] in ['critical', 'high']:
                    self.node_relationships[device['node_id']].append({
                        'target': freq_node['node_id'],
                        'relationship': 'leakage_source',
                        'strength': freq_node['strength_db']
                    })

        # Communication to device relationships
        for comm_node in self.nodes['communication_nodes']:
            for device in self.nodes['network_devices']:
                if device['risk_level'] in ['high', 'critical']:
                    self.node_relationships[comm_node['node_id']].append({
                        'target': device['node_id'],
                        'relationship': 'communication_path',
                        'confidence': comm_node.get('confidence', 0)
                    })

        # Monitoring to data relationships
        for monitor_node in self.nodes['monitoring_nodes']:
            for data_node in self.nodes['data_nodes']:
                if 'log' in data_node['source_name']:
                    self.node_relationships[monitor_node['node_id']].append({
                        'target': data_node['node_id'],
                        'relationship': 'data_producer',
                        'status': monitor_node['status']
                    })

    def calculate_node_metrics(self):
        """Calculate comprehensive node metrics."""
        total_nodes = sum(len(nodes) for nodes in self.nodes.values())

        self.node_metrics = {
            'total_nodes': total_nodes,
            'node_types': {node_type: len(nodes) for node_type, nodes in self.nodes.items()},
            'risk_distribution': self.calculate_risk_distribution(),
            'connectivity_stats': self.calculate_connectivity_stats(),
            'performance_metrics': self.calculate_performance_metrics()
        }

    def calculate_risk_distribution(self) -> Dict[str, int]:
        """Calculate risk level distribution across all nodes."""
        risk_counts = defaultdict(int)

        for node_list in self.nodes.values():
            for node in node_list:
                risk_level = node.get('risk_level', 'unknown')
                risk_counts[risk_level] += 1

        return dict(risk_counts)

    def calculate_connectivity_stats(self) -> Dict[str, Any]:
        """Calculate connectivity statistics."""
        total_relationships = sum(len(rels) for rels in self.node_relationships.values())

        return {
            'total_relationships': total_relationships,
            'average_connections_per_node': total_relationships / max(1, sum(len(nodes) for nodes in self.nodes.values())),
            'most_connected_nodes': self.find_most_connected_nodes()
        }

    def find_most_connected_nodes(self) -> List[Dict[str, Any]]:
        """Find nodes with most connections."""
        node_connections = [(node_id, len(rels)) for node_id, rels in self.node_relationships.items()]
        node_connections.sort(key=lambda x: x[1], reverse=True)

        return [{'node_id': node_id, 'connections': count} for node_id, count in node_connections[:5]]

    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics for nodes."""
        device_nodes = self.nodes['network_devices']
        if not device_nodes:
            return {'average_response_time': 0, 'slowest_device': None, 'fastest_device': None}

        response_times = [node['response_time_ms'] for node in device_nodes]
        avg_response = sum(response_times) / len(response_times)

        slowest = max(device_nodes, key=lambda x: x['response_time_ms'])
        fastest = min(device_nodes, key=lambda x: x['response_time_ms'])

        return {
            'average_response_time': round(avg_response, 2),
            'slowest_device': slowest['hostname'],
            'fastest_device': fastest['hostname'],
            'response_time_range': f"{fastest['response_time_ms']:.1f}ms - {slowest['response_time_ms']:.1f}ms"
        }

    def discover_all_nodes(self) -> Dict[str, Any]:
        """Discover all types of nodes in the system."""
        print("🔍 NETWORK Node Discovery System")
        print("=" * 50)

        # Discover all node types
        self.discover_network_device_nodes()
        self.discover_frequency_nodes()
        self.discover_communication_nodes()
        self.discover_monitoring_nodes()
        self.discover_data_nodes()
        self.discover_topology_nodes()

        # Build relationships and calculate metrics
        self.build_node_relationships()
        self.calculate_node_metrics()

        return {
            'nodes': self.nodes,
            'relationships': dict(self.node_relationships),
            'metrics': self.node_metrics,
            'discovery_timestamp': datetime.now().isoformat()
        }

    def generate_node_report(self) -> Dict[str, Any]:
        """Generate comprehensive node discovery report."""
        discovery_data = self.discover_all_nodes()

        report = {
            'report_title': 'NETWORK Node Discovery Report',
            'generated_at': datetime.now().isoformat(),
            'discovery_summary': {
                'total_nodes_discovered': discovery_data['metrics']['total_nodes'],
                'node_types_found': discovery_data['metrics']['node_types'],
                'risk_distribution': discovery_data['metrics']['risk_distribution'],
                'total_relationships': discovery_data['metrics']['connectivity_stats']['total_relationships']
            },
            'node_categories': {
                'network_devices': self.format_node_list(discovery_data['nodes']['network_devices']),
                'frequency_nodes': self.format_node_list(discovery_data['nodes']['frequency_nodes']),
                'communication_nodes': self.format_node_list(discovery_data['nodes']['communication_nodes']),
                'monitoring_nodes': self.format_node_list(discovery_data['nodes']['monitoring_nodes']),
                'data_nodes': self.format_node_list(discovery_data['nodes']['data_nodes']),
                'topology_nodes': self.format_node_list(discovery_data['nodes']['topology_nodes'])
            },
            'critical_findings': self.identify_critical_findings(discovery_data),
            'performance_insights': discovery_data['metrics']['performance_metrics'],
            'connectivity_analysis': discovery_data['metrics']['connectivity_stats']
        }

        return report

    def format_node_list(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format node list for reporting."""
        return [{
            'id': node['node_id'],
            'type': node['node_type'],
            'risk_level': node.get('risk_level', 'unknown'),
            'coordinates': node.get('coordinates', {}),
            'key_attributes': {k: v for k, v in node.items()
                             if k not in ['node_id', 'node_type', 'coordinates', 'connections']}
        } for node in nodes]

    def identify_critical_findings(self, discovery_data: Dict[str, Any]) -> List[str]:
        """Identify critical findings from node discovery."""
        findings = []

        # Check for critical risk nodes
        critical_count = discovery_data['metrics']['risk_distribution'].get('critical', 0)
        if critical_count > 0:
            findings.append(f"🚨 {critical_count} critical-risk nodes detected")

        # Check device performance
        perf = discovery_data['metrics']['performance_metrics']
        if perf.get('average_response_time', 0) > 100:
            findings.append(f"⚠️ High average device response time: {perf['average_response_time']}ms")

        # Check connectivity
        connectivity = discovery_data['metrics']['connectivity_stats']
        if connectivity['total_relationships'] == 0:
            findings.append("🔗 No node relationships detected - isolated network")

        # Check communication security
        comm_nodes = discovery_data['nodes']['communication_nodes']
        compromised_count = sum(1 for node in comm_nodes if node.get('risk_level') == 'critical')
        if compromised_count > 0:
            findings.append(f"🔒 {compromised_count} compromised communication channels detected")

        return findings

    def save_node_report(self, output_file: str = 'node_discovery_report.json'):
        """Save the node discovery report."""
        report = self.generate_node_report()

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✅ Node discovery report saved to {output_file}")
        return report


def main():
    """Main function to run node discovery."""
    discoverer = NodeDiscoverySystem()

    print("🔍 Discovering all network nodes...")
    report = discoverer.save_node_report()

    print("\n📊 Node Discovery Summary:")
    print(f"  • Total nodes found: {report['discovery_summary']['total_nodes_discovered']}")
    print(f"  • Node types: {', '.join(f'{k}({v})' for k, v in report['discovery_summary']['node_types_found'].items())}")
    print(f"  • Risk distribution: {report['discovery_summary']['risk_distribution']}")
    print(f"  • Total relationships: {report['discovery_summary']['total_relationships']}")

    print("\n🎯 Critical Findings:")
    for finding in report['critical_findings']:
        print(f"  • {finding}")

    print("\n📈 Performance Insights:")
    perf = report['performance_insights']
    print(f"  • Average response time: {perf.get('average_response_time', 'N/A')}ms")
    print(f"  • Response time range: {perf.get('response_time_range', 'N/A')}")
    print(f"  • Fastest device: {perf.get('fastest_device', 'N/A')}")
    print(f"  • Slowest device: {perf.get('slowest_device', 'N/A')}")

    print(f"\n✅ Report saved as 'node_discovery_report.json'")


if __name__ == "__main__":
    main()
