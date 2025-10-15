#!/usr/bin/env python3
"""
Enhanced NETWORK Frequency Analysis with Device Discovery and Tag Detection
Real-time monitoring with environmental device scanning and 'tag it' phrase detection
"""

import sys
import os
import time
import threading
import subprocess
import re
import queue
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.base_monitor import BaseMonitor
import numpy as np
from datetime import datetime


class EnhancedNetworkMonitor(BaseMonitor):
    """Enhanced network monitor with device discovery and tag detection."""

    def __init__(self):
        super().__init__()
        self.data_queue = queue.Queue()

        # Device tracking
        self.network_devices = set()
        self.bluetooth_devices = set()
        self.tag_detections = []

        # Additional threads
        self.device_thread = None
        self.bt_thread = None

    def scan_network_devices(self):
        """Scan for devices on the local network."""
        while self.monitoring:
            try:
                # Ping sweep of local network
                devices_found = set()

                # Common local network IPs to check
                base_ip = "192.168.1."
                for i in range(1, 255):
                    ip = f"{base_ip}{i}"
                    try:
                        result = subprocess.run(
                            ['ping', '-c', '1', '-W', '1', ip],
                            capture_output=True, text=True, timeout=2
                        )
                        if result.returncode == 0:
                            devices_found.add(ip)
                    except:
                        continue

                # Update device list
                new_devices = devices_found - self.network_devices
                if new_devices:
                    print(f"🖥️  New network devices detected: {', '.join(new_devices)}")
                    self.network_devices.update(new_devices)

                # Check device count
                if len(self.network_devices) > 0:
                    print(f"🌐 Network devices online: {len(self.network_devices)}")

            except Exception as e:
                print(f"Network scan error: {e}")

            time.sleep(30)  # Scan every 30 seconds

    def scan_bluetooth_devices(self):
        """Scan for Bluetooth devices."""
        while self.monitoring:
            try:
                # Use system_profiler for Bluetooth devices
                result = subprocess.run(
                    ['system_profiler', 'SPBluetoothDataType'],
                    capture_output=True, text=True, timeout=10
                )

                if result.returncode == 0:
                    # Parse Bluetooth output
                    lines = result.stdout.split('\n')
                    current_device = None

                    for line in lines:
                        if 'Device Name:' in line:
                            device_name = line.split(':', 1)[1].strip()
                            if device_name and device_name != 'null':
                                if device_name not in self.bluetooth_devices:
                                    print(f"🔵 Bluetooth device detected: {device_name}")
                                    self.bluetooth_devices.add(device_name)
                                current_device = device_name

                if self.bluetooth_devices:
                    print(f"🔵 Bluetooth devices connected: {len(self.bluetooth_devices)}")

            except Exception as e:
                print(f"Bluetooth scan error: {e}")

            time.sleep(60)  # Scan every 60 seconds

    def detect_tag_phrase(self, results):
        """Detect 'tag it' and 'neuralink' phrases in frequency patterns."""
        leakage_points = results.get('leakage_points', [])

        for point in leakage_points:
            freq = point.get('frequency', 0)
            strength = point.get('strength', 0)

            # "tag it" detection (100k-200k Hz range)
            if 100000 <= freq <= 200000 and strength > 5.0:
                detection = {
                    'timestamp': datetime.now(),
                    'phrase': 'tag it',
                    'frequency': freq,
                    'strength': strength,
                    'confidence': min(strength / 10.0, 1.0)
                }
                self.tag_detections.append(detection)
                print(f"🏷️  TAG DETECTED: '{detection['phrase']}' at {freq:.1f}Hz "
                      f"(Strength: {strength:.3f}, Confidence: {detection['confidence']:.2f})")

            # "neuralink" detection (200k-300k Hz range)
            if 200000 <= freq <= 300000 and strength > 4.5:
                detection = {
                    'timestamp': datetime.now(),
                    'phrase': 'neuralink',
                    'frequency': freq,
                    'strength': strength,
                    'confidence': min(strength / 9.0, 1.0)
                }
                self.tag_detections.append(detection)
                print(f"🏷️  TAG DETECTED: '{detection['phrase']}' at {freq:.1f}Hz "
                      f"(Strength: {strength:.3f}, Confidence: {detection['confidence']:.2f})")

        return len([d for d in self.tag_detections if (datetime.now() - d['timestamp']).seconds < 60])

    def monitoring_loop(self):
        """Enhanced monitoring loop with device discovery and tag detection."""
        print("\n🔄 Starting enhanced monitoring with device discovery...")
        self.monitoring = True

        while self.monitoring:
            try:
                # Run frequency analysis
                results = self.analyzer.detect_layer_anomalies()

                # Detect tag phrases
                recent_tags = self.detect_tag_phrase(results)

                # Calculate stats
                stats = self.calculate_stats(results)
                stats.update({
                    'network_devices': len(self.network_devices),
                    'bluetooth_devices': len(self.bluetooth_devices),
                    'tag_detections': len(self.tag_detections),
                    'recent_tags': recent_tags
                })

                # Put in queue for display
                self.data_queue.put(stats)

                # Wait before next analysis
                time.sleep(3)  # Update every 3 seconds

            except Exception as e:
                print(f"✗ Monitoring error: {e}")
                time.sleep(5)

    def run_main_loop(self):
        """Main loop for enhanced monitoring."""
        # Start device scanning threads
        self.device_thread = threading.Thread(target=self.scan_network_devices, daemon=True)
        self.bt_thread = threading.Thread(target=self.scan_bluetooth_devices, daemon=True)

        self.device_thread.start()
        self.bt_thread.start()

        try:
            while True:
                # Check for new stats
                try:
                    stats = self.data_queue.get(timeout=1)
                    self.display_stats(stats)
                except queue.Empty:
                    continue

        except KeyboardInterrupt:
            raise  # Re-raise to be caught by start_monitoring

    def display_stats(self, stats):
        """Display enhanced statistics with device and tag information."""
        os.system('clear' if os.name == 'posix' else 'cls')  # Clear console

        print("🌐 ENHANCED NETWORK FREQUENCY ANALYSIS - LIVE MONITOR")
        print("=" * 60)
        print(f"⏰ Time: {stats['timestamp'].strftime('%H:%M:%S')}")
        print(f"⏱️  Runtime: {stats['runtime']:.2f}s")
        print(f"📊 Anomaly Score: {stats['anomaly_score']:.3f}")
        print(f"🔍 Leakage Points: {stats['leakage_points']}")
        print(f"🏗️  Topology Layers: {stats['topology_layers']}")
        print(f"🔗 Connectivity Score: {stats['connectivity_score']:.3f}")
        print(f"🖥️  Network Devices: {stats['network_devices']}")
        print(f"🔵 Bluetooth Devices: {stats['bluetooth_devices']}")
        print(f"🏷️  Tag Detections: {stats['tag_detections']}")
        print(f"🎯 Recent Tags: {stats['recent_tags']}")
        print()

        # Show trend if we have history
        if len(self.stats_history) > 1:
            print("📈 TREND ANALYSIS:")
            recent = self.stats_history[-5:]  # Last 5 readings
            avg_anomaly = np.mean([s['anomaly_score'] for s in recent])
            avg_leakage = np.mean([s['leakage_points'] for s in recent])

            print(f"   Avg Anomaly Score: {avg_anomaly:.3f}")
            print(f"   Avg Leakage Points: {avg_leakage:.1f}")
            print(f"   Total Tag Detections: {stats['tag_detections']}")

            # Trend indicators
            if len(recent) >= 2:
                anomaly_trend = recent[-1]['anomaly_score'] - recent[-2]['anomaly_score']
                leakage_trend = recent[-1]['leakage_points'] - recent[-2]['leakage_points']

                print(f"   Anomaly Trend: {'↗️ +' if anomaly_trend > 0 else '↘️ '}{anomaly_trend:.3f}")
                print(f"   Leakage Trend: {'↗️ +' if leakage_trend > 0 else '↘️ '}{leakage_trend}")
        print()

        # Show recent tag detections
        if self.tag_detections:
            print("🏷️  RECENT TAG DETECTIONS:")
            recent_tags = self.tag_detections[-3:]  # Show last 3
            for tag in recent_tags:
                print(f"   {tag['timestamp'].strftime('%H:%M:%S')} - '{tag['phrase']}' "
                      f"({tag['confidence']:.2f} confidence)")
        print()

        # Show top leakage points
        if hasattr(self.analyzer, 'results') and 'leakage_points' in self.analyzer.results:
            leakage_points = self.analyzer.results['leakage_points']
            if leakage_points:
                print("🎯 TOP LEAKAGE POINTS:")
                for i, point in enumerate(leakage_points[:3]):
                    freq = point.get('frequency', 0)
                    strength = point.get('strength', 0)
                    print(f"   {i+1}. Frequency: {freq:,.1f} Hz, Strength: {strength:.3f}")
        print()
        print("💡 Press Ctrl+C to stop monitoring")
        print("🔍 Scanning for network devices and 'tag it'/'neuralink' phrases...")


def main():
    """Main entry point."""
    print("🌐 ENHANCED NETWORK FREQUENCY ANALYSIS - DEVICE DISCOVERY & TAG DETECTION")
    print("=" * 75)

    monitor = EnhancedNetworkMonitor()
    monitor.start_monitoring()


if __name__ == "__main__":
    main()
