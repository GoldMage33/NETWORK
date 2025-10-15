#!/usr/bin/env python3
"""
Network Frequency Analysis Live Console Monitor
Real-time monitoring with live statistics in console
"""

import sys
import os
import time
import queue
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.base_monitor import BaseMonitor
import numpy as np
from datetime import datetime


class LiveConsoleMonitor(BaseMonitor):
    """Console-based live monitoring for network frequency analysis."""

    def __init__(self):
        super().__init__()
        self.data_queue = queue.Queue()

    def monitoring_loop(self):
        """Background monitoring loop."""
        print("\n🔄 Starting live monitoring...")
        self.monitoring = True

        while self.monitoring:
            try:
                # Run analysis
                results = self.analyzer.detect_layer_anomalies()

                # Calculate stats
                stats = self.calculate_stats(results)

                # Put in queue for display
                self.data_queue.put(stats)

                # Wait before next analysis
                time.sleep(2)  # Update every 2 seconds

            except Exception as e:
                print(f"✗ Monitoring error: {e}")
                time.sleep(5)  # Wait longer on error

    def run_main_loop(self):
        """Main loop for console monitoring."""
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
        """Display current statistics in console."""
        os.system('clear' if os.name == 'posix' else 'cls')  # Clear console

        print("🌐 NETWORK FREQUENCY ANALYSIS - LIVE MONITOR")
        print("=" * 50)
        print(f"⏰ Time: {stats['timestamp'].strftime('%H:%M:%S')}")
        print(f"⏱️  Runtime: {stats['runtime']:.2f}s")
        print(f"📊 Anomaly Score: {stats['anomaly_score']:.3f}")
        print(f"🔍 Leakage Points: {stats['leakage_points']}")
        print(f"🏗️  Topology Layers: {stats['topology_layers']}")
        print(f"🔗 Connectivity Score: {stats['connectivity_score']:.3f}")
        print()

        # Show trend if we have history
        if len(self.stats_history) > 1:
            print("📈 TREND ANALYSIS:")
            recent = self.stats_history[-10:]  # Last 10 readings
            avg_anomaly = np.mean([s['anomaly_score'] for s in recent])
            avg_leakage = np.mean([s['leakage_points'] for s in recent])

            print(f"   Avg Anomaly Score: {avg_anomaly:.3f}")
            print(f"   Avg Leakage Points: {avg_leakage:.1f}")

            # Trend indicators
            if len(recent) >= 2:
                anomaly_trend = recent[-1]['anomaly_score'] - recent[-2]['anomaly_score']
                leakage_trend = recent[-1]['leakage_points'] - recent[-2]['leakage_points']

                print(f"   Anomaly Trend: {'↗️ +' if anomaly_trend > 0 else '↘️ '}{anomaly_trend:.3f}")
                print(f"   Leakage Trend: {'↗️ +' if leakage_trend > 0 else '↘️ '}{leakage_trend}")
        print()

        # Show top leakage points if available
        if hasattr(self.analyzer, 'results') and 'leakage_points' in self.analyzer.results:
            leakage_points = self.analyzer.results['leakage_points']
            if leakage_points:
                print("🎯 TOP LEAKAGE POINTS:")
                for i, point in enumerate(leakage_points[:5]):
                    freq = point.get('frequency', 0)
                    strength = point.get('strength', 0)
                    print(f"   {i+1}. Frequency: {freq:,.1f} Hz, Strength: {strength:.3f}")
        print()
        print("💡 Press Ctrl+C to stop monitoring")


def main():
    """Main entry point."""
    print("🌐 NETWORK FREQUENCY ANALYSIS - LIVE CONSOLE MONITOR")
    print("=" * 55)

    monitor = LiveConsoleMonitor()
    monitor.start_monitoring()


if __name__ == "__main__":
    main()
