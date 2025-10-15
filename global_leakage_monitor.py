#!/usr/bin/env python3
"""
NETWORK Global Leakage Monitor
Real-time monitoring of global cross-reference data leakage
"""

import json
import time
from datetime import datetime
import os


class GlobalLeakageMonitor:
    """Monitor global data leakage in real-time."""

    def __init__(self):
        self.report_file = 'global_leakage_report.json'
        self.last_check = None

    def get_current_status(self):
        """Get current global leakage status."""
        try:
            with open(self.report_file, 'r') as f:
                report = json.load(f)

            return {
                'timestamp': report.get('generated_at', 'Unknown'),
                'leakage_probability': report.get('overall_risk_assessment', {}).get('data_leakage_probability', 'Unknown'),
                'exposure_level': report.get('overall_risk_assessment', {}).get('global_exposure_level', 'Unknown'),
                'critical_devices': len(report.get('global_cross_reference', {}).get('device_exposure_analysis', {}).get('exposed_devices', [])),
                'leakage_points': report.get('local_network_summary', {}).get('leakage_points', 0),
                'communication_security': report.get('global_cross_reference', {}).get('tag_leakage_analysis', {}).get('communication_security', 'Unknown')
            }
        except Exception as e:
            return {'error': str(e)}

    def display_status(self):
        """Display current global leakage status."""
        status = self.get_current_status()

        if 'error' in status:
            print(f"❌ Error reading report: {status['error']}")
            return

        print("🌐 NETWORK Global Leakage Status")
        print("=" * 40)
        print(f"📅 Last Updated: {status['timestamp'][:19].replace('T', ' ')}")
        print(f"🎯 Data Leakage Probability: {status['leakage_probability']}")
        print(f"⚠️  Global Exposure Level: {status['exposure_level'].upper()}")
        print(f"🖥️  Critical Devices: {status['critical_devices']}")
        print(f"📡 Leakage Points: {status['leakage_points']:,}")
        print(f"🔒 Communication Security: {status['communication_security'].upper()}")

        # Risk indicators
        if status['exposure_level'] == 'high':
            print("\n🚨 HIGH RISK - Immediate action required!")
        elif status['exposure_level'] == 'medium':
            print("\n⚠️  MEDIUM RISK - Monitor closely")
        else:
            print("\n✅ LOW RISK - Continue monitoring")

    def monitor_loop(self, interval=300):  # 5 minutes default
        """Continuous monitoring loop."""
        print("🔄 Starting global leakage monitoring...")
        print(f"📊 Update interval: {interval} seconds")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                self.display_status()
                print(f"\n⏰ Next update in {interval} seconds...")
                time.sleep(interval)
                os.system('clear' if os.name == 'posix' else 'cls')

        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")

    def quick_check(self):
        """Quick status check."""
        status = self.get_current_status()

        if 'error' in status:
            return f"❌ Error: {status['error']}"

        exposure = status['exposure_level'].upper()
        probability = status['leakage_probability']

        if exposure == 'HIGH':
            return f"🚨 CRITICAL: {probability} leakage probability - {status['critical_devices']} exposed devices"
        elif exposure == 'MEDIUM':
            return f"⚠️  WARNING: {probability} leakage probability - monitor required"
        else:
            return f"✅ SECURE: {probability} leakage probability - status normal"


def main():
    """Main function."""
    import sys

    monitor = GlobalLeakageMonitor()

    if len(sys.argv) > 1:
        if sys.argv[1] == 'monitor':
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 300
            monitor.monitor_loop(interval)
        elif sys.argv[1] == 'quick':
            print(monitor.quick_check())
        else:
            print("Usage: python3 global_leakage_monitor.py [monitor|quick] [interval]")
    else:
        monitor.display_status()


if __name__ == "__main__":
    main()
