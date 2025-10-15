#!/usr/bin/env python3
"""
Network Frequency Analysis Background Monitor
Non-interactive version for running multiple instances
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.base_monitor import BaseMonitor


class BackgroundMonitor(BaseMonitor):
    """Background monitoring for network frequency analysis."""

    def monitoring_loop(self):
        """Background monitoring loop."""
        print("Starting background monitoring...")
        self.monitoring = True

        while self.monitoring:
            try:
                # Run analysis
                results = self.analyzer.detect_layer_anomalies()

                # Calculate stats
                stats = self.calculate_stats(results)

                # Log stats
                import logging
                logging.info(f"Stats - Anomaly: {stats['anomaly_score']:.3f}, "
                           f"Leakage: {stats['leakage_points']}, "
                           f"Layers: {stats['topology_layers']}, "
                           f"Connectivity: {stats['connectivity_score']:.3f}")
                print(f"[Instance {self.instance_id}] Analysis complete - Anomaly: {stats['anomaly_score']:.3f}")

                # Wait before next analysis
                time.sleep(5)  # Update every 5 seconds

            except Exception as e:
                import logging
                logging.error(f"Monitoring error: {e}")
                time.sleep(10)  # Wait longer on error

    def run_main_loop(self):
        """Main loop for background monitoring."""
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            raise  # Re-raise to be caught by start_monitoring


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='NETWORK Background Frequency Monitor')
    parser.add_argument('--instance', type=int, default=1, help='Instance ID')
    parser.add_argument('--log-file', type=str, help='Log file path')

    args = parser.parse_args()

    monitor = BackgroundMonitor(instance_id=args.instance, log_file=args.log_file)
    monitor.start_monitoring()


if __name__ == "__main__":
    main()
