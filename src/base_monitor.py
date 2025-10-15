"""
Base Monitor class for NETWORK frequency analysis monitoring.
Provides common functionality for all monitor types.
"""

import sys
import os
import time
import threading
import logging
from abc import ABC, abstractmethod
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.frequency_analyzer import FrequencyAnalyzer


class BaseMonitor(ABC):
    """
    Base class for NETWORK frequency analysis monitors.
    Provides common monitoring functionality and structure.
    """

    def __init__(self, resolution_hz: float = 100.0, instance_id: int = 1, log_file: str = None):
        """
        Initialize the base monitor.

        Args:
            resolution_hz (float): Frequency resolution in Hz
            instance_id (int): Instance ID for multi-instance monitoring
            log_file (str, optional): Path to log file
        """
        self.resolution_hz = resolution_hz
        self.instance_id = instance_id
        self.analyzer = FrequencyAnalyzer(resolution_hz=resolution_hz)
        self.monitoring = False
        self.monitor_thread = None
        self.stats_history = []
        self.start_time = datetime.now()

        # Setup logging
        if log_file:
            logging.basicConfig(
                filename=log_file,
                level=logging.INFO,
                format=f'%(asctime)s [Instance {instance_id}] %(message)s'
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format=f'%(asctime)s [Instance {instance_id}] %(message)s'
            )

        # Load initial data
        self.load_data()

    def load_data(self):
        """Load frequency data for analysis."""
        try:
            self.analyzer.load_audio_frequencies('data/sample_audio.csv')
            self.analyzer.load_radio_frequencies('data/sample_radio.csv')
            logging.info("Data loaded successfully")
            print("✓ Data loaded successfully")
        except FileNotFoundError as e:
            error_msg = f"Data files not found: {e}. Please ensure data/sample_audio.csv and data/sample_radio.csv exist."
            logging.error(error_msg)
            print(f"✗ {error_msg}")
            sys.exit(1)
        except Exception as e:
            error_msg = f"Error loading data: {e}"
            logging.error(error_msg)
            print(f"✗ {error_msg}")
            sys.exit(1)

    def start_monitoring(self):
        """Start the monitoring process."""
        logging.info(f"Initializing {self.__class__.__name__} (Instance {self.instance_id})")

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitor_thread.start()

        logging.info("Monitoring started.")
        print(f"✅ {self.__class__.__name__} started.")

        try:
            self.run_main_loop()
        except KeyboardInterrupt:
            logging.info("Monitoring stopped by user.")
            print(f"\n🛑 {self.__class__.__name__} stopped by user.")
            self.stop_monitoring()

    def stop_monitoring(self):
        """Stop the monitoring."""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        logging.info("Monitoring stopped.")
        print("✅ Monitoring stopped.")

    def calculate_stats(self, results):
        """
        Calculate common statistics from analysis results.

        Args:
            results (dict): Analysis results from detect_layer_anomalies()

        Returns:
            dict: Statistics dictionary
        """
        try:
            current_time = datetime.now()
            runtime = (current_time - self.start_time).total_seconds()

            # Validate results structure
            if not isinstance(results, dict):
                logging.warning(f"Invalid results type: {type(results)}, expected dict")
                results = {}

            stats = {
                'timestamp': current_time,
                'runtime': max(0, runtime),  # Ensure non-negative runtime
                'anomaly_score': float(results.get('anomaly_score', 0)),
                'leakage_points': len(results.get('leakage_points', [])),
                'topology_layers': results.get('network_topology', {}).get('layer_count', 0),
                'connectivity_score': results.get('network_topology', {}).get('connectivity', 0)
            }

            # Validate ranges
            stats['anomaly_score'] = max(0, min(1, stats['anomaly_score']))  # Clamp to [0,1]
            stats['leakage_points'] = max(0, stats['leakage_points'])  # Ensure non-negative
            stats['topology_layers'] = max(0, stats['topology_layers'])  # Ensure non-negative
            stats['connectivity_score'] = max(0, min(1, stats['connectivity_score']))  # Clamp to [0,1]

            # Add to history
            self.stats_history.append(stats)
            if len(self.stats_history) > 100:  # Keep last 100 readings
                self.stats_history.pop(0)

            return stats

        except Exception as e:
            logging.error(f"Error calculating stats: {e}")
            # Return safe default stats
            return {
                'timestamp': datetime.now(),
                'runtime': 0,
                'anomaly_score': 0.0,
                'leakage_points': 0,
                'topology_layers': 0,
                'connectivity_score': 0.0
            }

    @abstractmethod
    def monitoring_loop(self):
        """Abstract monitoring loop - must be implemented by subclasses."""
        pass

    @abstractmethod
    def run_main_loop(self):
        """Abstract main loop - must be implemented by subclasses."""
        pass
