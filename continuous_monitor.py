#!/usr/bin/env python3
"""
NETWORK Continuous Monitoring System with Live GUI
Runs frequency analysis continuously with live GUI and automatic CSV saving every minute
"""

import os
import signal
import sys
import time
import json
import pandas as pd
from datetime import datetime
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import queue
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.frequency_analyzer import FrequencyAnalyzer
from src.data_loader import DataLoader


class ContinuousMonitor:
    """Continuous monitoring system with live GUI and automatic CSV saving."""

    def __init__(self):
        self.is_running = False
        self.monitor_thread = None
        self.csv_save_thread = None
        self.last_csv_save = time.time()
        self.csv_save_interval = 60  # Save CSV every 60 seconds (1 minute)

        # Initialize components
        self.analyzer = FrequencyAnalyzer(resolution_hz=100.0)
        self.data_loader = DataLoader()

        # Create results directory
        self.results_base_dir = "results"
        self.today_dir = datetime.now().strftime("%Y%m%d")
        self.results_dir = os.path.join(self.results_base_dir, self.today_dir)
        os.makedirs(self.results_dir, exist_ok=True)

        # GUI components
        self.root = None
        self.gui_queue = queue.Queue()
        self.stats_vars = {}
        self.results_text = None
        self.leakage_text = None
        self.log_text = None
        self.status_var = None

        print("🌐 NETWORK Continuous Monitoring System Initialized")
        print(f"📁 Results Directory: {self.results_dir}")
        print("⏱️  Continuous monitoring with 1-minute CSV saves")

    def start_monitoring(self):
        """Start the continuous monitoring process."""
        if self.is_running:
            print("Monitoring already running!")
            return

        self.is_running = True
        self.last_csv_save = time.time()

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitor_thread.start()

        # Start CSV saving thread
        self.csv_save_thread = threading.Thread(target=self.csv_save_loop, daemon=True)
        self.csv_save_thread.start()

        print("✅ Continuous monitoring started")
        print("Press Ctrl+C to stop monitoring")

    def stop_monitoring(self):
        """Stop the continuous monitoring process."""
        self.is_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        if self.csv_save_thread and self.csv_save_thread.is_alive():
            self.csv_save_thread.join(timeout=2.0)
        print("🛑 Continuous monitoring stopped")

    def monitoring_loop(self):
        """Main monitoring loop that runs analysis continuously."""
        cycle_count = 0

        while self.is_running:
            try:
                cycle_count += 1
                cycle_start_time = datetime.now()

                # Perform analysis
                results = self.perform_analysis()

                # Update GUI if available
                if self.root and self.gui_queue:
                    self.update_gui_with_results(results, cycle_count)

                # Small delay to prevent overwhelming the system
                time.sleep(0.1)  # 100ms delay between cycles

            except Exception as e:
                print(f"❌ Error in monitoring cycle: {e}")
                time.sleep(1)  # Wait longer on error

    def perform_analysis(self):
        """Perform a single analysis cycle."""
        try:
            # Load hardware data (continuous collection)
            hw_data = self.data_loader.load_hardware_data(1.0, use_hardware=True)

            if not hw_data:
                return None

            # Set analyzer data
            if 'audio' in hw_data:
                self.analyzer.audio_data = hw_data['audio']
                audio_count = len(hw_data['audio'])
            else:
                audio_count = 0

            if 'radio' in hw_data:
                self.analyzer.radio_data = hw_data['radio']
                radio_count = len(hw_data['radio'])
            else:
                radio_count = 0

            # Perform analysis
            combined_data = self.analyzer.combine_frequency_data()
            results = self.analyzer.detect_layer_anomalies()

            # Add metadata
            results['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'audio_samples': audio_count,
                'radio_samples': radio_count,
                'combined_samples': len(combined_data),
                'frequency_resolution_hz': 100.0
            }

            return results

        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return None

    def save_cycle_results(self, results, cycle_count, timestamp):
        """Save results from a single analysis cycle."""
        if not results:
            return

        try:
            # Create timestamped filename
            timestamp_str = timestamp.strftime("%H%M%S")
            base_filename = f"analysis_cycle_{cycle_count:04d}_{timestamp_str}"

            # Save JSON results
            json_filename = f"{base_filename}.json"
            json_path = os.path.join(self.results_dir, json_filename)

            # Convert numpy types for JSON serialization
            serializable_results = self.convert_to_json_serializable(results)

            with open(json_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)

            # Save CSV data if available
            if hasattr(self.analyzer, 'combined_data') and self.analyzer.combined_data is not None:
                csv_filename = f"{base_filename}_data.csv"
                csv_path = os.path.join(self.results_dir, csv_filename)
                self.analyzer.combined_data.to_csv(csv_path, index=False)

            # Save summary text file
            summary_filename = f"{base_filename}_summary.txt"
            summary_path = os.path.join(self.results_dir, summary_filename)

            with open(summary_path, 'w') as f:
                f.write(f"NETWORK Analysis Cycle #{cycle_count}\n")
                f.write(f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")

                f.write("ANALYSIS RESULTS:\n")
                anomaly_score = results.get('anomaly_score', 'N/A')
                if isinstance(anomaly_score, (int, float)):
                    f.write(f"Anomaly Score: {anomaly_score:.3f}\n")
                else:
                    f.write(f"Anomaly Score: {anomaly_score}\n")

                f.write(f"Leakage Points: {len(results.get('leakage_points', []))}\n")
                f.write(f"Obscured Layers: {len(results.get('obscured_layers', []))}\n")

                topology = results.get('network_topology', {})
                layer_count = topology.get('layer_count', 'N/A')
                f.write(f"Network Layers: {layer_count}\n")

                connectivity_score = topology.get('connectivity_score', 'N/A')
                if isinstance(connectivity_score, (int, float)):
                    f.write(f"Connectivity Score: {connectivity_score:.3f}\n\n")
                else:
                    f.write(f"Connectivity Score: {connectivity_score}\n\n")

                f.write("TOP LEAKAGE POINTS:\n")
                leakage_points = results.get('leakage_points', [])[:5]
                for i, point in enumerate(leakage_points, 1):
                    f.write(f"{i}. {point.get('frequency', 0):.1f} Hz - Strength: {point.get('strength', 0):.3f}\n")

                f.write("\nMETADATA:\n")
                metadata = results.get('metadata', {})
                f.write(f"Audio Samples: {metadata.get('audio_samples', 0)}\n")
                f.write(f"Radio Samples: {metadata.get('radio_samples', 0)}\n")
                f.write(f"Analysis Timeframe: {metadata.get('analysis_timeframe_seconds', 0)}s\n")
                f.write(f"Frequency Resolution: {metadata.get('frequency_resolution_hz', 0)} Hz\n")

            print(f"  💾 Files saved: {json_filename}, {csv_filename}, {summary_filename}")

        except Exception as e:
            print(f"  ❌ Error saving results: {e}")

    def csv_save_loop(self):
        """Background thread that saves CSV files every minute."""
        while self.is_running:
            try:
                current_time = time.time()
                if current_time - self.last_csv_save >= self.csv_save_interval:
                    self.save_minute_csv()
                    self.last_csv_save = current_time

                time.sleep(1)  # Check every second

            except Exception as e:
                print(f"❌ Error in CSV save loop: {e}")
                time.sleep(5)
        """Convert numpy/pandas types to JSON-serializable Python types."""
        if isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            try:
                return [self.convert_to_json_serializable(item) for item in obj]
            except TypeError:
                return str(obj)
        else:
            return obj

    def save_minute_csv(self):
        """Save current combined data to CSV file with timestamp."""
        try:
            if hasattr(self.analyzer, 'combined_data') and self.analyzer.combined_data is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = f"network_analysis_{timestamp}.csv"
                csv_path = os.path.join(self.results_dir, csv_filename)

                self.analyzer.combined_data.to_csv(csv_path, index=False)
                print(f"💾 CSV saved: {csv_filename}")

        except Exception as e:
            print(f"❌ Error saving CSV: {e}")

    def update_gui_with_results(self, results, cycle_count):
        """Update GUI with analysis results."""
        if not results or not self.gui_queue:
            return

        try:
            # Prepare stats update
            stats_update = {
                'scan_count': cycle_count,
                'anomaly_score': results['anomaly_score'],
                'leakage_points': len(results['leakage_points']),
                'obscured_layers': len(results['obscured_layers']),
                'network_layers': results['network_topology'].get('layer_count', 0),
                'connectivity_score': results['network_topology'].get('connectivity_score', 0.0),
                'total_samples': results['metadata'].get('combined_samples', 0)
            }

            # Send updates to GUI
            self.gui_queue.put(('stats', stats_update))
            self.gui_queue.put(('results', results))

        except Exception as e:
            print(f"❌ Error updating GUI: {e}")

    def create_gui(self):
        """Create and return the GUI application."""
        return NetworkMonitorGUI(self)

    def get_status(self):
        """Get current monitoring status."""
        return {
            'is_running': self.is_running,
            'results_directory': self.results_dir,
            'csv_save_interval_seconds': self.csv_save_interval,
            'monitor_thread_alive': self.monitor_thread.is_alive() if self.monitor_thread else False,
            'csv_thread_alive': self.csv_save_thread.is_alive() if self.csv_save_thread else False
        }


def main():
    """Main function to run continuous monitoring with GUI."""
    import argparse

    parser = argparse.ArgumentParser(description='NETWORK Continuous Monitoring System with Live GUI')
    parser.add_argument('--no-gui', action='store_true',
                       help='Run without GUI (console only)')

    args = argparse.parse_args()

    # Create monitor
    monitor = ContinuousMonitor()

    if args.no_gui:
        # Console-only mode
        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            print("
🛑 Received signal, shutting down continuous monitoring...")
            monitor.stop_monitoring()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Start monitoring
            monitor.start_monitoring()

            # Keep main thread alive
            while monitor.is_running:
                time.sleep(1)

        except KeyboardInterrupt:
            print("
🛑 Shutting down continuous monitoring...")
            monitor.stop_monitoring()
            print("✅ Monitoring system stopped")
    else:
        # GUI mode
        gui = NetworkMonitorGUI(monitor)
        gui.run()


if __name__ == "__main__":
    main()


class NetworkMonitorGUI:
    """GUI application for real-time network frequency monitoring with continuous monitoring."""

    def __init__(self, monitor):
        self.monitor = monitor
        self.root = tk.Tk()
        self.root.title("NETWORK Continuous Frequency Analysis Monitor")
        self.root.geometry("1200x800")

        # Initialize GUI components
        self.create_widgets()
        self.monitor.root = self.root
        self.monitor.gui_queue = queue.Queue()
        self.monitor.results_text = self.results_text
        self.monitor.leakage_text = self.leakage_text
        self.monitor.log_text = self.log_text
        self.monitor.status_var = self.status_var
        self.monitor.stats_vars = self.stats_vars

        # Start GUI update loop
        self.update_gui()

    def create_widgets(self):
        """Create GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # Buttons
        self.start_btn = ttk.Button(control_frame, text="Start Continuous Monitoring",
                                   command=self.start_monitoring)
        self.start_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = ttk.Button(control_frame, text="Stop Monitoring",
                                  command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5)

        self.analyze_btn = ttk.Button(control_frame, text="Run Single Analysis",
                                     command=self.run_analysis)
        self.analyze_btn.grid(row=0, column=2, padx=5)

        self.export_btn = ttk.Button(control_frame, text="Export Current Data",
                                    command=self.export_data)
        self.export_btn.grid(row=0, column=3, padx=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Continuous monitoring with 1-minute CSV saves")
        status_label = ttk.Label(control_frame, textvariable=self.status_var, font=("Arial", 10, "bold"))
        status_label.grid(row=0, column=4, padx=20, sticky=tk.W)

        # Stats panel
        stats_frame = ttk.LabelFrame(main_frame, text="Live Statistics", padding="5")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

        # Stats labels
        self.stats_vars = {}
        stats_labels = [
            ("Total Samples", "total_samples"),
            ("Anomaly Score", "anomaly_score"),
            ("Leakage Points", "leakage_points"),
            ("Obscured Layers", "obscured_layers"),
            ("Network Layers", "network_layers"),
            ("Scan Count", "scan_count"),
            ("Connectivity Score", "connectivity_score")
        ]

        for i, (label, key) in enumerate(stats_labels):
            ttk.Label(stats_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            var = tk.StringVar()
            var.set("0")
            ttk.Label(stats_frame, textvariable=var, font=("Courier", 10)).grid(row=i, column=1, sticky=tk.W, pady=2)
            self.stats_vars[key] = var

        # Results panel
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="5")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, wrap=tk.WORD,
                                                     font=("Courier", 9))
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Leakage points display
        leakage_frame = ttk.LabelFrame(results_frame, text="Top Leakage Points")
        leakage_frame.pack(fill=tk.X, pady=(5, 0))

        self.leakage_text = tk.Text(leakage_frame, height=6, wrap=tk.WORD, font=("Courier", 8))
        leakage_scrollbar = ttk.Scrollbar(leakage_frame, orient=tk.VERTICAL, command=self.leakage_text.yview)
        self.leakage_text.configure(yscrollcommand=leakage_scrollbar.set)

        self.leakage_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        leakage_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Log panel
        log_frame = ttk.LabelFrame(main_frame, text="Activity Log", padding="5")
        log_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, wrap=tk.WORD, font=("Courier", 8))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.monitor.is_running:
            return

        self.monitor.start_monitoring()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Continuous monitoring active - CSV saves every minute")

        self.log_message("Started continuous monitoring with live GUI")

    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitor.stop_monitoring()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Monitoring stopped")

        self.log_message("Stopped continuous monitoring")

    def run_analysis(self):
        """Run a single analysis."""
        try:
            self.status_var.set("Running single analysis...")
            self.analyze_btn.config(state=tk.DISABLED)

            results = self.monitor.perform_analysis()

            if results:
                self.update_results_display(results)
                self.status_var.set("Single analysis complete")
                self.log_message(f"Single analysis complete - Anomaly score: {results['anomaly_score']:.3f}")
            else:
                self.status_var.set("Analysis failed")
                self.log_message("Single analysis failed")

        except Exception as e:
            self.status_var.set(f"Analysis error: {str(e)}")
            self.log_message(f"Analysis error: {str(e)}")
        finally:
            self.analyze_btn.config(state=tk.NORMAL)

    def update_results_display(self, results):
        """Update the results text area with analysis results."""
        self.results_text.delete(1.0, tk.END)

        # Summary
        summary = f"""NETWORK FREQUENCY ANALYSIS RESULTS
{'='*40}

SUMMARY:
• Resolution: {self.monitor.analyzer.resolution_hz:.1f} Hz
• Total Samples: {len(self.monitor.analyzer.combined_data) if self.monitor.analyzer.combined_data is not None else 0}
• Anomaly Score: {results['anomaly_score']:.3f}
• Leakage Points: {len(results['leakage_points'])}
• Obscured Layers: {len(results['obscured_layers'])}

NETWORK TOPOLOGY:
• Estimated Layers: {results['network_topology'].get('layer_count', 'Unknown')}
• Connectivity Score: {results['network_topology'].get('connectivity_score', 0.0):.3f}

"""
        self.results_text.insert(tk.END, summary)

        # Leakage points
        self.leakage_text.delete(1.0, tk.END)
        if results['leakage_points']:
            leakage_header = "TOP LEAKAGE POINTS:\n"
            leakage_header += "-" * 30 + "\n"
            self.leakage_text.insert(tk.END, leakage_header)

            for i, point in enumerate(results['leakage_points'][:10]):
                freq = point.get('frequency', 0)
                strength = point.get('strength', 0)
                line = f"{i+1:2d}. {freq:>12,.1f} Hz | Strength: {strength:>8.3f}\n"
                self.leakage_text.insert(tk.END, line)
        else:
            self.leakage_text.insert(tk.END, "No leakage points detected\n")

    def export_data(self):
        """Export current data to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"network_analysis_{timestamp}.csv"

            if hasattr(self.monitor.analyzer, 'combined_data') and self.monitor.analyzer.combined_data is not None:
                csv_path = os.path.join(self.monitor.results_dir, filename)
                self.monitor.analyzer.combined_data.to_csv(csv_path, index=False)
                self.log_message(f"Data exported to {filename}")
                messagebox.showinfo("Export Complete", f"Data exported to {filename}")
            else:
                messagebox.showwarning("Export Failed", "No data available to export")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")

    def update_gui(self):
        """Update GUI with data from queue."""
        try:
            while True:
                msg_type, data = self.data_queue.get_nowait()

                if msg_type == 'stats':
                    for key, value in data.items():
                        if key in self.stats_vars:
                            if isinstance(value, float):
                                self.stats_vars[key].set(f"{value:.3f}")
                            else:
                                self.stats_vars[key].set(str(value))

                elif msg_type == 'results':
                    self.update_results_display(data)

                elif msg_type == 'log':
                    self.log_message(data)

                elif msg_type == 'error':
                    self.log_message(f"Error: {data}")

        except queue.Empty:
            pass

        # Schedule next update
        self.root.after(100, self.update_gui)

    def log_message(self, message):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)

    def run(self):
        """Run the GUI application."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        """Handle window closing."""
        self.monitor.stop_monitoring()
        self.root.destroy()

    @property
    def data_queue(self):
        """Get the GUI queue from the monitor."""
        return self.monitor.gui_queue
