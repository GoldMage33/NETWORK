#!/usr/bin/env python3
"""
Network Frequency Analysis GUI Application
Real-time monitoring with live statistics and interactive controls
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.frequency_analyzer import FrequencyAnalyzer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from datetime import datetime
import queue


class NetworkMonitorGUI:
    """GUI application for real-time network frequency monitoring."""

    def __init__(self, root):
        self.root = root
        self.root.title("NETWORK Frequency Analysis Monitor")
        self.root.geometry("1200x800")

        # Initialize analyzer
        self.analyzer = FrequencyAnalyzer(resolution_hz=100.0)
        self.monitoring = False
        self.monitor_thread = None
        self.data_queue = queue.Queue()
        self.stats_history = []

        # Load initial data
        self.load_data()

        # Create GUI components
        self.create_widgets()

        # Initialize plots
        self.init_plots()

        # Start update loop
        self.update_gui()

    def load_data(self):
        """Load frequency data for analysis."""
        try:
            self.analyzer.load_audio_frequencies('data/sample_audio.csv')
            self.analyzer.load_radio_frequencies('data/sample_radio.csv')
            self.combined_data = self.analyzer.combine_frequency_data()
            self.status_var.set("Data loaded successfully")
        except Exception as e:
            self.status_var.set(f"Using sample data: {str(e)}")
            # Generate sample data if files don't exist
            self.analyzer.load_audio_frequencies('nonexistent.csv')
            self.analyzer.load_radio_frequencies('nonexistent.csv')
            self.combined_data = self.analyzer.combine_frequency_data()

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
        self.start_btn = ttk.Button(control_frame, text="Start Monitoring",
                                   command=self.start_monitoring)
        self.start_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = ttk.Button(control_frame, text="Stop Monitoring",
                                  command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5)

        self.analyze_btn = ttk.Button(control_frame, text="Run Analysis",
                                     command=self.run_analysis)
        self.analyze_btn.grid(row=0, column=2, padx=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.grid(row=0, column=3, padx=20, sticky=tk.W)

        # Stats panel
        stats_frame = ttk.LabelFrame(main_frame, text="Live Statistics", padding="5")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

        # Stats labels
        self.stats_vars = {}
        stats_labels = [
            "Total Samples", "Anomaly Score", "Leakage Points",
            "Obscured Layers", "Network Layers", "Scan Count"
        ]

        for i, label in enumerate(stats_labels):
            ttk.Label(stats_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, pady=2)
            var = tk.StringVar()
            var.set("0")
            ttk.Label(stats_frame, textvariable=var).grid(row=i, column=1, sticky=tk.W, pady=2)
            self.stats_vars[label] = var

        # Plot panel
        plot_frame = ttk.LabelFrame(main_frame, text="Live Frequency Spectrum", padding="5")
        plot_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Matplotlib figure
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Log panel
        log_frame = ttk.LabelFrame(main_frame, text="Activity Log", padding="5")
        log_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

    def init_plots(self):
        """Initialize matplotlib plots."""
        if hasattr(self, 'combined_data') and self.combined_data is not None:
            freq_data = self.combined_data.head(500)  # Show first 500 points

            # Frequency spectrum
            self.line1, = self.ax1.plot(freq_data['frequency'], freq_data['combined_amplitude'],
                                       'b-', linewidth=1, label='Combined')
            self.ax1.set_xlabel('Frequency (Hz)')
            self.ax1.set_ylabel('Amplitude')
            self.ax1.set_title('Live Frequency Spectrum')
            self.ax1.grid(True, alpha=0.3)
            self.ax1.legend()

            # Anomaly indicators
            self.anomaly_scatter = self.ax2.scatter([], [], c='red', s=50, alpha=0.7, label='Anomalies')
            self.ax2.set_xlabel('Frequency (Hz)')
            self.ax2.set_ylabel('Amplitude')
            self.ax2.set_title('Detected Anomalies')
            self.ax2.grid(True, alpha=0.3)
            self.ax2.legend()

            self.canvas.draw()

    def start_monitoring(self):
        """Start live monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Monitoring active...")

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.monitor_thread.start()

        self.log_message("Started live monitoring")

    def stop_monitoring(self):
        """Stop live monitoring."""
        self.monitoring = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Monitoring stopped")

        self.log_message("Stopped live monitoring")

    def monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        scan_count = 0

        while self.monitoring:
            try:
                # Simulate real-time data collection
                scan_count += 1

                # Run analysis
                results = self.analyzer.detect_layer_anomalies()

                # Update stats
                stats_update = {
                    'scan_count': scan_count,
                    'anomaly_score': results['anomaly_score'],
                    'leakage_points': len(results['leakage_points']),
                    'obscured_layers': len(results['obscured_layers']),
                    'network_layers': results['network_topology'].get('layer_count', 0),
                    'total_samples': len(self.combined_data) if self.combined_data is not None else 0
                }

                # Put results in queue for GUI update
                self.data_queue.put(('stats', stats_update))
                self.data_queue.put(('results', results))

                # Log activity
                self.data_queue.put(('log', f"Scan {scan_count}: Anomaly score {results['anomaly_score']:.3f}"))

                time.sleep(2)  # Update every 2 seconds

            except Exception as e:
                self.data_queue.put(('error', str(e)))
                time.sleep(5)  # Wait longer on error

    def run_analysis(self):
        """Run a single analysis."""
        try:
            self.status_var.set("Running analysis...")
            results = self.analyzer.detect_layer_anomalies()

            # Update stats
            stats_update = {
                'anomaly_score': results['anomaly_score'],
                'leakage_points': len(results['leakage_points']),
                'obscured_layers': len(results['obscured_layers']),
                'network_layers': results['network_topology'].get('layer_count', 0),
                'total_samples': len(self.combined_data) if self.combined_data is not None else 0
            }

            # Update GUI
            for key, value in stats_update.items():
                if key in self.stats_vars:
                    if isinstance(value, float):
                        self.stats_vars[key].set(f"{value:.3f}")
                    else:
                        self.stats_vars[key].set(str(value))

            # Update plots
            self.update_plots(results)

            self.status_var.set("Analysis complete")
            self.log_message(f"Analysis complete - Anomaly score: {results['anomaly_score']:.3f}")

        except Exception as e:
            self.status_var.set(f"Analysis error: {str(e)}")
            self.log_message(f"Analysis error: {str(e)}")

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
                    self.update_plots(data)

                elif msg_type == 'log':
                    self.log_message(data)

                elif msg_type == 'error':
                    self.log_message(f"Error: {data}")

        except queue.Empty:
            pass

        # Schedule next update
        self.root.after(100, self.update_gui)

    def update_plots(self, results):
        """Update matplotlib plots with new data."""
        try:
            if not hasattr(self, 'combined_data') or self.combined_data is None:
                return

            # Update frequency spectrum
            freq_data = self.combined_data.head(500)
            self.line1.set_data(freq_data['frequency'], freq_data['combined_amplitude'])

            # Update anomaly plot
            leakage_points = results.get('leakage_points', [])
            if leakage_points:
                freqs = [p['frequency'] for p in leakage_points[:20]]  # Show top 20
                amps = [p['strength'] for p in leakage_points[:20]]
                self.anomaly_scatter.set_offsets(np.column_stack([freqs, amps]))
            else:
                self.anomaly_scatter.set_offsets(np.empty((0, 2)))

            # Adjust axis limits
            if leakage_points:
                max_freq = max(p['frequency'] for p in leakage_points[:20]) if leakage_points else 10000
                max_amp = max(p['strength'] for p in leakage_points[:20]) if leakage_points else 1.0
                self.ax2.set_xlim(0, max_freq * 1.1)
                self.ax2.set_ylim(0, max_amp * 1.1)

            self.canvas.draw()

        except Exception as e:
            self.log_message(f"Plot update error: {str(e)}")

    def log_message(self, message):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)

    def on_closing(self):
        """Handle window closing."""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        self.root.destroy()


def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = NetworkMonitorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
