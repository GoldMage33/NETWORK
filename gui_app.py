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
from src.data_loader import DataLoader
import numpy as np
import pandas as pd
from datetime import datetime
import queue


class NetworkMonitorGUI:
    """GUI application for real-time network frequency monitoring."""

    def __init__(self, root):
        self.root = root
        self.root.title("NETWORK Frequency Analysis Monitor")
        self.root.geometry("1000x700")

        # Initialize analyzer and data loader
        self.analyzer = FrequencyAnalyzer(resolution_hz=100.0)
        self.data_loader = DataLoader()
        self.monitoring = False
        self.monitor_thread = None
        self.data_queue = queue.Queue()
        self.stats_history = []
        self.use_hardware = False

        # Load initial data
        self.load_data()

        # Create GUI components
        self.create_widgets()

        # Start update loop
        self.update_gui()

    def load_data(self):
        """Load frequency data for analysis using hardware input."""
        try:
            # Load hardware data (mandatory)
            hw_data = self.data_loader.load_hardware_data(1.0, use_hardware=True)
            if hw_data:
                if 'audio' in hw_data:
                    self.analyzer.audio_data = hw_data['audio']
                    print(f"✓ Loaded {len(hw_data['audio'])} hardware audio frequency points")
                if 'radio' in hw_data:
                    self.analyzer.radio_data = hw_data['radio']
                    print(f"✓ Loaded {len(hw_data['radio'])} hardware radio frequency points")
                self.combined_data = self.analyzer.combine_frequency_data()
                self.status_var.set("Hardware data loaded successfully")
            else:
                self.status_var.set("❌ Hardware data collection failed. Program requires hardware input.")
                self.combined_data = None
        except Exception as e:
            self.status_var.set(f"❌ Hardware data loading failed: {str(e)}")
            self.combined_data = None

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

        # Data source selection
        ttk.Label(control_frame, text="Data Source:").grid(row=0, column=0, padx=5)
        self.data_source = tk.StringVar(value="sample")
        ttk.Radiobutton(control_frame, text="Sample Data", variable=self.data_source,
                       value="sample", command=self.toggle_data_source).grid(row=0, column=1)
        ttk.Radiobutton(control_frame, text="Hardware", variable=self.data_source,
                       value="hardware", command=self.toggle_data_source).grid(row=0, column=2)

        # Hardware duration
        ttk.Label(control_frame, text="HW Duration:").grid(row=0, column=3, padx=5)
        self.hw_duration = tk.DoubleVar(value=1.0)
        ttk.Spinbox(control_frame, from_=0.1, to=10.0, increment=0.1,
                   textvariable=self.hw_duration, width=5).grid(row=0, column=4)

        # Buttons
        self.start_btn = ttk.Button(control_frame, text="Start Monitoring",
                                   command=self.start_monitoring)
        self.start_btn.grid(row=1, column=0, padx=5, pady=5)

        self.stop_btn = ttk.Button(control_frame, text="Stop Monitoring",
                                  command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_btn.grid(row=1, column=1, padx=5, pady=5)

        self.analyze_btn = ttk.Button(control_frame, text="Run Analysis",
                                     command=self.run_analysis)
        self.analyze_btn.grid(row=1, column=2, padx=5, pady=5)

        self.export_btn = ttk.Button(control_frame, text="Export Data",
                                    command=self.export_data)
        self.export_btn.grid(row=1, column=3, padx=5, pady=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_label = ttk.Label(control_frame, textvariable=self.status_var, font=("Arial", 10, "bold"))
        status_label.grid(row=1, column=4, padx=20, sticky=tk.W)

        # Stats panel
        stats_frame = ttk.LabelFrame(main_frame, text="Live Statistics", padding="5")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

        # Stats labels with better formatting
        self.stats_vars = {}
        stats_labels = [
            ("Total Samples", "total_samples"),
            ("Anomaly Score", "anomaly_score"),
            ("Leakage Points", "leakage_points"),
            ("Obscured Layers", "obscured_layers"),
            ("Network Layers", "network_layers"),
            ("Scan Count", "scan_count"),
            ("Runtime", "runtime"),
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

    def toggle_data_source(self):
        """Toggle between sample and hardware data sources."""
        self.use_hardware = (self.data_source.get() == "hardware")
        if self.use_hardware:
            self.status_var.set("Hardware mode selected - requires PyAudio/RTL-SDR")
        else:
            self.status_var.set("Sample data mode selected")

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
        start_time = time.time()

        while self.monitoring:
            try:
                scan_count += 1
                current_time = time.time() - start_time

                # Load fresh data if using hardware
                if self.use_hardware:
                    hw_data = self.data_loader.load_hardware_data(self.hw_duration.get(), use_hardware=True)
                    if hw_data:
                        if 'audio' in hw_data:
                            self.analyzer.audio_data = hw_data['audio']
                        if 'radio' in hw_data:
                            self.analyzer.radio_data = hw_data['radio']
                        self.combined_data = self.analyzer.combine_frequency_data()
                        self.log_message(f"Hardware data collected: {len(hw_data.get('audio', []))} audio, {len(hw_data.get('radio', []))} radio points")

                # Run analysis
                results = self.analyzer.detect_layer_anomalies()

                # Update stats
                stats_update = {
                    'scan_count': scan_count,
                    'runtime': current_time,
                    'anomaly_score': results['anomaly_score'],
                    'leakage_points': len(results['leakage_points']),
                    'obscured_layers': len(results['obscured_layers']),
                    'network_layers': results['network_topology'].get('layer_count', 0),
                    'connectivity_score': results['network_topology'].get('connectivity_score', 0.0),
                    'total_samples': len(self.combined_data) if self.combined_data is not None else 0
                }

                # Put results in queue for GUI update
                self.data_queue.put(('stats', stats_update))
                self.data_queue.put(('results', results))

                time.sleep(2)  # Update every 2 seconds

            except Exception as e:
                self.data_queue.put(('error', str(e)))
                time.sleep(5)  # Wait longer on error

    def run_analysis(self):
        """Run a single analysis."""
        try:
            self.status_var.set("Running analysis...")
            self.analyze_btn.config(state=tk.DISABLED)

            # Load hardware data if selected
            if self.use_hardware:
                hw_data = self.data_loader.load_hardware_data(self.hw_duration.get(), use_hardware=True)
                if hw_data:
                    if 'audio' in hw_data:
                        self.analyzer.audio_data = hw_data['audio']
                    if 'radio' in hw_data:
                        self.analyzer.radio_data = hw_data['radio']
                    self.combined_data = self.analyzer.combine_frequency_data()

            results = self.analyzer.detect_layer_anomalies()

            # Update stats
            stats_update = {
                'anomaly_score': results['anomaly_score'],
                'leakage_points': len(results['leakage_points']),
                'obscured_layers': len(results['obscured_layers']),
                'network_layers': results['network_topology'].get('layer_count', 0),
                'connectivity_score': results['network_topology'].get('connectivity_score', 0.0),
                'total_samples': len(self.combined_data) if self.combined_data is not None else 0
            }

            # Update GUI
            for key, value in stats_update.items():
                if key in self.stats_vars:
                    if isinstance(value, float):
                        self.stats_vars[key].set(f"{value:.3f}")
                    else:
                        self.stats_vars[key].set(str(value))

            # Update results display
            self.update_results_display(results)

            self.status_var.set("Analysis complete")
            self.log_message(f"Analysis complete - Anomaly score: {results['anomaly_score']:.3f}")

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
• Resolution: {self.analyzer.resolution_hz:.1f} Hz
• Total Samples: {len(self.combined_data) if self.combined_data is not None else 0}
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

            for i, point in enumerate(results['leakage_points'][:10]):  # Show top 10
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

            if self.combined_data is not None:
                self.analyzer.export_data(filename, 'combined')
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
