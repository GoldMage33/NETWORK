#!/usr/bin/env python3
"""
Network Frequency Analysis Web GUI Application
Real-time monitoring with live statistics and interactive controls
"""

import sys
import os
import time
import threading
import json
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.frequency_analyzer import FrequencyAnalyzer
from src.data_loader import DataLoader
import queue


class NetworkMonitorWeb:
    """Web-based GUI application for real-time network frequency monitoring."""

    def __init__(self):
        self.app = Flask(__name__)
        self.analyzer = FrequencyAnalyzer(resolution_hz=100.0)
        self.data_loader = DataLoader()
        self.monitoring = False
        self.monitor_thread = None
        self.data_queue = queue.Queue()
        self.stats_history = []
        self.use_hardware = False
        self.hw_duration = 1.0
        self.combined_data = None
        self.current_results = None

        # Load initial data
        self.load_data()

        # Setup routes
        self.setup_routes()

    def load_data(self):
        """Load frequency data for analysis using hardware input."""
        try:
            # Load hardware data (mandatory)
            hw_data = self.data_loader.load_hardware_data(self.hw_duration, use_hardware=True)
            if hw_data:
                if 'audio' in hw_data:
                    self.analyzer.audio_data = hw_data['audio']
                    print(f"✓ Loaded {len(hw_data['audio'])} hardware audio frequency points")
                if 'radio' in hw_data:
                    self.analyzer.radio_data = hw_data['radio']
                    print(f"✓ Loaded {len(hw_data['radio'])} hardware radio frequency points")
                self.combined_data = self.analyzer.combine_frequency_data()
            else:
                print("❌ Hardware data collection failed. Program requires hardware input.")
                self.combined_data = None
        except Exception as e:
            print(f"❌ Hardware data loading failed: {e}")
            self.combined_data = None

    def setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            return render_template_string(self.get_html_template())

        @self.app.route('/api/stats')
        def get_stats():
            stats = {
                'total_samples': len(self.combined_data) if self.combined_data is not None else 0,
                'anomaly_score': 0.0,
                'leakage_points': 0,
                'obscured_layers': 0,
                'network_layers': 0,
                'scan_count': 0,
                'runtime': 0,
                'connectivity_score': 0.0,
                'monitoring': self.monitoring,
                'data_available': self.combined_data is not None
            }

            if self.current_results:
                stats.update({
                    'anomaly_score': self.current_results['anomaly_score'],
                    'leakage_points': len(self.current_results['leakage_points']),
                    'obscured_layers': len(self.current_results['obscured_layers']),
                    'network_layers': self.current_results['network_topology'].get('layer_count', 0),
                    'connectivity_score': self.current_results['network_topology'].get('connectivity_score', 0.0),
                })

            return jsonify(stats)

        @self.app.route('/api/start_monitoring', methods=['POST'])
        def start_monitoring():
            if not self.monitoring:
                self.monitoring = True
                self.monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
                self.monitor_thread.start()
            return jsonify({'status': 'started'})

        @self.app.route('/api/stop_monitoring', methods=['POST'])
        def stop_monitoring():
            self.monitoring = False
            return jsonify({'status': 'stopped'})

        @self.app.route('/api/run_analysis', methods=['POST'])
        def run_analysis():
            try:
                if self.combined_data is None:
                    return jsonify({'status': 'error', 'message': 'No frequency data available. Please load data first.'})
                
                # Run the analysis
                results = self.analyzer.run_analysis()
                
                # Convert results to JSON-serializable format
                serializable_results = self.convert_to_json_serializable(results)
                
                return jsonify({'status': 'success', 'results': serializable_results})
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)})

        @self.app.route('/api/set_config', methods=['POST'])
        def set_config():
            data = request.get_json()
            self.use_hardware = data.get('use_hardware', False)
            self.hw_duration = data.get('hw_duration', 1.0)
            return jsonify({'status': 'updated'})

        @self.app.route('/api/frequency_data')
        def get_frequency_data():
            """Get current frequency data for real-time graphing."""
            if self.combined_data is not None and len(self.combined_data) > 0:
                # Sample data points for efficient transmission (max 1000 points)
                data = self.combined_data.copy()
                if len(data) > 1000:
                    step = len(data) // 1000
                    data = data.iloc[::step]
                
                # Convert to JSON-serializable format
                freq_data = {
                    'frequencies': data['frequency'].tolist(),
                    'audio_amplitudes': data['audio_amplitude'].tolist(),
                    'radio_amplitudes': data['radio_amplitude'].tolist(),
                    'combined_amplitudes': data['combined_amplitude'].tolist()
                }
                return jsonify(freq_data)
            return jsonify({'error': 'No frequency data available. Start monitoring with hardware data.'})

    def convert_to_json_serializable(self, obj):
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
                    hw_data = self.data_loader.load_hardware_data(self.hw_duration, use_hardware=True)
                    if hw_data:
                        if 'audio' in hw_data:
                            self.analyzer.audio_data = hw_data['audio']
                        if 'radio' in hw_data:
                            self.analyzer.radio_data = hw_data['radio']
                        # Recombine data for updated graph
                        self.combined_data = self.analyzer.combine_frequency_data()

                # Only run analysis if we have data
                if self.combined_data is not None:
                    self.current_results = self.analyzer.detect_layer_anomalies()

                time.sleep(1)  # Update every 1 second for smoother real-time updates

            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(2)  # Wait longer on error

    def get_html_template(self):
        """Get the streamlined HTML template for the web interface."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NETWORK Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: #0a0a0a;
            color: #00ff00;
            overflow-x: hidden;
        }
        .header {
            background: #1a1a1a;
            padding: 15px;
            text-align: center;
            border-bottom: 2px solid #00ff00;
            position: fixed;
            top: 0;
            width: 100%;
            z-index: 100;
        }
        .main {
            margin-top: 80px;
            padding: 20px;
        }
        .graph-container {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            height: 400px;
        }
        .controls {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #00ff00;
        }
        .stat-label {
            font-size: 11px;
            color: #888;
            margin-top: 5px;
        }
        button {
            background: #00ff00;
            color: #000;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            font-family: inherit;
        }
        button:hover { background: #00cc00; }
        button:disabled { background: #666; cursor: not-allowed; }
        .status {
            color: #00ff00;
            font-weight: bold;
        }
        .radio-group {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        .radio-group label {
            display: flex;
            align-items: center;
            gap: 5px;
            cursor: pointer;
        }
        input[type="number"] {
            background: #2a2a2a;
            border: 1px solid #555;
            color: #00ff00;
            padding: 5px;
            border-radius: 4px;
            width: 60px;
        }
        canvas {
            max-width: 100%;
            height: 100% !important;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌐 NETWORK FREQUENCY MONITOR</h1>
    </div>

    <div class="main">
        <div class="graph-container">
            <canvas id="frequencyChart"></canvas>
        </div>

        <div class="controls">
            <div class="radio-group">
                <label><input type="radio" name="data_source" value="sample" checked> Sample Data</label>
                <label><input type="radio" name="data_source" value="hardware"> Hardware Data</label>
            </div>
            <label>Duration: <input type="number" id="hw_duration" value="1.0" min="0.1" max="10" step="0.1">s</label>
            <button id="startBtn">▶ START</button>
            <button id="stopBtn" disabled>⏹ STOP</button>
            <button id="analyzeBtn">🔍 ANALYZE</button>
            <span class="status" id="status">Ready</span>
        </div>

        <div class="stats" id="statsGrid">
            <div class="stat-card"><div class="stat-value" id="anomaly_score">0.000</div><div class="stat-label">ANOMALY SCORE</div></div>
            <div class="stat-card"><div class="stat-value" id="leakage_points">0</div><div class="stat-label">LEAKAGE POINTS</div></div>
            <div class="stat-card"><div class="stat-value" id="network_layers">0</div><div class="stat-label">NETWORK LAYERS</div></div>
            <div class="stat-card"><div class="stat-value" id="connectivity_score">0.000</div><div class="stat-label">CONNECTIVITY</div></div>
        </div>
    </div>

    <script>
        let chart = null;
        let monitoring = false;
        let updateInterval;

        function initChart() {
            const ctx = document.getElementById('frequencyChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: [{
                        label: 'Audio Amplitude',
                        data: [],
                        borderColor: '#00ff00',
                        backgroundColor: 'rgba(0, 255, 0, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.1
                    }, {
                        label: 'Radio Amplitude',
                        data: [],
                        borderColor: '#0080ff',
                        backgroundColor: 'rgba(0, 128, 255, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {
                        x: {
                            type: 'linear',
                            position: 'bottom',
                            title: {
                                display: true,
                                text: 'Frequency (Hz)',
                                color: '#00ff00'
                            },
                            ticks: {
                                color: '#00ff00',
                                callback: function(value) {
                                    return (value / 1000).toFixed(1) + 'k';
                                }
                            },
                            grid: { color: '#333' }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Amplitude',
                                color: '#00ff00'
                            },
                            ticks: { color: '#00ff00' },
                            grid: { color: '#333' }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#00ff00' }
                        }
                    }
                }
            });
        }

        function updateChart() {
            fetch('/api/frequency_data')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        // Clear chart when no data available
                        chart.data.datasets[0].data = [];
                        chart.data.datasets[1].data = [];
                        chart.update('none');
                        return;
                    }
                    
                    if (data.frequencies && data.audio_amplitudes) {
                        const points = data.frequencies.map((freq, i) => ({
                            x: freq,
                            y: data.audio_amplitudes[i]
                        }));
                        const radioPoints = data.frequencies.map((freq, i) => ({
                            x: freq,
                            y: data.radio_amplitudes[i]
                        }));

                        chart.data.datasets[0].data = points;
                        chart.data.datasets[1].data = radioPoints;
                        chart.update('none');
                    }
                })
                .catch(err => {
                    console.log('Chart update error:', err);
                    // Clear chart on error
                    chart.data.datasets[0].data = [];
                    chart.data.datasets[1].data = [];
                    chart.update('none');
                });
        }

        function updateStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    if (!data.data_available) {
                        document.getElementById('anomaly_score').textContent = 'N/A';
                        document.getElementById('leakage_points').textContent = 'N/A';
                        document.getElementById('network_layers').textContent = 'N/A';
                        document.getElementById('connectivity_score').textContent = 'N/A';
                        document.getElementById('status').textContent = 'No data - Use hardware mode';
                    } else {
                        document.getElementById('anomaly_score').textContent = data.anomaly_score.toFixed(3);
                        document.getElementById('leakage_points').textContent = data.leakage_points;
                        document.getElementById('network_layers').textContent = data.network_layers;
                        document.getElementById('connectivity_score').textContent = data.connectivity_score.toFixed(3);
                        document.getElementById('status').textContent = monitoring ? 'Monitoring...' : 'Ready';
                    }

                    monitoring = data.monitoring;
                    document.getElementById('startBtn').disabled = monitoring;
                    document.getElementById('stopBtn').disabled = !monitoring;
                });
        }

        function startMonitoring() {
            fetch('/api/start_monitoring', { method: 'POST' })
                .then(() => {
                    document.getElementById('status').textContent = 'Monitoring...';
                    updateInterval = setInterval(() => {
                        updateStats();
                        updateChart();
                    }, 1000);
                });
        }

        function stopMonitoring() {
            fetch('/api/stop_monitoring', { method: 'POST' })
                .then(() => {
                    document.getElementById('status').textContent = 'Stopped';
                    if (updateInterval) clearInterval(updateInterval);
                });
        }

        function runAnalysis() {
            document.getElementById('status').textContent = 'Analyzing...';
            fetch('/api/run_analysis', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        document.getElementById('status').textContent = 'Analysis Complete';
                        updateStats();
                    } else {
                        document.getElementById('status').textContent = 'Analysis Error: ' + data.message;
                    }
                })
                .catch(err => {
                    document.getElementById('status').textContent = 'Analysis Error';
                    console.log('Analysis error:', err);
                });
        }

        function updateConfig() {
            const useHardware = document.querySelector('input[name="data_source"]:checked').value === 'hardware';
            const hwDuration = parseFloat(document.getElementById('hw_duration').value);

            fetch('/api/set_config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ use_hardware: useHardware, hw_duration: hwDuration })
            });
        }

        // Event listeners
        document.getElementById('startBtn').addEventListener('click', startMonitoring);
        document.getElementById('stopBtn').addEventListener('click', stopMonitoring);
        document.getElementById('analyzeBtn').addEventListener('click', runAnalysis);

        document.querySelectorAll('input[name="data_source"]').forEach(radio => {
            radio.addEventListener('change', updateConfig);
        });
        document.getElementById('hw_duration').addEventListener('change', updateConfig);

        // Initialize
        initChart();
        updateChart();
        updateStats();
    </script>
</body>
</html>
        """

    def run(self, host='localhost', port=5000):
        """Run the web application."""
        print(f"🌐 NETWORK Frequency Analysis Web GUI")
        print(f"📊 Open your browser to: http://{host}:{port}")
        print(f"🔧 Press Ctrl+C to stop the server")
        self.app.run(host=host, port=port, debug=False)


def main():
    """Main function to run the web GUI application."""
    monitor = NetworkMonitorWeb()
    monitor.run()


if __name__ == "__main__":
    main()
