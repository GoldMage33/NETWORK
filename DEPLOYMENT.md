# NETWORK Frequency Analysis System - Deployment Guide

## Overview
The NETWORK system is a comprehensive frequency analysis tool for detecting data leakage and obscured layers in layered networks. This deployment guide covers multiple deployment options.

## System Requirements
- Python 3.9+
- 4GB RAM minimum (8GB recommended)
- Linux/macOS/Windows

## Quick Start Deployment

### Option 1: Local Deployment
```bash
# Clone the repository
git clone <repository-url>
cd NETWORK

# Install dependencies
pip install -r requirements.txt

# Run the main analysis
python3 main.py

# Or start live monitoring
python3 live_console_monitor.py
```

### Option 2: Docker Deployment
```bash
# Build the container
docker build -t network-analyzer .

# Run the analysis
docker run --rm network-analyzer python3 main.py

# Run live monitoring
docker run -it network-analyzer python3 live_console_monitor.py
```

### Option 3: Background Service
```bash
# Run background monitoring
python3 background_monitor.py --instance 1 --log-file /var/log/network_monitor.log

# Run enhanced monitoring with device discovery
python3 enhanced_monitor.py
```

## Available Components

### Core Analysis
- `main.py` - Primary frequency analysis pipeline
- `src/frequency_analyzer.py` - Core analysis engine
- `src/data_loader.py` - Data loading and generation

### Monitoring Systems
- `live_console_monitor.py` - Real-time console monitoring
- `background_monitor.py` - Background multi-instance monitoring
- `enhanced_monitor.py` - Device discovery and tag detection

### Analysis Tools
- `global_cross_reference.py` - Cross-reference analysis
- `node_discovery.py` - Network node discovery
- `node_visualizer.py` - Network visualization
- `gui_app.py` - Interactive GUI application

## Configuration

### Environment Variables
```bash
export NETWORK_RESOLUTION_HZ=100.0  # Frequency resolution
export NETWORK_DATA_DIR=./data      # Data directory
export NETWORK_LOG_LEVEL=INFO       # Logging level
```

### Configuration Files
- `data/sample_audio.csv` - Audio frequency data
- `data/sample_radio.csv` - Radio frequency data
- `.vscode/settings.json` - VS Code configuration

## API Usage

### Basic Analysis
```python
from src.frequency_analyzer import FrequencyAnalyzer

analyzer = FrequencyAnalyzer(resolution_hz=100.0)
analyzer.load_audio_frequencies('data/audio.csv')
analyzer.load_radio_frequencies('data/radio.csv')
results = analyzer.detect_layer_anomalies()
print(f"Anomaly Score: {results['anomaly_score']:.3f}")
```

### Live Monitoring
```python
from live_console_monitor import LiveConsoleMonitor

monitor = LiveConsoleMonitor()
monitor.start_monitoring()  # Runs continuously
```

## Output Files

### Reports
- `global_leakage_analysis_report.md` - Comprehensive analysis report
- `comprehensive_connection_report.md` - Connection analysis
- `data/combined_frequency_analysis.csv` - Combined frequency data

### Visualizations
- `visualizations/network_topology_map.png` - Network topology
- `visualizations/node_topology_risk.png` - Risk assessment
- `visualizations/pattern_categorization_*.json` - Pattern analysis

### Metadata
- `network_metadata.json` - Network metadata
- `global_leakage_report.json` - Leakage detection results
- `node_discovery_report.json` - Node discovery results

## Monitoring and Maintenance

### Health Checks
```bash
# Check if monitoring is running
ps aux | grep python3 | grep monitor

# View logs
tail -f /var/log/network_monitor.log
```

### Performance Tuning
- Adjust `resolution_hz` for analysis precision vs speed
- Use background monitoring for continuous operation
- Monitor memory usage for large datasets

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Data Loading**: Check CSV file formats and paths
3. **Memory Issues**: Reduce resolution or use smaller datasets
4. **Permission Errors**: Ensure write access to output directories

### Debug Mode
```bash
python3 -c "import logging; logging.basicConfig(level=logging.DEBUG); from src.frequency_analyzer import FrequencyAnalyzer; analyzer = FrequencyAnalyzer(); print('Debug mode enabled')"
```

## Security Considerations
- The system analyzes frequency data for security research
- No sensitive data should be processed without proper authorization
- Monitor logs for any unauthorized access attempts
- Keep dependencies updated for security patches

## Support
For issues or questions:
1. Check the logs in `/var/log/network_monitor.log`
2. Review the test suite: `python3 -m pytest tests/`
3. Examine generated reports in the root directory
