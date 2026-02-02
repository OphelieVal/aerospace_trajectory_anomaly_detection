"""
Interactive Visualizer
Generates interactive HTML dashboards with animated flight paths and detailed analysis
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

class InteractiveVisualizer:
    def __init__(self, results_path):
        """Load anomaly detection results"""
        self.df = pd.read_csv(results_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
    def generate_flight_animation(self, icao24, output_path='output/flight_animation.html'):
        """
        Generate an animated flight path with anomaly detection
        Shows real-time simulation with red markers for anomalies
        """
        flight_data = self.df[self.df['icao24'] == icao24].sort_values('timestamp')
        
        if len(flight_data) == 0:
            print(f"No data found for flight {icao24}")
            return None
        
        # Prepare data for JavaScript
        path_points = []
        for idx, row in flight_data.iterrows():
            point = {
                'lat': float(row['latitude']),
                'lon': float(row['longitude']),
                'alt': float(row['altitude']),
                'speed': float(row['ground_speed']) if pd.notna(row['ground_speed']) else 0,
                'vspeed': float(row['vertical_speed']) if pd.notna(row['vertical_speed']) else 0,
                'time': row['timestamp'].strftime('%H:%M:%S'),
                'anomaly': bool(row['anomaly_combined']),
                'reason': str(row.get('anomaly_reason', '')) if row['anomaly_combined'] else ''
            }
            path_points.append(point)
        
        # Calculate bounds
        lat_min, lat_max = flight_data['latitude'].min(), flight_data['latitude'].max()
        lon_min, lon_max = flight_data['longitude'].min(), flight_data['longitude'].max()
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        
        callsign = flight_data['callsign'].iloc[0]
        duration = (flight_data['timestamp'].max() - flight_data['timestamp'].min()).total_seconds() / 60
        anomaly_count = flight_data['anomaly_combined'].sum()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Flight Animation - {callsign} ({icao24})</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a2e;
            color: #eee;
        }}
        #header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        h1 {{
            margin: 0;
            font-size: 28px;
            color: white;
        }}
        .subtitle {{
            margin-top: 8px;
            font-size: 14px;
            color: rgba(255,255,255,0.9);
        }}
        #container {{
            display: flex;
            height: calc(100vh - 100px);
        }}
        #map {{
            flex: 1;
            height: 100%;
        }}
        #controls {{
            width: 350px;
            background: #16213e;
            padding: 20px;
            overflow-y: auto;
            box-shadow: -2px 0 10px rgba(0,0,0,0.3);
        }}
        .control-section {{
            background: #0f3460;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .control-section h3 {{
            margin-bottom: 15px;
            color: #e94560;
            font-size: 16px;
            border-bottom: 2px solid #e94560;
            padding-bottom: 8px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px;
            background: rgba(255,255,255,0.05);
            border-radius: 5px;
        }}
        .stat-label {{
            color: #aaa;
            font-size: 13px;
        }}
        .stat-value {{
            color: #fff;
            font-weight: bold;
            font-size: 14px;
        }}
        .anomaly {{
            color: #ff4757;
        }}
        .normal {{
            color: #2ed573;
        }}
        button {{
            width: 100%;
            padding: 12px;
            margin: 8px 0;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
        }}
        #playBtn {{
            background: linear-gradient(135deg, #2ed573 0%, #26de81 100%);
            color: white;
        }}
        #playBtn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(46, 213, 115, 0.4);
        }}
        #pauseBtn {{
            background: linear-gradient(135deg, #ffa502 0%, #ff6348 100%);
            color: white;
        }}
        #pauseBtn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(255, 165, 2, 0.4);
        }}
        #resetBtn {{
            background: linear-gradient(135deg, #5f27cd 0%, #341f97 100%);
            color: white;
        }}
        #resetBtn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(95, 39, 205, 0.4);
        }}
        .speed-control {{
            margin: 15px 0;
        }}
        .speed-control label {{
            display: block;
            margin-bottom: 8px;
            color: #aaa;
            font-size: 13px;
        }}
        input[type="range"] {{
            width: 100%;
            height: 6px;
            background: #0f3460;
            outline: none;
            border-radius: 3px;
        }}
        input[type="range"]::-webkit-slider-thumb {{
            width: 18px;
            height: 18px;
            background: #e94560;
            cursor: pointer;
            border-radius: 50%;
        }}
        #progress {{
            margin-top: 10px;
            font-size: 12px;
            color: #aaa;
            text-align: center;
        }}
        .current-data {{
            background: #1a1a2e;
            padding: 12px;
            border-radius: 8px;
            margin-top: 10px;
            border-left: 4px solid #e94560;
        }}
        .data-item {{
            margin: 8px 0;
            font-size: 13px;
        }}
        .data-label {{
            color: #aaa;
            display: inline-block;
            width: 120px;
        }}
        .data-value {{
            color: #fff;
            font-weight: bold;
        }}
        .leaflet-popup-content {{
            color: #333;
            font-size: 13px;
        }}
        .leaflet-popup-content strong {{
            color: #e94560;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>✈️ Flight Animation: {callsign}</h1>
        <div class="subtitle">ICAO24: {icao24} | Duration: {duration:.1f} minutes | Anomalies: {anomaly_count}</div>
    </div>
    
    <div id="container">
        <div id="map"></div>
        
        <div id="controls">
            <div class="control-section">
                <h3>🎮 Animation Controls</h3>
                <button id="playBtn">▶ Play Animation</button>
                <button id="pauseBtn">⏸ Pause</button>
                <button id="resetBtn">↺ Reset</button>
                
                <div class="speed-control">
                    <label>Animation Speed: <span id="speedValue">1x</span></label>
                    <input type="range" id="speedSlider" min="1" max="20" value="5" step="1">
                </div>
                
                <div id="progress">Point 0 / {len(path_points)}</div>
            </div>
            
            <div class="control-section">
                <h3>📊 Flight Statistics</h3>
                <div class="stat-row">
                    <span class="stat-label">Total Points</span>
                    <span class="stat-value">{len(flight_data)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Anomalies</span>
                    <span class="stat-value anomaly">{anomaly_count} ({anomaly_count/len(flight_data)*100:.1f}%)</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Max Altitude</span>
                    <span class="stat-value">{flight_data['altitude'].max():.0f} m</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Max Speed</span>
                    <span class="stat-value">{flight_data['ground_speed'].max():.0f} m/s</span>
                </div>
            </div>
            
            <div class="control-section">
                <h3>📍 Current Position</h3>
                <div class="current-data">
                    <div class="data-item">
                        <span class="data-label">Time:</span>
                        <span class="data-value" id="currentTime">--:--:--</span>
                    </div>
                    <div class="data-item">
                        <span class="data-label">Altitude:</span>
                        <span class="data-value" id="currentAlt">-- m</span>
                    </div>
                    <div class="data-item">
                        <span class="data-label">Speed:</span>
                        <span class="data-value" id="currentSpeed">-- m/s</span>
                    </div>
                    <div class="data-item">
                        <span class="data-label">Vertical Speed:</span>
                        <span class="data-value" id="currentVSpeed">-- m/s</span>
                    </div>
                    <div class="data-item">
                        <span class="data-label">Status:</span>
                        <span class="data-value normal" id="currentStatus">Normal</span>
                    </div>
                    <div class="data-item" id="anomalyReason" style="display: none;">
                        <span class="data-label">Anomaly Type:</span>
                        <span class="data-value anomaly" id="reasonText"></span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Flight data
        const pathPoints = {json.dumps(path_points)};
        
        // Initialize map
        const map = L.map('map').setView([{center_lat}, {center_lon}], 8);
        
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19
        }}).addTo(map);
        
        // Draw complete path (semi-transparent)
        const pathCoordinates = pathPoints.map(p => [p.lat, p.lon]);
        L.polyline(pathCoordinates, {{
            color: '#3498db',
            weight: 2,
            opacity: 0.3
        }}).addTo(map);
        
        // Add start and end markers
        L.marker([pathPoints[0].lat, pathPoints[0].lon], {{
            icon: L.divIcon({{
                className: 'start-marker',
                html: '<div style="background: #2ed573; width: 20px; height: 20px; border-radius: 50%; border: 3px solid white;"></div>',
                iconSize: [20, 20]
            }})
        }}).addTo(map).bindPopup('<strong>Start</strong><br>Time: ' + pathPoints[0].time);
        
        L.marker([pathPoints[pathPoints.length-1].lat, pathPoints[pathPoints.length-1].lon], {{
            icon: L.divIcon({{
                className: 'end-marker',
                html: '<div style="background: #e74c3c; width: 20px; height: 20px; border-radius: 50%; border: 3px solid white;"></div>',
                iconSize: [20, 20]
            }})
        }}).addTo(map).bindPopup('<strong>End</strong><br>Time: ' + pathPoints[pathPoints.length-1].time);
        
        // Animation state
        let currentIndex = 0;
        let isPlaying = false;
        let animationInterval = null;
        let animationSpeed = 5; // Points per second
        
        // Animation elements
        let currentPolyline = null;
        let planeMarker = null;
        let anomalyMarkers = [];
        
        // Create plane marker
        const planeIcon = L.divIcon({{
            className: 'plane-marker',
            html: '<div style="font-size: 24px;">✈️</div>',
            iconSize: [24, 24],
            iconAnchor: [12, 12]
        }});
        
        function updateDisplay(index) {{
            const point = pathPoints[index];
            
            // Update current position display
            document.getElementById('currentTime').textContent = point.time;
            document.getElementById('currentAlt').textContent = point.alt.toFixed(0) + ' m';
            document.getElementById('currentSpeed').textContent = point.speed.toFixed(1) + ' m/s';
            document.getElementById('currentVSpeed').textContent = point.vspeed.toFixed(1) + ' m/s';
            
            const statusElement = document.getElementById('currentStatus');
            const reasonElement = document.getElementById('anomalyReason');
            
            if (point.anomaly) {{
                statusElement.textContent = 'ANOMALY DETECTED';
                statusElement.className = 'data-value anomaly';
                reasonElement.style.display = 'block';
                document.getElementById('reasonText').textContent = point.reason.replace(/;/g, ', ');
            }} else {{
                statusElement.textContent = 'Normal';
                statusElement.className = 'data-value normal';
                reasonElement.style.display = 'none';
            }}
            
            // Update progress
            document.getElementById('progress').textContent = `Point ${{index + 1}} / ${{pathPoints.length}}`;
            
            // Update map
            const currentPath = pathPoints.slice(0, index + 1).map(p => [p.lat, p.lon]);
            
            if (currentPolyline) {{
                map.removeLayer(currentPolyline);
            }}
            
            currentPolyline = L.polyline(currentPath, {{
                color: '#e94560',
                weight: 3,
                opacity: 0.8
            }}).addTo(map);
            
            // Update plane position
            if (planeMarker) {{
                planeMarker.setLatLng([point.lat, point.lon]);
            }} else {{
                planeMarker = L.marker([point.lat, point.lon], {{icon: planeIcon}})
                    .addTo(map)
                    .bindPopup(`<strong>Current Position</strong><br>
                                Time: ${{point.time}}<br>
                                Alt: ${{point.alt.toFixed(0)}} m<br>
                                Speed: ${{point.speed.toFixed(1)}} m/s`);
            }}
            
            // Add anomaly marker if detected
            if (point.anomaly) {{
                const anomalyMarker = L.circleMarker([point.lat, point.lon], {{
                    color: '#ff4757',
                    fillColor: '#ff4757',
                    fillOpacity: 0.8,
                    radius: 6
                }}).addTo(map).bindPopup(`<strong style="color: #e74c3c;">⚠ ANOMALY</strong><br>
                                          Time: ${{point.time}}<br>
                                          Reason: ${{point.reason.replace(/;/g, '<br>')}}<br>
                                          Alt: ${{point.alt.toFixed(0)}} m<br>
                                          Speed: ${{point.speed.toFixed(1)}} m/s`);
                anomalyMarkers.push(anomalyMarker);
            }}
            
            // Center map on plane
            map.panTo([point.lat, point.lon]);
        }}
        
        function animate() {{
            if (currentIndex < pathPoints.length - 1) {{
                currentIndex++;
                updateDisplay(currentIndex);
            }} else {{
                pause();
            }}
        }}
        
        function play() {{
            if (!isPlaying) {{
                isPlaying = true;
                const interval = 1000 / animationSpeed;
                animationInterval = setInterval(animate, interval);
                document.getElementById('playBtn').style.opacity = '0.5';
            }}
        }}
        
        function pause() {{
            isPlaying = false;
            if (animationInterval) {{
                clearInterval(animationInterval);
                animationInterval = null;
            }}
            document.getElementById('playBtn').style.opacity = '1';
        }}
        
        function reset() {{
            pause();
            currentIndex = 0;
            
            // Clear map
            if (currentPolyline) map.removeLayer(currentPolyline);
            if (planeMarker) map.removeLayer(planeMarker);
            anomalyMarkers.forEach(m => map.removeLayer(m));
            anomalyMarkers = [];
            currentPolyline = null;
            planeMarker = null;
            
            // Reset display
            updateDisplay(0);
            map.setView([{center_lat}, {center_lon}], 8);
        }}
        
        // Event listeners
        document.getElementById('playBtn').addEventListener('click', play);
        document.getElementById('pauseBtn').addEventListener('click', pause);
        document.getElementById('resetBtn').addEventListener('click', reset);
        
        document.getElementById('speedSlider').addEventListener('input', (e) => {{
            animationSpeed = parseInt(e.target.value);
            document.getElementById('speedValue').textContent = animationSpeed + 'x';
            
            if (isPlaying) {{
                pause();
                play();
            }}
        }});
        
        // Initialize display
        updateDisplay(0);
    </script>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Generated interactive flight animation: {output_path}")
        return output_path
    
    def generate_interactive_dashboard(self, output_path='output/dashboard.html'):
        """
        Generate comprehensive interactive dashboard with all flights
        """
        # Prepare summary statistics
        total_flights = self.df['icao24'].nunique()
        total_points = len(self.df)
        anomaly_points = self.df['anomaly_combined'].sum()
        anomaly_rate = (anomaly_points / total_points * 100) if total_points > 0 else 0
        
        # Get top anomalous flights
        anomalous_flights = self.df[self.df['anomaly_combined']].groupby('icao24').agg({
            'callsign': 'first',
            'anomaly_combined': 'sum',
            'latitude': 'first',
            'longitude': 'first'
        }).rename(columns={'anomaly_combined': 'anomaly_count'})
        anomalous_flights = anomalous_flights.sort_values('anomaly_count', ascending=False).head(20)
        
        # Prepare data for charts
        flights_data = []
        for icao24 in self.df['icao24'].unique()[:50]:  # Limit to 50 flights for performance
            flight = self.df[self.df['icao24'] == icao24]
            flights_data.append({
                'icao24': icao24,
                'callsign': flight['callsign'].iloc[0],
                'points': len(flight),
                'anomalies': int(flight['anomaly_combined'].sum()),
                'path': [[float(row['latitude']), float(row['longitude'])] 
                        for _, row in flight.iterrows()],
                'anomaly_points': [[float(row['latitude']), float(row['longitude'])]
                                  for _, row in flight[flight['anomaly_combined']].iterrows()]
            })
        
        # Anomaly type breakdown
        anomaly_types = {}
        if 'anomaly_reason' in self.df.columns:
            reasons = self.df[self.df['anomaly_rule']]['anomaly_reason'].str.split(';', expand=True).stack();
            for reason, count in reasons.value_counts().items():
                if reason:
                    anomaly_types[reason] = int(count)
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Aircraft Trajectory Analysis Dashboard</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0e27;
            color: #fff;
        }}
        #header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        h1 {{
            font-size: 36px;
            margin-bottom: 10px;
        }}
        .subtitle {{
            font-size: 16px;
            color: rgba(255,255,255,0.9);
        }}
        #stats-bar {{
            display: flex;
            justify-content: space-around;
            padding: 20px;
            background: #16213e;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        .stat-card {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
            border-radius: 12px;
            min-width: 200px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            border: 2px solid #1a1a2e;
        }}
        .stat-number {{
            font-size: 42px;
            font-weight: bold;
            color: #e94560;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 14px;
            color: #aaa;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        #main-content {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            padding: 20px;
        }}
        .panel {{
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .panel h2 {{
            color: #e94560;
            margin-bottom: 20px;
            font-size: 20px;
            border-bottom: 3px solid #e94560;
            padding-bottom: 10px;
        }}
        #map-container {{
            height: 600px;
            grid-column: 1;
            grid-row: 1 / 3;
        }}
        #map {{
            height: 100%;
            width: 100%;
            border-radius: 8px;
            overflow: hidden;
        }}
        #charts-container {{
            display: grid;
            gap: 20px;
        }}
        .chart-panel {{
            height: 280px;
        }}
        canvas {{
            max-height: 240px !important;
        }}
        #flight-list {{
            max-height: 400px;
            overflow-y: auto;
        }}
        .flight-item {{
            background: #0f3460;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            border-left: 4px solid transparent;
        }}
        .flight-item:hover {{
            background: #1a4d7a;
            border-left-color: #e94560;
            transform: translateX(5px);
        }}
        .flight-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }}
        .flight-callsign {{
            font-weight: bold;
            font-size: 16px;
            color: #fff;
        }}
        .flight-icao {{
            color: #aaa;
            font-size: 12px;
        }}
        .anomaly-badge {{
            background: #ff4757;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}
        .normal-badge {{
            background: #2ed573;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}
        .interpretation {{
            background: linear-gradient(135deg, #1a4d7a 0%, #0f3460 100%);
            padding: 20px;
            border-radius: 12px;
            margin: 20px;
            border-left: 6px solid #e94560;
        }}
        .interpretation h3 {{
            color: #e94560;
            margin-bottom: 15px;
            font-size: 22px;
        }}
        .interpretation p {{
            line-height: 1.8;
            color: #ccc;
            margin: 10px 0;
        }}
        .interpretation ul {{
            margin: 15px 0;
            padding-left: 20px;
        }}
        .interpretation li {{
            margin: 8px 0;
            color: #ddd;
            line-height: 1.6;
        }}
        .key-finding {{
            background: rgba(233, 69, 96, 0.1);
            border-left: 4px solid #e94560;
            padding: 12px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        ::-webkit-scrollbar {{
            width: 10px;
        }}
        ::-webkit-scrollbar-track {{
            background: #0f3460;
            border-radius: 5px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: #e94560;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>🛫 Aircraft Trajectory Analysis Dashboard</h1>
        <div class="subtitle">Global Flight Anomaly Detection System</div>
    </div>
    
    <div id="stats-bar">
        <div class="stat-card">
            <div class="stat-label">Total Flights</div>
            <div class="stat-number">{total_flights}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Trajectory Points</div>
            <div class="stat-number">{total_points:,}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Anomalies Detected</div>
            <div class="stat-number">{anomaly_points}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Anomaly Rate</div>
            <div class="stat-number">{anomaly_rate:.1f}%</div>
        </div>
    </div>
    
    <div id="main-content">
        <div id="map-container" class="panel">
            <h2>🗺️ Global Flight Map</h2>
            <div id="map"></div>
        </div>
        
        <div id="charts-container">
            <div class="panel chart-panel">
                <h2>📊 Anomaly Distribution</h2>
                <canvas id="anomalyChart"></canvas>
            </div>
            
            <div class="panel">
                <h2>⚠️ Top Anomalous Flights</h2>
                <div id="flight-list"></div>
            </div>
        </div>
    </div>
    
    <div class="interpretation">
        <h3>🔍 Analysis Interpretation & Key Findings</h3>
        <p>
            This dashboard presents the results of analyzing <strong>{total_flights} aircraft trajectories</strong> 
            using a hybrid anomaly detection system combining aerospace domain rules and machine learning algorithms.
        </p>
        
        <div class="key-finding">
            <strong>Detection Rate:</strong> {anomaly_rate:.2f}% of trajectory points were flagged as anomalous, 
            indicating {anomaly_points} unusual data points across all analyzed flights.
        </div>
        
        <p><strong>What the anomalies indicate:</strong></p>
        <ul>
            <li><strong>Excessive Speed Violations:</strong> Aircraft exceeding typical cruise speeds, possibly indicating military aircraft, data errors, or unusual operational conditions</li>
            <li><strong>Unusual Vertical Speeds:</strong> Rapid climbs or descents beyond normal parameters, which may indicate emergency procedures or aggressive flight maneuvers</li>
            <li><strong>Sharp Turns & High Turn Rates:</strong> Aggressive maneuvering uncommon in commercial aviation, possibly indicating training flights, aerobatic aircraft, or evasive actions</li>
            <li><strong>Altitude Anomalies:</strong> Unexpected altitude changes or inconsistent altitude-speed combinations</li>
        </ul>
        
        <p><strong>Interpretation Guidelines:</strong></p>
        <ul>
            <li><strong>Red markers on map:</strong> Indicate specific points where anomalies were detected</li>
            <li><strong>Click on flight paths:</strong> View detailed information about each trajectory</li>
            <li><strong>Hover over chart elements:</strong> See exact values and breakdowns</li>
            <li><strong>Multiple anomaly types:</strong> Some flights may trigger several detection rules simultaneously</li>
        </ul>
        
        <div class="key-finding">
            <strong>Important Note:</strong> Anomalies don't necessarily indicate safety issues. Many detected patterns 
            are legitimate operations (military flights, training exercises, special missions) or data quality issues 
            in the tracking system. This system highlights unusual patterns for further investigation.
        </div>
    </div>

    <script>
        const flightsData = {json.dumps(flights_data)};
        const anomalyTypes = {json.dumps(anomaly_types)};
        
        // Initialize map
        const map = L.map('map').setView([20, 0], 2);
        
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors',
            maxZoom: 18
        }}).addTo(map);
        
        // Draw all flight paths
        const pathLayers = {{}};
        const anomalyLayers = {{}};
        
        flightsData.forEach(flight => {{
            // Draw flight path
            const pathLayer = L.polyline(flight.path, {{
                color: '#3498db',
                weight: 2,
                opacity: 0.4
            }}).addTo(map);
            
            pathLayer.bindPopup(`
                <strong>${{flight.callsign}}</strong><br>
                ICAO24: ${{flight.icao24}}<br>
                Points: ${{flight.points}}<br>
                Anomalies: <span style="color: #e74c3c;">${{flight.anomalies}}</span>
            `);
            
            pathLayers[flight.icao24] = pathLayer;
            
            // Add anomaly markers
            if (flight.anomaly_points.length > 0) {{
                const anomalyGroup = L.layerGroup();
                flight.anomaly_points.forEach(point => {{
                    L.circleMarker(point, {{
                        color: '#ff4757',
                        fillColor: '#ff4757',
                        fillOpacity: 0.8,
                        radius: 5,
                        weight: 2
                    }}).addTo(anomalyGroup).bindPopup(`
                        <strong style="color: #e74c3c;">⚠ Anomaly Detected</strong><br>
                        Flight: ${{flight.callsign}}<br>
                        Position: ${{point[0].toFixed(4)}}, ${{point[1].toFixed(4)}}
                    `);
                }});
                anomalyGroup.addTo(map);
                anomalyLayers[flight.icao24] = anomalyGroup;
            }}
        }});
        
        // Populate flight list
        const flightList = document.getElementById('flight-list');
        const sortedFlights = flightsData.sort((a, b) => b.anomalies - a.anomalies).slice(0, 15);
        
        sortedFlights.forEach(flight => {{
            const item = document.createElement('div');
            item.className = 'flight-item';
            
            const anomalyRate = (flight.anomalies / flight.points * 100).toFixed(1);
            const badgeClass = flight.anomalies > 0 ? 'anomaly-badge' : 'normal-badge';
            const badgeText = flight.anomalies > 0 ? `${{flight.anomalies}} anomalies` : 'Normal';
            
            item.innerHTML = `
                <div class="flight-header">
                    <div>
                        <div class="flight-callsign">${{flight.callsign}}</div>
                        <div class="flight-icao">${{flight.icao24}}</div>
                    </div>
                    <span class="${{badgeClass}}">${{badgeText}}</span>
                </div>
                <div style="color: #aaa; font-size: 13px;">
                    Points: ${{flight.points}} | Anomaly Rate: ${{anomalyRate}}%
                </div>
            `;
            
            item.addEventListener('click', () => {{
                // Highlight selected flight
                Object.values(pathLayers).forEach(layer => {{
                    layer.setStyle({{ opacity: 0.2, weight: 2 }});
                }});
                
                if (pathLayers[flight.icao24]) {{
                    pathLayers[flight.icao24].setStyle({{ 
                        opacity: 1, 
                        weight: 4,
                        color: '#e94560'
                    }});
                    
                    // Zoom to flight
                    map.fitBounds(pathLayers[flight.icao24].getBounds(), {{padding: [50, 50]}});
                    
                    // Open popup
                    pathLayers[flight.icao24].openPopup();
                }}
            }});
            
            flightList.appendChild(item);
        }});
        
        // Create anomaly distribution chart
        const ctx = document.getElementById('anomalyChart');
        const labels = Object.keys(anomalyTypes);
        const data = Object.values(anomalyTypes);
        
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: labels.map(l => l.replace(/_/g, ' ')),
                datasets: [{{
                    label: 'Occurrences',
                    data: data,
                    backgroundColor: [
                        'rgba(255, 71, 87, 0.8)',
                        'rgba(255, 165, 2, 0.8)',
                        'rgba(46, 213, 115, 0.8)',
                        'rgba(52, 152, 219, 0.8)',
                        'rgba(155, 89, 182, 0.8)',
                        'rgba(241, 196, 15, 0.8)'
                    ],
                    borderColor: [
                        'rgba(255, 71, 87, 1)',
                        'rgba(255, 165, 2, 1)',
                        'rgba(46, 213, 115, 1)',
                        'rgba(52, 152, 219, 1)',
                        'rgba(155, 89, 182, 1)',
                        'rgba(241, 196, 15, 1)'
                    ],
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    tooltip: {{
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: '#e94560',
                        borderWidth: 2
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{
                            color: '#aaa'
                        }},
                        grid: {{
                            color: 'rgba(255, 255, 255, 0.1)'
                        }}
                    }},
                    x: {{
                        ticks: {{
                            color: '#aaa',
                            maxRotation: 45,
                            minRotation: 45
                        }},
                        grid: {{
                            color: 'rgba(255, 255, 255, 0.1)'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Generated interactive dashboard: {output_path}")
        return output_path
