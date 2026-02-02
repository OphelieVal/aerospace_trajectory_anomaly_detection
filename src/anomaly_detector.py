"""
Anomaly Detection System
Combines rule-based and machine learning approaches to detect unusual patterns
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    def __init__(self, processed_data_path):
        """Load processed trajectory data"""
        self.df = pd.read_csv(processed_data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.anomalies = pd.DataFrame()
        
    def rule_based_detection(self):
        """
        Detect anomalies using aerospace domain rules
        Returns DataFrame with anomaly flags and reasons
        """
        print("Running rule-based anomaly detection...")
        
        # Initialize anomaly flags
        self.df['anomaly_rule'] = False
        self.df['anomaly_reason'] = ''
        
        # Rule 1: Excessive ground speed (typical cruise: 200-300 m/s for commercial jets)
        excessive_speed = self.df['ground_speed'] > 350  # ~1260 km/h
        self.df.loc[excessive_speed, 'anomaly_rule'] = True
        self.df.loc[excessive_speed, 'anomaly_reason'] += 'EXCESSIVE_SPEED;'
        
        # Rule 2: Unusual vertical speed (typical climb/descent: < 15 m/s)
        unusual_vertical = np.abs(self.df['vertical_speed']) > 25
        self.df.loc[unusual_vertical, 'anomaly_rule'] = True
        self.df.loc[unusual_vertical, 'anomaly_reason'] += 'UNUSUAL_VERTICAL_SPEED;'
        
        # Rule 3: Extreme acceleration (typical: < 2 m/s²)
        extreme_accel = np.abs(self.df['ground_accel']) > 5
        self.df.loc[extreme_accel, 'anomaly_rule'] = True
        self.df.loc[extreme_accel, 'anomaly_reason'] += 'EXTREME_ACCELERATION;'
        
        # Rule 4: Sharp turns (typical turn rate: < 3 deg/s for commercial aircraft)
        sharp_turn = np.abs(self.df['turn_rate']) > 10
        self.df.loc[sharp_turn, 'anomaly_rule'] = True
        self.df.loc[sharp_turn, 'anomaly_reason'] += 'SHARP_TURN;'
        
        # Rule 5: Altitude anomalies
        very_low_speed = self.df['ground_speed'] < 20
        high_altitude = self.df['altitude'] > 15000  # meters (~49,000 ft)
        
        low_speed_high_alt = very_low_speed & high_altitude
        self.df.loc[low_speed_high_alt, 'anomaly_rule'] = True
        self.df.loc[low_speed_high_alt, 'anomaly_reason'] += 'LOW_SPEED_HIGH_ALTITUDE;'
        
        # Rule 6: Extreme altitude changes
        for icao24, group in self.df.groupby('icao24'):
            if len(group) < 5:
                continue
            
            # Check for sudden altitude jumps (> 500m in one step)
            alt_jump = group['altitude'].diff().abs() > 500
            self.df.loc[group[alt_jump].index, 'anomaly_rule'] = True
            self.df.loc[group[alt_jump].index, 'anomaly_reason'] += 'ALTITUDE_JUMP;'
        
        rule_anomalies = self.df[self.df['anomaly_rule']].copy()
        print(f"Found {len(rule_anomalies)} rule-based anomalies ({len(rule_anomalies)/len(self.df)*100:.2f}%)")
        
        # Anomaly breakdown
        reasons = rule_anomalies['anomaly_reason'].str.split(';', expand=True).stack().value_counts()
        print("\nAnomaly types:")
        print(reasons[reasons.index != ''])
        
        return rule_anomalies
    
    def ml_based_detection(self, contamination=0.05):
        """
        Detect anomalies using Isolation Forest
        contamination: expected proportion of anomalies in dataset
        """
        print(f"\nRunning ML-based anomaly detection (contamination={contamination})...")
        
        # Select features for ML model
        feature_cols = ['ground_speed', 'vertical_speed', 'ground_accel', 
                       'turn_rate', 'altitude', 'heading']
        
        # Create flight-level aggregated features
        flight_features = []
        
        for icao24, group in self.df.groupby('icao24'):
            if len(group) < 5:
                continue
                
            features = {
                'icao24': icao24,
                'callsign': group['callsign'].iloc[0],
                'mean_ground_speed': group['ground_speed'].mean(),
                'std_ground_speed': group['ground_speed'].std(),
                'max_ground_speed': group['ground_speed'].max(),
                'mean_vertical_speed': group['vertical_speed'].mean(),
                'std_vertical_speed': group['vertical_speed'].std(),
                'max_vertical_speed': group['vertical_speed'].max(),
                'mean_altitude': group['altitude'].mean(),
                'std_altitude': group['altitude'].std(),
                'max_altitude': group['altitude'].max(),
                'altitude_range': group['altitude'].max() - group['altitude'].min(),
                'mean_turn_rate': np.abs(group['turn_rate']).mean(),
                'max_turn_rate': np.abs(group['turn_rate']).max(),
                'mean_accel': np.abs(group['ground_accel']).mean(),
                'max_accel': np.abs(group['ground_accel']).max(),
                'trajectory_length': len(group),
                'duration': (group['timestamp'].max() - group['timestamp'].min()).total_seconds()
            }
            flight_features.append(features)
        
        flight_df = pd.DataFrame(flight_features)
        
        # Prepare features for ML
        ml_feature_cols = [col for col in flight_df.columns 
                          if col not in ['icao24', 'callsign']]
        X = flight_df[ml_feature_cols].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        predictions = iso_forest.fit_predict(X_scaled)
        
        # -1 indicates anomaly, 1 indicates normal
        flight_df['anomaly_ml'] = predictions == -1
        flight_df['anomaly_score'] = iso_forest.score_samples(X_scaled)
        
        # Merge back to main dataframe
        self.df = self.df.merge(
            flight_df[['icao24', 'anomaly_ml', 'anomaly_score']], 
            on='icao24', 
            how='left'
        )
        
        ml_anomalies = flight_df[flight_df['anomaly_ml']].copy()
        print(f"Found {len(ml_anomalies)} anomalous flights ({len(ml_anomalies)/len(flight_df)*100:.2f}%)")
        
        return ml_anomalies
    
    def detect_trajectory_shape_anomalies(self, eps_km=50, min_samples=3):
        """
        Detect unusual trajectory shapes using DBSCAN clustering
        Groups similar trajectories and identifies outliers
        """
        print("\nDetecting trajectory shape anomalies...")
        
        # Create trajectory shape features (start/end positions, bounding box)
        shape_features = []
        
        for icao24, group in self.df.groupby('icao24'):
            if len(group) < 10:
                continue
            
            features = {
                'icao24': icao24,
                'start_lat': group['latitude'].iloc[0],
                'start_lon': group['longitude'].iloc[0],
                'end_lat': group['latitude'].iloc[-1],
                'end_lon': group['longitude'].iloc[-1],
                'lat_range': group['latitude'].max() - group['latitude'].min(),
                'lon_range': group['longitude'].max() - group['longitude'].min(),
                'mean_lat': group['latitude'].mean(),
                'mean_lon': group['longitude'].mean()
            }
            shape_features.append(features)
        
        shape_df = pd.DataFrame(shape_features)
        
        if len(shape_df) < min_samples:
            print("Not enough trajectories for shape clustering")
            return pd.DataFrame()
        
        # Cluster trajectories by spatial characteristics
        X_shape = shape_df[['mean_lat', 'mean_lon', 'lat_range', 'lon_range']].values
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=eps_km/111, min_samples=min_samples)  # Rough conversion to degrees
        labels = clustering.fit_predict(X_shape)
        
        shape_df['cluster'] = labels
        shape_df['shape_anomaly'] = labels == -1  # -1 indicates noise/outlier
        
        # Merge back to main dataframe
        self.df = self.df.merge(
            shape_df[['icao24', 'shape_anomaly', 'cluster']], 
            on='icao24', 
            how='left'
        )
        
        shape_anomalies = shape_df[shape_df['shape_anomaly']].copy()
        print(f"Found {len(shape_anomalies)} unusual trajectory shapes")
        print(f"Identified {labels.max() + 1} trajectory clusters")
        
        return shape_anomalies
    
    def combine_anomaly_results(self):
        """Combine all anomaly detection methods"""
        print("\n" + "="*60)
        print("COMBINED ANOMALY DETECTION RESULTS")
        print("="*60)
        
        # Create combined anomaly flag
        self.df['anomaly_combined'] = (
            self.df['anomaly_rule'] | 
            self.df['anomaly_ml'].fillna(False) |
            self.df['shape_anomaly'].fillna(False)
        )
        
        # Summary statistics
        total_points = len(self.df)
        rule_anomalies = self.df['anomaly_rule'].sum()
        ml_anomalies = self.df['anomaly_ml'].sum() if 'anomaly_ml' in self.df else 0
        shape_anomalies = self.df['shape_anomaly'].sum() if 'shape_anomaly' in self.df else 0
        combined_anomalies = self.df['anomaly_combined'].sum()
        
        print(f"\nTotal trajectory points: {total_points}")
        print(f"Rule-based anomalies: {rule_anomalies} ({rule_anomalies/total_points*100:.2f}%)")
        print(f"ML-based anomalies: {ml_anomalies} ({ml_anomalies/total_points*100:.2f}%)")
        print(f"Shape anomalies: {shape_anomalies} ({shape_anomalies/total_points*100:.2f}%)")
        print(f"Combined anomalies: {combined_anomalies} ({combined_anomalies/total_points*100:.2f}%)")
        
        # Find flights with most anomalies
        anomaly_summary = self.df[self.df['anomaly_combined']].groupby('icao24').agg({
            'anomaly_rule': 'sum',
            'callsign': 'first',
            'timestamp': 'count'
        }).rename(columns={'timestamp': 'anomaly_count'})
        anomaly_summary = anomaly_summary.sort_values('anomaly_count', ascending=False)
        
        print("\nFlights with most anomalies:")
        print(anomaly_summary.head(10))
        
        return self.df
    
    def save_results(self, output_path):
        """Save anomaly detection results"""
        self.df.to_csv(output_path, index=False)
        print(f"\nSaved results to {output_path}")
        
        # Save summary of anomalous flights
        anomalous_flights = self.df[self.df['anomaly_combined']].groupby('icao24').first()
        anomalous_flights.to_csv(output_path.replace('.csv', '_anomalous_flights.csv'))
        print(f"Saved anomalous flights summary")


if __name__ == "__main__":
    # Example usage
    detector = AnomalyDetector("processed_trajectories.csv")
    
    # Run all detection methods
    rule_anomalies = detector.rule_based_detection()
    ml_anomalies = detector.ml_based_detection(contamination=0.05)
    shape_anomalies = detector.detect_trajectory_shape_anomalies(eps_km=100, min_samples=3)
    
    # Combine results
    results = detector.combine_anomaly_results()
    detector.save_results("anomaly_results.csv")