"""
Trajectory Processor
Cleans raw trajectory data and engineers features for anomaly detection
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from geopy.distance import geodesic

class TrajectoryProcessor:
    def __init__(self, raw_data_path):
        """Load raw trajectory data"""
        self.df = pd.read_csv(raw_data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='s')
        self.df = self.df.sort_values(['icao24', 'timestamp'])
        
    def clean_trajectories(self):
        """Remove outliers and invalid data points"""
        print("Cleaning trajectories...")
        initial_count = len(self.df)
        
        # Remove invalid coordinates
        self.df = self.df[
            (self.df['latitude'].between(-90, 90)) &
            (self.df['longitude'].between(-180, 180)) &
            (self.df['altitude'] >= 0)
        ]
        
        # Remove trajectories with too few points
        flight_counts = self.df.groupby('icao24').size()
        valid_flights = flight_counts[flight_counts >= 10].index
        self.df = self.df[self.df['icao24'].isin(valid_flights)]
        
        # Remove duplicate timestamps per flight
        self.df = self.df.drop_duplicates(subset=['icao24', 'timestamp'])
        
        print(f"Removed {initial_count - len(self.df)} invalid points")
        print(f"Remaining flights: {self.df['icao24'].nunique()}")
        
        return self
    
    def interpolate_missing_data(self, freq_seconds=10):
        """Interpolate trajectories to uniform time intervals"""
        print(f"Interpolating to {freq_seconds}s intervals...")
        
        interpolated_flights = []
        
        for icao24, group in self.df.groupby('icao24'):
            group = group.sort_values('timestamp')
            
            if len(group) < 3:
                continue
            
            # Create uniform time grid
            start_time = group['timestamp'].min()
            end_time = group['timestamp'].max()
            time_range = (end_time - start_time).total_seconds()
            
            if time_range < freq_seconds:
                continue
            
            new_times = pd.date_range(start_time, end_time, freq=f'{freq_seconds}S')
            
            # Convert to seconds for interpolation
            time_seconds = (group['timestamp'] - start_time).dt.total_seconds().values
            new_time_seconds = (new_times - start_time).total_seconds().values
            
            # Interpolate each dimension
            try:
                interp_lat = interp1d(time_seconds, group['latitude'].values, 
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                interp_lon = interp1d(time_seconds, group['longitude'].values,
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                interp_alt = interp1d(time_seconds, group['altitude'].values,
                                     kind='linear', bounds_error=False, fill_value='extrapolate')
                
                new_df = pd.DataFrame({
                    'icao24': icao24,
                    'callsign': group['callsign'].iloc[0],
                    'timestamp': new_times,
                    'latitude': interp_lat(new_time_seconds),
                    'longitude': interp_lon(new_time_seconds),
                    'altitude': interp_alt(new_time_seconds),
                    'on_ground': False
                })
                
                interpolated_flights.append(new_df)
            except Exception as e:
                print(f"Failed to interpolate {icao24}: {e}")
                continue
        
        if interpolated_flights:
            self.df = pd.concat(interpolated_flights, ignore_index=True)
            print(f"Interpolated {len(interpolated_flights)} flights")
        
        return self
    
    def engineer_features(self):
        """Calculate kinematic features from trajectory data"""
        print("Engineering features...")
        
        features_list = []
        
        for icao24, group in self.df.groupby('icao24'):
            group = group.sort_values('timestamp').reset_index(drop=True)
            
            if len(group) < 2:
                continue
            
            # Time differences (seconds)
            time_diff = group['timestamp'].diff().dt.total_seconds()
            
            # Ground speed calculation using geodesic distance
            distances = []
            for i in range(len(group) - 1):
                pos1 = (group.loc[i, 'latitude'], group.loc[i, 'longitude'])
                pos2 = (group.loc[i+1, 'latitude'], group.loc[i+1, 'longitude'])
                distances.append(geodesic(pos1, pos2).meters)
            distances = [0] + distances  # Pad first value
            
            ground_speed = np.array(distances) / time_diff.fillna(1).values  # m/s
            
            # Vertical speed (m/s)
            altitude_diff = group['altitude'].diff()
            vertical_speed = altitude_diff / time_diff.fillna(1)
            
            # Acceleration (m/s²)
            ground_accel = pd.Series(ground_speed).diff() / time_diff.fillna(1)
            
            # Heading (bearing between consecutive points)
            headings = []
            for i in range(len(group) - 1):
                lat1, lon1 = np.radians(group.loc[i, 'latitude']), np.radians(group.loc[i, 'longitude'])
                lat2, lon2 = np.radians(group.loc[i+1, 'latitude']), np.radians(group.loc[i+1, 'longitude'])
                
                dlon = lon2 - lon1
                x = np.sin(dlon) * np.cos(lat2)
                y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
                heading = np.degrees(np.arctan2(x, y))
                headings.append((heading + 360) % 360)
            headings = [headings[0] if headings else 0] + headings
            
            # Turn rate (degrees/second)
            heading_diff = pd.Series(headings).diff()
            # Handle 360-degree wraparound
            heading_diff = heading_diff.apply(lambda x: x - 360 if x > 180 else (x + 360 if x < -180 else x))
            turn_rate = heading_diff / time_diff.fillna(1)
            
            # Add features to dataframe
            group['ground_speed'] = ground_speed
            group['vertical_speed'] = vertical_speed.fillna(0)
            group['ground_accel'] = ground_accel.fillna(0)
            group['heading'] = headings
            group['turn_rate'] = turn_rate.fillna(0)
            
            # Trajectory statistics
            group['flight_duration'] = (group['timestamp'].max() - group['timestamp'].min()).total_seconds()
            group['max_altitude'] = group['altitude'].max()
            group['altitude_range'] = group['altitude'].max() - group['altitude'].min()
            
            features_list.append(group)
        
        if features_list:
            self.df = pd.concat(features_list, ignore_index=True)
            print(f"Engineered features for {self.df['icao24'].nunique()} flights")
        
        return self
    
    def save_processed_data(self, output_path):
        """Save processed trajectory data with features"""
        self.df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
        return self.df


if __name__ == "__main__":
    # Example usage
    processor = TrajectoryProcessor("raw_trajectories.csv")
    
    processed_df = (processor
                    .clean_trajectories()
                    .interpolate_missing_data(freq_seconds=10)
                    .engineer_features()
                    .save_processed_data("processed_trajectories.csv"))
    
    print("\nFeature statistics:")
    print(processed_df[['ground_speed', 'vertical_speed', 'ground_accel', 
                        'turn_rate', 'altitude']].describe())