"""
OpenSky Network Data Fetcher
Fetches historical flight trajectory data from OpenSky Network API
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from auth import get_access_token

class OpenSkyDataFetcher:
    def __init__(self):
        self.base_url = "https://opensky-network.org/api"
        self.token = get_access_token()
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    def fetch_flights_in_timerange(self, begin_timestamp, end_timestamp):
        """
        Fetch all flights that departed in the given time range
        Times should be Unix timestamps
        """
        url = f"{self.base_url}/flights/all"
        params = {
            "begin": int(begin_timestamp),
            "end": int(end_timestamp)
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching flights: {e}")
            return []
    
    def fetch_flight_track(self, icao24, time):
        """
        Fetch complete trajectory for a specific aircraft
        icao24: unique aircraft identifier
        time: Unix timestamp when aircraft was airborne
        """
        url = f"{self.base_url}/tracks/all"
        params = {
            "icao24": icao24,
            "time": int(time)
        }
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching track for {icao24}: {e}")
            return None
    
    def collect_batch_trajectories(self, start_date, end_date, output_file, max_flights=100):
        """
        Collect trajectories for multiple flights in a date range
        start_date, end_date: datetime objects
        output_file: path to save CSV
        max_flights: maximum number of flights to fetch
        """
        begin_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        
        print(f"Fetching flights between {start_date} and {end_date}...")
        flights = self.fetch_flights_in_timerange(begin_ts, end_ts)
        
        if not flights:
            print("No flights found in time range")
            return None
        
        print(f"Found {len(flights)} flights. Fetching trajectories...")
        
        all_trajectories = []
        flight_count = 0
        
        for flight in flights[:max_flights]:
            if flight_count >= max_flights:
                break
                
            icao24 = flight.get('icao24')
            callsign = flight.get('callsign', '').strip()
            departure_time = flight.get('firstSeen')
            
            if not icao24 or not departure_time:
                continue
            
            print(f"Fetching trajectory {flight_count+1}/{max_flights}: {callsign} ({icao24})")
            
            track_data = self.fetch_flight_track(icao24, departure_time)
            
            if track_data and 'path' in track_data:
                trajectory_df = self._parse_trajectory(track_data, icao24, callsign)
                if trajectory_df is not None and len(trajectory_df) > 0:
                    all_trajectories.append(trajectory_df)
                    flight_count += 1
            
            # Rate limiting - be nice to the API
            time.sleep(1)
        
        if all_trajectories:
            combined_df = pd.concat(all_trajectories, ignore_index=True)
            combined_df.to_csv(output_file, index=False)
            print(f"Saved {len(all_trajectories)} trajectories to {output_file}")
            return combined_df
        else:
            print("No trajectories collected")
            return None
    
    def _parse_trajectory(self, track_data, icao24, callsign):
        """Parse track data into a pandas DataFrame"""
        if not track_data or 'path' not in track_data:
            return None
        
        path = track_data['path']
        
        records = []
        for point in path:
            records.append({
                'icao24': icao24,
                'callsign': callsign,
                'timestamp': point[0],
                'latitude': point[1],
                'longitude': point[2],
                'altitude': point[3],  # meters (barometric altitude)
                'on_ground': point[4]
            })
        
        df = pd.DataFrame(records)
        df = df.dropna(subset=['latitude', 'longitude'])
        
        return df


if __name__ == "__main__":
    # Example usage: fetch last 24 hours of data
    fetcher = OpenSkyDataFetcher()
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=24)
    
    # Fetch 50 flights for testing
    trajectories = fetcher.collect_batch_trajectories(
        start_time, 
        end_time, 
        "raw_trajectories.csv",
        max_flights=50
    )
    
    if trajectories is not None:
        print(f"\nCollected {len(trajectories)} trajectory points")
        print(f"Unique flights: {trajectories['icao24'].nunique()}")
        print("\nSample data:")
        print(trajectories.head(10))