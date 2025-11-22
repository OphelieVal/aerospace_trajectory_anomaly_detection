import time as time_module
import requests
import pandas as pd
import os
from datetime import datetime, timezone
from typing import Optional, List
from auth import get_access_token

# ============================================================
#  OpenSky API - Track Retrieval Script
# ============================================================
# Documentation :
# https://opensky-network.org/apidoc/rest.html#operation/getTrackByAircraft
#
# This script downloads the historical trajectory of an aircraft
# using its ICAO24 identifier and a timestamp.
# ============================================================

class OpenSkyClient:
    """
    Wrapper for OpenSky API using OAuth2 access token.
    """

    BASE_URL = "https://opensky-network.org/api/tracks/"

    def __init__(self):
        self.token = None
        self.token_time = 0 # timestamp of last token retrieval
        self.token_ttl = 600 # 10 minutes
        self.headers = {}

    def _ensure_token(self):
        """
        Check if token is present and valid, otherwise get a new one.
        """
        now = time_module.time()
        if not self.token or now - self.token_time > self.token_ttl - 10:
            self.token = get_access_token()
            self.token_time = now
            self.headers = {"Authorization": f"Bearer {self.token}"}
            print("[INFO] New access token acquired.")

    def get_track(self, icao24: str, timestamp: int, retries: int = 3) -> Optional[pd.DataFrame]:
        self._ensure_token()
        endpoint = f"{self.BASE_URL}?icao24={icao24}&time={timestamp}"
        print(f"[INFO] Requesting track for {icao24} at time={timestamp}...")

        for attempt in range(1, retries + 1):
            try:
                response = requests.get(endpoint, headers=self.headers, timeout=10)
                if response.status_code == 401:
                    # Token expired
                    print("[WARNING] Unauthorized. Refreshing token...")
                    self.token = None
                    self._ensure_token()
                    continue

                if response.status_code == 404:
                    print(f"[WARNING] No track found for {icao24} at timestamp {timestamp}")
                    return None

                if response.status_code != 200:
                    print(f"[ERROR] Status {response.status_code}: {response.text}")
                    return None

                data = response.json()
                if "path" not in data or len(data["path"]) == 0:
                    print(f"[WARNING] No trajectory data returned for {icao24}")
                    return None

                df = pd.DataFrame(data["path"])
                # Adjust columns dynamically
                columns_expected = [
                    "time", "latitude", "longitude", "baro_altitude",
                    "heading", "on_ground", "velocity", "vertical_rate", "sensors"
                ]
                df.columns = columns_expected[:df.shape[1]]
                df["icao24"] = icao24

                return df

            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Attempt {attempt}: {e}")
                time_module.sleep(2)

        print(f"[ERROR] Failed to retrieve track for {icao24} after {retries} attempts.")
        return None

    def get_tracks_batch(self, icao24_list: List[str], timestamps: List[int]) -> List[pd.DataFrame]:
        """
        Retrieve multiple tracks for multiple ICAO24 and timestamps.
        Returns a list of DataFrames.
        """
        results = []
        for icao in icao24_list:
            for ts in timestamps:
                df = self.get_track(icao, ts)
                if df is not None:
                    results.append(df)
        return results


def save_track(df: pd.DataFrame, directory: str = "../data/raw"):
    """
    Save DataFrame to CSV in specified directory.
    """
    os.makedirs(directory, exist_ok=True)
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    icao24 = df["icao24"].iloc[0] if not df.empty else "unknown"
    filepath = os.path.join(directory, f"{icao24}_track_{timestamp_str}.csv")
    df.to_csv(filepath, index=False)
    print(f"[INFO] Track saved to {filepath}")


# ===========================
# Example usage
# ===========================
if __name__ == "__main__":
    client = OpenSkyClient()

    icao24_list = ["a24f71", "3c4b26"]  # multiple aircraft
    timestamps = [int(datetime.now(timezone.utc).timestamp()) - 3600]  # 1h ago

    tracks = client.get_tracks_batch(icao24_list, timestamps)
    for df in tracks:
        save_track(df)