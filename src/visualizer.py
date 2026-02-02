"""
Trajectory Visualizer
Creates plots and maps to visualize trajectories and detected anomalies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

class TrajectoryVisualizer:
    def __init__(self, results_path):
        """Load anomaly detection results"""
        self.df = pd.read_csv(results_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
    
    def plot_trajectory_map(self, icao24=None, max_flights=10, save_path=None):
        """
        Plot trajectories on a 2D map
        If icao24 is provided, plot that specific flight
        Otherwise plot up to max_flights
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        if icao24:
            flights_to_plot = [icao24]
        else:
            flights_to_plot = self.df['icao24'].unique()[:max_flights]
        
        # Left plot: All trajectories
        for flight_id in flights_to_plot:
            flight_data = self.df[self.df['icao24'] == flight_id]
            ax1.plot(flight_data['longitude'], flight_data['latitude'], 
                    linewidth=2, alpha=0.6, label=flight_data['callsign'].iloc[0])
        
        ax1.set_xlabel('Longitude', fontsize=12)
        ax1.set_ylabel('Latitude', fontsize=12)
        ax1.set_title('Aircraft Trajectories', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=8)
        
        # Right plot: Trajectories with anomalies highlighted
        for flight_id in flights_to_plot:
            flight_data = self.df[self.df['icao24'] == flight_id]
            normal_data = flight_data[~flight_data['anomaly_combined']]
            anomaly_data = flight_data[flight_data['anomaly_combined']]
            
            # Plot normal points
            ax2.plot(normal_data['longitude'], normal_data['latitude'], 
                    linewidth=2, alpha=0.6, color='blue')
            
            # Highlight anomalies
            if len(anomaly_data) > 0:
                ax2.scatter(anomaly_data['longitude'], anomaly_data['latitude'], 
                           color='red', s=50, alpha=0.8, zorder=5, marker='x')
        
        ax2.set_xlabel('Longitude', fontsize=12)
        ax2.set_ylabel('Latitude', fontsize=12)
        ax2.set_title('Trajectories with Anomalies (Red X)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved trajectory map to {save_path}")
        
        plt.show()
    
    def plot_altitude_profile(self, icao24=None, max_flights=5, save_path=None):
        """Plot altitude profiles over time"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        if icao24:
            flights_to_plot = [icao24]
        else:
            flights_to_plot = self.df['icao24'].unique()[:max_flights]
        
        for flight_id in flights_to_plot:
            flight_data = self.df[self.df['icao24'] == flight_id].sort_values('timestamp')
            
            # Normalize time to start from 0
            time_elapsed = (flight_data['timestamp'] - flight_data['timestamp'].min()).dt.total_seconds() / 60
            
            # Top plot: Altitude
            axes[0].plot(time_elapsed, flight_data['altitude'], 
                        linewidth=2, alpha=0.7, label=flight_data['callsign'].iloc[0])
            
            # Highlight anomalies
            anomalies = flight_data[flight_data['anomaly_combined']]
            if len(anomalies) > 0:
                time_elapsed_anom = (anomalies['timestamp'] - flight_data['timestamp'].min()).dt.total_seconds() / 60
                axes[0].scatter(time_elapsed_anom, anomalies['altitude'], 
                              color='red', s=100, alpha=0.8, zorder=5, marker='x')
            
            # Bottom plot: Vertical speed
            axes[1].plot(time_elapsed, flight_data['vertical_speed'], 
                        linewidth=2, alpha=0.7, label=flight_data['callsign'].iloc[0])
        
        axes[0].set_ylabel('Altitude (m)', fontsize=12)
        axes[0].set_title('Altitude Profile', fontsize=14, fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Time (minutes)', fontsize=12)
        axes[1].set_ylabel('Vertical Speed (m/s)', fontsize=12)
        axes[1].set_title('Vertical Speed', fontsize=14, fontweight='bold')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved altitude profile to {save_path}")
        
        plt.show()
    
    def plot_speed_analysis(self, save_path=None):
        """Plot speed distributions and anomalies"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Ground speed distribution
        normal_data = self.df[~self.df['anomaly_combined']]
        anomaly_data = self.df[self.df['anomaly_combined']]
        
        axes[0, 0].hist(normal_data['ground_speed'], bins=50, alpha=0.7, 
                       color='blue', label='Normal', edgecolor='black')
        axes[0, 0].hist(anomaly_data['ground_speed'], bins=50, alpha=0.7, 
                       color='red', label='Anomaly', edgecolor='black')
        axes[0, 0].set_xlabel('Ground Speed (m/s)', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Ground Speed Distribution', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Vertical speed distribution
        axes[0, 1].hist(normal_data['vertical_speed'], bins=50, alpha=0.7, 
                       color='blue', label='Normal', edgecolor='black')
        axes[0, 1].hist(anomaly_data['vertical_speed'], bins=50, alpha=0.7, 
                       color='red', label='Anomaly', edgecolor='black')
        axes[0, 1].set_xlabel('Vertical Speed (m/s)', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Vertical Speed Distribution', fontsize=13, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Acceleration distribution
        axes[1, 0].hist(normal_data['ground_accel'].clip(-10, 10), bins=50, 
                       alpha=0.7, color='blue', label='Normal', edgecolor='black')
        axes[1, 0].hist(anomaly_data['ground_accel'].clip(-10, 10), bins=50, 
                       alpha=0.7, color='red', label='Anomaly', edgecolor='black')
        axes[1, 0].set_xlabel('Acceleration (m/s²)', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].set_title('Acceleration Distribution', fontsize=13, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Turn rate distribution
        axes[1, 1].hist(normal_data['turn_rate'].clip(-20, 20), bins=50, 
                       alpha=0.7, color='blue', label='Normal', edgecolor='black')
        axes[1, 1].hist(anomaly_data['turn_rate'].clip(-20, 20), bins=50, 
                       alpha=0.7, color='red', label='Anomaly', edgecolor='black')
        axes[1, 1].set_xlabel('Turn Rate (deg/s)', fontsize=11)
        axes[1, 1].set_ylabel('Frequency', fontsize=11)
        axes[1, 1].set_title('Turn Rate Distribution', fontsize=13, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved speed analysis to {save_path}")
        
        plt.show()
    
    def plot_anomaly_timeline(self, icao24, save_path=None):
        """Plot detailed timeline for a specific flight showing all features and anomalies"""
        flight_data = self.df[self.df['icao24'] == icao24].sort_values('timestamp')
        
        if len(flight_data) == 0:
            print(f"No data found for flight {icao24}")
            return
        
        fig, axes = plt.subplots(5, 1, figsize=(16, 14))
        
        time_elapsed = (flight_data['timestamp'] - flight_data['timestamp'].min()).dt.total_seconds() / 60
        anomaly_times = time_elapsed[flight_data['anomaly_combined']].values
        
        # Helper function to shade anomaly regions
        def shade_anomalies(ax):
            for t in anomaly_times:
                ax.axvspan(t-0.5, t+0.5, alpha=0.2, color='red')
        
        # Altitude
        axes[0].plot(time_elapsed, flight_data['altitude'], linewidth=2, color='blue')
        shade_anomalies(axes[0])
        axes[0].set_ylabel('Altitude (m)', fontsize=11)
        axes[0].set_title(f'Flight {flight_data["callsign"].iloc[0]} ({icao24}) - Anomaly Timeline', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Ground speed
        axes[1].plot(time_elapsed, flight_data['ground_speed'], linewidth=2, color='green')
        shade_anomalies(axes[1])
        axes[1].set_ylabel('Ground Speed (m/s)', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Vertical speed
        axes[2].plot(time_elapsed, flight_data['vertical_speed'], linewidth=2, color='orange')
        shade_anomalies(axes[2])
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[2].set_ylabel('Vert Speed (m/s)', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        # Acceleration
        axes[3].plot(time_elapsed, flight_data['ground_accel'], linewidth=2, color='purple')
        shade_anomalies(axes[3])
        axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[3].set_ylabel('Acceleration (m/s²)', fontsize=11)
        axes[3].grid(True, alpha=0.3)
        
        # Turn rate
        axes[4].plot(time_elapsed, flight_data['turn_rate'], linewidth=2, color='brown')
        shade_anomalies(axes[4])
        axes[4].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[4].set_xlabel('Time (minutes)', fontsize=11)
        axes[4].set_ylabel('Turn Rate (deg/s)', fontsize=11)
        axes[4].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved timeline to {save_path}")
        
        plt.show()
    
    def generate_anomaly_report(self, output_path='anomaly_report.txt'):
        """Generate text report of anomalies"""
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("AIRCRAFT TRAJECTORY ANOMALY DETECTION REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Overall statistics
            f.write(f"Total trajectory points analyzed: {len(self.df)}\n")
            f.write(f"Total flights: {self.df['icao24'].nunique()}\n")
            f.write(f"Anomalous points: {self.df['anomaly_combined'].sum()}\n")
            f.write(f"Anomaly rate: {self.df['anomaly_combined'].mean()*100:.2f}%\n\n")
            
            # Anomaly types breakdown
            if 'anomaly_reason' in self.df.columns:
                f.write("Anomaly Types:\n")
                f.write("-"*70 + "\n")
                reasons = self.df[self.df['anomaly_rule']]['anomaly_reason'].str.split(';', expand=True).stack()
                reason_counts = reasons.value_counts()
                for reason, count in reason_counts.items():
                    if reason:
                        f.write(f"  {reason}: {count} occurrences\n")
                f.write("\n")
            
            # Most anomalous flights
            f.write("Top 10 Flights with Most Anomalies:\n")
            f.write("-"*70 + "\n")
            anomaly_counts = self.df[self.df['anomaly_combined']].groupby(['icao24', 'callsign']).size()
            anomaly_counts = anomaly_counts.sort_values(ascending=False).head(10)
            for (icao24, callsign), count in anomaly_counts.items():
                f.write(f"  {callsign} ({icao24}): {count} anomalous points\n")
        
        print(f"Generated anomaly report: {output_path}")


if __name__ == "__main__":
    # Example usage
    viz = TrajectoryVisualizer("anomaly_results.csv")
    
    # Generate various visualizations
    viz.plot_trajectory_map(max_flights=5, save_path="trajectory_map.png")
    viz.plot_altitude_profile(max_flights=3, save_path="altitude_profile.png")
    viz.plot_speed_analysis(save_path="speed_analysis.png")
    
    # Plot detailed timeline for first anomalous flight
    anomalous_flights = viz.df[viz.df['anomaly_combined']]['icao24'].unique()
    if len(anomalous_flights) > 0:
        viz.plot_anomaly_timeline(anomalous_flights[0], save_path="anomaly_timeline.png")
    
    # Generate text report
    viz.generate_anomaly_report()