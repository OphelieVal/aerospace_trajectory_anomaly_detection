"""
Main Pipeline
Orchestrates the complete anomaly detection workflow
"""

import sys
from datetime import datetime, timedelta
import argparse

from data_fetcher import OpenSkyDataFetcher
from trajectory_processor import TrajectoryProcessor
from anomaly_detector import AnomalyDetector
from visualizer import TrajectoryVisualizer


def run_full_pipeline(hours_back=24, max_flights=50, contamination=0.05):
    """
    Execute the complete anomaly detection pipeline
    
    Args:
        hours_back: How many hours back to fetch data
        max_flights: Maximum number of flights to analyze
        contamination: Expected proportion of anomalies for ML model
    """
    print("="*70)
    print("AIRCRAFT TRAJECTORY ANOMALY DETECTION SYSTEM")
    print("="*70)
    print()
    
    # Step 1: Data Collection
    print("STEP 1: DATA COLLECTION")
    print("-"*70)
    try:
        fetcher = OpenSkyDataFetcher()
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        print(f"Fetching data from {start_time} to {end_time}")
        trajectories = fetcher.collect_batch_trajectories(
            start_time, 
            end_time, 
            "raw_trajectories.csv",
            max_flights=max_flights
        )
        
        if trajectories is None or len(trajectories) == 0:
            print("ERROR: No data collected. Exiting.")
            return False
        
        print(f"✓ Successfully collected {len(trajectories)} trajectory points")
        print()
    except Exception as e:
        print(f"ERROR in data collection: {e}")
        return False
    
    # Step 2: Data Processing
    print("STEP 2: DATA PROCESSING & FEATURE ENGINEERING")
    print("-"*70)
    try:
        processor = TrajectoryProcessor("raw_trajectories.csv")
        processed_df = (processor
                       .clean_trajectories()
                       .interpolate_missing_data(freq_seconds=10)
                       .engineer_features()
                       .save_processed_data("processed_trajectories.csv"))
        
        print(f"✓ Successfully processed trajectories")
        print(f"  - Flights: {processed_df['icao24'].nunique()}")
        print(f"  - Total points: {len(processed_df)}")
        print()
    except Exception as e:
        print(f"ERROR in data processing: {e}")
        return False
    
    # Step 3: Anomaly Detection
    print("STEP 3: ANOMALY DETECTION")
    print("-"*70)
    try:
        detector = AnomalyDetector("processed_trajectories.csv")
        
        # Run all detection methods
        rule_anomalies = detector.rule_based_detection()
        ml_anomalies = detector.ml_based_detection(contamination=contamination)
        shape_anomalies = detector.detect_trajectory_shape_anomalies(eps_km=100, min_samples=3)
        
        # Combine results
        results = detector.combine_anomaly_results()
        detector.save_results("anomaly_results.csv")
        
        print(f"✓ Successfully detected anomalies")
        print()
    except Exception as e:
        print(f"ERROR in anomaly detection: {e}")
        return False
    
    # Step 4: Visualization
    print("STEP 4: VISUALIZATION")
    print("-"*70)
    try:
        from visualizer import TrajectoryVisualizer
        from interactive_visualizer import InteractiveVisualizer
         
        # Static visualizations
        viz = TrajectoryVisualizer("anomaly_results.csv")
        
        print("Generating trajectory map...")
        viz.plot_trajectory_map(max_flights=5, save_path="output/trajectory_map.png")
        
        print("Generating altitude profiles...")
        viz.plot_altitude_profile(max_flights=3, save_path="output/altitude_profile.png")
        
        print("Generating speed analysis...")
        viz.plot_speed_analysis(save_path="output/speed_analysis.png")
        
        # Plot timeline for most anomalous flight
        anomalous_flights = viz.df[viz.df['anomaly_combined']]['icao24'].unique()
        if len(anomalous_flights) > 0:
            print(f"Generating detailed timeline for flight {anomalous_flights[0]}...")
            viz.plot_anomaly_timeline(anomalous_flights[0], 
                                     save_path="output/anomaly_timeline.png")
        
        # Generate report
        print("Generating anomaly report...")
        viz.generate_anomaly_report('output/anomaly_report.txt')
        
        # Interactive visualizations
        print("\nGenerating interactive visualizations...")
        interactive_viz = InteractiveVisualizer("anomaly_results.csv")
        
        print("Creating interactive dashboard...")
        interactive_viz.generate_interactive_dashboard('output/dashboard.html')
        
        # Generate animation for most anomalous flight
        if len(anomalous_flights) > 0:
            print(f"Creating flight animation for {anomalous_flights[0]}...")
            interactive_viz.generate_flight_animation(anomalous_flights[0], 
                                                     'output/flight_animation.html')
        
        print(f"✓ Successfully generated all visualizations")
        print()
    except Exception as e:
        print(f"ERROR in visualization: {e}")
        import traceback
        traceback.print_exc()
        print(f"Continuing despite visualization error...")
    
    print("="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nOutput files:")
    print("  Static Visualizations:")
    print("    - raw_trajectories.csv: Raw trajectory data from OpenSky")
    print("    - processed_trajectories.csv: Cleaned and processed data with features")
    print("    - anomaly_results.csv: Complete results with anomaly flags")
    print("    - output/trajectory_map.png: Map visualization")
    print("    - output/altitude_profile.png: Altitude profiles")
    print("    - output/speed_analysis.png: Speed distributions")
    print("    - output/anomaly_timeline.png: Detailed timeline for anomalous flight")
    print("    - output/anomaly_report.txt: Text summary of findings")
    print("  Interactive Visualizations:")
    print("    - output/dashboard.html: 🌟 INTERACTIVE DASHBOARD (Open in browser!)")
    print("    - output/flight_animation.html: 🎬 ANIMATED FLIGHT PATH (Open in browser!)")
    print()
    print("💡 TIP: Open the .html files in your web browser for interactive exploration!")
    print()
    
    return True


def analyze_specific_flight(icao24):
    """
    Analyze a specific flight by ICAO24 identifier
    Assumes anomaly_results.csv already exists
    """
    print(f"Analyzing flight: {icao24}")
    
    try:
        from visualizer import TrajectoryVisualizer
        from interactive_visualizer import InteractiveVisualizer
        
        viz = TrajectoryVisualizer("anomaly_results.csv")
        
        # Check if flight exists
        if icao24 not in viz.df['icao24'].values:
            print(f"ERROR: Flight {icao24} not found in results")
            available_flights = viz.df['icao24'].unique()[:10]
            print(f"Available flights: {', '.join(available_flights)}")
            return False
        
        # Generate static visualizations for this flight
        print("Generating static visualizations...")
        viz.plot_trajectory_map(icao24=icao24, save_path=f"output/flight_{icao24}_map.png")
        viz.plot_altitude_profile(icao24=icao24, save_path=f"output/flight_{icao24}_altitude.png")
        viz.plot_anomaly_timeline(icao24, save_path=f"output/flight_{icao24}_timeline.png")
        
        # Generate interactive animation
        print("Generating interactive animation...")
        interactive_viz = InteractiveVisualizer("anomaly_results.csv")
        interactive_viz.generate_flight_animation(icao24, f"output/flight_{icao24}_animation.html")
        
        # Print flight statistics
        flight_data = viz.df[viz.df['icao24'] == icao24]
        print(f"\nFlight Statistics:")
        print(f"  Callsign: {flight_data['callsign'].iloc[0]}")
        print(f"  Total points: {len(flight_data)}")
        print(f"  Anomalous points: {flight_data['anomaly_combined'].sum()}")
        print(f"  Duration: {(flight_data['timestamp'].max() - flight_data['timestamp'].min())}")
        
        if 'anomaly_reason' in flight_data.columns:
            reasons = flight_data[flight_data['anomaly_rule']]['anomaly_reason'].str.split(';').explode()
            reason_counts = reasons.value_counts()
            print(f"\n  Anomaly types:")
            for reason, count in reason_counts.items():
                if reason:
                    print(f"    - {reason}: {count}")
        
        print(f"\n✓ Analysis complete. Output files:")
        print(f"    - output/flight_{icao24}_map.png")
        print(f"    - output/flight_{icao24}_altitude.png")
        print(f"    - output/flight_{icao24}_timeline.png")
        print(f"    - output/flight_{icao24}_animation.html (🎬 Interactive!)")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aircraft Trajectory Anomaly Detection System"
    )
    
    parser.add_argument(
        '--mode',
        choices=['full', 'analyze'],
        default='full',
        help='Run full pipeline or analyze specific flight'
    )
    
    parser.add_argument(
        '--hours',
        type=int,
        default=24,
        help='Hours of historical data to fetch (full mode only)'
    )
    
    parser.add_argument(
        '--max-flights',
        type=int,
        default=50,
        help='Maximum number of flights to analyze (full mode only)'
    )
    
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.05,
        help='Expected anomaly proportion for ML model (0.01-0.2)'
    )
    
    parser.add_argument(
        '--icao24',
        type=str,
        help='ICAO24 identifier for specific flight analysis (analyze mode only)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs('output', exist_ok=True)
    
    if args.mode == 'full':
        success = run_full_pipeline(
            hours_back=args.hours,
            max_flights=args.max_flights,
            contamination=args.contamination
        )
        sys.exit(0 if success else 1)
    
    elif args.mode == 'analyze':
        if not args.icao24:
            print("ERROR: --icao24 required for analyze mode")
            sys.exit(1)
        
        success = analyze_specific_flight(args.icao24)
        sys.exit(0 if success else 1)