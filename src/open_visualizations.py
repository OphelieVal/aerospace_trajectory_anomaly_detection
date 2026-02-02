"""
Quick Visualization Opener
Opens interactive HTML dashboards in your default web browser
"""

import os
import sys
import webbrowser
from pathlib import Path

def list_available_visualizations():
    """List all available HTML visualizations"""
    output_dir = Path("output")
    
    if not output_dir.exists():
        print("❌ Output directory not found. Have you run the pipeline yet?")
        print("Run: python main_pipeline.py")
        return []
    
    html_files = list(output_dir.glob("*.html"))
    
    if not html_files:
        print("❌ No HTML visualizations found in output/ directory")
        print("Run: python main_pipeline.py")
        return []
    
    return html_files

def open_visualization(filepath):
    """Open a visualization in the default web browser"""
    abs_path = filepath.absolute()
    url = f"file://{abs_path}"
    
    print(f"Opening {filepath.name} in your default browser...")
    webbrowser.open(url)
    print(f"✓ Opened: {url}")

def main():
    print("="*70)
    print("INTERACTIVE VISUALIZATION OPENER")
    print("="*70)
    print()
    
    html_files = list_available_visualizations()
    
    if not html_files:
        sys.exit(1)
    
    print(f"Found {len(html_files)} visualization(s):")
    print()
    
    # Categorize files
    dashboard = None
    animations = []
    
    for f in html_files:
        if f.name == "dashboard.html":
            dashboard = f
        elif "animation" in f.name:
            animations.append(f)
    
    # Show menu
    options = []
    
    if dashboard:
        print("  [1] 🌟 Main Dashboard (dashboard.html)")
        print("      - Interactive map with all flights")
        print("      - Anomaly distribution charts")
        print("      - Clickable flight paths")
        print("      - Comprehensive analysis")
        options.append(('1', dashboard))
        print()
    
    if animations:
        for i, anim_file in enumerate(animations, start=2):
            flight_id = anim_file.stem.replace('flight_', '').replace('_animation', '')
            print(f"  [{i}] 🎬 Flight Animation ({anim_file.name})")
            print(f"      - Animated flight path for {flight_id}")
            print(f"      - Real-time anomaly detection")
            print(f"      - Play/pause controls")
            options.append((str(i), anim_file))
            print()
    
    print(f"  [A] Open ALL visualizations")
    print(f"  [Q] Quit")
    print()
    
    choice = input("Select visualization to open: ").strip().upper()
    
    if choice == 'Q':
        print("Goodbye!")
        sys.exit(0)
    
    if choice == 'A':
        print("\nOpening all visualizations...")
        for _, filepath in options:
            open_visualization(filepath)
        print(f"\n✓ Opened {len(options)} visualization(s)")
        return
    
    # Find and open selected visualization
    for opt_key, filepath in options:
        if choice == opt_key:
            open_visualization(filepath)
            return
    
    print(f"❌ Invalid choice: {choice}")
    sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)