#!/usr/bin/env python3
"""
CLI Wrapper for DEXA Body Composition Analysis

This is a simplified wrapper around zscore_plot.py that provides a cleaner
command-line interface and helpful error messages.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path


def main():
    """Main CLI wrapper function."""
    parser = argparse.ArgumentParser(
        description='DEXA Body Composition Analysis - CLI Wrapper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py                           # Use example_config.json
  python run_analysis.py my_config.json           # Use custom config
  python run_analysis.py --config my_config.json  # Alternative syntax

The config file should be a JSON file with user info, scan history, and goals.
Run with --help-config to see the expected JSON format.
        """
    )
    
    parser.add_argument(
        'config_file',
        nargs='?',
        default='example_config.json',
        help='Path to JSON configuration file (default: example_config.json)'
    )
    
    parser.add_argument(
        '--config', '-c',
        dest='config_file_alt',
        help='Alternative way to specify config file path'
    )
    
    parser.add_argument(
        '--help-config',
        action='store_true',
        help='Show detailed help about the JSON configuration format'
    )
    
    args = parser.parse_args()
    
    if args.help_config:
        show_config_help()
        return 0
    
    # Determine which config file to use
    config_file = args.config_file_alt if args.config_file_alt else args.config_file
    
    # Validate config file exists
    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found: {config_file}")
        print()
        
        # Suggest creating example config if it doesn't exist
        if config_file == 'example_config.json':
            print("The example configuration file is missing.")
            print("Please ensure example_config.json exists in the current directory.")
        else:
            print("Please check the file path and try again.")
            
        print()
        print("Run with --help-config to see the expected JSON format.")
        return 1
    
    # Run the main analysis script
    try:
        print(f"Running DEXA analysis with config: {config_file}")
        print("=" * 60)
        
        result = subprocess.run([
            sys.executable, 'zscore_plot.py', 
            '--config', config_file
        ], check=True)
        
        print()
        print("Analysis completed successfully!")
        print("Generated files:")
        print("  - almi_plot.png        (ALMI percentile curves)")
        print("  - ffmi_plot.png        (FFMI percentile curves)")
        print("  - almi_stats_table.csv (Comprehensive data table)")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\nAnalysis failed with exit code: {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("Error: zscore_plot.py not found in current directory.")
        print("Please ensure you're running this script from the correct directory.")
        return 1
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        return 1


def show_config_help():
    """Show detailed help about the JSON configuration format."""
    help_text = """
JSON Configuration Format
=========================

The configuration file should be a JSON file with the following structure:

{
  "user_info": {
    "birth_date": "MM/DD/YYYY",
    "height_in": <height in inches>,
    "gender": "<male|female|m|f>"
  },
  "scan_history": [
    {
      "date": "MM/DD/YYYY",
      "total_lean_mass_lbs": <total lean mass in pounds>,
      "arms_lean_lbs": <arms lean mass in pounds>,
      "legs_lean_lbs": <legs lean mass in pounds>
    }
  ],
  "goal": {
    "target_percentile": <percentile as decimal (0.0-1.0)>,
    "target_age": <target age>,
    "description": "<optional description>"
  }
}

Field Descriptions:
------------------

user_info:
  - birth_date: Birth date in MM/DD/YYYY format
  - height_in: Height in inches (12-120)
  - gender: Gender as "male", "female", "m", or "f" (case insensitive)

scan_history:
  - Array of DEXA scan results (at least 1 required)
  - date: Scan date in MM/DD/YYYY format
  - total_lean_mass_lbs: Total lean mass in pounds (≥0)
  - arms_lean_lbs: Arms lean mass in pounds (≥0) 
  - legs_lean_lbs: Legs lean mass in pounds (≥0)

goal:
  - target_percentile: Target percentile as decimal (0.0 = 0%, 1.0 = 100%)
  - target_age: Target age for reaching the goal (18-120)
  - description: Optional description of the goal

Example:
--------
{
  "user_info": {
    "birth_date": "04/26/1982",
    "height_in": 66.0,
    "gender": "male"
  },
  "scan_history": [
    {
      "date": "04/07/2022",
      "total_lean_mass_lbs": 106.3,
      "arms_lean_lbs": 12.4,
      "legs_lean_lbs": 37.3
    },
    {
      "date": "11/25/2024",
      "total_lean_mass_lbs": 129.6,
      "arms_lean_lbs": 17.8,
      "legs_lean_lbs": 40.5
    }
  ],
  "goal": {
    "target_percentile": 0.90,
    "target_age": 45.0,
    "description": "Reach 90th percentile ALMI by age 45"
  }
}
    """
    print(help_text)


if __name__ == '__main__':
    exit(main())