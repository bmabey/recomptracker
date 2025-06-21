#!/usr/bin/env python3
"""
RecompTracker - Main CLI Script

This is the main entry point for RecompTracker body composition analysis. It provides
a comprehensive command-line interface with helpful error messages and
delegates the core analysis logic to the core module.
"""

import argparse
import os

from core import run_analysis


def main():
    """Main CLI function with comprehensive argument parsing."""
    parser = argparse.ArgumentParser(
        description="RecompTracker with Intelligent TLM Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py                           # Use example_config.json
  python run_analysis.py my_config.json           # Use custom config
  python run_analysis.py --config my_config.json  # Alternative syntax
  python run_analysis.py --suggest-goals          # Generate suggested goals

JSON config format:
  {
    "user_info": {
      "birth_date": "04/26/1982",
      "height_in": 66.0,
      "gender": "male",
      "training_level": "intermediate"  // optional: novice/intermediate/advanced
    },
    "scan_history": [
      {
        "date": "04/07/2022",
        "total_weight_lbs": 143.2,
        "total_lean_mass_lbs": 106.3,
        "fat_mass_lbs": 31.4,
        "body_fat_percentage": 22.8,
        "arms_lean_lbs": 12.4,
        "legs_lean_lbs": 37.3
      }
    ],
    "goals": {
      "almi": {
        "target_percentile": 0.90,
        "target_age": 45.0,        // or "?" for auto-calculated timeframe
        "target_body_fat_percentage": 20.0,  // optional: defaults to last scan's BF%
        "suggested": true,         // optional: enables suggested goal logic
        "description": "optional"
      },
      "ffmi": {
        "target_percentile": 0.85,
        "target_age": "?",         // auto-calculate based on progression rates
        "target_body_fat_percentage": 18.0,  // optional: defaults to last scan's BF%
        "description": "optional"
      }
    }
  }

Notes:
  - Goals section is optional (scan history analysis only if omitted)
  - Either almi or ffmi goals can be specified independently
  - training_level: If not specified, automatically detected from scan progression
  - target_age: Use "?" or null for automatic timeframe calculation
  - Suggested goals use conservative progression rates based on demographics and training level
        """,
    )

    parser.add_argument(
        "config_file",
        nargs="?",
        default="example_config.json",
        help="Path to JSON configuration file (default: example_config.json)",
    )

    parser.add_argument(
        "--config",
        "-c",
        dest="config_file_alt",
        help="Alternative way to specify config file path",
    )

    parser.add_argument(
        "--suggest-goals",
        "-s",
        action="store_true",
        help="Generate suggested goals for reaching 90th percentile with realistic timeframes",
    )

    parser.add_argument(
        "--target-percentile",
        "-p",
        type=float,
        default=0.90,
        help="Target percentile for suggested goals (default: 0.90 for 90th percentile)",
    )

    parser.add_argument(
        "--help-config",
        action="store_true",
        help="Show detailed help about the JSON configuration format",
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
        if config_file == "example_config.json":
            print("The example configuration file is missing.")
            print("Please ensure example_config.json exists in the current directory.")
        else:
            print("Please check the file path and try again.")

        print()
        print("Run with --help-config to see the expected JSON format.")
        return 1

    # Run the analysis using core module
    try:
        exit_code = run_analysis(
            config_path=config_file,
            suggest_goals=args.suggest_goals,
            target_percentile=args.target_percentile,
        )

        if exit_code == 0:
            print()
            print("Analysis completed successfully!")
            print("Generated files:")
            print("  - almi_plot.png        (ALMI percentile curves)")
            print("  - ffmi_plot.png        (FFMI percentile curves)")
            print("  - bf_plot.png          (Body fat percentage over time)")
            print("  - almi_stats_table.csv (Comprehensive data table)")

        return exit_code

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

goals (optional):
  - almi: Optional ALMI goal specification
    - target_percentile: Target percentile as decimal (0.0-1.0)
    - target_age: Target age for reaching the goal (18-120)
    - description: Optional description
  - ffmi: Optional FFMI goal specification
    - target_percentile: Target percentile as decimal (0.0-1.0)
    - target_age: Target age for reaching the goal (18-120)
    - description: Optional description

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
  "goals": {
    "almi": {
      "target_percentile": 0.90,
      "target_age": 45.0,
      "description": "Reach 90th percentile ALMI by age 45"
    },
    "ffmi": {
      "target_percentile": 0.85,
      "target_age": 50.0,
      "description": "Reach 85th percentile FFMI by age 50"
    }
  }
}

Notes:
- Goals section is entirely optional - analysis can run with DEXA scan history only
- ALMI and FFMI goals can be specified independently (either or both)
    """
    print(help_text)


if __name__ == "__main__":
    exit(main())
