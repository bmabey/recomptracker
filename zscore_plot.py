#!/usr/bin/env python3
"""
DEXA Body Composition Analysis - Legacy Wrapper

This is a legacy wrapper that maintains backward compatibility with the
original zscore_plot.py interface. All core logic has been moved to core.py.

For new usage, prefer run_analysis.py which provides a cleaner interface.
"""

import argparse
from core import run_analysis


def main():
    """Legacy main function for backward compatibility."""
    parser = argparse.ArgumentParser(
        description='DEXA Body Composition Analysis with Intelligent TLM Estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python zscore_plot.py --config example_config.json
  python zscore_plot.py  # Uses example_config.json by default

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
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='example_config.json',
        help='Path to JSON configuration file (default: example_config.json)'
    )
    
    parser.add_argument(
        '--suggest-goals', '-s',
        action='store_true',
        help='Generate suggested goals for reaching 90th percentile with realistic timeframes'
    )
    
    parser.add_argument(
        '--target-percentile', '-p',
        type=float,
        default=0.90,
        help='Target percentile for suggested goals (default: 0.90 for 90th percentile)'
    )
    
    args = parser.parse_args()
    
    # Delegate to core analysis function
    return run_analysis(
        config_path=args.config,
        suggest_goals=args.suggest_goals,
        target_percentile=args.target_percentile
    )


if __name__ == '__main__':
    exit(main())