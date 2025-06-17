# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python-based DEXA body composition analysis tool that calculates Z-scores and percentiles for body composition metrics (ALMI - Appendicular Lean Mass Index, FFMI - Fat-Free Mass Index) using LMS reference values from the LEAD cohort. The system processes scan history data and generates comprehensive visualizations with percentile curves and goal tracking.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run main analysis with default config
python run_analysis.py

# Run analysis with custom config
python run_analysis.py my_config.json
python run_analysis.py --config my_config.json

# Show configuration help
python run_analysis.py --help-config

# Run comprehensive test suite
python test_zscore_calculations.py

# Generate Z-scores for Excel data (legacy scientific app)
python scientific_zscore_app.py --filename="example_file.xlsx"

# Direct analysis script (bypassing CLI wrapper)
python zscore_plot.py --config example_config.json
```

## Architecture

### Core Components

1. **run_analysis.py**: User-friendly CLI wrapper that provides helpful error messages and delegates to zscore_plot.py
2. **zscore_plot.py**: Main analysis engine with four key sections:
   - Core calculation logic (age, Z-scores, T-scores, inverse Z-scores)
   - Data processing and orchestration (TLM estimation, LMS data loading)
   - Plotting logic (percentile curves, visualizations, CSV export)
   - Main execution orchestration
3. **scientific_zscore_app.py**: Legacy Excel-based Z-score calculator for batch processing
4. **test_zscore_calculations.py**: Comprehensive test suite covering all calculation functions

### Data Flow

- Input: JSON config files containing user info, scan history, and optional goals
- Processing: Intelligent TLM (Total Lean Mass) estimation using personalized ALM/TLM ratios
- Reference: LMS curves from `data/` directory (adults_LMS_*.csv files for different metrics and genders)
- Output: PNG plots (almi_plot.png, ffmi_plot.png) and CSV data export (almi_stats_table.csv)

### Key Features

- **Intelligent TLM Estimation**: Uses personalized ALM/TLM ratios when multiple scans are available, falls back to population-based ratios for single scans
- **Goal System**: Supports separate ALMI and FFMI goals with target percentiles and ages
- **Suggested Goals**: Automatic goal calculation based on training level detection from scan progression
- **Comprehensive Testing**: Full test coverage for mathematical functions and data processing

## Configuration Format

The system uses JSON configuration files with this structure:
- `user_info`: birth_date (MM/DD/YYYY), height_in, gender
- `scan_history`: Array of DEXA scans with date, total_lean_mass_lbs, arms_lean_lbs, legs_lean_lbs
- `goals` (optional): Separate almi and ffmi goal specifications with target_percentile and target_age

## Testing

Run the test suite with `python test_zscore_calculations.py`. Tests cover:
- Core mathematical calculations (Z-scores, T-scores, age calculations)
- TLM estimation algorithms
- Config parsing and validation
- Suggested goal calculation logic
- Integration scenarios

## Data Dependencies

Requires LMS reference data files in `data/` directory:
- `adults_LMS_appendicular_LMI_gender{0,1}.csv` for ALMI calculations
- `adults_LMS_LMI_gender{0,1}.csv` for FFMI calculations
- Gender codes: 0=male, 1=female