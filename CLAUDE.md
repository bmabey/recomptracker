# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python-based RecompTracker tool that calculates Z-scores and percentiles for body composition metrics (ALMI - Appendicular Lean Mass Index, FFMI - Fat-Free Mass Index) using LMS reference values from the LEAD cohort. The system processes DEXA scan history data with actual body fat percentages and generates comprehensive visualizations with percentile curves, change tracking, and actionable goal analysis.

## Common Commands

```bash
# Setup project (recommended)
task setup

# Launch web application
task webapp
task webapp-dev  # Development mode with extra features

# Run analysis
task run                           # With example config
task run-config -- my_config.json # With custom config
task help-config                   # Show configuration help

# Code Quality (IMPORTANT: Run these before committing!)
task fix                           # Auto-fix linting and formatting issues
task lint                          # Check linting and formatting
task typecheck                     # Run type checking with mypy
task ci                            # Run CI pipeline locally (lint + unit tests)
task ci-full                       # Full CI pipeline (lint + all tests)
task ci-strict                     # Strict CI (lint + typecheck + all tests)

# Testing
task test                          # Run all tests
task test-unit                     # Unit tests only
task test-integration              # Integration tests only
task test-webapp                   # Webapp integration tests only

# Dependencies
task sync                          # Sync dependencies
task add -- package-name          # Add dependency
task add-dev -- package-name      # Add dev dependency

# Maintenance
task clean                         # Clean generated files (*.png, *.csv)
task clean-env                     # Remove virtual environments

# Manual commands (if not using task runner)
uv sync                            # Setup with uv
uv run python run_analysis.py     # Run analysis
uv run streamlit run webapp.py    # Launch webapp
uv run python -m pytest tests/    # Run tests
```

## Development Workflow

**CRITICAL**: Always run code quality checks before committing:

1. **Make changes** to code
2. **Auto-fix issues**: `task fix` 
3. **Verify quality**: `task lint`
4. **Run tests**: `task test-unit` (at minimum)
5. **Full CI check**: `task ci-full` (recommended)
6. **Commit changes** only after all checks pass

For major changes, use `task ci-strict` to include type checking.

## Architecture

### Core Components

1. **run_analysis.py**: Main CLI entry point that provides comprehensive argument parsing and delegates to core module
2. **core.py**: Core analysis engine containing:
   - Core calculation logic (age, Z-scores, T-scores, inverse Z-scores)
   - Data processing and orchestration (TLM estimation, LMS data loading)
   - Plotting logic (percentile curves, visualizations, CSV export)
   - Main analysis orchestration function
3. **tests/**: Comprehensive test suite covering all calculation functions

### Data Flow

- Input: JSON config files containing user info, DEXA scan history, and optional goals
- Processing: Intelligent TLM (Total Lean Mass) estimation using personalized ALM/TLM ratios
- Reference: LMS curves from `data/` directory (adults_LMS_*.csv files for different metrics and genders)
- Output: PNG plots (almi_plot.png, ffmi_plot.png) and CSV data export (almi_stats_table.csv)

### Key Features

- **Accurate Body Fat Calculation**: Uses actual body fat percentages instead of calculated estimates, accounting for bone mass
- **Comprehensive Change Tracking**: Calculates changes since last scan and since first scan for all body composition metrics
- **Intelligent TLM Estimation**: Uses personalized ALM/TLM ratios when multiple scans are available, falls back to population-based ratios for single scans
- **Goal System**: Supports separate ALMI and FFMI goals with target percentiles and ages
- **Actionable Goal Deltas**: Shows exactly what changes are needed to reach goals (weight, lean mass, fat mass, percentiles)
- **Suggested Goals**: Automatic goal calculation based on training level detection from scan progression
- **Enhanced Output Table**: 20+ columns showing body composition values, changes, and progress tracking
- **Comprehensive Testing**: Full test coverage including body fat accuracy validation against ground truth body composition data

## Configuration Format

The system uses JSON configuration files with this structure:
- `user_info`: birth_date (MM/DD/YYYY), height_in, gender, training_level (optional)
- `scan_history`: Array of DEXA scans with **required fields**:
  - `date`, `total_weight_lbs`, `total_lean_mass_lbs`, `fat_mass_lbs`, `body_fat_percentage`
  - `arms_lean_lbs`, `legs_lean_lbs`
- `goals` (optional): Separate almi and ffmi goal specifications with target_percentile and target_age

**Note**: The system requires actual body fat percentages and fat mass values. Old configurations missing these fields will fail validation with clear error messages.

## Testing

Run the test suite with `task test` (recommended) or `uv run python -m pytest tests/`. Tests cover:
- Core mathematical calculations (Z-scores, T-scores, age calculations)
- Body fat percentage accuracy validation against ground truth body composition data
- TLM estimation algorithms
- Config parsing and validation with new required fields
- Suggested goal calculation logic
- Integration scenarios and goal processing

**Key Test Classes:**
- `TestBodyFatPercentageAccuracy`: Validates actual body fat percentages (22.8%, 18.5%, 20.9%, 11.1%, 11.9%)
- `TestBodyCompCalculations`: Core mathematical function validation
- `TestGoalProcessingIntegration`: End-to-end goal processing with change tracking

## Output Features

### Enhanced Data Table
The system generates a comprehensive table with 20+ columns:

**Basic Metrics:** Date, Age, Weight, Lean Mass, Fat Mass, Body Fat %, ALMI, FFMI

**Change Tracking:**
- **Since Last Scan (ΔX_L):** Changes between consecutive scans with +/- indicators
- **Since First Scan (ΔX_F):** Cumulative changes from baseline
- **Z-Score Changes:** Performance improvements in ALMI/FFMI percentiles

**Goal Rows:** Show target values and actionable deltas:
- Target body composition (weight, lean, fat, BF%)
- Changes needed from current state (e.g., -2.4 lbs weight, +2.8 lbs lean)
- Performance improvements required (e.g., +7.57 percentile points)

### CSV Export
Complete data export (`almi_stats_table.csv`) includes all calculated metrics, changes, and goal targets for detailed analysis and tracking.

## Data Dependencies

Requires LMS reference data files in `data/` directory:
- `adults_LMS_appendicular_LMI_gender{0,1}.csv` for ALMI calculations
- `adults_LMS_LMI_gender{0,1}.csv` for FFMI calculations
- Gender codes: 0=male, 1=female