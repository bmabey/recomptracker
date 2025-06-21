# RecompTracker

A Python tool that calculates Z-scores and percentiles for body composition metrics (ALMI, FFMI) using LMS reference values from the LEAD cohort. The system processes DEXA scan history with actual body fat percentages and generates comprehensive visualizations with percentile curves, change tracking, and actionable goal analysis.

## Operationalizing Peter Attia's Medicine 3.0 Philosophy

This tool directly implements Dr. Peter Attia's revolutionary **Medicine 3.0** approach to longevity and healthspan optimization. Rather than accepting "normal" population averages as healthy targets, Attia advocates for a proactive, data-driven strategy that aims for **elite percentiles** to build substantial physiological reserves against age-related decline.

### The Strategic Imperative: Building a Buffer Against Inevitable Decline

Attia's core insight is profound yet simple: **if you aspire to "kick ass" at 85, you can't afford to be average at 50**. With muscle mass declining 1-2% annually after age 50, and strength declining even faster at 4% per year, being "normal" today means being frail tomorrow. The solution is to build such a substantial buffer of muscle mass that even after decades of predictable decline, you never cross the threshold into frailty and dysfunction.

### Why ALMI is the Ultimate Longevity Metric

Attia emphasizes ALMI (Appendicular Lean Mass Index) as the "purer" measure of skeletal muscle compared to FFMI, because it isolates the metabolically active tissue in your arms and legs that responds to training. This isn't just about aesthetics—it's about engineering three critical pillars of healthspan:

1. **Metabolic Resilience**: Muscle is your body's primary glucose disposal site. Higher ALMI directly translates to better insulin sensitivity and protection against metabolic diseases
2. **Functional Reserve**: Starting with elite muscle mass ensures you maintain independence and physical capability deep into old age
3. **Strength Foundation**: While strength beats muscle mass for longevity prediction, muscle provides the biological substrate required to generate that strength

### From Philosophy to Action: Evidence-Based Goal Setting

Attia's recommendations are precise and ambitious:
- **Baseline Goal**: Achieve >75th percentile ALMI (supported by mortality data showing significant longevity benefits)
- **Aspirational Goal**: Target 90th-97th percentile (his personal standard for maximum healthspan engineering)

RecompTracker transforms these abstract percentiles into concrete, actionable intelligence:
- **Precise Targeting**: Know exactly where you stand relative to your age/sex cohort using validated LMS reference data
- **Goal Engineering**: Calculate the exact changes in weight, lean mass, and fat mass needed to reach your target percentile
- **Progress Quantification**: Track your percentile improvements over time, not just absolute numbers
- **Future Planning**: Reverse-engineer your current requirements based on desired function at advanced ages

### Beyond Traditional Fitness: A Longevity Investment Strategy

This isn't a typical fitness app—it's a **longevity investment calculator**. Traditional approaches focus on short-term aesthetics or performance. Attia's philosophy, operationalized here, treats muscle mass as a quantifiable asset in your "longevity bank account." Every percentage point improvement in ALMI percentile is a withdrawal from future frailty risk.

The tool embodies Attia's rejection of population-based "normal" ranges, which reflect a chronically unhealthy society. Instead, it provides the data infrastructure to pursue the "elite" metrics that correlate with exceptional healthspan—because in Attia's words, "normal" is simply a blueprint for decline.

### Democratizing Precision Longevity

What was once available only to Attia's high-net-worth patients is now accessible to anyone with DEXA scan data. RecompTracker democratizes the quantified, engineering approach to healthspan optimization that defines Medicine 3.0, giving users the same analytical framework used in elite longevity medicine.

This is how you move from hoping to age well to **engineering** a robust, functional future.

## Features

### Core Analysis
- **Accurate Body Fat Calculation**: Uses actual body fat percentages instead of calculated estimates
- **Comprehensive Change Tracking**: Calculates changes since last scan and since first scan
- **Intelligent TLM Estimation**: Uses personalized ALM/TLM ratios when multiple scans are available
- **Goal System**: Supports separate ALMI and FFMI goals with target percentiles and ages
- **Enhanced Output**: 20+ columns showing body composition values, changes, and progress tracking
- **CSV Export**: Complete data export for detailed analysis and tracking

### Interfaces
- **🌐 Web Interface**: Interactive Streamlit application with real-time validation and visualization
- **⌨️ Command Line**: Scriptable CLI for batch processing and automation
- **📊 Dual Visualizations**: ALMI and FFMI percentile curves with goal tracking
- **🎯 Smart Goals**: Automatic goal calculation based on training level and progression rates


## Reference
This code can be used to compute Z-scores of body composition parameters with the reference values for:

- adults published in:  
__Article Title__: Reference values of body composition parameters and visceral adipose tissue (VAT) by DXA in adults aged 18–81 years—results from the LEAD cohort  
__DOI__: 10.1038/s41430-020-0596-5, 2019EJCN0971  
__Link__: https://www.nature.com/articles/s41430-020-0596-5  
__Citation__: Ofenheimer, A., Breyer-Kohansal, R., Hartl, S. et al. Reference values of body composition parameters and visceral adipose tissue (VAT) by DXA in adults aged 18–81 years—results from the LEAD cohort. Eur J Clin Nutr (2020).

- children published in:  
__Article Title__: Reference charts for body composition parameters by dual‐energy X‐ray absorptiometry in European children and adolescents aged 6 to 18 years—Results from the Austrian LEAD (Lung, hEart , sociAl , boDy ) cohort  
__Link__: http://dx.doi.org/10.1111/ijpo.12695  
__Citation__: Ofenheimer, A, Breyer‐Kohansal, R, Hartl, S, et al. Reference charts for body composition parameters by dual‐energy X‐ray absorptiometry in European children and adolescents aged 6 to 18 years—Results from the Austrian LEAD (Lung, hEart , sociAl , boDy ) cohort. Pediatric Obesity. 2020;e12695. https://doi.org/10.1111/ijpo.12695



## Requirements

- Python 3.9+ (recommended: 3.11 or 3.13)
- Dependencies listed in requirements.txt
- pyenv (optional but recommended for Python version management)

## Setup

### Option 1: Automated Setup (Recommended)

Use the provided setup script that automatically detects your Python environment:

```bash
# Clone the repository
git clone <repository-url>
cd bodymetrics

# Run automated setup
task setup
```

The setup script will automatically:
- Detect pyenv, pipenv, poetry, or use standard venv
- Install compatible package versions
- Set up the virtual environment

### Option 2: Manual Setup

If you prefer manual setup or don't have `task` installed:

```bash
# Install task runner (macOS with Homebrew)
brew install go-task/tap/go-task

# Or use the setup script directly
./setup_env.sh

# Or manual venv setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option 3: Using Specific Tools

```bash
# Using pyenv + venv (recommended)
task pyenv-setup

# Using pipenv
task pipenv-setup

# Using standard venv
task venv-setup
```

## Usage

### Web Interface (Recommended)

Launch the interactive web application for an intuitive analysis experience:

```bash
# Activate environment (if not using pipenv/poetry)
source venv/bin/activate

# Launch web application
streamlit run webapp.py
```

The web interface will open in your browser at `http://localhost:8501` and provides:
- Interactive data input forms with real-time validation
- Editable DEXA scan history grid
- Goal setting with intelligent suggestions
- Live analysis updates and visualizations
- Downloadable results (CSV)
- Fake data generation for testing/demos

### Command Line Interface

For programmatic usage or batch processing:

```bash
# Activate environment (if not using pipenv/poetry)
source venv/bin/activate

# View configuration help
python run_analysis.py --help-config

# Run analysis with example config
python run_analysis.py example_config.json

# Run with custom config
python run_analysis.py my_config.json
```

### Using Task Runner

```bash
# Launch web application
task webapp

# Run CLI with example config
task run

# Run CLI with custom config
task run-config -- my_config.json

# Run tests
task test

# View configuration help
task help-config

# Clean generated files
task clean
```

### Configuration Format

Create a JSON configuration file with your DEXA scan data:

```json
{
  "user_info": {
    "birth_date": "MM/DD/YYYY",
    "height_in": 66.0,
    "gender": "male",
    "training_level": "intermediate"
  },
  "scan_history": [
    {
      "date": "04/07/2022",
      "total_weight_lbs": 150.0,
      "total_lean_mass_lbs": 106.3,
      "fat_mass_lbs": 25.2,
      "body_fat_percentage": 16.8,
      "arms_lean_lbs": 12.4,
      "legs_lean_lbs": 37.3
    }
  ],
  "goals": {
    "almi": {
      "target_percentile": 0.90,
      "target_age": 45.0,
      "description": "Reach 90th percentile ALMI by age 45"
    }
  }
}
```

**Required Fields:**
- `date`, `total_weight_lbs`, `total_lean_mass_lbs`, `fat_mass_lbs`, `body_fat_percentage`
- `arms_lean_lbs`, `legs_lean_lbs`

**Important:** The system requires actual body fat percentages and fat mass values for accurate calculations.

### Output

The analysis generates:
- **PNG plots**: `almi_plot.png`, `ffmi_plot.png` with percentile curves and data points
- **CSV export**: `almi_stats_table.csv` with comprehensive body composition tracking
- **Enhanced table**: 20+ columns including changes since last/first scan and goal progress



## Citation
If you find this code useful or if you use it for your own work, please cite the papers referenced above.






