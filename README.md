# RecompTracker

A Python tool that calculates Z-scores and percentiles for body composition metrics (ALMI, FFMI) using LMS reference values from the LEAD cohort. The system processes DEXA scan history with actual body fat percentages and generates comprehensive visualizations with percentile curves, change tracking, and actionable goal analysis.

## Operationalizing Peter Attia's Medicine 3.0 Philosophy

This tool directly implements [Dr. Peter Attia's](https://peterattiamd.com/) **Medicine 3.0** approach to longevity and healthspan optimization. Rather than accepting "normal" population averages as healthy targets, Attia advocates for a proactive, data-driven strategy that aims for **elite percentiles** to build substantial physiological reserves against age-related decline.

### The Strategic Imperative: Building a Buffer Against Inevitable Decline

Attia's core insight is straightforward: **if you aspire to "kick ass" at 85, you can't afford to be average at 50**. With muscle mass declining 1-2% annually after age 50, and strength declining even faster at 4% per year, being "normal" today means being frail tomorrow. The solution is to build such a substantial buffer of muscle mass that even after decades of predictable decline, you never cross the threshold into frailty and dysfunction.

### Why ALMI Over BMI: A Better Metric for Body Composition

Attia firmly rejects BMI as a useful tool for individual health assessment, famously noting that at the individual level, it tells him little more about a patient's health than their eye color ([read more](https://peterattiamd.com/should-bmi-be-used-measuring-obesity-at-the-individual-level/)). While BMI may have utility for large-scale epidemiological studies, it has fundamental limitations for personal health management because **it fails to distinguish between fat mass and lean mass**—two components that have opposing effects on metabolic health.

**The Problem with BMI:**
- Cannot differentiate between muscle and fat
- A highly muscular athlete and an obese individual can have identical BMI scores
- Provides no information about body composition or metabolic health
- Ignores the critical importance of muscle mass for longevity

**Why ALMI is Superior:**

ALMI (Appendicular Lean Mass Index) measures the lean mass specifically in your arms and legs, normalized for height. Attia considers this a "purer" measure of skeletal muscle compared to FFMI because it isolates the metabolically active tissue that responds to training. Here's why this matters:

1. **Metabolic Function**: Muscle tissue is your body's largest glucose disposal site. When you eat carbohydrates, healthy muscle takes up 80-90% of the circulating glucose. Higher ALMI directly correlates with better insulin sensitivity and protection against metabolic diseases ([learn more about insulin resistance](https://peterattiamd.com/peter-attia-on-how-insulin-resistance-manifests-in-the-muscle/)).

2. **Functional Capacity**: The lean mass in your arms and legs determines your ability to perform daily activities—carrying groceries, climbing stairs, getting up from chairs. This appendicular muscle mass is what you lose with age-related sarcopenia.

3. **Training Response**: Unlike organ mass or bone density, appendicular lean mass is what responds to resistance training and nutritional interventions. It's the tissue you can actually build and maintain through lifestyle choices ([see Attia's training approach](https://peterattiamd.com/ama71/)).

4. **Predictive Power**: Studies show that individuals in the lowest quartile for muscle mass have mortality rates more than double those in the highest quartile over 12-year follow-up periods ([more on muscle mass and longevity](https://peterattiamd.com/peter-attia-on-the-importance-of-preserving-strength-and-muscle-mass-as-we-age/)).

### From Philosophy to Action: Evidence-Based Goal Setting

Attia's recommendations are precise and ambitious:
- **Baseline Goal**: Achieve >75th percentile ALMI (supported by mortality data showing significant longevity benefits)
- **Aspirational Goal**: Target 90th-97th percentile (his personal standard for optimal healthspan planning)

RecompTracker transforms these abstract percentiles into concrete, actionable intelligence:
- **Precise Targeting**: Know exactly where you stand relative to your age/sex cohort using validated LMS reference data
- **Goal Engineering**: Calculate the exact changes in weight, lean mass, and fat mass needed to reach your target percentile
- **Progress Quantification**: Track your percentile improvements over time, not just absolute numbers
- **Future Planning**: Reverse-engineer your current requirements based on desired function at advanced ages

### Beyond Traditional Fitness: A Longevity Investment Strategy

RecompTracker is a **longevity planning tool**. Traditional approaches focus on short-term aesthetics or performance. Attia's philosophy, operationalized here, treats muscle mass as a quantifiable asset in your "longevity bank account." Every percentage point improvement in ALMI percentile is a withdrawal from future frailty risk.

The tool embodies Attia's rejection of population-based "normal" ranges, which reflect a chronically unhealthy society. Instead, it provides the data infrastructure to pursue the "elite" metrics that correlate with better healthspan outcomes—because in Attia's words, "normal" is simply a blueprint for decline.

### A Personal Tool, Shared

This started as a personal project to help track my own DEXA scan progress using Attia's percentile-based approach. Rather than keep it to myself, I'm sharing it in case others find it useful for their own body composition goals.

**🤖 Fun fact**: This entire tool (including this text!) was [vibe coded](https://youtu.be/JeNS1ZNHQs8) using [Claude Code](https://claude.ai/code) in about a day!

**Learn More:**
- [Peter Attia's complete approach to body composition](https://peterattiamd.com/improving-body-composition/)
- [Why DEXA scans matter for longevity](https://peterattiamd.com/ai-dexa-and-mortality/)
- [Tim Ferriss interview with Peter Attia (detailed transcript)](https://tim.blog/2023/03/17/peter-attia-outlive-transcript/)

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



### Data Source
The LMS reference data files used by this tool were obtained from: https://github.com/FlorianKrach/scientific-LMS-zscore-app

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
cd recomptracker

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

# For development (shows Load Example button)
STREAMLIT_ENV=development streamlit run webapp.py
```

The web interface will open in your browser at `http://localhost:8501` and provides:
- Interactive data input forms with real-time validation
- Editable DEXA scan history grid
- Goal setting with intelligent suggestions
- Live analysis updates and visualizations
- Downloadable results (CSV)
- Fake data generation for testing/demos

**Environment Variables:**
- `STREAMLIT_ENV=development` - Shows development features like the "Load Example" button

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






