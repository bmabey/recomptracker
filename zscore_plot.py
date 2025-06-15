"""
================================================================================
README: Comprehensive DEXA Body Composition Analysis Script
================================================================================

PURPOSE:
This script performs a comprehensive analysis of DEXA scan data. It processes
a user's historical scan results, compares them against a reference population
using the LMS method, and visualizes the data with percentile curves, historical
trends, and user-defined goals. The script features intelligent TLM (Total Lean
Mass) estimation using personalized ALM/TLM ratios when multiple scans are available,
or population-based ratios when only a single scan exists.

HOW TO USE:
1.  Ensure you have the required LMS reference data files in the data/ directory:
    - `adults_LMS_appendicular_LMI_gender0.csv` (for male ALMI)
    - `adults_LMS_LMI_gender0.csv` (for male LMI/FFMI)
2.  Run the script: `python zscore_plot.py`
3.  For comprehensive testing: `python test_zscore_calculations.py`
4.  The analysis will:
    - Extract the hardcoded DEXA scan data
    - Intelligently estimate TLM gain needed using ALM/TLM ratio method
    - Load the LMS reference data
    - Calculate all metrics (ALMI, FFMI, Z-scores, T-scores, etc.)
    - Generate and save plots named `almi_plot.png` and `ffmi_plot.png`
    - Export the table data to `almi_stats_table.csv`
    - Print a final summary table to the console

SECTIONS:
- SECTION 1: CORE CALCULATION LOGIC
  Contains the fundamental mathematical functions for age calculation, Z-scores,
  inverse Z-scores (for plotting percentiles), and T-scores.
- SECTION 2: DATA PROCESSING AND ORCHESTRATION
  Contains functions that manage the overall workflow, including extracting
  DEXA data, intelligent TLM estimation using ALM/TLM ratios, loading LMS
  reference files, and processing scans and goals to produce a complete dataset.
- SECTION 3: PLOTTING LOGIC
  Contains the function responsible for generating the final plots, including
  the percentile curves, data points, and CSV export functionality.
- SECTION 4: MAIN EXECUTION BLOCK
  The entry point of the script that orchestrates the data processing and
  plot generation. Tests are now located in test_zscore_calculations.py.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.stats as stats
from datetime import datetime
import os
import json
import argparse
from jsonschema import validate, ValidationError

# JSON Schema for configuration validation
CONFIG_SCHEMA = {
    "type": "object",
    "required": ["user_info", "scan_history", "goal"],
    "properties": {
        "user_info": {
            "type": "object",
            "required": ["birth_date", "height_in", "gender"],
            "properties": {
                "birth_date": {"type": "string", "pattern": "^\\d{2}/\\d{2}/\\d{4}$"},
                "height_in": {"type": "number", "minimum": 12, "maximum": 120},
                "gender": {"type": "string", "pattern": "^(?i)(m|f|male|female)$"}
            },
            "additionalProperties": False
        },
        "scan_history": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["date", "total_lean_mass_lbs", "arms_lean_lbs", "legs_lean_lbs"],
                "properties": {
                    "date": {"type": "string", "pattern": "^\\d{2}/\\d{2}/\\d{4}$"},
                    "total_lean_mass_lbs": {"type": "number", "minimum": 0},
                    "arms_lean_lbs": {"type": "number", "minimum": 0},
                    "legs_lean_lbs": {"type": "number", "minimum": 0}
                },
                "additionalProperties": False
            }
        },
        "goal": {
            "type": "object",
            "required": ["target_percentile", "target_age"],
            "properties": {
                "target_percentile": {"type": "number", "minimum": 0, "maximum": 1},
                "target_age": {"type": "number", "minimum": 18, "maximum": 120},
                "description": {"type": "string"}
            },
            "additionalProperties": False
        }
    },
    "additionalProperties": False
}

# ---------------------------------------------------------------------------
# SECTION 1: CORE CALCULATION LOGIC
# ---------------------------------------------------------------------------

def calculate_age_precise(birth_date_str, scan_date_str):
    """
    Calculates the precise age in decimal years between two dates.

    Args:
        birth_date_str (str): The birth date in "MM/DD/YYYY" format.
        scan_date_str (str): The scan date in "MM/DD/YYYY" format.

    Returns:
        float: The age in decimal years, accounting for leap years.
    """
    birth_date = datetime.strptime(birth_date_str, "%m/%d/%Y")
    scan_date = datetime.strptime(scan_date_str, "%m/%d/%Y")
    time_difference = scan_date - birth_date
    return time_difference.days / 365.2425

def compute_zscore(y, L, M, S, eps=1e-5):
    """
    Calculates the Z-score for a given value y using the LMS method.

    This function takes a measured value and the LMS parameters (which describe
    the reference population's distribution) to determine how many standard
    deviations the value is from the median of that population, after
    accounting for skewness.

    Args:
        y (float): The measured value (e.g., an ALMI or FFMI value).
        L (float): The lambda parameter (Box-Cox power for skewness).
        M (float): The mu parameter (the median of the reference population).
        S (float): The sigma parameter (the coefficient of variation).
        eps (float): A small epsilon value to handle L values that are very close to zero.

    Returns:
        float: The calculated Z-score, or NaN if any input is invalid or y is non-positive.
    """
    if y <= 0 or pd.isna(y) or pd.isna(L) or pd.isna(M) or pd.isna(S):
        return np.nan
    if abs(L) < eps:
        # When L is close to zero, the formula simplifies to a logarithmic form.
        return np.log(y / M) / S
    else:
        # Standard Box-Cox transformation formula for Z-score.
        return (((y / M)**L) - 1) / (L * S)

def get_value_from_zscore(z, l_val, m_val, s_val, eps=1e-5):
    """
    Calculates a metric value from a Z-score (inverse LMS transformation).

    This is the inverse of the `compute_zscore` function. It's used to find the
    metric value that corresponds to a specific percentile (represented by its
    Z-score) for a given age.

    Args:
        z (float): The Z-score (e.g., 0 for the 50th percentile, 1.28 for the 90th).
        l_val (float): The L (lambda) value for the given age.
        m_val (float): The M (mu/median) value for the given age.
        s_val (float): The S (sigma) value for the given age.
        eps (float): A small epsilon value for handling L values close to zero.

    Returns:
        float: The metric value (e.g., ALMI) corresponding to the Z-score.
    """
    if pd.isna(z) or pd.isna(l_val) or pd.isna(m_val) or pd.isna(s_val):
        return np.nan
    if abs(l_val) < eps:
        return m_val * np.exp(s_val * z)
    else:
        term = l_val * s_val * z + 1
        if term < 0 and (1.0 / l_val) % 1 != 0:
            return np.nan  # Avoids complex numbers for real-valued plots
        return m_val * np.power(term, (1.0 / l_val))

def calculate_t_score(value, young_adult_median, young_adult_sd):
    """
    Calculates a T-score against a young adult reference.

    A T-score compares a given value to the mean of a healthy, young adult
    reference population (typically age 30), expressed in standard deviations.

    Args:
        value (float): The user's measured value.
        young_adult_median (float): The median value for the age 30 reference group.
        young_adult_sd (float): The standard deviation for the age 30 reference group.

    Returns:
        float: The calculated T-score.
    """
    if pd.isna(value) or pd.isna(young_adult_median) or pd.isna(young_adult_sd) or young_adult_sd == 0:
        return np.nan
    return (value - young_adult_median) / young_adult_sd

def calculate_z_percentile(value, age, L_func, M_func, S_func):
    """
    Calculates Z-score and percentile for a given metric value and age by
    wrapping the core z-score calculation.

    Args:
        value (float): The metric value (e.g., ALMI).
        age (float): The age of the user at the time of measurement.
        L_func (scipy.interpolate.interp1d): Interpolation function for L.
        M_func (scipy.interpolate.interp1d): Interpolation function for M.
        S_func (scipy.interpolate.interp1d): Interpolation function for S.

    Returns:
        tuple[float, float]: A tuple containing the Z-score and the percentile.
    """
    if L_func is None or M_func is None or S_func is None:
        return np.nan, np.nan
    try:
        l_val, m_val, s_val = L_func(age), M_func(age), S_func(age)
        z_score = compute_zscore(value, l_val, m_val, s_val)
        if pd.isna(z_score):
            return np.nan, np.nan
        percentile = stats.norm.cdf(z_score) * 100
        return z_score, percentile
    except Exception:
        return np.nan, np.nan


# ---------------------------------------------------------------------------
# SECTION 2: DATA PROCESSING AND ORCHESTRATION
# ---------------------------------------------------------------------------

def load_config_json(config_path):
    """
    Loads configuration from a JSON file containing user info, scan history, and goals.

    Args:
        config_path (str): Path to the JSON configuration file.

    Returns:
        dict: Configuration dictionary with user_info, scan_history, and goal sections.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        json.JSONDecodeError: If the JSON is malformed.
        ValidationError: If the JSON doesn't match the required schema.
    """
    print(f"Loading configuration from {config_path}...")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate against JSON schema
    validate(config, CONFIG_SCHEMA)
    
    print(f"Successfully loaded config with {len(config['scan_history'])} scans")
    return config

def parse_gender(gender_str):
    """
    Converts user-friendly gender string to gender code for LMS data loading.
    
    Args:
        gender_str (str): Gender as string (m, f, male, female - case insensitive)
        
    Returns:
        int: Gender code (0 for male, 1 for female)
    """
    gender_lower = gender_str.lower()
    if gender_lower in ['m', 'male']:
        return 0
    elif gender_lower in ['f', 'female']:
        return 1
    else:
        raise ValueError(f"Invalid gender: {gender_str}. Must be one of: m, f, male, female")

def extract_data_from_config(config):
    """
    Extracts user_info and scan_history from loaded config in the format expected by existing functions.

    Args:
        config (dict): Configuration dictionary from load_config_json.

    Returns:
        tuple[dict, list]: A tuple containing user_info dict and scan_history list.
    """
    # Convert config format to expected format for backward compatibility
    user_info = {
        "birth_date_str": config['user_info']['birth_date'],
        "height_in": config['user_info']['height_in'],
        "gender_code": parse_gender(config['user_info']['gender'])
    }
    
    # Convert scan format (date -> date_str for backward compatibility)
    scan_history = []
    for scan in config['scan_history']:
        scan_converted = {
            'date_str': scan['date'],
            'total_lean_mass_lbs': scan['total_lean_mass_lbs'],
            'arms_lean_lbs': scan['arms_lean_lbs'],
            'legs_lean_lbs': scan['legs_lean_lbs']
        }
        scan_history.append(scan_converted)
    
    return user_info, scan_history

def get_alm_tlm_ratio(processed_data, goal_params, lms_functions, user_info):
    """
    Calculates ALM/TLM ratio using personal history when available, 
    otherwise falls back to population ratio at target age.
    
    Args:
        processed_data (list): List of processed scan data
        goal_params (dict): Dictionary with target age and percentile
        lms_functions (dict): Dictionary of LMS interpolation functions
        user_info (dict): Dictionary with user height and other info
        
    Returns:
        float: ALM/TLM ratio to use for TLM estimation
    """
    lbs_to_kg = 1 / 2.20462
    
    if len(processed_data) >= 2:
        # Use personal ratio from recent scans
        recent_scans = processed_data[-min(3, len(processed_data)):]  # Last 3 or fewer scans
        ratios = []
        for scan in recent_scans:
            alm_kg = scan['alm_lbs'] * lbs_to_kg
            tlm_kg = scan['total_lean_mass_lbs'] * lbs_to_kg
            ratios.append(alm_kg / tlm_kg)
        
        personal_ratio = np.mean(ratios)
        print(f"Using personal ALM/TLM ratio of {personal_ratio:.3f} from {len(recent_scans)} recent scans")
        return personal_ratio
    
    else:
        # Use population ratio at TARGET AGE from LMS data
        target_age = goal_params['target_age']
        height_m_sq = (user_info['height_in'] * 0.0254) ** 2
        
        # Calculate population medians at target age
        almi_median = lms_functions['almi_M'](target_age)  # ALMI at target age
        lmi_median = lms_functions['lmi_M'](target_age)    # LMI at target age
        
        # Convert to absolute masses
        alm_median_kg = almi_median * height_m_sq
        tlm_median_kg = lmi_median * height_m_sq
        
        population_ratio = alm_median_kg / tlm_median_kg
        print(f"Using population ALM/TLM ratio of {population_ratio:.3f} at target age {target_age} (single scan fallback)")
        return population_ratio

def load_lms_data(metric, gender_code, data_path="./data/"):
    """
    Loads LMS data from a CSV file and creates interpolation functions.

    Args:
        metric (str): The body composition metric (e.g., 'appendicular_LMI').
        gender_code (int): The gender code (0 for male, 1 for female).
        data_path (str): The path to the directory containing the LMS data files.

    Returns:
        tuple[interp1d, interp1d, interp1d]: A tuple of interpolation
                                              functions for L, M, and S.
                                              Returns (None, None, None) on failure.
    """
    filename = f"adults_LMS_{metric}_gender{gender_code}.csv"
    file_path = os.path.join(data_path, filename)
    try:
        df_lms = pd.read_csv(file_path)
        # Standardize column names (handle both 'L' and 'lambda', etc.)
        df_lms.rename(columns={'lambda': 'L', 'mu': 'M', 'sigma': 'S'}, inplace=True)
        if not all(col in df_lms.columns for col in ['age', 'L', 'M', 'S']):
            return None, None, None
        
        L_func = interp1d(df_lms['age'], df_lms['L'], kind='cubic', fill_value="extrapolate")
        M_func = interp1d(df_lms['age'], df_lms['M'], kind='cubic', fill_value="extrapolate")
        S_func = interp1d(df_lms['age'], df_lms['S'], kind='cubic', fill_value="extrapolate")
        return L_func, M_func, S_func
    except FileNotFoundError:
        print(f"LMS data file not found: {file_path}")
        return None, None, None
    except Exception as e:
        print(f"Error reading or processing LMS file {file_path}: {e}")
        return None, None, None

def process_scans_and_goal(user_info, scan_history, goal_params, lms_functions):
    """
    Calculates all metrics for historical scans and the user's goal.

    This function orchestrates the main data processing pipeline, taking raw
    scan data and enriching it with calculated metrics like ALMI, FFMI, Z-scores,
    T-scores, and percentiles. It also calculates the values for the user's goal.

    Args:
        user_info (dict): Dictionary with user's birth date, height, etc.
        scan_history (list): List of dictionaries, each with raw data for a scan.
        goal_params (dict): Dictionary defining the user's goal.
        lms_functions (dict): Dictionary containing the loaded LMS interpolation functions.

    Returns:
        pd.DataFrame: A comprehensive DataFrame with all historical and goal data.
    """
    print("Processing scan history and goal...")
    height_m = user_info['height_in'] * 0.0254
    height_m_sq = height_m ** 2
    lbs_to_kg = 1 / 2.20462

    ref_age_30 = 30.0
    almi_m_30 = lms_functions['almi_M'](ref_age_30)
    almi_s_30 = lms_functions['almi_S'](ref_age_30)
    almi_sd_30 = almi_m_30 * almi_s_30
    
    lmi_m_30 = lms_functions['lmi_M'](ref_age_30)
    lmi_s_30 = lms_functions['lmi_S'](ref_age_30)
    lmi_sd_30 = lmi_m_30 * lmi_s_30

    processed_data = []
    for scan in scan_history:
        point = scan.copy()
        point['age_at_scan'] = calculate_age_precise(user_info['birth_date_str'], point['date_str'])
        point['alm_lbs'] = point['arms_lean_lbs'] + point['legs_lean_lbs']
        point['almi_kg_m2'] = (point['alm_lbs'] * lbs_to_kg) / height_m_sq
        point['ffmi_kg_m2'] = (point['total_lean_mass_lbs'] * lbs_to_kg) / height_m_sq
        
        point['almi_z_score'], point['almi_percentile'] = calculate_z_percentile(point['almi_kg_m2'], point['age_at_scan'], lms_functions['almi_L'], lms_functions['almi_M'], lms_functions['almi_S'])
        point['ffmi_lmi_z_score'], point['ffmi_lmi_percentile'] = calculate_z_percentile(point['ffmi_kg_m2'], point['age_at_scan'], lms_functions['lmi_L'], lms_functions['lmi_M'], lms_functions['lmi_S'])
        
        point['almi_t_score'] = calculate_t_score(point['almi_kg_m2'], almi_m_30, almi_sd_30)
        point['ffmi_lmi_t_score'] = calculate_t_score(point['ffmi_kg_m2'], lmi_m_30, lmi_sd_30)
        processed_data.append(point)

    target_z = stats.norm.ppf(goal_params['target_percentile'])
    l_goal, m_goal, s_goal = lms_functions['almi_L'](goal_params['target_age']), lms_functions['almi_M'](goal_params['target_age']), lms_functions['almi_S'](goal_params['target_age'])
    target_almi = get_value_from_zscore(target_z, l_goal, m_goal, s_goal)
    target_alm_kg = target_almi * height_m_sq
    current_alm_kg = (processed_data[-1]['alm_lbs']) * lbs_to_kg
    goal_params['alm_to_add_kg'] = target_alm_kg - current_alm_kg
    
    # Intelligently estimate TLM gain using ALM/TLM ratio
    alm_tlm_ratio = get_alm_tlm_ratio(processed_data, goal_params, lms_functions, user_info)
    target_tlm_kg = target_alm_kg / alm_tlm_ratio
    current_tlm_kg = (processed_data[-1]['total_lean_mass_lbs']) * lbs_to_kg
    goal_params['estimated_tlm_gain_kg'] = target_tlm_kg - current_tlm_kg
    
    print(f"Calculated TLM gain needed: {goal_params['estimated_tlm_gain_kg']:.2f} kg ({goal_params['estimated_tlm_gain_kg']/(1/2.20462):.2f} lbs)")
    
    target_ffmi = target_tlm_kg / height_m_sq
    goal_ffmi_z, goal_ffmi_p = calculate_z_percentile(target_ffmi, goal_params['target_age'], lms_functions['lmi_L'], lms_functions['lmi_M'], lms_functions['lmi_S'])

    goal_row = {
        'date_str': f"Goal (Age {goal_params['target_age']})", 'age_at_scan': goal_params['target_age'],
        'almi_kg_m2': target_almi, 'almi_z_score': target_z,
        'almi_percentile': goal_params['target_percentile'] * 100,
        'almi_t_score': calculate_t_score(target_almi, almi_m_30, almi_sd_30),
        'ffmi_kg_m2': target_ffmi, 'ffmi_lmi_z_score': goal_ffmi_z,
        'ffmi_lmi_percentile': goal_ffmi_p,
        'ffmi_lmi_t_score': calculate_t_score(target_ffmi, lmi_m_30, lmi_sd_30)
    }
    processed_data.append(goal_row)
    
    return pd.DataFrame(processed_data)

# ---------------------------------------------------------------------------
# SECTION 3: PLOTTING LOGIC
# ---------------------------------------------------------------------------

def plot_metric_with_table(df_results, metric_to_plot, lms_functions, goal_params):
    """
    Generates and saves a plot for a given metric (ALMI or FFMI/LMI)
    with historical data and goal. Also exports the data table to a CSV file.

    Args:
        df_results (pd.DataFrame): The comprehensive DataFrame with all calculated data.
        metric_to_plot (str): The primary metric to plot ('ALMI' or 'FFMI').
        lms_functions (dict): Dictionary of loaded LMS interpolation functions.
        goal_params (dict): Dictionary defining the user's goal parameters.
    """
    print(f"Generating plot for {metric_to_plot.upper()}...")
    is_almi_plot = 'almi' in metric_to_plot.lower()
    
    value_col = 'almi_kg_m2' if is_almi_plot else 'ffmi_kg_m2'
    L_func = lms_functions['almi_L'] if is_almi_plot else lms_functions['lmi_L']
    M_func = lms_functions['almi_M'] if is_almi_plot else lms_functions['lmi_M']
    S_func = lms_functions['almi_S'] if is_almi_plot else lms_functions['lmi_S']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    percentiles_to_plot = {'3%': -1.88, '10%': -1.28, '25%': -0.67, '50%': 0, '75%': 0.67, '90%': 1.28, '97%': 1.88}
    ages_plot = np.linspace(18, 81, 200)
    for name, z_val in percentiles_to_plot.items():
        l, m, s = L_func(ages_plot), M_func(ages_plot), S_func(ages_plot)
        values = np.vectorize(get_value_from_zscore)(z_val, l, m, s)
        ax.plot(ages_plot, values, label=name, alpha=0.7, linewidth=1.5)
        
    historical_data = df_results[df_results['date_str'].str.find('Goal') == -1]
    goal_data = df_results[df_results['date_str'].str.find('Goal') != -1].iloc[0]
    
    # Plot individual scans with numbered markers and detailed legend entries
    percentile_col = 'almi_percentile' if is_almi_plot else 'ffmi_lmi_percentile'
    
    for i, (idx, row) in enumerate(historical_data.iterrows()):
        scan_num = i + 1
        date_label = datetime.strptime(row['date_str'], '%m/%d/%Y').strftime('%m/%d/%y')
        percentile = row[percentile_col]
        
        # Plot individual point with number marker
        ax.plot(row['age_at_scan'], row[value_col], 
               marker=f'${scan_num}$', markersize=10, 
               color='blue', markeredgecolor='darkblue', markeredgewidth=1,
               label=f"{date_label} {percentile:.1f}%",
               linestyle='None', zorder=5)
    
    # Plot goal marker
    ax.plot(goal_data['age_at_scan'], goal_data[value_col], marker='*', color='gold', markersize=20, markeredgecolor='black', zorder=6, label=f"Goal ({goal_params['target_percentile']*100:.0f}th %ile)", linestyle='None')

    lbs_to_kg = 1 / 2.20462
    alm_add_str = f"{goal_params['alm_to_add_kg']:.2f} kg ({goal_params['alm_to_add_kg'] / lbs_to_kg:.2f} lbs)"
    tlm_add_str = f"{goal_params['estimated_tlm_gain_kg']:.2f} kg ({goal_params['estimated_tlm_gain_kg'] / lbs_to_kg:.2f} lbs)"
    title = (f"{metric_to_plot.upper()} Percentile Curves for Adult Males with Scan History & Goal\n(Data Source: LEAD Cohort)\n"
             f"To reach 90th %ile ALMI at age 45: Add ~{alm_add_str} ALM. Estimated Total Lean Mass gain needed: ~{tlm_add_str}.")
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Age (Years)', fontsize=12)
    ax.set_ylabel(f'{metric_to_plot.upper()} (kg/m²)', fontsize=12)
    
    # Organize legend with percentiles first, then scans, then goal
    handles, labels = ax.get_legend_handles_labels()
    
    # Separate different types of legend entries
    percentile_handles = []
    percentile_labels = []
    scan_handles = []
    scan_labels = []
    goal_handles = []
    goal_labels = []
    
    for handle, label in zip(handles, labels):
        if '%' in label and 'Scan' not in label and 'Goal' not in label:
            percentile_handles.append(handle)
            percentile_labels.append(label)
        elif 'Scan' in label:
            scan_handles.append(handle)
            scan_labels.append(label)
        elif 'Goal' in label:
            goal_handles.append(handle)
            goal_labels.append(label)
    
    # Combine in organized order
    organized_handles = percentile_handles + scan_handles + goal_handles
    organized_labels = percentile_labels + scan_labels + goal_labels
    
    ax.legend(organized_handles, organized_labels, loc='upper left', title="Percentiles & Scan Data", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Export table data to CSV (only for ALMI plot to avoid duplicate files)
    if is_almi_plot:
        table_data = df_results[[
            'date_str', 'age_at_scan', 'almi_kg_m2', 'almi_z_score', 'almi_percentile',
            'almi_t_score', 'ffmi_kg_m2', 'ffmi_lmi_z_score', 'ffmi_lmi_percentile', 'ffmi_lmi_t_score'
        ]].copy()
        
        # Rename columns for CSV clarity
        table_data.columns = ['Date', 'Age', 'ALMI_kg_m2', 'ALMI_Z_Score', 'ALMI_Percentile',
                             'ALMI_T_Score', 'FFMI_kg_m2', 'FFMI_Z_Score', 'FFMI_Percentile', 'FFMI_T_Score']
        
        # Save raw data to CSV without formatting
        csv_filename = "almi_stats_table.csv"
        table_data.to_csv(csv_filename, index=False)
        print(f"Table data saved as: {csv_filename}")
    
    output_filename = f"{metric_to_plot.lower()}_plot.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Plot saved as: {output_filename}")
    plt.close()

# ---------------------------------------------------------------------------
# SECTION 4: MAIN EXECUTION BLOCK
# ---------------------------------------------------------------------------

def main():
    """Main function for DEXA body composition analysis."""
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
      "gender": "male"
    },
    "scan_history": [
      {
        "date": "04/07/2022",
        "total_lean_mass_lbs": 106.3,
        "arms_lean_lbs": 12.4,
        "legs_lean_lbs": 37.3
      }
    ],
    "goal": {
      "target_percentile": 0.90,
      "target_age": 45.0
    }
  }
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='example_config.json',
        help='Path to JSON configuration file (default: example_config.json)'
    )
    
    args = parser.parse_args()
    
    print("DEXA Body Composition Analysis with Intelligent TLM Estimation")
    print("=" * 65)
    print("Note: Run 'python test_zscore_calculations.py' for comprehensive testing\n")
    
    try:
        # Step 1: Load Configuration
        config = load_config_json(args.config)
        user_info, scan_history = extract_data_from_config(config)
        goal_params = config['goal']
        
        # Display loaded configuration
        print(f"User Info:")
        print(f"  - Birth Date: {config['user_info']['birth_date']}")
        print(f"  - Height: {config['user_info']['height_in']} inches")
        print(f"  - Gender: {config['user_info']['gender']}")
        print(f"\nGoal:")
        print(f"  - Target ALMI Percentile: {goal_params['target_percentile']*100:.0f}th")
        print(f"  - Target Age: {goal_params['target_age']:.1f}")
        if 'description' in goal_params:
            print(f"  - Description: {goal_params['description']}")
        print("  - Total Lean Mass Gain: Will be calculated intelligently based on ALM/TLM ratio\n")

        # Step 2: Load LMS Data
        lms_functions = {
            'almi_L': None, 'almi_M': None, 'almi_S': None,
            'lmi_L': None, 'lmi_M': None, 'lmi_S': None
        }
        lms_functions['almi_L'], lms_functions['almi_M'], lms_functions['almi_S'] = load_lms_data(
            metric='appendicular_LMI', gender_code=user_info['gender_code'])
        lms_functions['lmi_L'], lms_functions['lmi_M'], lms_functions['lmi_S'] = load_lms_data(
            metric='LMI', gender_code=user_info['gender_code'])

        if not all(lms_functions.values()):
            print("Failed to load all necessary LMS data. Aborting analysis.")
            return 1
        
        # Step 3: Process data and generate results DataFrame
        df_results = process_scans_and_goal(user_info, scan_history, goal_params, lms_functions)
        
        # Step 4: Generate Plots
        plot_metric_with_table(df_results, 'ALMI', lms_functions, goal_params)
        plot_metric_with_table(df_results, 'FFMI', lms_functions, goal_params)
        
        # Step 5: Print Final Table
        print("\n--- Final Comprehensive Data Table ---")
        df_display = df_results[[
            'date_str', 'age_at_scan', 'almi_kg_m2', 'almi_z_score', 'almi_percentile', 'almi_t_score',
            'ffmi_kg_m2', 'ffmi_lmi_z_score', 'ffmi_lmi_percentile', 'ffmi_lmi_t_score'
        ]].copy()
        df_display.columns = [
            'Date', 'Age', 'ALMI', 'ALMI Z', 'ALMI %', 'ALMI T', 'FFMI', 'FFMI Z', 'FFMI %', 'FFMI T'
        ]
        for col in df_display.columns:
             if df_display[col].dtype == 'float64':
                 df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        print(df_display.to_markdown(index=False))
        
        return 0
        
    except (FileNotFoundError, json.JSONDecodeError, ValidationError, KeyError, ValueError) as e:
        print(f"Error: {e}")
        print(f"\nPlease check your configuration file: {args.config}")
        print("Run with --help to see the expected JSON format.")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())

