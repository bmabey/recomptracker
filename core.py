"""
Core DEXA Body Composition Analysis Logic

This module contains all the core calculation logic, data processing functions,
and plotting functionality for DEXA body composition analysis. This is the 
computational engine that powers the analysis scripts.

Sections:
- Core calculation logic (age, Z-scores, T-scores)
- Suggested goal logic and lean mass gain rates  
- Data processing and orchestration
- Plotting logic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.stats as stats
from datetime import datetime
import os
import json
from jsonschema import validate, ValidationError

# Constants for lean mass gain rates (kg/month)
LEAN_MASS_GAIN_RATES = {
    'male': {
        'novice': 1.0,
        'intermediate': 0.35,
        'advanced': 0.15
    },
    'female': {
        'novice': 0.5,
        'intermediate': 0.2,
        'advanced': 0.08
    }
}

# Healthy body fat percentage ranges by gender
# Based on American Council on Exercise (ACE) and athletic performance standards
HEALTHY_BF_RANGES = {
    'male': {
        'athletic': (6, 13),      # Athletes
        'fitness': (14, 17),     # Fitness enthusiasts  
        'acceptable': (18, 24),  # General health
        'overweight': (25, 100)  # Above healthy range
    },
    'female': {
        'athletic': (16, 20),    # Athletes
        'fitness': (21, 24),     # Fitness enthusiasts
        'acceptable': (25, 31),  # General health  
        'overweight': (32, 100)  # Above healthy range
    }
}

# Age adjustment factor for lean mass gain rates (per decade over 30)
AGE_ADJUSTMENT_FACTOR = 0.1  # 10% reduction per decade over 30

# JSON Schema for configuration validation
CONFIG_SCHEMA = {
    "type": "object",
    "required": ["user_info", "scan_history"],
    "properties": {
        "user_info": {
            "type": "object",
            "required": ["birth_date", "height_in", "gender"],
            "properties": {
                "birth_date": {"type": "string", "pattern": "^\\d{2}/\\d{2}/\\d{4}$"},
                "height_in": {"type": "number", "minimum": 12, "maximum": 120},
                "gender": {"type": "string", "pattern": "^(m|f|male|female|M|F|Male|Female|MALE|FEMALE)$"},
                "training_level": {"type": "string", "pattern": "^(novice|intermediate|advanced|Novice|Intermediate|Advanced|NOVICE|INTERMEDIATE|ADVANCED)$"}
            },
            "additionalProperties": False
        },
        "scan_history": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["date", "total_weight_lbs", "total_lean_mass_lbs", "fat_mass_lbs", "body_fat_percentage", "arms_lean_lbs", "legs_lean_lbs"],
                "properties": {
                    "date": {"type": "string", "pattern": "^\\d{2}/\\d{2}/\\d{4}$"},
                    "total_weight_lbs": {"type": "number", "minimum": 0},
                    "total_lean_mass_lbs": {"type": "number", "minimum": 0},
                    "fat_mass_lbs": {"type": "number", "minimum": 0},
                    "body_fat_percentage": {"type": "number", "minimum": 0, "maximum": 100},
                    "arms_lean_lbs": {"type": "number", "minimum": 0},
                    "legs_lean_lbs": {"type": "number", "minimum": 0}
                },
                "additionalProperties": False
            }
        },
        "goals": {
            "type": "object",
            "properties": {
                "almi": {
                    "type": "object",
                    "required": ["target_percentile"],
                    "properties": {
                        "target_percentile": {"type": "number", "minimum": 0, "maximum": 1},
                        "target_age": {
                            "oneOf": [
                                {"type": "number", "minimum": 18, "maximum": 120},
                                {"type": "string", "pattern": "^\\?$"},
                                {"type": "null"}
                            ]
                        },
                        "suggested": {"type": "boolean"},
                        "description": {"type": "string"},
                        "target_body_fat_percentage": {"type": "number", "minimum": 0, "maximum": 100}
                    },
                    "additionalProperties": False
                },
                "ffmi": {
                    "type": "object",
                    "required": ["target_percentile"],
                    "properties": {
                        "target_percentile": {"type": "number", "minimum": 0, "maximum": 1},
                        "target_age": {
                            "oneOf": [
                                {"type": "number", "minimum": 18, "maximum": 120},
                                {"type": "string", "pattern": "^\\?$"},
                                {"type": "null"}
                            ]
                        },
                        "suggested": {"type": "boolean"},
                        "description": {"type": "string"},
                        "target_body_fat_percentage": {"type": "number", "minimum": 0, "maximum": 100}
                    },
                    "additionalProperties": False
                }
            },
            "additionalProperties": False
        }
    },
    "additionalProperties": False
}


# ---------------------------------------------------------------------------
# CORE CALCULATION LOGIC
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
        eps (float): A small epsilon value to handle L values that are very close to zero.

    Returns:
        float: The metric value corresponding to the given Z-score, or NaN if inputs are invalid.
    """
    if pd.isna(z) or pd.isna(l_val) or pd.isna(m_val) or pd.isna(s_val):
        return np.nan
    
    if abs(l_val) < eps:
        # When L is close to zero, use the logarithmic form.
        return m_val * np.exp(s_val * z)
    else:
        # Standard inverse Box-Cox transformation.
        return m_val * ((l_val * s_val * z + 1) ** (1 / l_val))

def calculate_t_score(value, young_adult_median, young_adult_sd):
    """
    Calculates the T-score for a given value using young adult reference values.

    T-scores are commonly used in bone density analysis but can be applied to
    body composition metrics to show how a value compares to the peak values
    typically seen in young adults.

    Args:
        value (float): The measured value.
        young_adult_median (float): The median value for young adults (ages 20-30).
        young_adult_sd (float): The standard deviation for young adults.

    Returns:
        float: The T-score, or NaN if any input is invalid or the SD is zero.
    """
    if pd.isna(value) or pd.isna(young_adult_median) or pd.isna(young_adult_sd) or young_adult_sd == 0:
        return np.nan
    return (value - young_adult_median) / young_adult_sd

def calculate_z_percentile(value, age, L_func, M_func, S_func):
    """
    Calculates the Z-score and percentile for a given value at a specific age.

    This function combines the LMS interpolation with Z-score calculation to
    determine both the Z-score and its corresponding percentile for any age.

    Args:
        value (float): The measured value (e.g., ALMI, FFMI).
        age (float): The age in decimal years.
        L_func (callable): Interpolation function for L values.
        M_func (callable): Interpolation function for M values.
        S_func (callable): Interpolation function for S values.

    Returns:
        tuple: (Z-score, percentile) both as floats, or (NaN, NaN) if calculation fails.
    """
    if pd.isna(value) or pd.isna(age):
        return np.nan, np.nan
    
    try:
        L_val = L_func(age)
        M_val = M_func(age)
        S_val = S_func(age)
        z_score = compute_zscore(value, L_val, M_val, S_val)
        if pd.isna(z_score):
            return np.nan, np.nan
        percentile = stats.norm.cdf(z_score)
        return z_score, percentile
    except (ValueError, TypeError):
        return np.nan, np.nan


# ---------------------------------------------------------------------------
# SUGGESTED GOAL LOGIC AND LEAN MASS GAIN RATES
# ---------------------------------------------------------------------------

def get_gender_string(gender_code):
    """Convert gender code to string."""
    return "male" if gender_code == 0 else "female"

def detect_training_level_from_scans(processed_data, user_info):
    """
    Detects training level from scan progression patterns.
    
    Args:
        processed_data (pd.DataFrame or list): DataFrame with scan history and metrics or list of dicts
        user_info (dict): User information dictionary
        
    Returns:
        str: Detected training level ('novice', 'intermediate', 'advanced')
    """
    # If training level is explicitly provided, use it
    if 'training_level' in user_info and user_info['training_level']:
        level = user_info['training_level'].lower()
        return level, f"User-specified training level: {level}"
    
    # Need at least 2 scans for progression analysis
    if len(processed_data) < 2:
        print("  Insufficient scan history for training level detection - defaulting to intermediate")
        return 'intermediate', "Insufficient scan history - defaulting to intermediate"
    
    # Handle both DataFrame and list formats
    if hasattr(processed_data, 'sort_values'):
        # DataFrame format
        processed_data = processed_data.sort_values('scan_date')
        lean_gains = []
        
        for i in range(1, len(processed_data)):
            prev_scan = processed_data.iloc[i-1]
            curr_scan = processed_data.iloc[i]
            
            # Calculate time difference in months
            time_diff_days = (curr_scan['scan_date'] - prev_scan['scan_date']).days
            time_diff_months = time_diff_days / 30.44  # Average days per month
            
            if time_diff_months > 0.5:  # Only consider gaps > 2 weeks
                lean_gain_lbs = curr_scan['total_lean_mass_lbs'] - prev_scan['total_lean_mass_lbs']
                lean_gain_kg_per_month = (lean_gain_lbs * 0.453592) / time_diff_months
                lean_gains.append(lean_gain_kg_per_month)
    else:
        # List format - simplified analysis for tests
        lean_gains = []
        
        for i in range(1, len(processed_data)):
            prev_scan = processed_data[i-1]
            curr_scan = processed_data[i]
            
            # For test data without dates, assume 6 months between scans
            time_diff_months = 6.0
            
            if 'total_lean_mass_lbs' in curr_scan and 'total_lean_mass_lbs' in prev_scan:
                lean_gain_lbs = curr_scan['total_lean_mass_lbs'] - prev_scan['total_lean_mass_lbs']
                lean_gain_kg_per_month = (lean_gain_lbs * 0.453592) / time_diff_months
                lean_gains.append(lean_gain_kg_per_month)
    
    if not lean_gains:
        print("  No sufficient time gaps between scans for progression analysis - defaulting to intermediate")
        return 'intermediate', "No sufficient time gaps between scans for progression analysis - defaulting to intermediate"
    
    # Calculate average monthly lean mass gain rate
    avg_gain_rate = np.mean(lean_gains)
    
    # Classification thresholds based on research (conservative estimates)
    # These are monthly rates, adjusted for gender
    gender_str = get_gender_string(user_info['gender_code'])
    if gender_str == 'male':
        novice_threshold = 0.8     # >0.8 kg/month suggests novice gains
        advanced_threshold = 0.2   # <0.2 kg/month suggests advanced/slow gains
    else:  # female
        novice_threshold = 0.4     # >0.4 kg/month suggests novice gains  
        advanced_threshold = 0.1   # <0.1 kg/month suggests advanced/slow gains
    
    if avg_gain_rate > novice_threshold:
        detected_level = 'novice'
        explanation = f"Detected novice level: rapid progression {avg_gain_rate:.2f} kg/month"
        print(f"  {explanation}")
    elif avg_gain_rate < advanced_threshold:
        detected_level = 'advanced'
        explanation = f"Detected advanced level: slow progression {avg_gain_rate:.2f} kg/month, experienced trainee"
        print(f"  {explanation}")
    else:
        detected_level = 'intermediate'
        explanation = f"Detected intermediate level: moderate progression {avg_gain_rate:.2f} kg/month"
        print(f"  {explanation}")
    
    return detected_level, explanation

def get_conservative_gain_rate(user_info, training_level, current_age):
    """
    Gets conservative lean mass gain rate based on demographics and training level.
    
    Args:
        user_info (dict): User information
        training_level (str): Training level ('novice', 'intermediate', 'advanced')
        current_age (float): Current age
        
    Returns:
        tuple: (rate, explanation) - Conservative gain rate in kg/month and explanation string
    """
    gender_str = get_gender_string(user_info['gender_code'])
    
    # Use the constant for consistency with tests
    base_rate = LEAN_MASS_GAIN_RATES[gender_str][training_level]
    
    # Age adjustment factor (muscle building capacity decreases with age)
    if current_age <= 30:
        age_factor = 1.0  # No adjustment for age 30 and under
    else:
        # Linear reduction of 10% per decade over 30
        decades_over_30 = (current_age - 30) / 10
        age_factor = max(1 - (AGE_ADJUSTMENT_FACTOR * decades_over_30), 0.5)  # Minimum 50% of base rate
    
    adjusted_rate = base_rate * age_factor
    
    if current_age <= 30:
        explanation = f"Conservative {training_level} rate for {gender_str}: {base_rate:.2f} kg/month (age {current_age:.0f})"
    else:
        explanation = f"Conservative {training_level} rate for {gender_str}: {base_rate:.2f} kg/month, age-adjusted to {adjusted_rate:.2f} kg/month (age {current_age:.0f})"
    
    print(f"  {explanation}")
    
    return adjusted_rate, explanation

def determine_training_level(user_info, processed_data):
    """
    Determines training level using explicit specification or detection from scans.
    
    Args:
        user_info (dict): User information
        processed_data (pd.DataFrame): Processed scan data
        
    Returns:
        tuple: (level, explanation) - Training level and explanation string
    """
    # Check if explicitly provided
    if 'training_level' in user_info and user_info['training_level']:
        level = user_info['training_level'].lower()
        explanation = f"User-specified training level: {level}"
        return level, explanation
    
    # Detect from scan progression
    level, explanation = detect_training_level_from_scans(processed_data, user_info)
    return level, explanation

def calculate_suggested_goal(goal_params, user_info, processed_data, lms_functions, metric='almi'):
    """
    Calculates suggested goals with realistic timeframes based on training level and progression rates.
    
    Args:
        goal_params (dict): Goal parameters including target_percentile
        user_info (dict): User information
        processed_data (pd.DataFrame): Processed scan data
        lms_functions (dict): LMS interpolation functions
        metric (str): Metric type ('almi' or 'ffmi')
        
    Returns:
        tuple: (updated_goal, messages) where updated_goal contains target_age and messages is list of explanations
    """
    # Initialize messages list to collect explanations
    messages = []
    
    # Get the most recent scan data (handle both DataFrame and list)
    if hasattr(processed_data, 'iloc'):
        latest_scan = processed_data.iloc[-1]
    else:
        latest_scan = processed_data[-1]
    current_age = latest_scan['age_at_scan']
    
    # Determine training level
    training_level, level_explanation = determine_training_level(user_info, processed_data)
    messages.append(level_explanation)
    
    # Get conservative gain rate
    monthly_gain_rate_kg, gain_explanation = get_conservative_gain_rate(user_info, training_level, current_age)
    messages.append(gain_explanation)
    
    # Get current metric value and target percentile
    current_metric = latest_scan[f'{metric}_kg_m2']
    target_percentile = goal_params['target_percentile']
    
    # Get current percentile from scan data (already calculated in percentage format)
    current_percentile = latest_scan[f'{metric}_percentile']
    
    # Check if user is already at or above target percentile
    # Note: current_percentile is in percentage format (0-100), target_percentile is in decimal format (0-1)
    current_percentile_decimal = current_percentile / 100
    if current_percentile_decimal >= target_percentile:
        print(f"  ✓ Already at {current_percentile:.1f}th percentile for {metric.upper()}, which is above target {target_percentile*100:.0f}th percentile")
        
        # If user is already above 90th percentile, don't suggest any goal
        if current_percentile_decimal >= 0.90:
            print(f"  🎯 You're already above the 90th percentile - no goal suggestion needed!")
            return None, messages  # Return None to indicate no goal should be suggested
        
        # Only suggest higher percentile if user is below 90th percentile
        new_target_percentile = min(0.90, current_percentile_decimal + 0.05)
        print(f"  Suggesting a higher target: {new_target_percentile*100:.0f}th percentile instead")
        
        updated_goal = goal_params.copy()
        updated_goal['target_percentile'] = new_target_percentile
        updated_goal['target_age'] = current_age + 2  # Default 2-year timeframe
        updated_goal['suggested'] = True
        messages.append(f"Suggesting a higher target: {new_target_percentile*100:.0f}th percentile instead")
        
        return updated_goal, messages
    
    # Binary search to find the target age where we can achieve the goal
    min_age = current_age
    max_age = min(current_age + 10, 70)  # Search up to 10 years or age 70
    
    def can_reach_goal_at_age(target_age):
        """Check if goal is achievable at target_age given gain rates."""
        # Map metric names to LMS function keys
        lms_key = 'almi' if metric == 'almi' else 'lmi'  # ffmi uses lmi functions
        
        # Get LMS values for target age
        L_func = lms_functions[f'{lms_key}_L']
        M_func = lms_functions[f'{lms_key}_M'] 
        S_func = lms_functions[f'{lms_key}_S']
        
        L_val = L_func(target_age)
        M_val = M_func(target_age)
        S_val = S_func(target_age)
        
        # Get target metric value for this percentile at target age
        target_z = stats.norm.ppf(target_percentile)
        target_metric_value = get_value_from_zscore(target_z, L_val, M_val, S_val)
        
        if pd.isna(target_metric_value):
            return False
            
        # Calculate required gain
        required_gain = target_metric_value - current_metric
        
        if required_gain <= 0:
            return True  # Already at or above target
            
        # Convert metric gain to lean mass gain (approximate)
        height_m = user_info['height_in'] * 0.0254
        height_m2 = height_m ** 2
        
        if metric == 'almi':
            # For ALMI, we need ALM gain. Assume ALM is ~45% of total lean mass
            alm_gain_needed_kg = required_gain * height_m2
            # Convert ALM gain to total lean mass gain (ALM/TLM ratio ~0.45)
            tlm_gain_needed_kg = alm_gain_needed_kg / 0.45
        else:  # ffmi
            # For FFMI, direct lean mass gain
            tlm_gain_needed_kg = required_gain * height_m2
            
        # Calculate time needed
        time_needed_months = time_needed_months = tlm_gain_needed_kg / monthly_gain_rate_kg if monthly_gain_rate_kg > 0 else float('inf')
        time_needed_years = time_needed_months / 12
        
        # Check if achievable within timeframe
        return (current_age + time_needed_years) <= target_age
    
    # Binary search for optimal target age
    best_age = None
    for _ in range(20):  # Max iterations
        mid_age = (min_age + max_age) / 2
        
        if can_reach_goal_at_age(mid_age):
            best_age = mid_age
            max_age = mid_age
        else:
            min_age = mid_age
            
        if abs(max_age - min_age) < 0.1:  # Converged
            break
    
    if best_age is None:
        # If no feasible solution found, use a reasonable timeframe
        best_age = current_age + 2  # 2 years as fallback
        messages.append(f"Could not find feasible timeframe for {target_percentile*100:.0f}th percentile {metric.upper()}")
        messages.append(f"Using 2-year timeframe as fallback (age {best_age:.1f})")
        print(f"  Could not find feasible timeframe for {target_percentile*100:.0f}th percentile {metric.upper()}")
        print(f"  Using 2-year timeframe as fallback (age {best_age:.1f})")
    else:
        time_to_goal = best_age - current_age
        messages.append(f"Calculated feasible timeframe: {time_to_goal:.1f} years (age {best_age:.1f}) based on {training_level} progression rates")
        print(f"  ✓ Calculated feasible timeframe: {time_to_goal:.1f} years (age {best_age:.1f}) based on {training_level} progression rates")
    
    # Update goal parameters
    updated_goal = goal_params.copy()
    updated_goal['target_age'] = best_age
    updated_goal['suggested'] = True
    
    return updated_goal, messages


# ---------------------------------------------------------------------------
# DATA PROCESSING AND ORCHESTRATION
# ---------------------------------------------------------------------------

def load_config_json(config_path, quiet=False):
    """
    Loads and validates a JSON configuration file.
    
    Args:
        config_path (str): Path to the JSON configuration file.
        quiet (bool): If True, suppress print statements
        
    Returns:
        dict: Configuration dictionary with user_info, scan_history, and goal sections.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        json.JSONDecodeError: If the JSON is malformed.
        ValidationError: If the JSON doesn't match the required schema.
    """
    if not quiet:
        print(f"Loading configuration from {config_path}...")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate against JSON schema
    validate(config, CONFIG_SCHEMA)
    
    if not quiet:
        print(f"Successfully loaded config with {len(config['scan_history'])} scans")
    return config

def parse_gender(gender_str):
    """
    Converts user-friendly gender string to gender code for LMS data loading.
    
    Args:
        gender_str (str): Gender as string (m, f, male, female - case insensitive)
        
    Returns:
        int: Gender code (0 for male, 1 for female)
        
    Raises:
        ValueError: If gender string is not recognized
    """
    gender_lower = gender_str.lower()
    if gender_lower in ['m', 'male']:
        return 0
    elif gender_lower in ['f', 'female']:
        return 1
    else:
        raise ValueError(f"Unrecognized gender: {gender_str}. Use 'm', 'f', 'male', or 'female'.")

def extract_data_from_config(config):
    """
    Extracts and processes user info, scan history, and goals from config.
    
    Args:
        config (dict): Validated configuration dictionary
        
    Returns:
        tuple: (user_info, scan_history, almi_goal, ffmi_goal)
    """
    # Process user info
    user_info = config['user_info'].copy()
    user_info['gender_code'] = parse_gender(user_info['gender'])
    
    # Add birth_date_str for test compatibility
    if 'birth_date' in user_info:
        user_info['birth_date_str'] = user_info['birth_date']
    
    # Extract scan history and add date_str for compatibility
    scan_history = []
    for scan in config['scan_history']:
        scan_copy = scan.copy()
        if 'date' in scan_copy:
            scan_copy['date_str'] = scan_copy['date']
        scan_history.append(scan_copy)
    
    # Extract goals (optional)
    goals = config.get('goals', {})
    almi_goal = goals.get('almi')
    ffmi_goal = goals.get('ffmi')
    
    return user_info, scan_history, almi_goal, ffmi_goal

def get_alm_tlm_ratio(processed_data, goal_params, lms_functions, user_info):
    """
    Intelligently estimates ALM/TLM ratio using available scan data.
    
    Uses personalized ratios from recent scans when available, or falls back
    to population-based estimates for single scans.
    
    Args:
        processed_data (pd.DataFrame or list): DataFrame with scan history or list of dicts
        goal_params (dict): Goal parameters  
        lms_functions (dict): LMS interpolation functions
        user_info (dict): User information
        
    Returns:
        float: ALM/TLM ratio to use for calculations
    """
    # If we have multiple scans, use personalized ratio from recent data
    if len(processed_data) >= 2:
        if hasattr(processed_data, 'tail'):
            # DataFrame format
            recent_scans = processed_data.tail(min(3, len(processed_data)))
            alm_values = recent_scans['alm_kg']
            tlm_values = recent_scans['total_lean_mass_lbs'] * 0.453592  # Convert to kg
            
            # Calculate ratio for each scan and take the mean
            ratios = alm_values / tlm_values
            personal_ratio = ratios.mean()
        else:
            # List format (for tests)
            recent_scans = processed_data[-min(3, len(processed_data)):]
            ratios = []
            
            for scan in recent_scans:
                if 'alm_lbs' in scan:
                    alm_kg = scan['alm_lbs'] * 0.453592
                    tlm_kg = scan['total_lean_mass_lbs'] * 0.453592
                    ratios.append(alm_kg / tlm_kg)
            
            if ratios:
                personal_ratio = sum(ratios) / len(ratios)
            else:
                # Fallback to population ratio if no ALM data
                gender_str = get_gender_string(user_info['gender_code'])
                return 0.45 if gender_str == 'male' else 0.42
        
        print(f"Using personal ALM/TLM ratio of {personal_ratio:.3f} from {len(recent_scans)} recent scans")
        return personal_ratio
    
    # For single scans, use population-based estimates from LMS functions
    if goal_params and 'target_age' in goal_params and goal_params['target_age']:
        target_age = goal_params['target_age']
    else:
        # Use current age if no target age specified
        if hasattr(processed_data, 'iloc'):
            target_age = processed_data.iloc[-1]['age_at_scan']
        else:
            target_age = processed_data[-1]['age_at_scan']
    
    # Calculate ratio from LMS functions at target age
    try:
        if 'almi_M' in lms_functions and 'lmi_M' in lms_functions:
            almi_at_age = lms_functions['almi_M'](target_age)
            lmi_at_age = lms_functions['lmi_M'](target_age)
            population_ratio = almi_at_age / lmi_at_age
        else:
            # Fallback to fixed ratios if LMS functions not available
            gender_str = get_gender_string(user_info['gender_code'])
            population_ratio = 0.45 if gender_str == 'male' else 0.42
    except:
        # Fallback to fixed ratios if calculation fails
        gender_str = get_gender_string(user_info['gender_code'])
        population_ratio = 0.45 if gender_str == 'male' else 0.42
    
    gender_str = get_gender_string(user_info['gender_code'])
    print(f"Using population-based ALM/TLM ratio of {population_ratio:.3f} for {gender_str} (single scan)")
    return population_ratio

def load_lms_data(metric, gender_code, data_path="./data/"):
    """
    Loads LMS reference data and creates interpolation functions.
    
    Args:
        metric (str): Either 'appendicular_LMI' for ALMI or 'LMI' for FFMI.
        gender_code (int): 0 for male, 1 for female.
        data_path (str): Path to the directory containing LMS CSV files.
        
    Returns:
        tuple: (L_func, M_func, S_func) - interpolation functions for LMS parameters.
               Returns (None, None, None) if loading fails.
    """
    filename = f"adults_LMS_{metric}_gender{gender_code}.csv"
    filepath = os.path.join(data_path, filename)
    
    if not os.path.exists(filepath):
        print(f"Error: LMS data file not found: {filepath}")
        return None, None, None
    
    try:
        df = pd.read_csv(filepath)
        
        # Handle different column naming conventions
        column_mapping = {
            'age': 'Age',
            'lambda': 'L', 
            'mu': 'M',
            'sigma': 'S'
        }
        
        # Rename columns if they use lowercase names
        df_renamed = df.rename(columns=column_mapping)
        required_columns = ['Age', 'L', 'M', 'S']
        
        if not all(col in df_renamed.columns for col in required_columns):
            print(f"Error: LMS file {filepath} missing required columns: {required_columns}")
            print(f"Available columns: {list(df.columns)}")
            return None, None, None
        
        df = df_renamed
        
        # Create interpolation functions
        L_func = interp1d(df['Age'], df['L'], kind='linear', bounds_error=False, fill_value='extrapolate')
        M_func = interp1d(df['Age'], df['M'], kind='linear', bounds_error=False, fill_value='extrapolate')
        S_func = interp1d(df['Age'], df['S'], kind='linear', bounds_error=False, fill_value='extrapolate')
        
        print(f"Successfully loaded LMS data: {filename}")
        return L_func, M_func, S_func
        
    except Exception as e:
        print(f"Error loading LMS data from {filepath}: {e}")
        return None, None, None

def process_scans_and_goal(user_info, scan_history, almi_goal, ffmi_goal, lms_functions):
    """
    Main processing function that handles scan data and goal calculations.
    
    This function:
    1. Processes all scans to calculate body composition metrics
    2. Calculates changes between scans
    3. Processes goals and adds goal rows if specified
    4. Returns comprehensive DataFrame with all metrics and goal calculations
    
    Args:
        user_info (dict): User information including birth_date, height, gender
        scan_history (list): List of scan dictionaries
        almi_goal (dict): ALMI goal parameters (optional)
        ffmi_goal (dict): FFMI goal parameters (optional)  
        lms_functions (dict): Dictionary containing LMS interpolation functions
        
    Returns:
        tuple: (df_results, goal_calculations) where df_results is the main DataFrame
               and goal_calculations contains goal-specific information
    """
    print("Processing scan history and goals...")
    
    # Convert scan history to DataFrame and sort by date
    df = pd.DataFrame(scan_history)
    
    # Handle both 'date' and 'date_str' field names for compatibility
    if 'date' in df.columns:
        df['scan_date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    elif 'date_str' in df.columns:
        df['scan_date'] = pd.to_datetime(df['date_str'], format='%m/%d/%Y')
    else:
        # For test data without dates, create dummy dates
        df['scan_date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='6M')
    
    df = df.sort_values('scan_date').reset_index(drop=True)
    
    # Calculate basic metrics for each scan
    results = []
    goal_calculations = {}
    
    for i, scan in df.iterrows():
        # Calculate age at scan
        # Handle both birth_date and birth_date_str for test compatibility
        birth_date_key = 'birth_date' if 'birth_date' in user_info else 'birth_date_str'
        scan_date_key = 'date' if 'date' in scan else 'date_str'
        age_at_scan = calculate_age_precise(user_info[birth_date_key], scan[scan_date_key])
        
        # Convert height to meters
        height_m = user_info['height_in'] * 0.0254
        height_m2 = height_m ** 2
        
        # Calculate appendicular lean mass (ALM) from arms + legs
        alm_kg = (scan['arms_lean_lbs'] + scan['legs_lean_lbs']) * 0.453592
        
        # Calculate ALMI and FFMI
        almi_kg_m2 = alm_kg / height_m2
        ffmi_kg_m2 = (scan['total_lean_mass_lbs'] * 0.453592) / height_m2
        
        # Calculate Z-scores and percentiles using LMS functions
        almi_z, almi_percentile = calculate_z_percentile(
            almi_kg_m2, age_at_scan, 
            lms_functions['almi_L'], lms_functions['almi_M'], lms_functions['almi_S']
        )
        
        ffmi_z, ffmi_percentile = calculate_z_percentile(
            ffmi_kg_m2, age_at_scan,
            lms_functions['lmi_L'], lms_functions['lmi_M'], lms_functions['lmi_S']
        )
        
        # Store all calculated values
        result = {
            'date_str': scan[scan_date_key],
            'scan_date': scan['scan_date'] if 'scan_date' in scan else pd.to_datetime(scan[scan_date_key], format='%m/%d/%Y'),
            'age_at_scan': age_at_scan,
            'total_weight_lbs': scan['total_weight_lbs'],
            'total_lean_mass_lbs': scan['total_lean_mass_lbs'],
            'fat_mass_lbs': scan['fat_mass_lbs'],
            'body_fat_percentage': scan['body_fat_percentage'],
            'arms_lean_lbs': scan['arms_lean_lbs'],
            'legs_lean_lbs': scan['legs_lean_lbs'],
            'alm_kg': alm_kg,
            'almi_kg_m2': almi_kg_m2,
            'ffmi_kg_m2': ffmi_kg_m2,
            'almi_z_score': almi_z,
            'almi_percentile': almi_percentile * 100,
            'ffmi_z_score': ffmi_z,
            'ffmi_percentile': ffmi_percentile * 100
        }
        results.append(result)
    
    # Convert to DataFrame
    processed_data = pd.DataFrame(results)
    
    # Calculate changes between scans
    for i in range(len(processed_data)):
        if i == 0:
            # First scan - no changes to calculate
            processed_data.loc[i, 'weight_change_last'] = np.nan
            processed_data.loc[i, 'lean_change_last'] = np.nan
            processed_data.loc[i, 'fat_change_last'] = np.nan
            processed_data.loc[i, 'bf_change_last'] = np.nan
            processed_data.loc[i, 'almi_z_change_last'] = np.nan
            processed_data.loc[i, 'ffmi_z_change_last'] = np.nan
            processed_data.loc[i, 'almi_pct_change_last'] = np.nan
            processed_data.loc[i, 'ffmi_pct_change_last'] = np.nan
        else:
            # Calculate changes from previous scan
            prev_idx = i - 1
            processed_data.loc[i, 'weight_change_last'] = processed_data.loc[i, 'total_weight_lbs'] - processed_data.loc[prev_idx, 'total_weight_lbs']
            processed_data.loc[i, 'lean_change_last'] = processed_data.loc[i, 'total_lean_mass_lbs'] - processed_data.loc[prev_idx, 'total_lean_mass_lbs']
            processed_data.loc[i, 'fat_change_last'] = processed_data.loc[i, 'fat_mass_lbs'] - processed_data.loc[prev_idx, 'fat_mass_lbs']
            processed_data.loc[i, 'bf_change_last'] = processed_data.loc[i, 'body_fat_percentage'] - processed_data.loc[prev_idx, 'body_fat_percentage']
            processed_data.loc[i, 'almi_z_change_last'] = processed_data.loc[i, 'almi_z_score'] - processed_data.loc[prev_idx, 'almi_z_score']
            processed_data.loc[i, 'ffmi_z_change_last'] = processed_data.loc[i, 'ffmi_z_score'] - processed_data.loc[prev_idx, 'ffmi_z_score']
            processed_data.loc[i, 'almi_pct_change_last'] = processed_data.loc[i, 'almi_percentile'] - processed_data.loc[prev_idx, 'almi_percentile']
            processed_data.loc[i, 'ffmi_pct_change_last'] = processed_data.loc[i, 'ffmi_percentile'] - processed_data.loc[prev_idx, 'ffmi_percentile']
        
        # Calculate changes from first scan
        if i == 0:
            processed_data.loc[i, 'weight_change_first'] = 0
            processed_data.loc[i, 'lean_change_first'] = 0
            processed_data.loc[i, 'fat_change_first'] = 0
            processed_data.loc[i, 'bf_change_first'] = 0
        else:
            processed_data.loc[i, 'weight_change_first'] = processed_data.loc[i, 'total_weight_lbs'] - processed_data.loc[0, 'total_weight_lbs']
            processed_data.loc[i, 'lean_change_first'] = processed_data.loc[i, 'total_lean_mass_lbs'] - processed_data.loc[0, 'total_lean_mass_lbs']
            processed_data.loc[i, 'fat_change_first'] = processed_data.loc[i, 'fat_mass_lbs'] - processed_data.loc[0, 'fat_mass_lbs']
            processed_data.loc[i, 'bf_change_first'] = processed_data.loc[i, 'body_fat_percentage'] - processed_data.loc[0, 'body_fat_percentage']
    
    # Process goals and add goal rows if specified
    goal_rows = []
    
    if almi_goal:
        print(f"Processing {'suggested ' if almi_goal.get('suggested') else ''}ALMI goal: {almi_goal['target_percentile']*100:.0f}th percentile")
        
        # Handle suggested goals (auto-calculate target age)
        if almi_goal.get('target_age') in [None, "?"] or almi_goal.get('suggested'):
            almi_goal, almi_messages = calculate_suggested_goal(almi_goal, user_info, processed_data, lms_functions, 'almi')
            if almi_messages:
                goal_calculations['messages'] = goal_calculations.get('messages', []) + almi_messages
        
        # Only create goal row if we have a valid goal (not None)
        if almi_goal is not None:
            goal_row, goal_calc = create_goal_row(
                almi_goal, user_info, processed_data, lms_functions, 'almi'
            )
            if goal_row is not None:
                goal_rows.append(goal_row)
                goal_calculations['almi'] = goal_calc
    
    if ffmi_goal:
        print(f"Processing {'suggested ' if ffmi_goal.get('suggested') else ''}FFMI goal: {ffmi_goal['target_percentile']*100:.0f}th percentile")
        
        # Handle suggested goals (auto-calculate target age)  
        if ffmi_goal.get('target_age') in [None, "?"] or ffmi_goal.get('suggested'):
            ffmi_goal, ffmi_messages = calculate_suggested_goal(ffmi_goal, user_info, processed_data, lms_functions, 'ffmi')
            if ffmi_messages:
                goal_calculations['messages'] = goal_calculations.get('messages', []) + ffmi_messages
            
        # Only create goal row if we have a valid goal (not None)
        if ffmi_goal is not None:
            goal_row, goal_calc = create_goal_row(
                ffmi_goal, user_info, processed_data, lms_functions, 'ffmi'
            )
            if goal_row is not None:
                goal_rows.append(goal_row)
                goal_calculations['ffmi'] = goal_calc
    
    # Combine scan data with goal rows
    if goal_rows:
        goal_df = pd.DataFrame(goal_rows)
        df_results = pd.concat([processed_data, goal_df], ignore_index=True)
    else:
        df_results = processed_data
    
    return df_results, goal_calculations


def get_bf_category(bf_percentage, gender):
    """Determine the category of body fat percentage for a given gender."""
    ranges = HEALTHY_BF_RANGES[gender]
    
    if ranges['athletic'][0] <= bf_percentage <= ranges['athletic'][1]:
        return 'athletic'
    elif ranges['fitness'][0] <= bf_percentage <= ranges['fitness'][1]:
        return 'fitness'
    elif ranges['acceptable'][0] <= bf_percentage <= ranges['acceptable'][1]:
        return 'acceptable'
    elif bf_percentage >= ranges['overweight'][0]:
        return 'overweight'
    else:
        # Handle edge cases between ranges - assign to closest range
        if bf_percentage < ranges['athletic'][0]:
            return 'athletic'  # Below athletic range (very lean)
        elif bf_percentage < ranges['fitness'][0]:
            return 'athletic'  # Between athletic and fitness
        elif bf_percentage < ranges['acceptable'][0]:
            return 'fitness'   # Between fitness and acceptable
        else:
            return 'acceptable'  # Between acceptable and overweight


def calculate_target_bf_percentage(current_bf, gender, goal_duration_months, training_level='intermediate'):
    """
    Calculate target body fat percentage when none is specified.
    
    Args:
        current_bf (float): Current body fat percentage
        gender (str): User's gender ('male' or 'female')
        goal_duration_months (float): Duration until goal age in months
        training_level (str): Training level for determining feasible rate of change
        
    Returns:
        float: Target body fat percentage
    """
    ranges = HEALTHY_BF_RANGES[gender]
    current_category = get_bf_category(current_bf, gender)
    
    # If already in athletic or fitness range, maintain current BF%
    if current_category in ['athletic', 'fitness']:
        return current_bf
    
    # If in acceptable range, aim for upper fitness range (easier to maintain)
    if current_category == 'acceptable':
        if gender == 'male':
            return 17.0  # Upper fitness range for males
        else:
            return 24.0  # Upper fitness range for females
    
    # If overweight, calculate feasible target based on duration
    # Conservative fat loss rate: 0.5-1% BF per month for overweight individuals
    # More aggressive: 1-2% BF per month with proper training
    max_bf_loss_rate = 1.0 if training_level in ['intermediate', 'advanced'] else 0.5
    max_feasible_loss = goal_duration_months * max_bf_loss_rate
    
    # Target the fitness range, but don't exceed feasible loss rate
    target_fitness_bf = ranges['fitness'][1]  # Upper fitness range
    feasible_target = max(current_bf - max_feasible_loss, target_fitness_bf)
    
    # Ensure we don't go below healthy minimums
    healthy_minimum = ranges['athletic'][0]
    return max(feasible_target, healthy_minimum)


def create_goal_row(goal_params, user_info, processed_data, lms_functions, metric):
    """
    Creates a goal row with target body composition values and required changes.
    
    Args:
        goal_params (dict): Goal parameters
        user_info (dict): User information
        processed_data (pd.DataFrame): Processed scan data
        lms_functions (dict): LMS interpolation functions
        metric (str): 'almi' or 'ffmi'
        
    Returns:
        tuple: (goal_row_dict, goal_calculations_dict)
    """
    try:
        target_age = goal_params['target_age']
        target_percentile = goal_params['target_percentile']
        
        # Map metric names to LMS function keys
        lms_key = 'almi' if metric == 'almi' else 'lmi'  # ffmi uses lmi functions
        
        # Get LMS values for target age
        L_func = lms_functions[f'{lms_key}_L']
        M_func = lms_functions[f'{lms_key}_M'] 
        S_func = lms_functions[f'{lms_key}_S']
        
        L_val = L_func(target_age)
        M_val = M_func(target_age)
        S_val = S_func(target_age)
        
        # Calculate target metric value
        target_z = stats.norm.ppf(target_percentile)
        target_metric_value = get_value_from_zscore(target_z, L_val, M_val, S_val)
        
        if pd.isna(target_metric_value):
            print(f"  Warning: Could not calculate target {metric.upper()} value")
            return None, None
        
        # Get current state from last scan
        current_scan = processed_data.iloc[-1]
        current_metric = current_scan[f'{metric}_kg_m2']
        
        # Calculate required metric change
        metric_change_needed = target_metric_value - current_metric
        
        # Convert to lean mass changes using ALM/TLM ratio
        height_m = user_info['height_in'] * 0.0254
        height_m2 = height_m ** 2
        
        # Get ALM/TLM ratio for calculations
        alm_tlm_ratio = get_alm_tlm_ratio(processed_data, goal_params, lms_functions, user_info)
        
        if metric == 'almi':
            # ALMI change -> ALM change -> TLM change
            alm_change_needed_kg = metric_change_needed * height_m2
            tlm_change_needed_kg = alm_change_needed_kg / alm_tlm_ratio
        else:  # ffmi
            # FFMI change -> direct TLM change
            tlm_change_needed_kg = metric_change_needed * height_m2
            alm_change_needed_kg = tlm_change_needed_kg * alm_tlm_ratio
        
        # Convert to pounds
        alm_change_needed_lbs = alm_change_needed_kg * 2.20462
        tlm_change_needed_lbs = tlm_change_needed_kg * 2.20462
        
        # Calculate target body composition
        if 'target_body_fat_percentage' in goal_params and goal_params['target_body_fat_percentage'] is not None:
            # Use explicitly specified target BF%
            target_body_fat_pct = goal_params['target_body_fat_percentage']
        else:
            # Calculate intelligent target BF% based on health and feasibility
            current_bf = current_scan['body_fat_percentage']
            current_age = current_scan['age_at_scan']
            goal_duration_months = (target_age - current_age) * 12  # Convert years to months
            training_level = user_info.get('training_level', 'intermediate')
            # Handle both gender string and gender_code
            if 'gender' in user_info:
                gender = user_info['gender']
            else:
                gender = get_gender_string(user_info['gender_code'])
            
            target_body_fat_pct = calculate_target_bf_percentage(
                current_bf, gender, goal_duration_months, training_level
            )
            
            # Show BF% targeting rationale
            current_category = get_bf_category(current_bf, gender)
            if current_category in ['athletic', 'fitness']:
                print(f"  Current BF% ({current_bf:.1f}%) is in {current_category} range - maintaining current level")
            elif current_category == 'acceptable':
                print(f"  Current BF% ({current_bf:.1f}%) is acceptable - targeting upper fitness range ({target_body_fat_pct:.1f}%)")
            else:
                print(f"  Current BF% ({current_bf:.1f}%) is above healthy range - targeting feasible improvement to {target_body_fat_pct:.1f}% over {goal_duration_months:.1f} months")
        
        target_lean_mass_lbs = current_scan['total_lean_mass_lbs'] + tlm_change_needed_lbs
        
        # Calculate target weight and fat mass
        # If BF% is specified, calculate weight to achieve that BF% with target lean mass
        target_fat_mass_lbs = (target_lean_mass_lbs * target_body_fat_pct) / (100 - target_body_fat_pct)
        target_weight_lbs = target_lean_mass_lbs + target_fat_mass_lbs
        
        # Calculate changes from current state
        weight_change = target_weight_lbs - current_scan['total_weight_lbs']
        lean_change = target_lean_mass_lbs - current_scan['total_lean_mass_lbs']
        fat_change = target_fat_mass_lbs - current_scan['fat_mass_lbs']
        bf_change = target_body_fat_pct - current_scan['body_fat_percentage']
        
        # Calculate percentile changes
        current_percentile = current_scan[f'{metric}_percentile']  # This is in percentage format (0-100)
        percentile_change = (target_percentile * 100) - current_percentile  # Convert target to percentage first
        
        # Calculate Z-score changes
        current_z = current_scan[f'{metric}_z_score']
        z_change = target_z - current_z
        
        print(f"{metric.upper()} goal calculations: ALM to add: {alm_change_needed_lbs:.1f} lbs ({alm_change_needed_kg:.2f} kg), Est. TLM gain: {tlm_change_needed_lbs:.1f} lbs ({tlm_change_needed_kg:.2f} kg), Target BF: {target_body_fat_pct:.1f}%, Total weight change: {weight_change:+.1f} lbs")
        
        # Create goal row
        goal_row = {
            'date_str': f"{metric.upper()} Goal (Age {target_age})",
            'scan_date': pd.Timestamp.now(),  # Placeholder
            'age_at_scan': target_age,
            'total_weight_lbs': target_weight_lbs,
            'total_lean_mass_lbs': target_lean_mass_lbs,
            'fat_mass_lbs': target_fat_mass_lbs,
            'body_fat_percentage': target_body_fat_pct,
            'arms_lean_lbs': np.nan,  # Not calculated for goals
            'legs_lean_lbs': np.nan,  # Not calculated for goals
            'alm_kg': np.nan,  # Not calculated for goals
            'almi_kg_m2': target_metric_value if metric == 'almi' else current_scan['almi_kg_m2'],
            'ffmi_kg_m2': target_metric_value if metric == 'ffmi' else current_scan['ffmi_kg_m2'],
            'almi_z_score': target_z if metric == 'almi' else current_scan['almi_z_score'],
            'almi_percentile': target_percentile * 100 if metric == 'almi' else current_scan['almi_percentile'],
            'ffmi_z_score': target_z if metric == 'ffmi' else current_scan['ffmi_z_score'],
            'ffmi_percentile': target_percentile * 100 if metric == 'ffmi' else current_scan['ffmi_percentile'],
            'weight_change_last': weight_change,
            'lean_change_last': lean_change,
            'fat_change_last': fat_change,
            'bf_change_last': bf_change,
            'almi_z_change_last': z_change if metric == 'almi' else current_scan.get('almi_z_change_last', 0),
            'ffmi_z_change_last': z_change if metric == 'ffmi' else current_scan.get('ffmi_z_change_last', 0),
            'almi_pct_change_last': percentile_change if metric == 'almi' else current_scan.get('almi_pct_change_last', 0),
            'ffmi_pct_change_last': percentile_change if metric == 'ffmi' else current_scan.get('ffmi_pct_change_last', 0),
            'weight_change_first': target_weight_lbs - processed_data.iloc[0]['total_weight_lbs'],
            'lean_change_first': target_lean_mass_lbs - processed_data.iloc[0]['total_lean_mass_lbs'],
            'fat_change_first': target_fat_mass_lbs - processed_data.iloc[0]['fat_mass_lbs'],
            'bf_change_first': target_body_fat_pct - processed_data.iloc[0]['body_fat_percentage']
        }
        
        # Goal calculations for plotting
        goal_calculations = {
            'target_age': target_age,
            'target_percentile': target_percentile,
            'target_metric_value': target_metric_value,
            'target_z_score': target_z,
            'metric_change_needed': metric_change_needed,
            'lean_change_needed_lbs': tlm_change_needed_lbs,
            'alm_change_needed_lbs': alm_change_needed_lbs,
            'alm_change_needed_kg': alm_change_needed_kg,
            'tlm_change_needed_lbs': tlm_change_needed_lbs,
            'tlm_change_needed_kg': tlm_change_needed_kg,
            'weight_change': weight_change,
            'lean_change': lean_change,
            'fat_change': fat_change,
            'bf_change': bf_change,
            'percentile_change': percentile_change,
            'z_change': z_change,
            'target_body_composition': {
                'weight_lbs': target_weight_lbs,
                'lean_mass_lbs': target_lean_mass_lbs,
                'fat_mass_lbs': target_fat_mass_lbs,
                'body_fat_percentage': target_body_fat_pct
            },
            # Backwards compatibility field names for tests
            'alm_to_add_kg': alm_change_needed_kg,
            'estimated_tlm_gain_kg': tlm_change_needed_kg,
            'tlm_to_add_kg': tlm_change_needed_kg
        }
        
        return goal_row, goal_calculations
        
    except Exception as e:
        print(f"  Error creating {metric.upper()} goal row: {e}")
        return None, None


# ---------------------------------------------------------------------------
# PLOTTING LOGIC
# ---------------------------------------------------------------------------

def create_metric_plot(df_results, metric_to_plot, lms_functions, goal_calculations, return_figure=False):
    """
    Creates comprehensive plots showing percentile curves, data points, and optionally returns figure.
    
    This function generates detailed visualizations that include:
    - LMS-based percentile curves (3rd, 10th, 25th, 50th, 75th, 90th, 97th)
    - User's actual data points plotted over time
    - Goal markers and projections if goals are specified
    
    Args:
        df_results (pd.DataFrame): Complete results DataFrame with scan history and goals
        metric_to_plot (str): Either 'ALMI' or 'FFMI' 
        lms_functions (dict): Dictionary containing LMS interpolation functions
        goal_calculations (dict): Goal calculation results
        return_figure (bool): If True, returns matplotlib figure object instead of saving
        
    Returns:
        matplotlib.figure.Figure or None: Figure object if return_figure=True, None otherwise
    """
    if not return_figure:
        print(f"Generating plot for {metric_to_plot}...")
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define age range for percentile curves
    age_range = np.linspace(18, 80, 100)
    
    # Select appropriate LMS functions
    if metric_to_plot == 'ALMI':
        L_func = lms_functions['almi_L']
        M_func = lms_functions['almi_M']
        S_func = lms_functions['almi_S']
        y_column = 'almi_kg_m2'
        y_label = 'ALMI (kg/m²)'
        plot_title = 'Appendicular Lean Mass Index (ALMI) Percentiles'
    else:  # FFMI
        L_func = lms_functions['lmi_L']
        M_func = lms_functions['lmi_M']
        S_func = lms_functions['lmi_S']
        y_column = 'ffmi_kg_m2'
        y_label = 'FFMI (kg/m²)'
        plot_title = 'Fat-Free Mass Index (FFMI) Percentiles'
    
    # Define percentiles to plot
    percentiles = [0.03, 0.10, 0.25, 0.50, 0.75, 0.90, 0.97]
    percentile_labels = ['3rd', '10th', '25th', '50th', '75th', '90th', '97th']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECCA7', '#DDA0DD', '#FFB347']
    
    # Plot percentile curves
    for i, (percentile, label, color) in enumerate(zip(percentiles, percentile_labels, colors)):
        z_score = stats.norm.ppf(percentile)
        curve_values = []
        
        for age in age_range:
            try:
                L_val = L_func(age)
                M_val = M_func(age)
                S_val = S_func(age)
                value = get_value_from_zscore(z_score, L_val, M_val, S_val)
                curve_values.append(value)
            except:
                curve_values.append(np.nan)
        
        ax.plot(age_range, curve_values, color=color, linewidth=2, label=f'{label} percentile', alpha=0.8)
    
    # Filter data for actual scans (not goal rows)
    scan_data = df_results[~df_results['date_str'].str.contains('Goal', na=False)]
    
    # Plot actual data points
    if len(scan_data) > 0:
        ax.scatter(scan_data['age_at_scan'], scan_data[y_column], 
                   color='red', s=100, zorder=5, label='Your scans', edgecolors='black', linewidth=1)
        
        # Connect points with lines if multiple scans
        if len(scan_data) > 1:
            ax.plot(scan_data['age_at_scan'], scan_data[y_column], 
                    color='red', linewidth=2, alpha=0.7, zorder=4)
    
    # Plot goal if available
    goal_key = metric_to_plot.lower()
    if goal_key in goal_calculations:
        goal_calc = goal_calculations[goal_key]
        goal_age = goal_calc['target_age']
        goal_value = goal_calc['target_metric_value']
        
        ax.scatter([goal_age], [goal_value], color='gold', s=150, marker='*', 
                   zorder=6, label='Goal', edgecolors='black', linewidth=1)
        
        # Draw line from last scan to goal
        if len(scan_data) > 0:
            last_scan = scan_data.iloc[-1]
            ax.plot([last_scan['age_at_scan'], goal_age], 
                    [last_scan[y_column], goal_value],
                    color='gold', linewidth=3, linestyle='--', alpha=0.8, zorder=3)
    
    # Customize plot
    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if return_figure:
        return fig
    else:
        # Save plot
        filename = f"{metric_to_plot.lower()}_plot.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved as: {filename}")
        return None

def plot_metric_with_table(df_results, metric_to_plot, lms_functions, goal_calculations):
    """
    Legacy function that creates plots and saves them to disk.
    Uses the new create_metric_plot function for actual plotting.
    """
    # Use the new function to create and save the plot
    create_metric_plot(df_results, metric_to_plot, lms_functions, goal_calculations, return_figure=False)
    
    # Export data table to CSV (only for ALMI to avoid duplication)
    if metric_to_plot == 'ALMI':
        csv_filename = 'almi_stats_table.csv'
        df_results.to_csv(csv_filename, index=False)
        print(f"Table data saved as: {csv_filename}")

def create_body_fat_plot(df_results, user_info, return_figure=False):
    """
    Creates a line plot showing body fat percentage progression over time.
    
    Args:
        df_results (pd.DataFrame): Complete results DataFrame with scan history
        user_info (dict): User information including gender for health ranges
        return_figure (bool): If True, returns matplotlib figure object instead of saving
        
    Returns:
        matplotlib.figure.Figure or None: Figure object if return_figure=True, None otherwise
    """
    if not return_figure:
        print("Generating body fat percentage plot...")
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter data for actual scans (not goal rows)
    scan_data = df_results[~df_results['date_str'].str.contains('Goal', na=False)]
    
    if len(scan_data) == 0:
        if return_figure:
            return fig
        else:
            plt.close()
            print("No scan data found for body fat plot")
            return None
    
    # Get gender for health ranges
    if 'gender' in user_info:
        gender = user_info['gender']
    else:
        gender = get_gender_string(user_info['gender_code'])
    
    # Add healthy body fat percentage ranges as background shading
    ranges = HEALTHY_BF_RANGES[gender]
    
    # Create age range for background shading
    age_min = scan_data['age_at_scan'].min() - 1
    age_max = scan_data['age_at_scan'].max() + 1
    
    # Add background shading for health ranges
    ax.axhspan(ranges['athletic'][0], ranges['athletic'][1], 
               color='lightgreen', alpha=0.2, label='Athletic Range')
    ax.axhspan(ranges['fitness'][0], ranges['fitness'][1], 
               color='lightblue', alpha=0.2, label='Fitness Range')
    ax.axhspan(ranges['acceptable'][0], ranges['acceptable'][1], 
               color='lightyellow', alpha=0.2, label='Acceptable Range')
    
    # Plot the actual data line
    ax.plot(scan_data['age_at_scan'], scan_data['body_fat_percentage'], 
            color='red', linewidth=3, marker='o', markersize=8, 
            label='Your Body Fat %', zorder=10)
    
    # Add data point annotations with values
    for _, scan in scan_data.iterrows():
        ax.annotate(f'{scan["body_fat_percentage"]:.1f}%', 
                   (scan['age_at_scan'], scan['body_fat_percentage']),
                   textcoords="offset points", xytext=(0,10), ha='center',
                   fontsize=10, fontweight='bold', color='darkred')
    
    # Customize plot
    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel('Body Fat Percentage (%)', fontsize=12)
    ax.set_title('Body Fat Percentage Over Time', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y-axis limits based on data and health ranges
    y_min = min(scan_data['body_fat_percentage'].min() - 2, ranges['athletic'][0] - 1)
    y_max = max(scan_data['body_fat_percentage'].max() + 2, ranges['acceptable'][1] + 1)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    if return_figure:
        return fig
    else:
        # Save plot
        filename = "bf_plot.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Body fat plot saved as: {filename}")
        return None

def create_plotly_body_fat_plot(df_results, user_info):
    """
    Creates interactive Plotly plot for body fat percentage over time.
    
    Args:
        df_results (pd.DataFrame): Complete results DataFrame with scan history
        user_info (dict): User information including gender for health ranges
        
    Returns:
        plotly.graph_objects.Figure: Interactive plotly figure
    """
    # Import plotly here to avoid import errors in CLI-only environments
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        raise ImportError("Plotly is required for interactive body fat plots")
    
    # Create figure
    fig = go.Figure()
    
    # Filter data for actual scans (not goal rows)
    scan_data = df_results[~df_results['date_str'].str.contains('Goal', na=False)]
    
    if len(scan_data) == 0:
        fig.add_annotation(
            text="No scan data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Get gender for health ranges
    if 'gender' in user_info:
        gender = user_info['gender']
    else:
        gender = get_gender_string(user_info['gender_code'])
    
    # Add healthy body fat percentage ranges as background shading
    ranges = HEALTHY_BF_RANGES[gender]
    
    # Create age range for background shading
    age_min = scan_data['age_at_scan'].min() - 1
    age_max = scan_data['age_at_scan'].max() + 1
    
    # Add background shapes for health ranges
    fig.add_hrect(
        y0=ranges['athletic'][0], y1=ranges['athletic'][1],
        fillcolor="lightgreen", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="Athletic Range", annotation_position="top left"
    )
    fig.add_hrect(
        y0=ranges['fitness'][0], y1=ranges['fitness'][1],
        fillcolor="lightblue", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="Fitness Range", annotation_position="top left"
    )
    fig.add_hrect(
        y0=ranges['acceptable'][0], y1=ranges['acceptable'][1],
        fillcolor="lightyellow", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="Acceptable Range", annotation_position="top left"
    )
    
    # Create hover text with comprehensive information
    hover_text = []
    for _, scan in scan_data.iterrows():
        # Calculate body fat category
        bf_pct = scan['body_fat_percentage']
        bf_category = get_bf_category(bf_pct, gender)
        
        # Build hover info
        hover_info = [
            f"<b>Scan Date:</b> {scan['date_str']}",
            f"<b>Age:</b> {scan['age_at_scan']:.1f} years",
            f"<b>Body Fat:</b> {bf_pct:.1f}%",
            f"<b>Category:</b> {bf_category.title()}",
            "",
            f"<b>Weight:</b> {scan['total_weight_lbs']:.1f} lbs",
            f"<b>Lean Mass:</b> {scan['total_lean_mass_lbs']:.1f} lbs",
            f"<b>Fat Mass:</b> {scan['fat_mass_lbs']:.1f} lbs"
        ]
        
        # Add change information if available
        if pd.notna(scan.get('bf_change_last')):
            change_last = scan['bf_change_last']
            change_sign = "+" if change_last >= 0 else ""
            hover_info.extend([
                "",
                f"<b>Change from last scan:</b> {change_sign}{change_last:.1f}%"
            ])
        
        if pd.notna(scan.get('bf_change_first')):
            change_first = scan['bf_change_first']
            change_sign = "+" if change_first >= 0 else ""
            hover_info.extend([
                f"<b>Change from first scan:</b> {change_sign}{change_first:.1f}%"
            ])
        
        hover_text.append("<br>".join(hover_info))
    
    # Add the main data line with markers
    fig.add_trace(go.Scatter(
        x=scan_data['age_at_scan'],
        y=scan_data['body_fat_percentage'],
        mode='lines+markers',
        name='Your Body Fat %',
        line=dict(color='red', width=3),
        marker=dict(
            color='red',
            size=10,
            line=dict(color='black', width=1)
        ),
        hovertemplate='%{text}<extra></extra>',
        text=hover_text
    ))
    
    # Customize layout
    y_min = min(scan_data['body_fat_percentage'].min() - 2, ranges['athletic'][0] - 1)
    y_max = max(scan_data['body_fat_percentage'].max() + 2, ranges['acceptable'][1] + 1)
    
    fig.update_layout(
        title=dict(
            text='Body Fat Percentage Over Time',
            font=dict(size=16, family="Arial", color="black"),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Age (years)', font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        yaxis=dict(
            title=dict(text='Body Fat Percentage (%)', font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor='lightgray',
            gridwidth=0.5,
            range=[y_min, y_max]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font_size=12
        ),
        height=600,
        showlegend=True
    )
    
    return fig


# ---------------------------------------------------------------------------
# COMPARISON TABLE FUNCTIONS
# ---------------------------------------------------------------------------

def create_scan_comparison_table(df_results, return_html=False):
    """
    Creates a comparison table showing first scan vs most recent scan with changes.
    
    Args:
        df_results (pd.DataFrame): Complete results DataFrame with scan history
        return_html (bool): If True, returns HTML table with color coding for web interface
        
    Returns:
        str: Formatted table as string (CLI) or HTML (web)
    """
    # Filter out goal rows to get only actual scans
    scan_data = df_results[~df_results['date_str'].str.contains('Goal', na=False)]
    
    if len(scan_data) < 1:
        return "No scan data available for comparison"
    
    # Get first and last scans
    first_scan = scan_data.iloc[0]
    last_scan = scan_data.iloc[-1] if len(scan_data) > 1 else first_scan
    
    # Prepare data for the table
    table_data = []
    
    # First scan row (no changes)
    first_row = [
        first_scan['date_str'],
        f"{first_scan['total_weight_lbs']:.1f}",
        "-",  # No change for first scan
        f"{first_scan['total_lean_mass_lbs']:.1f}",
        "-",  # No change for first scan
        f"{first_scan['fat_mass_lbs']:.1f}",
        "-",  # No change for first scan
        f"{first_scan['body_fat_percentage']:.1f}%",
        "-"   # No change for first scan
    ]
    
    # Most recent scan row (with changes from first)
    if len(scan_data) > 1:
        weight_change = last_scan['weight_change_first']
        lean_change = last_scan['lean_change_first'] 
        fat_change = last_scan['fat_change_first']
        bf_change = last_scan['bf_change_first']
        
        last_row = [
            last_scan['date_str'],
            f"{last_scan['total_weight_lbs']:.1f}",
            f"{weight_change:+.1f}" if pd.notna(weight_change) else "N/A",
            f"{last_scan['total_lean_mass_lbs']:.1f}",
            f"{lean_change:+.1f}" if pd.notna(lean_change) else "N/A",
            f"{last_scan['fat_mass_lbs']:.1f}",
            f"{fat_change:+.1f}" if pd.notna(fat_change) else "N/A",
            f"{last_scan['body_fat_percentage']:.1f}%",
            f"{bf_change:+.1f}%" if pd.notna(bf_change) else "N/A"
        ]
    else:
        # Only one scan - same as first row
        last_row = first_row.copy()
    
    headers = ['Date', 'Total Mass', 'Change', 'Lean Mass', 'Change', 'Fat Mass', 'Change', 'Body Fat %', 'Change']
    table_data = [first_row, last_row]
    
    if return_html:
        # Create HTML table with color coding
        html = '<table class="scan-comparison-table" style="border-collapse: collapse; margin: 20px 0;">\n'
        
        # Headers
        html += '  <thead>\n    <tr>\n'
        for header in headers:
            html += f'      <th style="border: 1px solid #ddd; padding: 8px; background-color: #f5f5f5; text-align: center;">{header}</th>\n'
        html += '    </tr>\n  </thead>\n'
        
        # Data rows
        html += '  <tbody>\n'
        for i, row in enumerate(table_data):
            html += '    <tr>\n'
            for j, cell in enumerate(row):
                # Determine cell styling based on content and position
                style = "border: 1px solid #ddd; padding: 8px; text-align: center;"
                
                # Color code change cells (odd indices are change columns) in the last row
                if j % 2 == 1 and j > 0 and i == 1:  # Change columns, last row only
                    if cell != "-" and cell != "N/A":
                        # Parse the change value
                        try:
                            if cell.endswith('%'):
                                change_val = float(cell.replace('%', '').replace('+', ''))
                            else:
                                change_val = float(cell.replace('+', ''))
                            
                            # Color coding logic based on column position
                            if j == 3:  # Lean mass change - positive is good
                                if change_val > 0:
                                    style += " background-color: #d4edda; color: #155724;"  # Green
                                elif change_val < 0:
                                    style += " background-color: #f8d7da; color: #721c24;"  # Red
                            elif j == 5:  # Fat mass change - negative is good (fat loss)
                                if change_val < 0:
                                    style += " background-color: #d4edda; color: #155724;"  # Green
                                elif change_val > 0:
                                    style += " background-color: #f8d7da; color: #721c24;"  # Red
                            elif j == 7:  # Body fat % change - negative is good (BF% reduction)
                                if change_val < 0:
                                    style += " background-color: #d4edda; color: #155724;"  # Green
                                elif change_val > 0:
                                    style += " background-color: #f8d7da; color: #721c24;"  # Red
                            # Weight change (j == 1) remains neutral - no color coding
                        except ValueError:
                            pass  # Keep default styling if can't parse
                
                # Also color code the actual values for improved visualization in the last row
                elif j % 2 == 0 and j > 0 and i == 1:  # Value columns (even indices), last row only
                    # Color the actual values based on corresponding changes
                    change_cell_idx = j + 1
                    if change_cell_idx < len(row):
                        change_cell = row[change_cell_idx]
                        if change_cell != "-" and change_cell != "N/A":
                            try:
                                if change_cell.endswith('%'):
                                    change_val = float(change_cell.replace('%', '').replace('+', ''))
                                else:
                                    change_val = float(change_cell.replace('+', ''))
                                
                                # Apply lighter background colors to value cells
                                if j == 2:  # Lean mass value - positive change is good
                                    if change_val > 0:
                                        style += " background-color: #e8f5e8;"  # Light green
                                    elif change_val < 0:
                                        style += " background-color: #fce8e8;"  # Light red
                                elif j == 4:  # Fat mass value - negative change is good
                                    if change_val < 0:
                                        style += " background-color: #e8f5e8;"  # Light green
                                    elif change_val > 0:
                                        style += " background-color: #fce8e8;"  # Light red
                                elif j == 6:  # Body fat % value - negative change is good
                                    if change_val < 0:
                                        style += " background-color: #e8f5e8;"  # Light green
                                    elif change_val > 0:
                                        style += " background-color: #fce8e8;"  # Light red
                            except ValueError:
                                pass
                
                html += f'      <td style="{style}">{cell}</td>\n'
            html += '    </tr>\n'
        
        html += '  </tbody>\n</table>'
        return html
    else:
        # Return plain text table for CLI
        try:
            from tabulate import tabulate
            return tabulate(table_data, headers=headers, tablefmt='pipe')
        except ImportError:
            # Fallback to simple formatting
            header_str = ' | '.join(f"{h:>12}" for h in headers)
            separator = '-' * len(header_str)
            rows = [' | '.join(f"{cell:>12}" for cell in row) for row in table_data]
            return f"{header_str}\n{separator}\n" + "\n".join(rows)


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS FOR WEB INTERFACE
# ---------------------------------------------------------------------------

def generate_fake_profile():
    """
    Generates a realistic fake user profile for testing/demo purposes.
    
    Returns:
        dict: User info dictionary with birth_date, height_in, gender, training_level
    """
    import random
    from datetime import datetime, timedelta
    
    # Generate realistic age (25-45 years old)
    current_year = datetime.now().year
    age = random.randint(25, 45)
    birth_year = current_year - age
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)  # Safe day range
    birth_date = f"{birth_month:02d}/{birth_day:02d}/{birth_year}"
    
    # Generate realistic height
    gender = random.choice(['male', 'female'])
    if gender == 'male':
        height_in = random.uniform(66, 74)  # 5'6" to 6'2"
    else:
        height_in = random.uniform(60, 68)  # 5'0" to 5'8"
    
    training_level = random.choice(['novice', 'intermediate', 'advanced'])
    
    return {
        'birth_date': birth_date,
        'height_in': round(height_in, 1),
        'gender': gender,
        'training_level': training_level
    }

def generate_fake_scans(user_info, num_scans=4):
    """
    Generates realistic fake DEXA scan data showing progression over time.
    Targets 30-50th ALMI percentile starting range for realistic profiles.
    
    Args:
        user_info (dict): User profile with gender and training level
        num_scans (int): Number of scans to generate (2-6)
        
    Returns:
        list: List of scan dictionaries with realistic progression
    """
    import random
    from datetime import datetime, timedelta
    import scipy.stats as stats
    
    scans = []
    gender = user_info['gender']
    training_level = user_info.get('training_level', 'intermediate')
    height_in = user_info['height_in']
    height_m = height_in * 0.0254
    height_m2 = height_m ** 2
    
    # Load LMS data for ALMI calculations
    gender_code = 0 if gender == 'male' else 1
    almi_L, almi_M, almi_S = load_lms_data('appendicular_LMI', gender_code)
    
    # Generate a realistic starting age (25-45)
    from dateutil import parser
    if isinstance(user_info.get('birth_date'), str):
        birth_date = parser.parse(user_info['birth_date'])
        current_age = (datetime.now() - birth_date).days / 365.25
    else:
        current_age = random.uniform(25, 45)
    
    # Target ALMI percentile between 32-52th percentile for realistic starting point
    # Slightly higher range to account for calculation variations
    target_percentile = random.uniform(0.32, 0.52)
    target_z = stats.norm.ppf(target_percentile)
    
    # Calculate target ALMI value for this age/percentile
    l_val = almi_L(current_age)
    m_val = almi_M(current_age)
    s_val = almi_S(current_age)
    target_almi = get_value_from_zscore(target_z, l_val, m_val, s_val)
    
    # Calculate target ALM from ALMI
    target_alm_kg = target_almi * height_m2
    target_alm_lbs = target_alm_kg / 0.453592
    
    # Calculate realistic body composition around this ALMI target
    alm_ratio = random.uniform(0.42, 0.46) if gender == 'male' else random.uniform(0.38, 0.42)
    target_lean_mass = target_alm_lbs / alm_ratio
    
    if gender == 'male':
        initial_bf_pct = random.uniform(15, 25)
        weight_range_factor = random.uniform(0.9, 1.1)
    else:
        initial_bf_pct = random.uniform(20, 30)
        weight_range_factor = random.uniform(0.9, 1.1)
    
    # Calculate weight from lean mass and body fat
    initial_fat_mass = (target_lean_mass * initial_bf_pct) / (100 - initial_bf_pct)
    initial_weight = (target_lean_mass + initial_fat_mass) * weight_range_factor
    
    # Recalculate composition based on final weight
    initial_fat_mass = initial_weight * (initial_bf_pct / 100)
    initial_lean_mass = initial_weight - initial_fat_mass
    initial_alm = target_alm_lbs  # Use our target ALM
    initial_arms_lean = initial_alm * random.uniform(0.33, 0.37)  # ~35% of ALM in arms
    initial_legs_lean = initial_alm * random.uniform(0.63, 0.67)  # ~65% of ALM in legs
    
    # Generate progression rates based on training level (adjusted for longer time periods)
    # These rates are now per 12-18 month periods instead of 3-6 months
    if training_level == 'novice':
        lean_gain_rate = random.uniform(6.0, 12.0)  # lbs per scan period (12-18 months)
        fat_loss_rate = random.uniform(4.0, 10.0)
    elif training_level == 'intermediate':
        lean_gain_rate = random.uniform(2.0, 8.0)
        fat_loss_rate = random.uniform(2.0, 6.0)
    else:  # advanced
        lean_gain_rate = random.uniform(1.0, 4.0)
        fat_loss_rate = random.uniform(1.0, 3.0)
    
    # Generate scans over 36-60 months (3-5 years) for more realistic long-term tracking
    total_period_days = random.randint(1095, 1825)  # 3-5 years
    start_date = datetime.now() - timedelta(days=total_period_days)
    
    for i in range(num_scans):
        # Date progression (12-18 months between scans)
        if i == 0:
            scan_date = start_date
        else:
            days_gap = random.randint(365, 550)  # 12-18 months between scans
            scan_date = scans[-1]['scan_date'] + timedelta(days=days_gap)
        
        # Progressive body composition changes
        if i == 0:
            weight = initial_weight
            lean_mass = initial_lean_mass
            fat_mass = initial_fat_mass
            alm = initial_alm
        else:
            # Add some randomness to progression
            lean_change = lean_gain_rate * random.uniform(0.7, 1.3)
            fat_change = -fat_loss_rate * random.uniform(0.5, 1.2)
            
            lean_mass = max(scans[-1]['total_lean_mass_lbs'] + lean_change, initial_lean_mass * 0.9)
            fat_mass = max(scans[-1]['fat_mass_lbs'] + fat_change, initial_fat_mass * 0.4)
            weight = lean_mass + fat_mass
            
            # ALM should progress proportionally with lean mass, but maintain realistic ratio
            prev_alm = scans[-1]['arms_lean_lbs'] + scans[-1]['legs_lean_lbs']
            alm_change = lean_change * alm_ratio  # ALM changes proportionally
            alm = max(prev_alm + alm_change, initial_alm * 0.95)
        
        # Calculate derived values
        body_fat_percentage = (fat_mass / weight) * 100
        arms_lean = alm * random.uniform(0.33, 0.37)  # Maintain consistent arm/leg distribution
        legs_lean = alm * random.uniform(0.63, 0.67)
        
        scan = {
            'date': scan_date.strftime("%m/%d/%Y"),
            'scan_date': scan_date,  # Keep for sorting, will be removed
            'total_weight_lbs': round(weight, 1),
            'total_lean_mass_lbs': round(lean_mass, 1),
            'fat_mass_lbs': round(fat_mass, 1),
            'body_fat_percentage': round(body_fat_percentage, 1),
            'arms_lean_lbs': round(arms_lean, 1),
            'legs_lean_lbs': round(legs_lean, 1)
        }
        scans.append(scan)
    
    # Remove the helper scan_date field
    for scan in scans:
        del scan['scan_date']
    
    return scans

def get_metric_explanations():
    """
    Returns explanatory text for metrics and tooltips.
    
    Returns:
        dict: Dictionary with explanations for different metrics
    """
    return {
        'header_info': {
            'title': 'DEXA Body Composition Analysis',
            'subtitle': 'Analyze your body composition using scientifically validated percentile curves',
            'almi_explanation': '''
            **ALMI (Appendicular Lean Mass Index)** measures the lean muscle mass in your arms and legs 
            relative to your height. It's calculated as (Arms Lean Mass + Legs Lean Mass) ÷ Height². 
            This metric is important for assessing functional muscle mass and overall strength potential.
            ''',
            'ffmi_explanation': '''
            **FFMI (Fat-Free Mass Index)** measures your total lean body mass relative to your height. 
            It's calculated as Total Lean Mass ÷ Height². This gives a normalized measure of your overall 
            muscle mass that accounts for differences in height.
            ''',
            'percentiles_explanation': '''
            **Percentiles** show how you compare to a reference population. For example, the 75th percentile 
            means you have more muscle mass than 75% of people your age and gender. The reference data comes 
            from the [LEAD cohort study](https://www.nature.com/articles/s41430-020-0596-5) of healthy adults.
            ''',
            'population_source': '''
            Reference data is from the [LEAD cohort](https://www.nature.com/articles/s41430-020-0596-5) (Leadership in Exercise and Active Decisions), 
            a comprehensive study of body composition in healthy adults across different ages.
            '''
        },
        'tooltips': {
            'z_score': 'Z-score: How many standard deviations you are from the population median. Positive values are above average.',
            'percentile': 'Percentile: The percentage of the population with lower values than yours.',
            'almi': 'ALMI (Appendicular Lean Mass Index): Measures lean muscle mass in your arms and legs relative to height (kg/m²). Higher values indicate more functional muscle mass.',
            'ffmi': 'FFMI (Fat-Free Mass Index): Measures your total lean body mass relative to height (kg/m²). Higher values indicate more overall muscle mass.',
            'training_level': 'Training level affects goal suggestions and muscle gain rate estimates.',
            'goal_age': 'Target age to reach your goal. Use "?" for automatic calculation based on realistic progression rates.',
            'target_percentile': 'The percentile you want to reach (e.g., 0.75 = 75th percentile).',
            'body_fat_percentage': 'Percentage of total body weight that is fat mass, measured by DEXA scan.',
            'lean_mass': 'Total muscle, bone, and organ mass excluding fat.',
            'arms_lean': 'Lean mass in both arms combined.',
            'legs_lean': 'Lean mass in both legs combined.'
        }
    }

def validate_user_input(field_name, value, user_data=None):
    """
    Validates user input for real-time feedback in the web interface.
    
    Args:
        field_name (str): Name of the field being validated
        value: The value to validate
        user_data (dict): Other user data for cross-field validation
        
    Returns:
        tuple: (is_valid, error_message)
    """
    from datetime import datetime
    
    if field_name == 'birth_date':
        try:
            birth_date = datetime.strptime(value, "%m/%d/%Y")
            age = (datetime.now() - birth_date).days / 365.25
            if age < 18:
                return False, "Age must be at least 18 years"
            if age > 80:
                return False, "Age must be less than 80 years"
            return True, ""
        except ValueError:
            return False, "Please use MM/DD/YYYY format"
    
    elif field_name == 'height_in':
        try:
            height = float(value)
            if height < 12:
                return False, "Height must be at least 12 inches"
            if height > 120:
                return False, "Height must be less than 120 inches"
            return True, ""
        except (ValueError, TypeError):
            return False, "Please enter a valid number"
    
    elif field_name == 'scan_date':
        try:
            scan_date = datetime.strptime(value, "%m/%d/%Y")
            if user_data and 'birth_date' in user_data:
                birth_date = datetime.strptime(user_data['birth_date'], "%m/%d/%Y")
                if scan_date <= birth_date:
                    return False, "Scan date must be after birth date"
            return True, ""
        except ValueError:
            return False, "Please use MM/DD/YYYY format"
    
    elif field_name in ['total_weight_lbs', 'total_lean_mass_lbs', 'fat_mass_lbs', 
                        'arms_lean_lbs', 'legs_lean_lbs']:
        try:
            weight = float(value)
            if weight <= 0:
                return False, "Value must be greater than 0"
            if weight > 1000:
                return False, "Value seems unreasonably high"
            return True, ""
        except (ValueError, TypeError):
            return False, "Please enter a valid number"
    
    elif field_name == 'body_fat_percentage':
        try:
            bf_pct = float(value)
            if bf_pct <= 0:
                return False, "Body fat percentage must be greater than 0"
            if bf_pct >= 100:
                return False, "Body fat percentage must be less than 100"
            return True, ""
        except (ValueError, TypeError):
            return False, "Please enter a valid number"
    
    elif field_name == 'target_percentile':
        try:
            percentile = float(value)
            if percentile <= 0:
                return False, "Percentile must be greater than 0"
            if percentile >= 1:
                return False, "Percentile must be less than 1 (e.g., 0.75 for 75th percentile)"
            return True, ""
        except (ValueError, TypeError):
            return False, "Please enter a decimal between 0 and 1"
    
    elif field_name == 'target_age':
        if value == "?" or value == "":
            return True, ""
        try:
            age = float(value)
            if age < 18:
                return False, "Target age must be at least 18"
            if age > 80:
                return False, "Target age must be less than 80"
            return True, ""
        except (ValueError, TypeError):
            return False, "Please enter a valid age or '?' for auto-calculation"
    
    return True, ""

# ---------------------------------------------------------------------------
# MAIN ANALYSIS FUNCTION  
# ---------------------------------------------------------------------------

def run_analysis_from_data(user_info, scan_history, almi_goal=None, ffmi_goal=None):
    """
    Runs analysis directly from data dictionaries (for web interface).
    
    Args:
        user_info (dict): User information dictionary
        scan_history (list): List of scan dictionaries
        almi_goal (dict): ALMI goal parameters (optional)
        ffmi_goal (dict): FFMI goal parameters (optional)
        
    Returns:
        tuple: (df_results, goal_calculations, figures, comparison_table_html)
    """
    # Load LMS Data
    lms_functions = {
        'almi_L': None, 'almi_M': None, 'almi_S': None,
        'lmi_L': None, 'lmi_M': None, 'lmi_S': None
    }
    lms_functions['almi_L'], lms_functions['almi_M'], lms_functions['almi_S'] = load_lms_data(
        metric='appendicular_LMI', gender_code=user_info['gender_code'])
    lms_functions['lmi_L'], lms_functions['lmi_M'], lms_functions['lmi_S'] = load_lms_data(
        metric='LMI', gender_code=user_info['gender_code'])

    if not all(lms_functions.values()):
        raise ValueError("Failed to load all necessary LMS data")
    
    # Process data and generate results DataFrame
    df_results, goal_calculations = process_scans_and_goal(user_info, scan_history, almi_goal, ffmi_goal, lms_functions)
    
    # Generate plots
    almi_fig = create_metric_plot(df_results, 'ALMI', lms_functions, goal_calculations, return_figure=True)
    ffmi_fig = create_metric_plot(df_results, 'FFMI', lms_functions, goal_calculations, return_figure=True)
    bf_fig = create_body_fat_plot(df_results, user_info, return_figure=True)
    figures = {'ALMI': almi_fig, 'FFMI': ffmi_fig, 'BODY_FAT': bf_fig}
    
    # Generate comparison table for web interface
    comparison_table_html = create_scan_comparison_table(df_results, return_html=True)
    
    return df_results, goal_calculations, figures, comparison_table_html

def run_analysis(config_path='example_config.json', suggest_goals=False, target_percentile=0.90, 
                 training_level_override=None, return_results=False):
    """
    Main analysis function that orchestrates the entire DEXA analysis workflow.
    
    Args:
        config_path (str): Path to JSON configuration file
        suggest_goals (bool): Whether to generate suggested goals
        target_percentile (float): Target percentile for suggested goals
        training_level_override (str): Override training level detection ('novice', 'intermediate', 'advanced')
        return_results (bool): If True, returns analysis results instead of printing and saving
        
    Returns:
        int or tuple: Exit code (0 for success, 1 for error) if return_results=False,
                     or (df_results, goal_calculations, figures) if return_results=True
    """
    if not return_results:
        print("DEXA Body Composition Analysis with Intelligent TLM Estimation")
        print("=" * 65)
        print("Note: Run 'python test_zscore_calculations.py' for comprehensive testing\n")
    
    try:
        # Step 1: Load Configuration
        config = load_config_json(config_path, quiet=return_results)
        user_info, scan_history, almi_goal, ffmi_goal = extract_data_from_config(config)
        
        # Apply training level override if provided
        if training_level_override:
            user_info['training_level'] = training_level_override.lower()
        
        # Display loaded configuration
        print(f"User Info:")
        print(f"  - Birth Date: {config['user_info']['birth_date']}")
        print(f"  - Height: {config['user_info']['height_in']} inches")
        print(f"  - Gender: {config['user_info']['gender']}")
        if user_info.get('training_level'):
            print(f"  - Training Level: {user_info['training_level']}")
        
        print(f"\nGoals:")
        if almi_goal:
            target_age_str = str(almi_goal['target_age']) if almi_goal.get('target_age') not in [None, "?"] else "auto-calculated"
            print(f"  - ALMI Goal: {almi_goal['target_percentile']*100:.0f}th percentile at age {target_age_str}")
            if 'description' in almi_goal:
                print(f"    Description: {almi_goal['description']}")
        if ffmi_goal:
            target_age_str = str(ffmi_goal['target_age']) if ffmi_goal.get('target_age') not in [None, "?"] else "auto-calculated"
            print(f"  - FFMI Goal: {ffmi_goal['target_percentile']*100:.0f}th percentile at age {target_age_str}")
            if 'description' in ffmi_goal:
                print(f"    Description: {ffmi_goal['description']}")
        if not almi_goal and not ffmi_goal:
            print("  - No goals specified (scan history analysis only)")
        print()

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
        df_results, goal_calculations = process_scans_and_goal(user_info, scan_history, almi_goal, ffmi_goal, lms_functions)
        
        # Step 4: Generate Plots
        if return_results:
            # Return figure objects for web interface
            almi_fig = create_metric_plot(df_results, 'ALMI', lms_functions, goal_calculations, return_figure=True)
            ffmi_fig = create_metric_plot(df_results, 'FFMI', lms_functions, goal_calculations, return_figure=True)
            bf_fig = create_body_fat_plot(df_results, user_info, return_figure=True)
            figures = {'ALMI': almi_fig, 'FFMI': ffmi_fig, 'BODY_FAT': bf_fig}
            comparison_table_html = create_scan_comparison_table(df_results, return_html=True)
            return df_results, goal_calculations, figures, comparison_table_html
        else:
            # Save plots to disk for CLI interface
            plot_metric_with_table(df_results, 'ALMI', lms_functions, goal_calculations)
            plot_metric_with_table(df_results, 'FFMI', lms_functions, goal_calculations)
            create_body_fat_plot(df_results, user_info, return_figure=False)
            
            # Step 5: Print Comparison Table
            print("\n--- First vs Most Recent Scan Comparison ---")
            comparison_table = create_scan_comparison_table(df_results, return_html=False)
            print(comparison_table)
            
            # Step 6: Print Final Table
            print("\n--- Final Comprehensive Data Table ---")
            # Select columns for display
            display_columns = [
                'date_str', 'age_at_scan', 
                'total_weight_lbs', 'total_lean_mass_lbs', 'fat_mass_lbs', 'body_fat_percentage',
                'almi_kg_m2', 'ffmi_kg_m2',
                'weight_change_last', 'lean_change_last', 'fat_change_last', 'bf_change_last',
                'weight_change_first', 'lean_change_first', 'fat_change_first', 'bf_change_first',
                'almi_z_change_last', 'ffmi_z_change_last', 'almi_pct_change_last', 'ffmi_pct_change_last'
            ]
            display_names = [
                'Date', 'Age', 
                'Weight', 'Lean', 'Fat', 'BF%',
                'ALMI', 'FFMI',
                'ΔW_L', 'ΔL_L', 'ΔF_L', 'ΔBF_L',
                'ΔW_F', 'ΔL_F', 'ΔF_F', 'ΔBF_F',
                'ΔALMI_Z_L', 'ΔFFMI_Z_L', 'ΔALMI_%_L', 'ΔFFMI_%_L'
            ]
                
            df_display = df_results[display_columns].copy()
            df_display.columns = display_names
            
            # Format numeric columns with appropriate precision
            for col in df_display.columns:
                if df_display[col].dtype == 'float64':
                    if 'Δ' in col and ('Z' in col or '%' in col):
                        # Z-scores and percentile changes - more precision
                        df_display[col] = df_display[col].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "N/A")
                    elif 'Δ' in col:
                        # Other changes - show with +/- sign
                        df_display[col] = df_display[col].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "N/A")
                    else:
                        # Regular values
                        df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            
            # Use tabulate for better formatting
            try:
                from tabulate import tabulate
                print(tabulate(df_display, headers='keys', tablefmt='pipe', showindex=False))
            except ImportError:
                # Fallback to pandas display if tabulate not available
                print(df_display.to_string(index=False))
            
            return 0
        
    except (FileNotFoundError, json.JSONDecodeError, ValidationError, KeyError, ValueError) as e:
        if return_results:
            raise e  # Re-raise for web interface to handle
        else:
            print(f"Error: {e}")
            print(f"\nPlease check your configuration file: {config_path}")
            return 1
    except Exception as e:
        if return_results:
            raise e  # Re-raise for web interface to handle
        else:
            print(f"Unexpected error: {e}")
            return 1