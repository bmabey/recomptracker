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
        processed_data (pd.DataFrame): DataFrame with scan history and metrics
        user_info (dict): User information dictionary
        
    Returns:
        str: Detected training level ('novice', 'intermediate', 'advanced')
    """
    # If training level is explicitly provided, use it
    if 'training_level' in user_info and user_info['training_level']:
        return user_info['training_level'].lower()
    
    # Need at least 2 scans for progression analysis
    if len(processed_data) < 2:
        print("  Insufficient scan history for training level detection - defaulting to intermediate")
        return 'intermediate'
    
    # Calculate lean mass gain rates (convert to kg per month)
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
    
    if not lean_gains:
        print("  No sufficient time gaps between scans for progression analysis - defaulting to intermediate")
        return 'intermediate'
    
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
        print(f"  Detected novice level: rapid progression {avg_gain_rate:.2f} kg/month")
    elif avg_gain_rate < advanced_threshold:
        detected_level = 'advanced'
        print(f"  Detected advanced level: slow progression {avg_gain_rate:.2f} kg/month")
    else:
        detected_level = 'intermediate'
        print(f"  Detected intermediate level: moderate progression {avg_gain_rate:.2f} kg/month")
    
    return detected_level

def get_conservative_gain_rate(user_info, training_level, current_age):
    """
    Gets conservative lean mass gain rate based on demographics and training level.
    
    Args:
        user_info (dict): User information
        training_level (str): Training level ('novice', 'intermediate', 'advanced')
        current_age (float): Current age
        
    Returns:
        float: Conservative gain rate in kg/month
    """
    gender_str = get_gender_string(user_info['gender_code'])
    
    # Base rates by gender and training level (kg/month)
    # These are conservative estimates based on research literature
    base_rates = {
        'male': {
            'novice': 1.0,       # Can gain muscle rapidly initially
            'intermediate': 0.35, # Moderate gains with consistent training
            'advanced': 0.15      # Very slow gains, near genetic potential
        },
        'female': {
            'novice': 0.5,       # Lower absolute gains but similar relative
            'intermediate': 0.2,  # Moderate gains
            'advanced': 0.08     # Very slow gains
        }
    }
    
    base_rate = base_rates[gender_str][training_level]
    
    # Age adjustment factor (muscle building capacity decreases with age)
    if current_age < 25:
        age_factor = 1.0      # Peak muscle building years
    elif current_age < 35:
        age_factor = 0.95     # Slight decrease
    elif current_age < 45:
        age_factor = 0.9      # Moderate decrease
    elif current_age < 55:
        age_factor = 0.8      # More significant decrease
    else:
        age_factor = 0.7      # Substantial decrease but still possible
    
    adjusted_rate = base_rate * age_factor
    print(f"  Conservative {training_level} rate for {gender_str}: {base_rate:.2f} kg/month, age-adjusted to {adjusted_rate:.2f} kg/month (age {current_age:.0f})")
    
    return adjusted_rate

def determine_training_level(user_info, processed_data):
    """
    Determines training level using explicit specification or detection from scans.
    
    Args:
        user_info (dict): User information
        processed_data (pd.DataFrame): Processed scan data
        
    Returns:
        str: Training level ('novice', 'intermediate', 'advanced')
    """
    # Check if explicitly provided
    if 'training_level' in user_info and user_info['training_level']:
        return user_info['training_level'].lower()
    
    # Detect from scan progression
    return detect_training_level_from_scans(processed_data, user_info)

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
        dict: Updated goal parameters with calculated target_age
    """
    # Get the most recent scan data
    latest_scan = processed_data.iloc[-1]
    current_age = latest_scan['age_at_scan']
    
    # Determine training level
    training_level = determine_training_level(user_info, processed_data)
    
    # Get conservative gain rate
    monthly_gain_rate_kg = get_conservative_gain_rate(user_info, training_level, current_age)
    
    # Get current metric value and target percentile
    current_metric = latest_scan[f'{metric}_kg_m2']
    target_percentile = goal_params['target_percentile']
    
    # Binary search to find the target age where we can achieve the goal
    min_age = current_age
    max_age = min(current_age + 10, 70)  # Search up to 10 years or age 70
    
    def can_reach_goal_at_age(target_age):
        """Check if goal is achievable at target_age given gain rates."""
        # Get LMS values for target age
        L_func = lms_functions[f'{metric}_L']
        M_func = lms_functions[f'{metric}_M'] 
        S_func = lms_functions[f'{metric}_S']
        
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
        print(f"  Could not find feasible timeframe for {target_percentile*100:.0f}th percentile {metric.upper()}")
        print(f"  Using 2-year timeframe as fallback (age {best_age:.1f})")
    else:
        time_to_goal = best_age - current_age
        print(f"  ✓ Calculated feasible timeframe: {time_to_goal:.1f} years (age {best_age:.1f}) based on {training_level} progression rates")
    
    # Update goal parameters
    updated_goal = goal_params.copy()
    updated_goal['target_age'] = best_age
    updated_goal['suggested'] = True
    
    return updated_goal


# ---------------------------------------------------------------------------
# DATA PROCESSING AND ORCHESTRATION
# ---------------------------------------------------------------------------

def load_config_json(config_path):
    """
    Loads and validates a JSON configuration file.
    
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
    
    # Extract scan history
    scan_history = config['scan_history']
    
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
        processed_data (pd.DataFrame): DataFrame with scan history
        goal_params (dict): Goal parameters  
        lms_functions (dict): LMS interpolation functions
        user_info (dict): User information
        
    Returns:
        float: ALM/TLM ratio to use for calculations
    """
    # If we have multiple scans, use personalized ratio from recent data
    if len(processed_data) >= 2:
        # Use last 3 scans for stability, or all if fewer than 3
        recent_scans = processed_data.tail(min(3, len(processed_data)))
        alm_values = recent_scans['alm_kg']
        tlm_values = recent_scans['total_lean_mass_lbs'] * 0.453592  # Convert to kg
        
        # Calculate ratio for each scan and take the mean
        ratios = alm_values / tlm_values
        personal_ratio = ratios.mean()
        
        print(f"Using personal ALM/TLM ratio of {personal_ratio:.3f} from {len(recent_scans)} recent scans")
        return personal_ratio
    
    # For single scans, use population-based estimates
    # These are based on research literature for healthy adults
    gender_str = get_gender_string(user_info['gender_code'])
    
    if gender_str == 'male':
        population_ratio = 0.45  # Males typically have ~45% ALM/TLM
    else:
        population_ratio = 0.42  # Females typically have ~42% ALM/TLM
    
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
    df['scan_date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = df.sort_values('scan_date').reset_index(drop=True)
    
    # Calculate basic metrics for each scan
    results = []
    goal_calculations = {}
    
    for i, scan in df.iterrows():
        # Calculate age at scan
        age_at_scan = calculate_age_precise(user_info['birth_date'], scan['date'])
        
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
            'date_str': scan['date'],
            'scan_date': scan['scan_date'],
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
            'almi_percentile': almi_percentile,
            'ffmi_z_score': ffmi_z,
            'ffmi_percentile': ffmi_percentile
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
            processed_data.loc[i, 'almi_pct_change_last'] = (processed_data.loc[i, 'almi_percentile'] - processed_data.loc[prev_idx, 'almi_percentile']) * 100
            processed_data.loc[i, 'ffmi_pct_change_last'] = (processed_data.loc[i, 'ffmi_percentile'] - processed_data.loc[prev_idx, 'ffmi_percentile']) * 100
        
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
            almi_goal = calculate_suggested_goal(almi_goal, user_info, processed_data, lms_functions, 'almi')
        
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
            ffmi_goal = calculate_suggested_goal(ffmi_goal, user_info, processed_data, lms_functions, 'ffmi')
            
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
        
        # Get LMS values for target age
        L_func = lms_functions[f'{metric}_L']
        M_func = lms_functions[f'{metric}_M'] 
        S_func = lms_functions[f'{metric}_S']
        
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
        target_body_fat_pct = goal_params.get('target_body_fat_percentage', current_scan['body_fat_percentage'])
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
        current_percentile = current_scan[f'{metric}_percentile']
        percentile_change = (target_percentile - current_percentile) * 100
        
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
            'almi_percentile': target_percentile if metric == 'almi' else current_scan['almi_percentile'],
            'ffmi_z_score': target_z if metric == 'ffmi' else current_scan['ffmi_z_score'],
            'ffmi_percentile': target_percentile if metric == 'ffmi' else current_scan['ffmi_percentile'],
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
            'target_body_composition': {
                'weight_lbs': target_weight_lbs,
                'lean_mass_lbs': target_lean_mass_lbs,
                'fat_mass_lbs': target_fat_mass_lbs,
                'body_fat_percentage': target_body_fat_pct
            }
        }
        
        return goal_row, goal_calculations
        
    except Exception as e:
        print(f"  Error creating {metric.upper()} goal row: {e}")
        return None, None


# ---------------------------------------------------------------------------
# PLOTTING LOGIC
# ---------------------------------------------------------------------------

def plot_metric_with_table(df_results, metric_to_plot, lms_functions, goal_calculations):
    """
    Creates comprehensive plots showing percentile curves, data points, and exports CSV data.
    
    This function generates detailed visualizations that include:
    - LMS-based percentile curves (3rd, 10th, 25th, 50th, 75th, 90th, 97th)
    - User's actual data points plotted over time
    - Goal markers and projections if goals are specified
    - Exports comprehensive data table to CSV
    
    Args:
        df_results (pd.DataFrame): Complete results DataFrame with scan history and goals
        metric_to_plot (str): Either 'ALMI' or 'FFMI' 
        lms_functions (dict): Dictionary containing LMS interpolation functions
        goal_calculations (dict): Goal calculation results
    """
    print(f"Generating plot for {metric_to_plot}...")
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
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
        
        plt.plot(age_range, curve_values, color=color, linewidth=2, label=f'{label} percentile', alpha=0.8)
    
    # Filter data for actual scans (not goal rows)
    scan_data = df_results[~df_results['date_str'].str.contains('Goal', na=False)]
    
    # Plot actual data points
    if len(scan_data) > 0:
        plt.scatter(scan_data['age_at_scan'], scan_data[y_column], 
                   color='red', s=100, zorder=5, label='Your scans', edgecolors='black', linewidth=1)
        
        # Connect points with lines if multiple scans
        if len(scan_data) > 1:
            plt.plot(scan_data['age_at_scan'], scan_data[y_column], 
                    color='red', linewidth=2, alpha=0.7, zorder=4)
    
    # Plot goal if available
    goal_key = metric_to_plot.lower()
    if goal_key in goal_calculations:
        goal_calc = goal_calculations[goal_key]
        goal_age = goal_calc['target_age']
        goal_value = goal_calc['target_metric_value']
        
        plt.scatter([goal_age], [goal_value], color='gold', s=150, marker='*', 
                   zorder=6, label='Goal', edgecolors='black', linewidth=1)
        
        # Draw line from last scan to goal
        if len(scan_data) > 0:
            last_scan = scan_data.iloc[-1]
            plt.plot([last_scan['age_at_scan'], goal_age], 
                    [last_scan[y_column], goal_value],
                    color='gold', linewidth=3, linestyle='--', alpha=0.8, zorder=3)
    
    # Customize plot
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(plot_title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    filename = f"{metric_to_plot.lower()}_plot.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as: {filename}")
    
    # Export data table to CSV (only for ALMI to avoid duplication)
    if metric_to_plot == 'ALMI':
        csv_filename = 'almi_stats_table.csv'
        df_results.to_csv(csv_filename, index=False)
        print(f"Table data saved as: {csv_filename}")


# ---------------------------------------------------------------------------
# MAIN ANALYSIS FUNCTION  
# ---------------------------------------------------------------------------

def run_analysis(config_path='example_config.json', suggest_goals=False, target_percentile=0.90):
    """
    Main analysis function that orchestrates the entire DEXA analysis workflow.
    
    Args:
        config_path (str): Path to JSON configuration file
        suggest_goals (bool): Whether to generate suggested goals
        target_percentile (float): Target percentile for suggested goals
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    print("DEXA Body Composition Analysis with Intelligent TLM Estimation")
    print("=" * 65)
    print("Note: Run 'python test_zscore_calculations.py' for comprehensive testing\n")
    
    try:
        # Step 1: Load Configuration
        config = load_config_json(config_path)
        user_info, scan_history, almi_goal, ffmi_goal = extract_data_from_config(config)
        
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
        plot_metric_with_table(df_results, 'ALMI', lms_functions, goal_calculations)
        plot_metric_with_table(df_results, 'FFMI', lms_functions, goal_calculations)
        
        # Step 5: Print Final Table
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
        print(f"Error: {e}")
        print(f"\nPlease check your configuration file: {config_path}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1