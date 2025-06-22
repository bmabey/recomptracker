"""
Core RecompTracker Analysis Logic

This module contains all the core calculation logic, data processing functions,
and plotting functionality for RecompTracker. This is the
computational engine that powers the analysis scripts.

Sections:
- Core calculation logic (age, Z-scores, T-scores)
- Suggested goal logic and lean mass gain rates
- Data processing and orchestration
- Plotting logic
"""

import functools
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from jsonschema import ValidationError, validate
from scipy.interpolate import interp1d

# Import shared data models
from shared_models import (
    AnalysisResults,
    BodyComposition,
    GoalConfig,
    GoalResults,
    ScanData,
    TrainingLevel,
    UserProfile,
    convert_dict_to_goal_config,
    convert_dict_to_user_profile,
)

# Constants for lean mass gain rates (kg/month)
# Based on realistic muscle building research (Helms et al., Krieger, etc.)
LEAN_MASS_GAIN_RATES = {
    "male": {"novice": 0.45, "intermediate": 0.25, "advanced": 0.12},
    "female": {"novice": 0.25, "intermediate": 0.15, "advanced": 0.06},
}

# Healthy body fat percentage ranges by gender
# Based on American Council on Exercise (ACE) and athletic performance standards
HEALTHY_BF_RANGES = {
    "male": {
        "athletic": (6, 13),  # Athletes
        "fitness": (14, 17),  # Fitness enthusiasts
        "acceptable": (18, 24),  # General health
        "overweight": (25, 100),  # Above healthy range
    },
    "female": {
        "athletic": (16, 20),  # Athletes
        "fitness": (21, 24),  # Fitness enthusiasts
        "acceptable": (25, 31),  # General health
        "overweight": (32, 100),  # Above healthy range
    },
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
                "gender": {
                    "type": "string",
                    "pattern": "^(m|f|male|female|M|F|Male|Female|MALE|FEMALE)$",
                },
                "training_level": {
                    "type": "string",
                    "pattern": "^(novice|intermediate|advanced|Novice|Intermediate|Advanced|NOVICE|INTERMEDIATE|ADVANCED)$",
                },
            },
            "additionalProperties": False,
        },
        "scan_history": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "date",
                    "total_weight_lbs",
                    "total_lean_mass_lbs",
                    "fat_mass_lbs",
                    "body_fat_percentage",
                    "arms_lean_lbs",
                    "legs_lean_lbs",
                ],
                "properties": {
                    "date": {"type": "string", "pattern": "^\\d{2}/\\d{2}/\\d{4}$"},
                    "total_weight_lbs": {"type": "number", "minimum": 0},
                    "total_lean_mass_lbs": {"type": "number", "minimum": 0},
                    "fat_mass_lbs": {"type": "number", "minimum": 0},
                    "body_fat_percentage": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "arms_lean_lbs": {"type": "number", "minimum": 0},
                    "legs_lean_lbs": {"type": "number", "minimum": 0},
                },
                "additionalProperties": False,
            },
        },
        "goals": {
            "type": "object",
            "properties": {
                "almi": {
                    "type": "object",
                    "required": ["target_percentile"],
                    "properties": {
                        "target_percentile": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "suggested": {"type": "boolean"},
                        "description": {"type": "string"},
                        "target_body_fat_percentage": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                        },
                    },
                    "additionalProperties": False,
                },
                "ffmi": {
                    "type": "object",
                    "required": ["target_percentile"],
                    "properties": {
                        "target_percentile": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "suggested": {"type": "boolean"},
                        "description": {"type": "string"},
                        "target_body_fat_percentage": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
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
        return (((y / M) ** L) - 1) / (L * S)


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
    if (
        pd.isna(value)
        or pd.isna(young_adult_median)
        or pd.isna(young_adult_sd)
        or young_adult_sd == 0
    ):
        return np.nan
    return (value - young_adult_median) / young_adult_sd


def calculate_tscore_reference_values(gender_code):
    """
    Calculate T-score reference values (mean and SD) from young adults aged 20-30.

    This function derives T-score reference values by sampling from the LMS distribution
    for young adults (ages 20-30) to estimate the actual population mean and standard deviation.
    The approach:
    1. Use LMS parameters for ages 20-30 to generate simulated population samples
    2. Calculate empirical mean and standard deviation from these samples
    3. Return values suitable for T-score calculation

    Args:
        gender_code (int): 0 for male, 1 for female

    Returns:
        tuple: (mu_peak, sigma_peak) - mean and standard deviation for young adults,
               or (NaN, NaN) if calculation fails
    """
    try:
        # Load ALMI LMS data for the specified gender
        data_file = f"data/adults_LMS_appendicular_LMI_gender{gender_code}.csv"

        if not os.path.exists(data_file):
            print(f"Warning: T-score reference data not found: {data_file}")
            return np.nan, np.nan

        df = pd.read_csv(data_file)

        # Filter to young adult age range (20-30 years) - peak muscle mass period
        young_adult_data = df[(df["age"] >= 20) & (df["age"] <= 30)]

        if len(young_adult_data) == 0:
            print("Warning: No young adult data found in age range 20-30")
            return np.nan, np.nan

        # For T-score calculation, we need to estimate the actual population variance
        # from the LMS distribution. We'll sample from the distribution at multiple ages
        # and combine to get empirical population statistics.

        all_samples = []
        n_samples_per_age = 1000  # Sample size per age year

        for _, row in young_adult_data.iterrows():
            L, M, S = row["lambda"], row["mu"], row["sigma"]

            # Generate samples from the LMS distribution for this age
            # Use percentile sampling approach for robustness
            percentiles = np.linspace(0.01, 0.99, n_samples_per_age)
            z_scores = stats.norm.ppf(percentiles)

            # Convert Z-scores to ALMI values using LMS transformation
            samples = []
            for z in z_scores:
                almi_value = get_value_from_zscore(z, L, M, S)
                if not pd.isna(almi_value) and almi_value > 0:
                    samples.append(almi_value)

            all_samples.extend(samples)

        if len(all_samples) == 0:
            print("Warning: No valid samples generated from LMS distribution")
            return np.nan, np.nan

        # Calculate empirical population statistics
        mu_peak = np.mean(all_samples)
        sigma_peak = np.std(all_samples, ddof=1)  # Sample standard deviation

        # Sanity check the results
        if sigma_peak == 0:
            print("Warning: Zero standard deviation in simulated population")
            return mu_peak, np.nan

        if sigma_peak < 0.1:
            print("Warning: Suspiciously low standard deviation in T-score reference")

        return mu_peak, sigma_peak

    except Exception as e:
        print(f"Error calculating T-score reference values: {e}")
        return np.nan, np.nan


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


@functools.lru_cache(maxsize=1024)
def calculate_percentile_cached(value, age, metric, gender_code, data_path="./data/"):
    """
    Calculates percentile for a given value with caching for performance.

    This is a high-level function that loads LMS data and calculates percentiles
    with automatic caching of both LMS data and percentile calculations.
    Optimized for Monte Carlo simulations with repeated calculations.

    Args:
        value (float): The measured value (e.g., ALMI, FFMI).
        age (float): The age in decimal years.
        metric (str): Either 'appendicular_LMI' for ALMI or 'LMI' for FFMI.
        gender_code (int): 0 for male, 1 for female.
        data_path (str): Path to the directory containing LMS CSV files.

    Returns:
        float: The percentile (0.0 to 1.0), or NaN if calculation fails.
    """
    # Load cached LMS data
    L_func, M_func, S_func = load_lms_data(metric, gender_code, data_path)

    if L_func is None:
        return np.nan

    # Calculate Z-score and percentile using existing logic
    z_score, percentile = calculate_z_percentile(value, age, L_func, M_func, S_func)

    return percentile


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
    if "training_level" in user_info and user_info["training_level"]:
        level = user_info["training_level"].lower()
        return level, f"user-specified training level: {level}"

    # Need at least 2 scans for progression analysis
    if len(processed_data) < 2:
        print(
            "  Insufficient scan history for training level detection - defaulting to intermediate"
        )
        return (
            "intermediate",
            "Insufficient scan history - defaulting to intermediate (realistic baseline)",
        )

    # Handle both DataFrame and list formats
    if hasattr(processed_data, "sort_values"):
        # DataFrame format
        processed_data = processed_data.sort_values("scan_date")
        lean_gains = []

        for i in range(1, len(processed_data)):
            prev_scan = processed_data.iloc[i - 1]
            curr_scan = processed_data.iloc[i]

            # Calculate time difference in months
            time_diff_days = (curr_scan["scan_date"] - prev_scan["scan_date"]).days
            time_diff_months = time_diff_days / 30.44  # Average days per month

            if time_diff_months > 0.5:  # Only consider gaps > 2 weeks
                lean_gain_lbs = (
                    curr_scan["total_lean_mass_lbs"] - prev_scan["total_lean_mass_lbs"]
                )
                lean_gain_kg_per_month = (lean_gain_lbs * 0.453592) / time_diff_months
                lean_gains.append(lean_gain_kg_per_month)
    else:
        # List format - simplified analysis for tests
        lean_gains = []

        for i in range(1, len(processed_data)):
            prev_scan = processed_data[i - 1]
            curr_scan = processed_data[i]

            # Calculate time difference based on date_str if available
            if "date_str" in curr_scan and "date_str" in prev_scan:
                from datetime import datetime

                prev_date = datetime.strptime(prev_scan["date_str"], "%m/%d/%Y")
                curr_date = datetime.strptime(curr_scan["date_str"], "%m/%d/%Y")
                time_diff_days = (curr_date - prev_date).days
                time_diff_months = time_diff_days / 30.44  # Average days per month
            else:
                # For test data without dates, assume 6 months between scans
                time_diff_months = 6.0

            if (
                "total_lean_mass_lbs" in curr_scan
                and "total_lean_mass_lbs" in prev_scan
            ):
                lean_gain_lbs = (
                    curr_scan["total_lean_mass_lbs"] - prev_scan["total_lean_mass_lbs"]
                )
                lean_gain_kg_per_month = (lean_gain_lbs * 0.453592) / time_diff_months
                lean_gains.append(lean_gain_kg_per_month)

    if not lean_gains:
        print(
            "  No sufficient time gaps between scans for progression analysis - defaulting to intermediate"
        )
        return (
            "intermediate",
            "No sufficient time gaps between scans for progression analysis - defaulting to intermediate",
        )

    # Calculate average monthly lean mass gain rate
    avg_gain_rate = np.mean(lean_gains)

    # Classification thresholds based on research (conservative estimates)
    # These are monthly rates, adjusted for gender
    gender_str = get_gender_string(user_info["gender_code"])
    if gender_str == "male":
        novice_threshold = 0.8  # >0.8 kg/month suggests novice gains
        advanced_threshold = 0.3  # <0.3 kg/month suggests advanced/slow gains
    else:  # female
        novice_threshold = 0.4  # >0.4 kg/month suggests novice gains
        advanced_threshold = 0.1  # <0.1 kg/month suggests advanced/slow gains

    if avg_gain_rate > novice_threshold:
        detected_level = "novice"
        explanation = f"Detected novice level: novice gains {avg_gain_rate:.2f} kg/month, early training phase"
        print(f"  {explanation}")
    elif avg_gain_rate < advanced_threshold:
        detected_level = "advanced"
        explanation = f"Detected advanced level: slow progression {avg_gain_rate:.2f} kg/month, experienced trainee"
        print(f"  {explanation}")
    else:
        detected_level = "intermediate"
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
    gender_str = get_gender_string(user_info["gender_code"])

    # Use the constant for consistency with tests
    base_rate = LEAN_MASS_GAIN_RATES[gender_str][training_level]

    # Age adjustment factor (muscle building capacity decreases with age)
    if current_age <= 30:
        age_factor = 1.0  # No adjustment for age 30 and under
    else:
        # Linear reduction of 10% per decade over 30
        decades_over_30 = (current_age - 30) / 10
        age_factor = max(
            1 - (AGE_ADJUSTMENT_FACTOR * decades_over_30), 0.5
        )  # Minimum 50% of base rate

    adjusted_rate = base_rate * age_factor

    if current_age <= 30:
        explanation = f"Conservative {training_level} rate for {gender_str}: {base_rate:.2f} kg/month (age {current_age:.0f})"
    else:
        explanation = f"Conservative {training_level} rate for {gender_str}: {base_rate:.2f} kg/month, age-adjusted to {adjusted_rate:.2f} kg/month (age {current_age:.0f})"

    print(f"  {explanation}")

    return adjusted_rate, explanation


def calculate_progressive_gain_over_time(
    user_info, initial_training_level, current_age, years
):
    """
    Calculates realistic total lean mass gain over multiple years using progressive rates.

    Models diminishing returns as trainees advance from novice → intermediate → advanced.

    Args:
        user_info (dict): User information
        initial_training_level (str): Starting training level
        current_age (float): Current age
        years (float): Time period in years

    Returns:
        tuple: (total_gain_kg, explanation) - Total achievable gain and explanation
    """
    gender_str = get_gender_string(user_info["gender_code"])

    # Define progression timeline (years when levels typically change)
    novice_duration = 1.0  # First year: novice gains
    intermediate_duration = 2.0  # Years 2-3: intermediate gains
    # Year 4+: advanced gains

    total_gain_kg = 0.0
    explanations = []

    # Calculate age-adjusted rates for each level
    rates = {}
    for level in ["novice", "intermediate", "advanced"]:
        base_rate = LEAN_MASS_GAIN_RATES[gender_str][level]
        # Age adjustment
        if current_age <= 30:
            age_factor = 1.0
        else:
            decades_over_30 = (current_age - 30) / 10
            age_factor = max(1 - (AGE_ADJUSTMENT_FACTOR * decades_over_30), 0.5)
        rates[level] = base_rate * age_factor

    remaining_years = years
    current_level = initial_training_level

    # Year 1: Novice phase (if starting as novice or if user is truly new)
    if current_level == "novice" and remaining_years > 0:
        novice_years = min(remaining_years, novice_duration)
        novice_gain = rates["novice"] * 12 * novice_years  # Convert to kg/year
        total_gain_kg += novice_gain
        remaining_years -= novice_years
        explanations.append(
            f"Year 1: {novice_gain:.1f} kg at novice rate ({rates['novice']:.2f} kg/month)"
        )
        current_level = "intermediate"

    # Years 2-3: Intermediate phase
    if current_level in ["novice", "intermediate"] and remaining_years > 0:
        intermediate_years = min(remaining_years, intermediate_duration)
        intermediate_gain = rates["intermediate"] * 12 * intermediate_years
        total_gain_kg += intermediate_gain
        remaining_years -= intermediate_years
        year_start = years - remaining_years - intermediate_years + 1
        year_end = years - remaining_years
        if intermediate_years > 0:
            explanations.append(
                f"Years {year_start:.0f}-{year_end:.0f}: {intermediate_gain:.1f} kg at intermediate rate ({rates['intermediate']:.2f} kg/month)"
            )
        current_level = "advanced"

    # Year 4+: Advanced phase
    if remaining_years > 0:
        advanced_gain = rates["advanced"] * 12 * remaining_years
        total_gain_kg += advanced_gain
        year_start = years - remaining_years + 1
        if remaining_years > 0:
            explanations.append(
                f"Years {year_start:.0f}+: {advanced_gain:.1f} kg at advanced rate ({rates['advanced']:.2f} kg/month)"
            )

    # If starting as intermediate/advanced, just use that rate for the entire period
    if initial_training_level == "intermediate" and years <= intermediate_duration:
        total_gain_kg = rates["intermediate"] * 12 * years
        explanations = [
            f"All {years:.1f} years: {total_gain_kg:.1f} kg at intermediate rate ({rates['intermediate']:.2f} kg/month)"
        ]
    elif initial_training_level == "advanced":
        total_gain_kg = rates["advanced"] * 12 * years
        explanations = [
            f"All {years:.1f} years: {total_gain_kg:.1f} kg at advanced rate ({rates['advanced']:.2f} kg/month)"
        ]

    explanation = f"Progressive gain model: {'; '.join(explanations)}"

    return total_gain_kg, explanation


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
    if "training_level" in user_info and user_info["training_level"]:
        level = user_info["training_level"].lower()
        explanation = f"user-specified training level: {level}"
        return level, explanation

    # Detect from scan progression
    level, explanation = detect_training_level_from_scans(processed_data, user_info)
    return level, explanation


def calculate_suggested_goal(
    goal_params, user_info, processed_data, lms_functions, metric=None
):
    """
    Calculates suggested goals with auto-calculated realistic timeframes.

    Supports both new dataclass format and legacy dict format for backward compatibility.

    Args:
        goal_params (dict or GoalConfig): Goal parameters including target_percentile
        user_info (dict or UserProfile): User information
        processed_data (pd.DataFrame): Processed scan data
        lms_functions (dict): LMS interpolation functions
        metric (str, optional): Metric type ('almi' or 'ffmi'), auto-detected if not provided

    Returns:
        tuple or GoalResults: (updated_goal, messages) for legacy format or GoalResults for new format
    """

    # Handle both new dataclass format and legacy dict format
    if hasattr(goal_params, "metric_type"):
        # New dataclass format - this is the future path
        goal_config = goal_params
        user_profile = user_info
        metric = goal_config.metric_type

        # Convert to legacy format for internal processing
        goal_params_dict = {
            "target_percentile": goal_config.target_percentile,
            "suggested": goal_config.suggested,
            "target_body_fat_percentage": goal_config.target_body_fat_percentage,
        }
        user_info_dict = {
            "birth_date": user_profile.birth_date,
            "height_in": user_profile.height_in,
            "gender": user_profile.gender,
            "gender_code": user_profile.gender_code,
            "training_level": user_profile.training_level.value,
        }
        is_legacy_call = False
    else:
        # Legacy dict format - for backward compatibility
        goal_params_dict = goal_params.copy()
        user_info_dict = user_info
        if metric is None:
            # Try to infer metric from context, default to almi
            metric = "almi"
        is_legacy_call = True

    # Get the most recent scan data (handle both DataFrame and list)
    if hasattr(processed_data, "iloc"):
        latest_scan = processed_data.iloc[-1]
    else:
        latest_scan = processed_data[-1]
    current_age = latest_scan["age_at_scan"]

    # Determine training level
    training_level, level_explanation = determine_training_level(
        user_info_dict, processed_data
    )

    # Get conservative gain rate
    monthly_gain_rate_kg, gain_explanation = get_conservative_gain_rate(
        user_info_dict, training_level, current_age
    )

    # Get current metric value and target percentile
    current_metric = latest_scan[f"{metric}_kg_m2"]
    target_percentile = goal_params_dict["target_percentile"]

    # Get current percentile from scan data (already calculated in percentage format)
    current_percentile = latest_scan[f"{metric}_percentile"]

    # Check if user is already at or above target percentile
    # Note: current_percentile is in percentage format (0-100), target_percentile is in decimal format (0-1)
    current_percentile_decimal = current_percentile / 100
    if current_percentile_decimal >= target_percentile:
        print(
            f"  ✓ Already at {current_percentile:.1f}th percentile for {metric.upper()}, which is above target {target_percentile * 100:.0f}th percentile"
        )

        # If user is already above 90th percentile, don't suggest any goal
        if current_percentile_decimal >= 0.90:
            print(
                "  🎯 You're already above the 90th percentile - no goal suggestion needed!"
            )
            if is_legacy_call:
                return None, [level_explanation, gain_explanation]
            else:
                return None

        # Only suggest higher percentile if user is below 90th percentile
        new_target_percentile = min(0.90, current_percentile_decimal + 0.05)
        print(
            f"  Suggesting a higher target: {new_target_percentile * 100:.0f}th percentile instead"
        )

        # Update goal params with new target
        goal_params_dict["target_percentile"] = new_target_percentile
        goal_params_dict["suggested"] = True
        target_percentile = new_target_percentile

    # Binary search to find the target age where we can achieve the goal
    min_age = current_age
    max_age = min(current_age + 10, 70)  # Search up to 10 years or age 70

    def can_reach_goal_at_age(target_age):
        """Check if goal is achievable at target_age given gain rates."""
        # Map metric names to LMS function keys
        lms_key = "almi" if metric == "almi" else "lmi"  # ffmi uses lmi functions

        # Get LMS values for target age
        L_func = lms_functions[f"{lms_key}_L"]
        M_func = lms_functions[f"{lms_key}_M"]
        S_func = lms_functions[f"{lms_key}_S"]

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
        height_m = user_info_dict["height_in"] * 0.0254
        height_m2 = height_m**2

        if metric == "almi":
            # For ALMI, we need ALM gain. Assume ALM is ~45% of total lean mass
            alm_gain_needed_kg = required_gain * height_m2
            # Convert ALM gain to total lean mass gain (ALM/TLM ratio ~0.45)
            tlm_gain_needed_kg = alm_gain_needed_kg / 0.45
        else:  # ffmi
            # For FFMI, direct lean mass gain
            tlm_gain_needed_kg = required_gain * height_m2

        # Calculate time needed using progressive gain model
        years_available = target_age - current_age
        if years_available <= 0:
            return False

        # Calculate total achievable gain over the available timeframe
        achievable_gain_kg, _ = calculate_progressive_gain_over_time(
            user_info_dict, training_level, current_age, years_available
        )

        # Check if achievable gain meets requirement
        return achievable_gain_kg >= tlm_gain_needed_kg

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
        print(
            f"  Could not find feasible timeframe for {target_percentile * 100:.0f}th percentile {metric.upper()}"
        )
        print(f"  Using 2-year timeframe as fallback (age {best_age:.1f})")
    else:
        time_to_goal = best_age - current_age

        # Get progressive gain explanation for this timeframe
        _, progressive_explanation = calculate_progressive_gain_over_time(
            user_info_dict, training_level, current_age, time_to_goal
        )

        print(
            f"  ✓ Calculated feasible timeframe: {time_to_goal:.1f} years (age {best_age:.1f}) using progressive gain model"
        )
        print(f"  {progressive_explanation}")

    # Prepare messages for legacy return format
    messages = [level_explanation, gain_explanation]
    if best_age is None:
        messages.append(
            f"Could not find feasible timeframe for {target_percentile * 100:.0f}th percentile {metric.upper()}"
        )
        messages.append(f"Using 2-year timeframe as fallback (age {best_age:.1f})")
    else:
        time_to_goal = best_age - current_age
        _, progressive_explanation = calculate_progressive_gain_over_time(
            user_info_dict, training_level, current_age, time_to_goal
        )
        messages.append(
            f"Calculated feasible timeframe: {time_to_goal:.1f} years (age {best_age:.1f}) using progressive gain model"
        )
        messages.append(progressive_explanation)

    # Update goal parameters for legacy return
    updated_goal = goal_params_dict.copy()
    updated_goal["target_age"] = best_age
    updated_goal["suggested"] = goal_params_dict.get(
        "suggested", True
    )  # Preserve original or default to True

    if is_legacy_call:
        # Return legacy format (tuple)
        return updated_goal, messages
    else:
        # For new dataclass format, use create_goal_row to get detailed calculations
        temp_goal_dict = {
            "target_percentile": target_percentile,
            "target_age": best_age,
            "suggested": goal_params_dict.get("suggested", True),
            "target_body_fat_percentage": goal_params_dict.get(
                "target_body_fat_percentage"
            ),
        }

        goal_row, goal_calculations = create_goal_row(
            temp_goal_dict, user_info_dict, processed_data, lms_functions, metric
        )

        if goal_calculations is None:
            return None

        # Convert to GoalResults dataclass
        return GoalResults(
            target_age=best_age,
            target_percentile=target_percentile,
            target_metric_value=goal_calculations["target_metric_value"],
            target_z_score=goal_calculations["target_z_score"],
            metric_change_needed=goal_calculations["metric_change_needed"],
            lean_change_needed_lbs=goal_calculations["lean_change_needed_lbs"],
            alm_change_needed_lbs=goal_calculations["alm_change_needed_lbs"],
            alm_change_needed_kg=goal_calculations["alm_change_needed_kg"],
            tlm_change_needed_lbs=goal_calculations["tlm_change_needed_lbs"],
            tlm_change_needed_kg=goal_calculations["tlm_change_needed_kg"],
            weight_change=goal_calculations["weight_change"],
            lean_change=goal_calculations["lean_change"],
            fat_change=goal_calculations["fat_change"],
            bf_change=goal_calculations["bf_change"],
            percentile_change=goal_calculations["percentile_change"],
            z_change=goal_calculations["z_change"],
            target_body_composition=goal_calculations["target_body_composition"],
            suggested=goal_params_dict.get("suggested", True),
            target_almi=goal_calculations.get("target_almi"),
            target_ffmi=goal_calculations.get("target_ffmi"),
        )


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

    with open(config_path, "r") as f:
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
    if gender_lower in ["m", "male"]:
        return 0
    elif gender_lower in ["f", "female"]:
        return 1
    else:
        raise ValueError(
            f"Unrecognized gender: {gender_str}. Use 'm', 'f', 'male', or 'female'."
        )


def extract_data_from_config(config):
    """
    Extracts and processes user info, scan history, and goals from config.

    Args:
        config (dict): Validated configuration dictionary

    Returns:
        tuple: (user_info, scan_history, almi_goal, ffmi_goal)
    """
    # Process user info
    user_info = config["user_info"].copy()
    user_info["gender_code"] = parse_gender(user_info["gender"])

    # Add birth_date_str for test compatibility
    if "birth_date" in user_info:
        user_info["birth_date_str"] = user_info["birth_date"]

    # Extract scan history and add date_str for compatibility
    scan_history = []
    for scan in config["scan_history"]:
        scan_copy = scan.copy()
        if "date" in scan_copy:
            scan_copy["date_str"] = scan_copy["date"]
        scan_history.append(scan_copy)

    # Extract goals (optional)
    goals = config.get("goals", {})
    almi_goal = goals.get("almi")
    ffmi_goal = goals.get("ffmi")

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
        if hasattr(processed_data, "tail"):
            # DataFrame format
            recent_scans = processed_data.tail(min(3, len(processed_data)))
            alm_values = recent_scans["alm_kg"]
            tlm_values = recent_scans["total_lean_mass_lbs"] * 0.453592  # Convert to kg

            # Calculate ratio for each scan and take the mean
            ratios = alm_values / tlm_values
            personal_ratio = ratios.mean()
        else:
            # List format (for tests)
            recent_scans = processed_data[-min(3, len(processed_data)) :]
            ratios = []

            for scan in recent_scans:
                if "alm_lbs" in scan:
                    alm_kg = scan["alm_lbs"] * 0.453592
                    tlm_kg = scan["total_lean_mass_lbs"] * 0.453592
                    ratios.append(alm_kg / tlm_kg)

            if ratios:
                personal_ratio = sum(ratios) / len(ratios)
            else:
                # Fallback to population ratio if no ALM data
                gender_str = get_gender_string(user_info["gender_code"])
                return 0.45 if gender_str == "male" else 0.42

        print(
            f"Using personal ALM/TLM ratio of {personal_ratio:.3f} from {len(recent_scans)} recent scans"
        )
        return personal_ratio

    # For single scans, use population-based estimates from LMS functions
    if goal_params and "target_age" in goal_params and goal_params["target_age"]:
        target_age = goal_params["target_age"]
    else:
        # Use current age if no target age specified
        if hasattr(processed_data, "iloc"):
            target_age = processed_data.iloc[-1]["age_at_scan"]
        else:
            target_age = processed_data[-1]["age_at_scan"]

    # Calculate ratio from LMS functions at target age
    try:
        if "almi_M" in lms_functions and "lmi_M" in lms_functions:
            almi_at_age = lms_functions["almi_M"](target_age)
            lmi_at_age = lms_functions["lmi_M"](target_age)
            population_ratio = almi_at_age / lmi_at_age
        else:
            # Fallback to fixed ratios if LMS functions not available
            gender_str = get_gender_string(user_info["gender_code"])
            population_ratio = 0.45 if gender_str == "male" else 0.42
    except:
        # Fallback to fixed ratios if calculation fails
        gender_str = get_gender_string(user_info["gender_code"])
        population_ratio = 0.45 if gender_str == "male" else 0.42

    gender_str = get_gender_string(user_info["gender_code"])
    print(
        f"Using population-based ALM/TLM ratio of {population_ratio:.3f} for {gender_str} (single scan)"
    )
    return population_ratio


@functools.lru_cache(maxsize=8)
def load_lms_data(metric, gender_code, data_path="./data/"):
    """
    Loads LMS reference data and creates interpolation functions.

    This function is cached to avoid repeated file I/O operations.
    Cache stores up to 8 combinations (2 metrics × 2 genders × 2 paths typically).

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
        column_mapping = {"age": "Age", "lambda": "L", "mu": "M", "sigma": "S"}

        # Rename columns if they use lowercase names
        df_renamed = df.rename(columns=column_mapping)
        required_columns = ["Age", "L", "M", "S"]

        if not all(col in df_renamed.columns for col in required_columns):
            print(
                f"Error: LMS file {filepath} missing required columns: {required_columns}"
            )
            print(f"Available columns: {list(df.columns)}")
            return None, None, None

        df = df_renamed

        # Create interpolation functions
        L_func = interp1d(
            df["Age"],
            df["L"],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        M_func = interp1d(
            df["Age"],
            df["M"],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        S_func = interp1d(
            df["Age"],
            df["S"],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        print(f"Successfully loaded LMS data: {filename}")
        return L_func, M_func, S_func

    except Exception as e:
        print(f"Error loading LMS data from {filepath}: {e}")
        return None, None, None


def process_scans_and_goal(
    user_info, scan_history, almi_goal, ffmi_goal, lms_functions
):
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

    # Calculate T-score reference values for this gender
    almi_mu_peak, almi_sigma_peak = calculate_tscore_reference_values(
        user_info["gender_code"]
    )
    print(f"ALMI T-score reference: μ={almi_mu_peak:.3f}, σ={almi_sigma_peak:.3f}")

    # Convert scan history to DataFrame and sort by date
    df = pd.DataFrame(scan_history)

    # Handle both 'date' and 'date_str' field names for compatibility
    if "date" in df.columns:
        df["scan_date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    elif "date_str" in df.columns:
        df["scan_date"] = pd.to_datetime(df["date_str"], format="%m/%d/%Y")
    else:
        # For test data without dates, create dummy dates
        df["scan_date"] = pd.date_range(start="2020-01-01", periods=len(df), freq="6M")

    df = df.sort_values("scan_date").reset_index(drop=True)

    # Calculate basic metrics for each scan
    results = []
    goal_calculations = {}

    for i, scan in df.iterrows():
        # Calculate age at scan
        # Handle both birth_date and birth_date_str for test compatibility
        birth_date_key = "birth_date" if "birth_date" in user_info else "birth_date_str"
        scan_date_key = "date" if "date" in scan else "date_str"
        age_at_scan = calculate_age_precise(
            user_info[birth_date_key], scan[scan_date_key]
        )

        # Convert height to meters
        height_m = user_info["height_in"] * 0.0254
        height_m2 = height_m**2

        # Calculate appendicular lean mass (ALM) from arms + legs
        alm_kg = (scan["arms_lean_lbs"] + scan["legs_lean_lbs"]) * 0.453592

        # Calculate ALMI and FFMI
        almi_kg_m2 = alm_kg / height_m2
        ffmi_kg_m2 = (scan["total_lean_mass_lbs"] * 0.453592) / height_m2

        # Calculate Z-scores and percentiles using LMS functions
        almi_z, almi_percentile = calculate_z_percentile(
            almi_kg_m2,
            age_at_scan,
            lms_functions["almi_L"],
            lms_functions["almi_M"],
            lms_functions["almi_S"],
        )

        ffmi_z, ffmi_percentile = calculate_z_percentile(
            ffmi_kg_m2,
            age_at_scan,
            lms_functions["lmi_L"],
            lms_functions["lmi_M"],
            lms_functions["lmi_S"],
        )

        # Calculate T-scores using young adult reference values
        almi_t_score = calculate_t_score(almi_kg_m2, almi_mu_peak, almi_sigma_peak)
        # For FFMI T-score, we use same reference values as ALMI for now
        # In future, we could calculate separate FFMI reference values
        ffmi_t_score = calculate_t_score(ffmi_kg_m2, almi_mu_peak, almi_sigma_peak)

        # Store all calculated values
        result = {
            "date_str": scan[scan_date_key],
            "scan_date": scan["scan_date"]
            if "scan_date" in scan
            else pd.to_datetime(scan[scan_date_key], format="%m/%d/%Y"),
            "age_at_scan": age_at_scan,
            "total_weight_lbs": scan["total_weight_lbs"],
            "total_lean_mass_lbs": scan["total_lean_mass_lbs"],
            "fat_mass_lbs": scan["fat_mass_lbs"],
            "body_fat_percentage": scan["body_fat_percentage"],
            "arms_lean_lbs": scan["arms_lean_lbs"],
            "legs_lean_lbs": scan["legs_lean_lbs"],
            "alm_kg": alm_kg,
            "almi_kg_m2": almi_kg_m2,
            "ffmi_kg_m2": ffmi_kg_m2,
            "almi_z_score": almi_z,
            "almi_percentile": almi_percentile * 100,
            "almi_t_score": almi_t_score,
            "ffmi_z_score": ffmi_z,
            "ffmi_percentile": ffmi_percentile * 100,
            "ffmi_t_score": ffmi_t_score,
        }
        results.append(result)

    # Convert to DataFrame
    processed_data = pd.DataFrame(results)

    # Calculate changes between scans
    for i in range(len(processed_data)):
        if i == 0:
            # First scan - no changes to calculate
            processed_data.loc[i, "weight_change_last"] = np.nan
            processed_data.loc[i, "lean_change_last"] = np.nan
            processed_data.loc[i, "fat_change_last"] = np.nan
            processed_data.loc[i, "bf_change_last"] = np.nan
            processed_data.loc[i, "almi_z_change_last"] = np.nan
            processed_data.loc[i, "ffmi_z_change_last"] = np.nan
            processed_data.loc[i, "almi_t_change_last"] = np.nan
            processed_data.loc[i, "ffmi_t_change_last"] = np.nan
            processed_data.loc[i, "almi_pct_change_last"] = np.nan
            processed_data.loc[i, "ffmi_pct_change_last"] = np.nan
        else:
            # Calculate changes from previous scan
            prev_idx = i - 1
            processed_data.loc[i, "weight_change_last"] = (
                processed_data.loc[i, "total_weight_lbs"]
                - processed_data.loc[prev_idx, "total_weight_lbs"]
            )
            processed_data.loc[i, "lean_change_last"] = (
                processed_data.loc[i, "total_lean_mass_lbs"]
                - processed_data.loc[prev_idx, "total_lean_mass_lbs"]
            )
            processed_data.loc[i, "fat_change_last"] = (
                processed_data.loc[i, "fat_mass_lbs"]
                - processed_data.loc[prev_idx, "fat_mass_lbs"]
            )
            processed_data.loc[i, "bf_change_last"] = (
                processed_data.loc[i, "body_fat_percentage"]
                - processed_data.loc[prev_idx, "body_fat_percentage"]
            )
            processed_data.loc[i, "almi_z_change_last"] = (
                processed_data.loc[i, "almi_z_score"]
                - processed_data.loc[prev_idx, "almi_z_score"]
            )
            processed_data.loc[i, "ffmi_z_change_last"] = (
                processed_data.loc[i, "ffmi_z_score"]
                - processed_data.loc[prev_idx, "ffmi_z_score"]
            )
            processed_data.loc[i, "almi_t_change_last"] = (
                processed_data.loc[i, "almi_t_score"]
                - processed_data.loc[prev_idx, "almi_t_score"]
            )
            processed_data.loc[i, "ffmi_t_change_last"] = (
                processed_data.loc[i, "ffmi_t_score"]
                - processed_data.loc[prev_idx, "ffmi_t_score"]
            )
            processed_data.loc[i, "almi_pct_change_last"] = (
                processed_data.loc[i, "almi_percentile"]
                - processed_data.loc[prev_idx, "almi_percentile"]
            )
            processed_data.loc[i, "ffmi_pct_change_last"] = (
                processed_data.loc[i, "ffmi_percentile"]
                - processed_data.loc[prev_idx, "ffmi_percentile"]
            )

        # Calculate changes from first scan
        if i == 0:
            processed_data.loc[i, "weight_change_first"] = 0
            processed_data.loc[i, "lean_change_first"] = 0
            processed_data.loc[i, "fat_change_first"] = 0
            processed_data.loc[i, "bf_change_first"] = 0
            processed_data.loc[i, "almi_t_change_first"] = 0
            processed_data.loc[i, "ffmi_t_change_first"] = 0
        else:
            processed_data.loc[i, "weight_change_first"] = (
                processed_data.loc[i, "total_weight_lbs"]
                - processed_data.loc[0, "total_weight_lbs"]
            )
            processed_data.loc[i, "lean_change_first"] = (
                processed_data.loc[i, "total_lean_mass_lbs"]
                - processed_data.loc[0, "total_lean_mass_lbs"]
            )
            processed_data.loc[i, "fat_change_first"] = (
                processed_data.loc[i, "fat_mass_lbs"]
                - processed_data.loc[0, "fat_mass_lbs"]
            )
            processed_data.loc[i, "bf_change_first"] = (
                processed_data.loc[i, "body_fat_percentage"]
                - processed_data.loc[0, "body_fat_percentage"]
            )
            processed_data.loc[i, "almi_t_change_first"] = (
                processed_data.loc[i, "almi_t_score"]
                - processed_data.loc[0, "almi_t_score"]
            )
            processed_data.loc[i, "ffmi_t_change_first"] = (
                processed_data.loc[i, "ffmi_t_score"]
                - processed_data.loc[0, "ffmi_t_score"]
            )

    # Process goals and add goal rows if specified
    goal_rows = []

    if almi_goal:
        print(
            f"Processing {'suggested ' if almi_goal.get('suggested') else ''}ALMI goal: {almi_goal['target_percentile'] * 100:.0f}th percentile"
        )

        # Convert to dataclass format and handle suggested goals (auto-calculate target age)
        if almi_goal.get("target_age") in [None, "?"] or almi_goal.get("suggested"):
            # Convert user_info and scan_history to UserProfile for new function
            try:
                user_profile = convert_dict_to_user_profile(user_info, scan_history)
                goal_config = convert_dict_to_goal_config(
                    {**almi_goal, "metric_type": "almi"}
                )

                goal_results = calculate_suggested_goal(
                    goal_config, user_profile, processed_data, lms_functions
                )

                if goal_results is not None:
                    # Convert back to legacy dict format for create_goal_row
                    almi_goal = {
                        "target_percentile": goal_results.target_percentile,
                        "target_age": goal_results.target_age,
                        "suggested": goal_results.suggested,
                        "target_body_fat_percentage": goal_config.target_body_fat_percentage,
                    }
                    # For new dataclass format, no separate messages are returned - they're integrated
                    # Add a placeholder for backward compatibility
                    goal_calculations["messages"] = goal_calculations.get(
                        "messages", []
                    )
                else:
                    almi_goal = None

            except Exception as e:
                print(f"Error in goal calculation: {e}")
                # Fall back to 2-year timeframe
                almi_goal["target_age"] = user_info.get("current_age", 30) + 2
                almi_goal["suggested"] = True

        # Only create goal row if we have a valid goal (not None)
        if almi_goal is not None:
            goal_row, goal_calc = create_goal_row(
                almi_goal, user_info, processed_data, lms_functions, "almi"
            )
            if goal_row is not None:
                goal_rows.append(goal_row)
                goal_calculations["almi"] = goal_calc

    if ffmi_goal:
        print(
            f"Processing {'suggested ' if ffmi_goal.get('suggested') else ''}FFMI goal: {ffmi_goal['target_percentile'] * 100:.0f}th percentile"
        )

        # Convert to dataclass format and handle suggested goals (auto-calculate target age)
        if ffmi_goal.get("target_age") in [None, "?"] or ffmi_goal.get("suggested"):
            # Convert user_info and scan_history to UserProfile for new function
            try:
                user_profile = convert_dict_to_user_profile(user_info, scan_history)
                goal_config = convert_dict_to_goal_config(
                    {**ffmi_goal, "metric_type": "ffmi"}
                )

                goal_results = calculate_suggested_goal(
                    goal_config, user_profile, processed_data, lms_functions
                )

                if goal_results is not None:
                    # Convert back to legacy dict format for create_goal_row
                    ffmi_goal = {
                        "target_percentile": goal_results.target_percentile,
                        "target_age": goal_results.target_age,
                        "suggested": goal_results.suggested,
                        "target_body_fat_percentage": goal_config.target_body_fat_percentage,
                    }
                    # For new dataclass format, no separate messages are returned - they're integrated
                    # Add a placeholder for backward compatibility
                    goal_calculations["messages"] = goal_calculations.get(
                        "messages", []
                    )
                else:
                    ffmi_goal = None

            except Exception as e:
                print(f"Error in goal calculation: {e}")
                # Fall back to 2-year timeframe
                ffmi_goal["target_age"] = user_info.get("current_age", 30) + 2
                ffmi_goal["suggested"] = True

        # Only create goal row if we have a valid goal (not None)
        if ffmi_goal is not None:
            goal_row, goal_calc = create_goal_row(
                ffmi_goal, user_info, processed_data, lms_functions, "ffmi"
            )
            if goal_row is not None:
                goal_rows.append(goal_row)
                goal_calculations["ffmi"] = goal_calc

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

    if ranges["athletic"][0] <= bf_percentage <= ranges["athletic"][1]:
        return "athletic"
    elif ranges["fitness"][0] <= bf_percentage <= ranges["fitness"][1]:
        return "fitness"
    elif ranges["acceptable"][0] <= bf_percentage <= ranges["acceptable"][1]:
        return "acceptable"
    elif bf_percentage >= ranges["overweight"][0]:
        return "overweight"
    else:
        # Handle edge cases between ranges - assign to closest range
        if bf_percentage < ranges["athletic"][0]:
            return "athletic"  # Below athletic range (very lean)
        elif bf_percentage < ranges["fitness"][0]:
            return "athletic"  # Between athletic and fitness
        elif bf_percentage < ranges["acceptable"][0]:
            return "fitness"  # Between fitness and acceptable
        else:
            return "acceptable"  # Between acceptable and overweight


def calculate_target_bf_percentage(
    current_bf, gender, goal_duration_months, training_level="intermediate"
):
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
    if current_category in ["athletic", "fitness"]:
        return current_bf

    # If in acceptable range, aim for upper fitness range (easier to maintain)
    if current_category == "acceptable":
        if gender == "male":
            return 17.0  # Upper fitness range for males
        else:
            return 24.0  # Upper fitness range for females

    # If overweight, calculate feasible target based on duration
    # Conservative fat loss rate: 0.5-1% BF per month for overweight individuals
    # More aggressive: 1-2% BF per month with proper training
    max_bf_loss_rate = 1.0 if training_level in ["intermediate", "advanced"] else 0.5
    max_feasible_loss = goal_duration_months * max_bf_loss_rate

    # Target the fitness range, but don't exceed feasible loss rate
    target_fitness_bf = ranges["fitness"][1]  # Upper fitness range
    feasible_target = max(current_bf - max_feasible_loss, target_fitness_bf)

    # Ensure we don't go below healthy minimums
    healthy_minimum = ranges["athletic"][0]
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
        target_age = goal_params["target_age"]
        target_percentile = goal_params["target_percentile"]

        # Map metric names to LMS function keys
        lms_key = "almi" if metric == "almi" else "lmi"  # ffmi uses lmi functions

        # Get LMS values for target age
        L_func = lms_functions[f"{lms_key}_L"]
        M_func = lms_functions[f"{lms_key}_M"]
        S_func = lms_functions[f"{lms_key}_S"]

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
        current_metric = current_scan[f"{metric}_kg_m2"]

        # Calculate required metric change
        metric_change_needed = target_metric_value - current_metric

        # Convert to lean mass changes using ALM/TLM ratio
        height_m = user_info["height_in"] * 0.0254
        height_m2 = height_m**2

        # Get ALM/TLM ratio for calculations
        alm_tlm_ratio = get_alm_tlm_ratio(
            processed_data, goal_params, lms_functions, user_info
        )

        if metric == "almi":
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
        if (
            "target_body_fat_percentage" in goal_params
            and goal_params["target_body_fat_percentage"] is not None
        ):
            # Use explicitly specified target BF%
            target_body_fat_pct = goal_params["target_body_fat_percentage"]
        else:
            # Calculate intelligent target BF% based on health and feasibility
            current_bf = current_scan["body_fat_percentage"]
            current_age = current_scan["age_at_scan"]
            goal_duration_months = (
                target_age - current_age
            ) * 12  # Convert years to months
            training_level = user_info.get("training_level", "intermediate")
            # Handle both gender string and gender_code
            if "gender" in user_info:
                gender = user_info["gender"]
            else:
                gender = get_gender_string(user_info["gender_code"])

            target_body_fat_pct = calculate_target_bf_percentage(
                current_bf, gender, goal_duration_months, training_level
            )

            # Show BF% targeting rationale
            current_category = get_bf_category(current_bf, gender)
            if current_category in ["athletic", "fitness"]:
                print(
                    f"  Current BF% ({current_bf:.1f}%) is in {current_category} range - maintaining current level"
                )
            elif current_category == "acceptable":
                print(
                    f"  Current BF% ({current_bf:.1f}%) is acceptable - targeting upper fitness range ({target_body_fat_pct:.1f}%)"
                )
            else:
                print(
                    f"  Current BF% ({current_bf:.1f}%) is above healthy range - targeting feasible improvement to {target_body_fat_pct:.1f}% over {goal_duration_months:.1f} months"
                )

        target_lean_mass_lbs = (
            current_scan["total_lean_mass_lbs"] + tlm_change_needed_lbs
        )

        # Calculate target weight and fat mass
        # If BF% is specified, calculate weight to achieve that BF% with target lean mass
        target_fat_mass_lbs = (target_lean_mass_lbs * target_body_fat_pct) / (
            100 - target_body_fat_pct
        )
        target_weight_lbs = target_lean_mass_lbs + target_fat_mass_lbs

        # Calculate changes from current state
        weight_change = target_weight_lbs - current_scan["total_weight_lbs"]
        lean_change = target_lean_mass_lbs - current_scan["total_lean_mass_lbs"]
        fat_change = target_fat_mass_lbs - current_scan["fat_mass_lbs"]
        bf_change = target_body_fat_pct - current_scan["body_fat_percentage"]

        # Calculate percentile changes
        current_percentile = current_scan[
            f"{metric}_percentile"
        ]  # This is in percentage format (0-100)
        percentile_change = (
            target_percentile * 100
        ) - current_percentile  # Convert target to percentage first

        # Calculate Z-score changes
        current_z = current_scan[f"{metric}_z_score"]
        z_change = target_z - current_z

        print(
            f"{metric.upper()} goal calculations: ALM to add: {alm_change_needed_lbs:.1f} lbs ({alm_change_needed_kg:.2f} kg), Est. TLM gain: {tlm_change_needed_lbs:.1f} lbs ({tlm_change_needed_kg:.2f} kg), Target BF: {target_body_fat_pct:.1f}%, Total weight change: {weight_change:+.1f} lbs"
        )

        # Create goal row
        goal_row = {
            "date_str": f"{metric.upper()} Goal (Age {target_age})",
            "scan_date": pd.Timestamp.now(),  # Placeholder
            "age_at_scan": target_age,
            "total_weight_lbs": target_weight_lbs,
            "total_lean_mass_lbs": target_lean_mass_lbs,
            "fat_mass_lbs": target_fat_mass_lbs,
            "body_fat_percentage": target_body_fat_pct,
            "arms_lean_lbs": np.nan,  # Not calculated for goals
            "legs_lean_lbs": np.nan,  # Not calculated for goals
            "alm_kg": np.nan,  # Not calculated for goals
            "almi_kg_m2": target_metric_value
            if metric == "almi"
            else current_scan["almi_kg_m2"],
            "ffmi_kg_m2": target_metric_value
            if metric == "ffmi"
            else current_scan["ffmi_kg_m2"],
            "almi_z_score": target_z
            if metric == "almi"
            else current_scan["almi_z_score"],
            "almi_percentile": target_percentile * 100
            if metric == "almi"
            else current_scan["almi_percentile"],
            "ffmi_z_score": target_z
            if metric == "ffmi"
            else current_scan["ffmi_z_score"],
            "ffmi_percentile": target_percentile * 100
            if metric == "ffmi"
            else current_scan["ffmi_percentile"],
            "weight_change_last": weight_change,
            "lean_change_last": lean_change,
            "fat_change_last": fat_change,
            "bf_change_last": bf_change,
            "almi_z_change_last": z_change
            if metric == "almi"
            else current_scan.get("almi_z_change_last", 0),
            "ffmi_z_change_last": z_change
            if metric == "ffmi"
            else current_scan.get("ffmi_z_change_last", 0),
            "almi_pct_change_last": percentile_change
            if metric == "almi"
            else current_scan.get("almi_pct_change_last", 0),
            "ffmi_pct_change_last": percentile_change
            if metric == "ffmi"
            else current_scan.get("ffmi_pct_change_last", 0),
            "weight_change_first": target_weight_lbs
            - processed_data.iloc[0]["total_weight_lbs"],
            "lean_change_first": target_lean_mass_lbs
            - processed_data.iloc[0]["total_lean_mass_lbs"],
            "fat_change_first": target_fat_mass_lbs
            - processed_data.iloc[0]["fat_mass_lbs"],
            "bf_change_first": target_body_fat_pct
            - processed_data.iloc[0]["body_fat_percentage"],
        }

        # Goal calculations for plotting
        goal_calculations = {
            "target_age": target_age,
            "target_percentile": target_percentile,
            "target_metric_value": target_metric_value,
            "target_z_score": target_z,
            "metric_change_needed": metric_change_needed,
            "lean_change_needed_lbs": tlm_change_needed_lbs,
            "alm_change_needed_lbs": alm_change_needed_lbs,
            "alm_change_needed_kg": alm_change_needed_kg,
            "tlm_change_needed_lbs": tlm_change_needed_lbs,
            "tlm_change_needed_kg": tlm_change_needed_kg,
            "weight_change": weight_change,
            "lean_change": lean_change,
            "fat_change": fat_change,
            "bf_change": bf_change,
            "percentile_change": percentile_change,
            "z_change": z_change,
            "target_body_composition": {
                "weight_lbs": target_weight_lbs,
                "lean_mass_lbs": target_lean_mass_lbs,
                "fat_mass_lbs": target_fat_mass_lbs,
                "body_fat_percentage": target_body_fat_pct,
            },
            # Backwards compatibility field names for tests
            "alm_to_add_kg": alm_change_needed_kg,
            "estimated_tlm_gain_kg": tlm_change_needed_kg,
            "tlm_to_add_kg": tlm_change_needed_kg,
            # Test expected field names (for field name consistency)
            "target_z": target_z,
            "suggested": goal_params.get("suggested", False),
        }

        # Add metric-specific target field name
        if metric == "almi":
            goal_calculations["target_almi"] = target_metric_value
        else:  # ffmi
            goal_calculations["target_ffmi"] = target_metric_value

        return goal_row, goal_calculations

    except Exception as e:
        print(f"  Error creating {metric.upper()} goal row: {e}")
        return None, None


# ---------------------------------------------------------------------------
# PLOTTING LOGIC
# ---------------------------------------------------------------------------


def create_metric_plot(
    df_results, metric_to_plot, lms_functions, goal_calculations, return_figure=False
):
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
    if metric_to_plot == "ALMI":
        L_func = lms_functions["almi_L"]
        M_func = lms_functions["almi_M"]
        S_func = lms_functions["almi_S"]
        y_column = "almi_kg_m2"
        y_label = "ALMI (kg/m²)"
        plot_title = "Appendicular Lean Mass Index (ALMI) Percentiles"
    else:  # FFMI
        L_func = lms_functions["lmi_L"]
        M_func = lms_functions["lmi_M"]
        S_func = lms_functions["lmi_S"]
        y_column = "ffmi_kg_m2"
        y_label = "FFMI (kg/m²)"
        plot_title = "Fat-Free Mass Index (FFMI) Percentiles"

    # Define percentiles to plot
    percentiles = [0.03, 0.10, 0.25, 0.50, 0.75, 0.90, 0.97]
    percentile_labels = ["3rd", "10th", "25th", "50th", "75th", "90th", "97th"]
    colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FECCA7",
        "#DDA0DD",
        "#FFB347",
    ]

    # Plot percentile curves
    for _i, (percentile, label, color) in enumerate(
        zip(percentiles, percentile_labels, colors)
    ):
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

        ax.plot(
            age_range,
            curve_values,
            color=color,
            linewidth=2,
            label=f"{label} percentile",
            alpha=0.8,
        )

    # Filter data for actual scans (not goal rows)
    scan_data = df_results[~df_results["date_str"].str.contains("Goal", na=False)]

    # Plot actual data points
    if len(scan_data) > 0:
        ax.scatter(
            scan_data["age_at_scan"],
            scan_data[y_column],
            color="red",
            s=100,
            zorder=5,
            label="Your scans",
            edgecolors="black",
            linewidth=1,
        )

        # Connect points with lines if multiple scans
        if len(scan_data) > 1:
            ax.plot(
                scan_data["age_at_scan"],
                scan_data[y_column],
                color="red",
                linewidth=2,
                alpha=0.7,
                zorder=4,
            )

    # Plot goal if available
    goal_key = metric_to_plot.lower()
    if goal_key in goal_calculations:
        goal_calc = goal_calculations[goal_key]
        goal_age = goal_calc["target_age"]
        goal_value = goal_calc["target_metric_value"]

        ax.scatter(
            [goal_age],
            [goal_value],
            color="gold",
            s=150,
            marker="*",
            zorder=6,
            label="Goal",
            edgecolors="black",
            linewidth=1,
        )

        # Draw line from last scan to goal
        if len(scan_data) > 0:
            last_scan = scan_data.iloc[-1]
            ax.plot(
                [last_scan["age_at_scan"], goal_age],
                [last_scan[y_column], goal_value],
                color="gold",
                linewidth=3,
                linestyle="--",
                alpha=0.8,
                zorder=3,
            )

    # Customize plot
    ax.set_xlabel("Age (years)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(plot_title, fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if return_figure:
        return fig
    else:
        # Save plot
        filename = f"{metric_to_plot.lower()}_plot.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved as: {filename}")
        return None


def plot_metric_with_table(
    df_results, metric_to_plot, lms_functions, goal_calculations
):
    """
    Legacy function that creates plots and saves them to disk.
    Uses the new create_metric_plot function for actual plotting.
    """
    # Use the new function to create and save the plot
    create_metric_plot(
        df_results,
        metric_to_plot,
        lms_functions,
        goal_calculations,
        return_figure=False,
    )

    # Export data table to CSV (only for ALMI to avoid duplication)
    if metric_to_plot == "ALMI":
        csv_filename = "almi_stats_table.csv"

        # Create CSV export with new table structure (main data + Changes row)
        main_columns = [
            "date_str",
            "age_at_scan",
            "total_weight_lbs",
            "total_lean_mass_lbs",
            "fat_mass_lbs",
            "body_fat_percentage",
            "almi_kg_m2",
            "ffmi_kg_m2",
        ]
        main_names = ["Date", "Age", "Weight", "Lean", "Fat", "BF%", "ALMI", "FFMI"]

        # Create main data table for CSV
        df_csv = df_results[main_columns].copy()
        df_csv.columns = main_names

        # Create changes row for CSV
        last_scan_idx = len(df_results) - 1
        first_scan_idx = 0
        changes_data = {}
        changes_data["Date"] = "Changes"  # Plain text for CSV
        # Calculate age change for CSV
        age_change = (
            df_results.loc[last_scan_idx, "age_at_scan"]
            - df_results.loc[first_scan_idx, "age_at_scan"]
        )
        changes_data["Age"] = age_change

        # Map change columns
        change_mapping = {
            "Weight": "weight_change_last",
            "Lean": "lean_change_last",
            "Fat": "fat_change_last",
            "BF%": "bf_change_last",
            "ALMI": "almi_z_change_last",
            "FFMI": "ffmi_z_change_last",
        }

        for display_col, change_col in change_mapping.items():
            if change_col in df_results.columns:
                change_val = df_results.loc[last_scan_idx, change_col]
                if pd.notna(change_val):
                    changes_data[display_col] = change_val  # Raw numeric value for CSV
                else:
                    changes_data[display_col] = ""
            else:
                changes_data[display_col] = ""

        # Add changes row and export
        changes_row = pd.DataFrame([changes_data])
        df_csv_with_changes = pd.concat([df_csv, changes_row], ignore_index=True)
        df_csv_with_changes.to_csv(csv_filename, index=False)
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
    scan_data = df_results[~df_results["date_str"].str.contains("Goal", na=False)]

    if len(scan_data) == 0:
        if return_figure:
            return fig
        else:
            plt.close()
            print("No scan data found for body fat plot")
            return None

    # Get gender for health ranges
    if "gender" in user_info:
        gender = user_info["gender"]
    else:
        gender = get_gender_string(user_info["gender_code"])

    # Add healthy body fat percentage ranges as background shading
    ranges = HEALTHY_BF_RANGES[gender]

    # Create age range for background shading
    scan_data["age_at_scan"].min() - 1
    scan_data["age_at_scan"].max() + 1

    # Add background shading for health ranges
    ax.axhspan(
        ranges["athletic"][0],
        ranges["athletic"][1],
        color="lightgreen",
        alpha=0.2,
        label="Athletic Range",
    )
    ax.axhspan(
        ranges["fitness"][0],
        ranges["fitness"][1],
        color="lightblue",
        alpha=0.2,
        label="Fitness Range",
    )
    ax.axhspan(
        ranges["acceptable"][0],
        ranges["acceptable"][1],
        color="lightyellow",
        alpha=0.2,
        label="Acceptable Range",
    )

    # Plot the actual data line
    ax.plot(
        scan_data["age_at_scan"],
        scan_data["body_fat_percentage"],
        color="red",
        linewidth=3,
        marker="o",
        markersize=8,
        label="Your Body Fat %",
        zorder=10,
    )

    # Add data point annotations with values
    for _, scan in scan_data.iterrows():
        ax.annotate(
            f"{scan['body_fat_percentage']:.1f}%",
            (scan["age_at_scan"], scan["body_fat_percentage"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="darkred",
        )

    # Customize plot
    ax.set_xlabel("Age (years)", fontsize=12)
    ax.set_ylabel("Body Fat Percentage (%)", fontsize=12)
    ax.set_title("Body Fat Percentage Over Time", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    # Set reasonable y-axis limits based on data and health ranges
    y_min = min(scan_data["body_fat_percentage"].min() - 2, ranges["athletic"][0] - 1)
    y_max = max(scan_data["body_fat_percentage"].max() + 2, ranges["acceptable"][1] + 1)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    if return_figure:
        return fig
    else:
        # Save plot
        filename = "bf_plot.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
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
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly is required for interactive body fat plots")

    # Create figure
    fig = go.Figure()

    # Filter data for actual scans (not goal rows)
    scan_data = df_results[~df_results["date_str"].str.contains("Goal", na=False)]

    if len(scan_data) == 0:
        fig.add_annotation(
            text="No scan data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Get gender for health ranges
    if "gender" in user_info:
        gender = user_info["gender"]
    else:
        gender = get_gender_string(user_info["gender_code"])

    # Add healthy body fat percentage ranges as background shading
    ranges = HEALTHY_BF_RANGES[gender]

    # Create age range for background shading
    scan_data["age_at_scan"].min() - 1
    scan_data["age_at_scan"].max() + 1

    # Add background shapes for health ranges
    fig.add_hrect(
        y0=ranges["athletic"][0],
        y1=ranges["athletic"][1],
        fillcolor="lightgreen",
        opacity=0.2,
        layer="below",
        line_width=0,
        annotation_text="Athletic Range",
        annotation_position="top left",
    )
    fig.add_hrect(
        y0=ranges["fitness"][0],
        y1=ranges["fitness"][1],
        fillcolor="lightblue",
        opacity=0.2,
        layer="below",
        line_width=0,
        annotation_text="Fitness Range",
        annotation_position="top left",
    )
    fig.add_hrect(
        y0=ranges["acceptable"][0],
        y1=ranges["acceptable"][1],
        fillcolor="lightyellow",
        opacity=0.2,
        layer="below",
        line_width=0,
        annotation_text="Acceptable Range",
        annotation_position="top left",
    )

    # Create hover text with comprehensive information
    hover_text = []
    for _, scan in scan_data.iterrows():
        # Calculate body fat category
        bf_pct = scan["body_fat_percentage"]
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
            f"<b>Fat Mass:</b> {scan['fat_mass_lbs']:.1f} lbs",
        ]

        # Add change information if available
        if pd.notna(scan.get("bf_change_last")):
            change_last = scan["bf_change_last"]
            change_sign = "+" if change_last >= 0 else ""
            hover_info.extend(
                ["", f"<b>Change from last scan:</b> {change_sign}{change_last:.1f}%"]
            )

        if pd.notna(scan.get("bf_change_first")):
            change_first = scan["bf_change_first"]
            change_sign = "+" if change_first >= 0 else ""
            hover_info.extend(
                [f"<b>Change from first scan:</b> {change_sign}{change_first:.1f}%"]
            )

        hover_text.append("<br>".join(hover_info))

    # Add the main data line with markers
    fig.add_trace(
        go.Scatter(
            x=scan_data["age_at_scan"],
            y=scan_data["body_fat_percentage"],
            mode="lines+markers",
            name="Your Body Fat %",
            line={"color": "red", "width": 3},
            marker={"color": "red", "size": 10, "line": {"color": "black", "width": 1}},
            hovertemplate="%{text}<extra></extra>",
            text=hover_text,
        )
    )

    # Customize layout
    y_min = min(scan_data["body_fat_percentage"].min() - 2, ranges["athletic"][0] - 1)
    y_max = max(scan_data["body_fat_percentage"].max() + 2, ranges["acceptable"][1] + 1)

    fig.update_layout(
        title={
            "text": "Body Fat Percentage Over Time",
            "font": {"size": 16, "family": "Arial", "color": "black"},
            "x": 0,
        },
        xaxis={
            "title": {"text": "Age (years)", "font": {"size": 14}},
            "tickfont": {"size": 12},
            "gridcolor": "lightgray",
            "gridwidth": 0.5,
        },
        yaxis={
            "title": {"text": "Body Fat Percentage (%)", "font": {"size": 14}},
            "tickfont": {"size": 12},
            "gridcolor": "lightgray",
            "gridwidth": 0.5,
            "range": [y_min, y_max],
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        hoverlabel={"bgcolor": "white", "bordercolor": "black", "font_size": 12},
        height=600,
        showlegend=True,
    )

    return fig


def create_plotly_dual_mode_plot(
    df_results,
    metric_to_plot,
    lms_functions,
    goal_calculations,
    almi_mu_peak=None,
    almi_sigma_peak=None,
):
    """
    Creates interactive Plotly plots with optional T-score toggle overlay for ALMI.

    This function generates visualizations that:
    - Always show percentile curves to maintain consistent y-axis range
    - Keep same markers and data points in both modes
    - Add T-score information as overlay layers when toggled (ALMI only)
    - Use consistent legend font sizes to avoid jarring transitions
    - Toggle is additive: T-score mode adds layers without removing percentiles

    Args:
        df_results (pd.DataFrame): Complete results DataFrame with scan history and goals
        metric_to_plot (str): Either 'ALMI' or 'FFMI'
        lms_functions (dict): Dictionary containing LMS interpolation functions
        goal_calculations (dict): Goal calculation results
        almi_mu_peak (float, optional): Young adult reference mean for T-score calculation (ALMI only)
        almi_sigma_peak (float, optional): Young adult reference SD for T-score calculation (ALMI only)

    Returns:
        plotly.graph_objects.Figure: Interactive plotly figure with optional T-score toggle
    """
    import plotly.graph_objects as go
    import scipy.stats as stats
    from plotly.subplots import make_subplots

    # Define age range for curves
    age_range = np.linspace(18, 80, 100)

    # Select appropriate LMS functions and labels
    if metric_to_plot == "ALMI":
        L_func = lms_functions["almi_L"]
        M_func = lms_functions["almi_M"]
        S_func = lms_functions["almi_S"]
        y_column = "almi_kg_m2"
        y_label = "ALMI (kg/m²)"
        plot_title = "Appendicular Lean Mass Index (ALMI)"
        t_score_column = "almi_t_score"
        z_score_column = "almi_z_score"
        percentile_column = "almi_percentile"
    else:  # FFMI
        L_func = lms_functions["lmi_L"]
        M_func = lms_functions["lmi_M"]
        S_func = lms_functions["lmi_S"]
        y_column = "ffmi_kg_m2"
        y_label = "FFMI (kg/m²)"
        plot_title = "Fat-Free Mass Index (FFMI)"
        t_score_column = "ffmi_t_score"
        z_score_column = "ffmi_z_score"
        percentile_column = "ffmi_percentile"

    # Create figure (no secondary y-axis needed for seamless approach)
    fig = go.Figure()

    # Define percentiles and colors - these ALWAYS stay visible
    # Reversed order to match visual appearance in plot (highest to lowest)
    percentiles = [0.97, 0.90, 0.75, 0.50, 0.25, 0.10, 0.03]
    percentile_labels = ["97th", "90th", "75th", "50th", "25th", "10th", "3rd"]
    colors = [
        "#FFB347",
        "#DDA0DD",
        "#FECCA7",
        "#96CEB4",
        "#45B7D1",
        "#4ECDC4",
        "#FF6B6B",
    ]

    # Add percentile curves (ALWAYS visible for consistent axis range)
    for i, (percentile, label, color) in enumerate(
        zip(percentiles, percentile_labels, colors)
    ):
        z_score = stats.norm.ppf(percentile)
        curve_values = []

        for age in age_range:
            try:
                L_val = L_func(age)
                M_val = M_func(age)
                S_val = S_func(age)
                from core import get_value_from_zscore

                value = get_value_from_zscore(z_score, L_val, M_val, S_val)
                curve_values.append(value)
            except:
                curve_values.append(None)

        fig.add_trace(
            go.Scatter(
                x=age_range,
                y=curve_values,
                mode="lines",
                name=f"{label} percentile",
                line={"color": color, "width": 2},
                hovertemplate=f"{label} percentile<br>Age: %{{x:.1f}} years<br>{y_label}: %{{y:.2f}}<extra></extra>",
                visible=True,  # Always visible for consistent axis range
                legendgroup="percentiles",
            )
        )

    # Define T-score zones for ALMI only (after goal marker)
    # T-scores compare to peak young adult muscle mass (ages 20-30)
    t_score_bands = []
    enable_tscore = (
        metric_to_plot == "ALMI"
        and almi_mu_peak is not None
        and almi_sigma_peak is not None
    )

    if enable_tscore:
        # Reversed order to match visual appearance in plot (highest to lowest)
        t_score_bands = [
            (2, 4, "#228B22", "Elite Zone"),
            (0, 2, "#90EE90", "Peak Zone"),
            (-1, 0, "#FFD700", "Approaching Peak"),
            (-2, -1, "#FFA500", "Below Peak"),
            (-4, -2, "#FF6B6B", "Well Below Peak"),
        ]

    # Filter data for actual scans (not goal rows)
    scan_data = df_results[~df_results["date_str"].str.contains("Goal", na=False)]

    # Add actual data points with comprehensive hover information
    if len(scan_data) > 0:
        # Create hover text with both Z-scores and T-scores
        hover_text = []
        for _, scan in scan_data.iterrows():
            hover_info = [
                f"<b>Scan Date:</b> {scan['date_str']}",
                f"<b>Age:</b> {scan['age_at_scan']:.1f} years",
                f"<b>{y_label}:</b> {scan[y_column]:.2f}",
                f"<b>Z-Score:</b> {scan[z_score_column]:.2f}",
                f"<b>Percentile:</b> {scan[percentile_column]:.1f}%",
                f"<b>T-Score:</b> {scan[t_score_column]:.2f}",
                "",
                f"<b>Weight:</b> {scan['total_weight_lbs']:.1f} lbs",
                f"<b>Lean Mass:</b> {scan['total_lean_mass_lbs']:.1f} lbs",
                f"<b>Fat Mass:</b> {scan['fat_mass_lbs']:.1f} lbs",
                f"<b>Body Fat:</b> {scan['body_fat_percentage']:.1f}%",
            ]
            hover_text.append("<br>".join(hover_info))

        # Add scan points (same markers for both modes)
        fig.add_trace(
            go.Scatter(
                x=scan_data["age_at_scan"],
                y=scan_data[y_column],
                mode="markers",
                name="Your Scans",
                marker={
                    "color": "red",
                    "size": 12,
                    "line": {"color": "black", "width": 1},
                },
                hovertemplate="%{text}<extra></extra>",
                text=hover_text,
                visible=True,  # Always visible - same markers in both modes
                legendgroup="scans",
            )
        )

        # Connect points with lines if multiple scans
        if len(scan_data) > 1:
            fig.add_trace(
                go.Scatter(
                    x=scan_data["age_at_scan"],
                    y=scan_data[y_column],
                    mode="lines",
                    name="Progression",
                    line={"color": "red", "width": 2},
                    opacity=0.7,
                    showlegend=False,
                    hoverinfo="skip",
                    visible=True,  # Always visible - same line in both modes
                    legendgroup="progression",
                )
            )

    # Add goal if available (similar implementation for both modes)
    goal_key = metric_to_plot.lower()
    if goal_key in goal_calculations:
        goal_calc = goal_calculations[goal_key]
        goal_age = goal_calc["target_age"]
        goal_value = goal_calc["target_metric_value"]
        goal_percentile = goal_calc["target_percentile"]

        # Build comprehensive goal hover information
        goal_z_score = goal_calc.get("target_z_score", 0)
        goal_t_score = calculate_t_score(goal_value, almi_mu_peak, almi_sigma_peak)
        target_body_comp = goal_calc.get("target_body_composition", {})

        goal_hover_text = "<br>".join(
            [
                f"<b>🎯 {metric_to_plot} Goal</b>",
                f"<b>Target Age:</b> {goal_age:.1f} years",
                f"<b>Target {y_label}:</b> {goal_value:.2f}",
                f"<b>Target Z-Score:</b> {goal_z_score:.2f}",
                f"<b>Target Percentile:</b> {goal_percentile * 100:.0f}%",
                f"<b>Target T-Score:</b> {goal_t_score:.2f}",
                "",
                "<b>Target Body Composition:</b>",
                f"<b>Weight:</b> {target_body_comp.get('weight_lbs', 0):.1f} lbs",
                f"<b>Lean Mass:</b> {target_body_comp.get('lean_mass_lbs', 0):.1f} lbs",
                f"<b>Fat Mass:</b> {target_body_comp.get('fat_mass_lbs', 0):.1f} lbs",
                f"<b>Body Fat:</b> {target_body_comp.get('body_fat_percentage', 0):.1f}%",
            ]
        )

        # Goal marker (same for both modes)
        fig.add_trace(
            go.Scatter(
                x=[goal_age],
                y=[goal_value],
                mode="markers",
                name="Goal",
                marker={
                    "color": "gold",
                    "size": 15,
                    "symbol": "star",
                    "line": {"color": "black", "width": 1},
                },
                hovertemplate="%{text}<extra></extra>",
                text=[goal_hover_text],
                visible=True,  # Always visible - same marker in both modes
                legendgroup="goal",
            )
        )

    # Add T-score zones AFTER goal marker (ALMI only - so they appear below goal in legend)
    if enable_tscore:
        for t_min, t_max, color, label in t_score_bands:
            # Convert T-scores to metric values for band boundaries
            metric_min = almi_mu_peak + t_min * almi_sigma_peak
            metric_max = almi_mu_peak + t_max * almi_sigma_peak

            # Create filled area for peak zone band
            band_y = [metric_min, metric_max, metric_max, metric_min, metric_min]
            band_x = [18, 18, 80, 80, 18]

            fig.add_trace(
                go.Scatter(
                    x=band_x,
                    y=band_y,
                    fill="toself",
                    fillcolor=color,
                    opacity=0.3,
                    line={"width": 0},
                    name=label,
                    hoverinfo="skip",
                    visible=False,  # Hidden by default, shown when T-score toggle is on
                    legendgroup="tscores",
                )
            )

        # Add T-score grid lines AFTER goal marker (ALMI only - hidden by default)
        for t_score in range(-3, 4):
            metric_value = almi_mu_peak + t_score * almi_sigma_peak
            fig.add_trace(
                go.Scatter(
                    x=[18, 80],
                    y=[metric_value, metric_value],
                    mode="lines",
                    line={"color": "gray", "dash": "dash", "width": 1},
                    opacity=0.5,
                    name=f"T-score {t_score}",
                    hoverinfo="skip",
                    visible=False,  # Hidden by default, shown when T-score toggle is on
                    showlegend=False,
                )
            )

    # Calculate fixed y-axis range that works for both percentiles and T-score overlays
    # This ensures seamless toggle without axis range changes

    # Get range from percentile curves (3rd to 97th percentile at age range)
    percentile_y_values = []
    for percentile in [0.03, 0.97]:  # Min and max percentiles
        z_score = stats.norm.ppf(percentile)
        for age in [20, 30, 40, 50, 60, 70]:  # Sample ages
            try:
                L_val = L_func(age)
                M_val = M_func(age)
                S_val = S_func(age)
                value = get_value_from_zscore(z_score, L_val, M_val, S_val)
                if not pd.isna(value):
                    percentile_y_values.append(value)
            except:
                continue

    # Get range from T-score bands (ALMI only)
    t_score_y_values = []
    if enable_tscore:
        for t_score in [-3, 3]:  # Reasonable T-score range
            metric_value = almi_mu_peak + t_score * almi_sigma_peak
            t_score_y_values.append(metric_value)

    # Get range from actual scan data
    scan_y_values = []
    if len(scan_data) > 0:
        scan_y_values = scan_data[y_column].dropna().tolist()

    # Get range from goal data
    goal_y_values = []
    if goal_key in goal_calculations:
        goal_y_values = [goal_calculations[goal_key]["target_metric_value"]]

    # Combine all y-values and calculate fixed range with padding
    all_y_values = (
        percentile_y_values + t_score_y_values + scan_y_values + goal_y_values
    )
    if all_y_values:
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        y_padding = (y_max - y_min) * 0.1  # 10% padding
        fixed_y_min = y_min - y_padding
        fixed_y_max = y_max + y_padding
    else:
        # Fallback range if no data
        fixed_y_min = 4.0
        fixed_y_max = 12.0

    # Customize layout
    fig.update_layout(
        title={
            "text": f"{plot_title} - Interactive Analysis",
            "font": {"size": 16, "family": "Arial", "color": "black"},
            "x": 0,
        },
        xaxis={
            "title": {"text": "Age (years)", "font": {"size": 14}},
            "tickfont": {"size": 12},
            "gridcolor": "lightgray",
            "gridwidth": 0.5,
            "range": [18, 80],
        },
        yaxis={
            "title": {"text": y_label, "font": {"size": 14}},
            "tickfont": {"size": 12},
            "gridcolor": "lightgray",
            "gridwidth": 0.5,
            "range": [fixed_y_min, fixed_y_max],  # Fixed range for seamless toggle
        },
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
            "font": {"size": 10},
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        hoverlabel={"bgcolor": "white", "bordercolor": "black", "font_size": 12},
        height=650
        if enable_tscore
        else 600,  # Increased height only if T-score annotations needed
        margin={
            "r": 150,
            "b": 100 if enable_tscore else 50,
        },  # Added bottom margin only if T-score annotations needed
        # Add single toggle button for T-score overlay (ALMI only)
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": "Add T-score Overlay",
                        "method": "update",
                        "args": [
                            {
                                "visible": [True]
                                * len(percentiles)  # percentile curves stay visible
                                + [True]  # scan points stay the same
                                + (
                                    [True] if len(scan_data) > 1 else []
                                )  # progression line stays the same
                                + (
                                    [True] if goal_key in goal_calculations else []
                                )  # goal marker stays the same
                                + [True]
                                * len(
                                    t_score_bands
                                )  # T-score bands shown (now after goal)
                                + [True] * 7  # T-score grid lines shown
                            },
                            {
                                "annotations[0].visible": True,  # Show T-score explanation
                            }
                            if enable_tscore
                            else {},
                        ],
                        "args2": [
                            {
                                "visible": [True]
                                * len(percentiles)  # percentile curves stay visible
                                + [True]  # scan points stay the same
                                + (
                                    [True] if len(scan_data) > 1 else []
                                )  # progression line stays the same
                                + (
                                    [True] if goal_key in goal_calculations else []
                                )  # goal marker stays the same
                                + [False]
                                * len(
                                    t_score_bands
                                )  # T-score bands hidden (now after goal)
                                + [False] * 7  # T-score grid lines hidden
                            },
                            {
                                "annotations[0].visible": False,  # Hide T-score explanation
                            }
                            if enable_tscore
                            else {},
                        ],
                    },
                ],
                "active": 0,  # Start with T-score overlay off (uses args2 by default)
                "direction": "down",
                "showactive": False,  # Don't show active state styling
                "x": 0.01,
                "xanchor": "left",
                "y": 1.05,
                "yanchor": "top",
                "type": "buttons",
            }
        ]
        if enable_tscore
        else [],  # Only show toggle button for ALMI
        # Add informative annotation for T-score overlay only (ALMI only)
        annotations=[
            {
                "text": "<b>T-score Overlay:</b> Compare to PEAK young adult muscle mass (ages 20-30). T ≥ +2 = Elite Zone, T ≥ 0 = Peak Zone, T < -2 = Well Below Peak.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.02,
                "y": -0.12,
                "xanchor": "left",
                "yanchor": "top",
                "showarrow": False,
                "font": {"size": 11, "color": "darkgreen"},
                "bgcolor": "rgba(240, 255, 240, 0.8)",
                "bordercolor": "lightgreen",
                "borderwidth": 1,
                "visible": False,  # Hidden by default, shown when T-score mode is active
            },
        ]
        if enable_tscore
        else [],  # Only show annotation for ALMI
    )

    return fig


def create_tscore_plot(
    df_results, metric_to_plot, almi_mu_peak, almi_sigma_peak, return_figure=False
):
    """
    Creates T-score plot showing horizontal risk bands and T-score axis.

    This function generates T-score visualizations that include:
    - Horizontal T-score risk bands (≥0 green, -1 yellow, -2 orange, <-2 red)
    - Grid lines at T-score intervals
    - Open square markers for scan data points
    - Right-hand T-score axis
    - Small thermometer inset showing latest T-score

    Args:
        df_results (pd.DataFrame): Complete results DataFrame with scan history
        metric_to_plot (str): Either 'ALMI' or 'FFMI'
        almi_mu_peak (float): Young adult reference mean for T-score calculation
        almi_sigma_peak (float): Young adult reference SD for T-score calculation
        return_figure (bool): If True, returns matplotlib figure object instead of saving

    Returns:
        matplotlib.figure.Figure or None: Figure object if return_figure=True, None otherwise
    """
    if not return_figure:
        print(f"Generating T-score plot for {metric_to_plot}...")

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get y-column and labels
    if metric_to_plot == "ALMI":
        y_column = "almi_kg_m2"
        y_label = "ALMI (kg/m²)"
        plot_title = "Appendicular Lean Mass Index (ALMI) T-scores"
    else:  # FFMI
        y_column = "ffmi_kg_m2"
        y_label = "FFMI (kg/m²)"
        plot_title = "Fat-Free Mass Index (FFMI) T-scores"

    # Calculate ALMI range for T-score bands
    min_almi = 4.0  # Reasonable minimum for plotting
    max_almi = 12.0  # Reasonable maximum for plotting

    # Create horizontal T-score risk bands
    t_score_bands = [
        (-4, -2, "#FF6B6B", "Severe Risk (T < -2.0)"),
        (-2, -1, "#FFA500", "Moderate Risk (-2.0 ≤ T < -1.0)"),
        (-1, 0, "#FFD700", "Mild Risk (-1.0 ≤ T < 0)"),
        (0, 4, "#90EE90", "Normal (T ≥ 0)"),
    ]

    # Draw T-score bands as horizontal stripes
    for t_min, t_max, color, label in t_score_bands:
        # Convert T-scores to ALMI values for band boundaries
        almi_min = almi_mu_peak + t_min * almi_sigma_peak
        almi_max = almi_mu_peak + t_max * almi_sigma_peak

        # Clip to plotting range
        almi_min = max(almi_min, min_almi)
        almi_max = min(almi_max, max_almi)

        ax.axhspan(almi_min, almi_max, alpha=0.3, color=color, label=label)

    # Add horizontal grid lines at T-score intervals
    for t_score in range(-3, 4):
        almi_value = almi_mu_peak + t_score * almi_sigma_peak
        if min_almi <= almi_value <= max_almi:
            ax.axhline(
                y=almi_value, color="gray", linestyle="--", alpha=0.5, linewidth=1
            )

    # Filter data for actual scans (not goal rows)
    scan_data = df_results[~df_results["date_str"].str.contains("Goal", na=False)]

    # Plot actual data points with open squares
    if len(scan_data) > 0:
        ax.scatter(
            scan_data["age_at_scan"],
            scan_data[y_column],
            marker="s",  # Square markers
            facecolors="none",  # Open squares
            edgecolors="red",
            s=120,
            linewidth=2,
            zorder=5,
            label="Your scans",
        )

        # Connect points with lines if multiple scans
        if len(scan_data) > 1:
            ax.plot(
                scan_data["age_at_scan"],
                scan_data[y_column],
                color="red",
                linewidth=2,
                alpha=0.7,
                zorder=4,
            )

    # Set up main y-axis (ALMI values)
    ax.set_xlabel("Age (years)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(plot_title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(min_almi, max_almi)
    ax.set_xlim(18, 80)

    # Create right-hand T-score axis
    ax2 = ax.twinx()

    # Calculate T-score values for the y-axis
    y_ticks = ax.get_yticks()
    t_score_ticks = [(y - almi_mu_peak) / almi_sigma_peak for y in y_ticks]
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels([f"{t:.1f}" for t in t_score_ticks])
    ax2.set_ylabel("T-score", fontsize=12)
    ax2.set_ylim(ax.get_ylim())

    # Add thermometer inset showing latest T-score
    if len(scan_data) > 0:
        latest_scan = scan_data.iloc[-1]
        latest_t_score = latest_scan[f"{metric_to_plot.lower()}_t_score"]

        # Create inset axes for thermometer
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        inset_ax = inset_axes(
            ax,
            width="8%",
            height="25%",
            loc="upper right",
            bbox_to_anchor=(0.95, 0.95, 1, 1),
            bbox_transform=ax.transAxes,
        )

        # Draw thermometer
        t_range = np.linspace(-3, 2, 100)
        colors = []
        for t in t_range:
            if t < -2:
                colors.append("#FF6B6B")  # Red
            elif t < -1:
                colors.append("#FFA500")  # Orange
            elif t < 0:
                colors.append("#FFD700")  # Yellow
            else:
                colors.append("#90EE90")  # Green

        # Plot thermometer background
        for i, (t, color) in enumerate(zip(t_range[:-1], colors[:-1])):
            inset_ax.barh(
                t,  # y position
                1,  # width
                height=t_range[1] - t_range[0],
                left=0,  # x position
                color=color,
                alpha=0.7,
                edgecolor="none",
            )

        # Add current T-score marker
        inset_ax.barh(
            latest_t_score - 0.05,  # y position
            1.2,  # width
            height=0.1,
            left=0,  # x position
            color="black",
            alpha=0.8,
        )
        inset_ax.text(
            1.3,
            latest_t_score,
            f"{latest_t_score:.1f}",
            va="center",
            fontweight="bold",
            fontsize=10,
        )

        inset_ax.set_xlim(0, 2)
        inset_ax.set_ylim(-3, 2)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([-2, -1, 0, 1])
        inset_ax.set_title("Current\nT-score", fontsize=8)

    # Add legend
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()

    if return_figure:
        return fig
    else:
        # Save the plot
        filename = f"{metric_to_plot.lower()}_tscore_plot.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"T-score plot saved as {filename}")
        plt.show()


# ---------------------------------------------------------------------------
# COMPARISON TABLE FUNCTIONS
# ---------------------------------------------------------------------------


def create_scan_comparison_table(df_results, return_html=False):
    """
    Creates a comparison table showing scan data with Changes row at bottom.

    Args:
        df_results (pd.DataFrame): Complete results DataFrame with scan history
        return_html (bool): If True, returns HTML table with color coding for web interface

    Returns:
        str: Formatted table as string (CLI) or HTML (web)
    """
    # Filter out goal rows to get only actual scans
    scan_data = df_results[~df_results["date_str"].str.contains("Goal", na=False)]

    if len(scan_data) < 1:
        return "No scan data available for comparison"

    # Prepare data for the table with new structure
    table_data = []
    headers = ["Date", "Age", "Weight", "Lean", "Fat", "BF%", "ALMI", "FFMI"]

    # Add all scan rows
    for _, scan in scan_data.iterrows():
        row = [
            scan["date_str"],
            f"{scan['age_at_scan']:.1f}",
            f"{scan['total_weight_lbs']:.1f}",
            f"{scan['total_lean_mass_lbs']:.1f}",
            f"{scan['fat_mass_lbs']:.1f}",
            f"{scan['body_fat_percentage']:.1f}%",
            f"{scan['almi_kg_m2']:.2f}",
            f"{scan['ffmi_kg_m2']:.2f}",
        ]
        table_data.append(row)

    # Add Changes row if there are multiple scans
    if len(scan_data) > 1:
        last_scan = scan_data.iloc[-1]
        first_scan = scan_data.iloc[0]

        # Calculate age change
        age_change = last_scan["age_at_scan"] - first_scan["age_at_scan"]
        changes_row = [
            "Changes",
            f"+{age_change:.1f} years",
        ]  # Start with Changes label and age change

        # Add changes for each metric
        for col_name in [
            "weight_change_first",
            "lean_change_first",
            "fat_change_first",
            "bf_change_first",
        ]:
            if col_name in last_scan and pd.notna(last_scan[col_name]):
                if col_name == "bf_change_first":
                    changes_row.append(f"{last_scan[col_name]:+.1f}%")
                else:
                    changes_row.append(f"{last_scan[col_name]:+.1f}")
            else:
                changes_row.append("N/A")

        # Add ALMI and FFMI changes (from first scan to last scan)
        almi_change = last_scan["almi_kg_m2"] - first_scan["almi_kg_m2"]
        ffmi_change = last_scan["ffmi_kg_m2"] - first_scan["ffmi_kg_m2"]
        changes_row.append(f"{almi_change:+.2f}")
        changes_row.append(f"{ffmi_change:+.2f}")

        table_data.append(changes_row)

    if return_html:
        # Create HTML table with color coding
        html = '<table class="scan-comparison-table" style="border-collapse: collapse; margin: 20px 0;">\n'

        # Headers
        html += "  <thead>\n    <tr>\n"
        for header in headers:
            html += f'      <th style="border: 1px solid #ddd; padding: 8px; background-color: #f5f5f5; text-align: center;">{header}</th>\n'
        html += "    </tr>\n  </thead>\n"

        # Data rows
        html += "  <tbody>\n"
        for i, row in enumerate(table_data):
            is_changes_row = row[0] == "Changes"

            # Add thick border before Changes row
            if is_changes_row:
                html += '    <tr style="border-top: 3px solid #333;">\n'
            else:
                html += "    <tr>\n"

            for j, cell in enumerate(row):
                # Determine cell styling based on content and position
                style = "border: 1px solid #ddd; padding: 8px; text-align: center;"

                # Special styling for Changes row
                if is_changes_row:
                    if j == 0:  # "Changes" label
                        style += " background-color: #f5f5f5; font-weight: bold;"
                    elif j == 1:  # Age change - neutral white background
                        style += " background-color: white;"
                    elif cell != "" and cell != "N/A":
                        # Color code changes based on column
                        try:
                            if cell.endswith("%"):
                                change_val = float(
                                    cell.replace("%", "").replace("+", "")
                                )
                            else:
                                change_val = float(cell.replace("+", ""))

                            # Calculate color intensity based on magnitude (0.2 to 1.0 opacity)
                            abs_change = abs(change_val)
                            if j in [2, 3, 4, 5]:  # Physical metrics
                                max_expected = (
                                    10.0 if j in [2, 3, 4] else 5.0
                                )  # Weight/lean/fat vs BF%
                            elif j in [6, 7]:  # ALMI/FFMI metric changes (kg/m²)
                                max_expected = (
                                    2.0  # Reasonable max for ALMI/FFMI changes in kg/m²
                                )
                            else:  # Other metrics
                                max_expected = 2.0

                            intensity = min(
                                0.2 + (abs_change / max_expected) * 0.8, 1.0
                            )

                            # Color coding logic based on column position
                            if j == 2:  # Weight change - neutral white
                                style += " background-color: white;"
                            elif j == 3:  # Lean mass change - positive is good
                                if change_val > 0:
                                    style += f" background-color: rgba(212, 237, 218, {intensity}); color: #155724;"  # Green
                                elif change_val < 0:
                                    style += f" background-color: rgba(248, 215, 218, {intensity}); color: #721c24;"  # Red
                            elif (
                                j == 4
                            ):  # Fat mass change - negative is good (fat loss)
                                if change_val < 0:
                                    style += f" background-color: rgba(212, 237, 218, {intensity}); color: #155724;"  # Green
                                elif change_val > 0:
                                    style += f" background-color: rgba(248, 215, 218, {intensity}); color: #721c24;"  # Red
                            elif (
                                j == 5
                            ):  # Body fat % change - negative is good (BF% reduction)
                                if change_val < 0:
                                    style += f" background-color: rgba(212, 237, 218, {intensity}); color: #155724;"  # Green
                                elif change_val > 0:
                                    style += f" background-color: rgba(248, 215, 218, {intensity}); color: #721c24;"  # Red
                            elif j in [
                                6,
                                7,
                            ]:  # ALMI/FFMI metric changes - positive is good
                                if change_val > 0:
                                    style += f" background-color: rgba(212, 237, 218, {intensity}); color: #155724;"  # Green
                                elif change_val < 0:
                                    style += f" background-color: rgba(248, 215, 218, {intensity}); color: #721c24;"  # Red
                        except ValueError:
                            pass  # Keep default styling if can't parse
                else:
                    # Color coding for regular data rows (all rows after first scan)
                    if i > 0 and j > 1:  # Skip first row and Date/Age columns
                        # Calculate changes from previous row for color coding
                        try:
                            current_val = float(
                                cell.replace("%", "").replace("+", "").replace("-", "")
                            )
                            if i == 1:  # Second row - compare to first row
                                prev_val = float(
                                    table_data[0][j]
                                    .replace("%", "")
                                    .replace("+", "")
                                    .replace("-", "")
                                )
                            else:  # Later rows - compare to previous row
                                prev_val = float(
                                    table_data[i - 1][j]
                                    .replace("%", "")
                                    .replace("+", "")
                                    .replace("-", "")
                                )

                            change_val = current_val - prev_val
                            abs_change = abs(change_val)

                            # Calculate intensity
                            if j in [2, 3, 4, 5]:  # Physical metrics
                                max_expected = (
                                    5.0 if j in [2, 3, 4] else 2.5
                                )  # Weight/lean/fat vs BF%
                            else:  # Z-scores/indices
                                max_expected = 1.0

                            intensity = min(
                                0.1 + (abs_change / max_expected) * 0.4, 0.5
                            )  # Lighter for data rows

                            # Apply color coding
                            if j == 2:  # Weight - neutral
                                style += " background-color: white;"
                            elif j == 3:  # Lean mass - positive is good
                                if change_val > 0.1:
                                    style += f" background-color: rgba(212, 237, 218, {intensity});"
                                elif change_val < -0.1:
                                    style += f" background-color: rgba(248, 215, 218, {intensity});"
                            elif j == 4:  # Fat mass - negative is good
                                if change_val < -0.1:
                                    style += f" background-color: rgba(212, 237, 218, {intensity});"
                                elif change_val > 0.1:
                                    style += f" background-color: rgba(248, 215, 218, {intensity});"
                            elif j == 5:  # Body fat % - negative is good
                                if change_val < -0.1:
                                    style += f" background-color: rgba(212, 237, 218, {intensity});"
                                elif change_val > 0.1:
                                    style += f" background-color: rgba(248, 215, 218, {intensity});"
                            elif j in [6, 7]:  # ALMI/FFMI - positive is good
                                if change_val > 0.05:
                                    style += f" background-color: rgba(212, 237, 218, {intensity});"
                                elif change_val < -0.05:
                                    style += f" background-color: rgba(248, 215, 218, {intensity});"
                        except (ValueError, IndexError):
                            pass  # Keep default styling if can't parse

                html += f'      <td style="{style}">{cell}</td>\n'
            html += "    </tr>\n"

        html += "  </tbody>\n</table>"
        return html
    else:
        # Return plain text table for CLI
        try:
            from tabulate import tabulate

            table_output = tabulate(table_data, headers=headers, tablefmt="pipe")

            # Add thick border before Changes row
            lines = table_output.split("\n")
            changes_line_idx = None
            for i, line in enumerate(lines):
                if "Changes" in line and "|" in line:
                    changes_line_idx = i
                    break

            if changes_line_idx:
                # Insert thick border line before Changes row
                separator_line = lines[1]  # Use header separator format
                thick_border = separator_line.replace("-", "=")  # Make it thicker
                lines.insert(changes_line_idx, thick_border)

            return "\n".join(lines)
        except ImportError:
            # Fallback to simple formatting
            header_str = " | ".join(f"{h:>12}" for h in headers)
            separator = "-" * len(header_str)
            thick_separator = "=" * len(header_str)
            rows = []
            for row in table_data:
                if row[0] == "Changes":
                    rows.append(thick_separator)  # Add thick separator before Changes
                rows.append(" | ".join(f"{cell:>12}" for cell in row))
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
    from datetime import datetime

    # Generate realistic age (25-45 years old)
    current_year = datetime.now().year
    age = random.randint(25, 45)
    birth_year = current_year - age
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)  # Safe day range
    birth_date = f"{birth_month:02d}/{birth_day:02d}/{birth_year}"

    # Generate realistic height
    gender = random.choice(["male", "female"])
    if gender == "male":
        height_in = random.uniform(66, 74)  # 5'6" to 6'2"
    else:
        height_in = random.uniform(60, 68)  # 5'0" to 5'8"

    training_level = random.choice(["novice", "intermediate", "advanced"])

    return {
        "birth_date": birth_date,
        "height_in": round(height_in, 1),
        "gender": gender,
        "training_level": training_level,
    }


def generate_fake_scans(user_info, num_scans=4):
    """
    Generates realistic fake DEXA scan data with varied progression patterns.
    Creates different profile types including plateaus, setbacks, and varied endpoints.

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
    gender = user_info["gender"]
    training_level = user_info.get("training_level", "intermediate")
    height_in = user_info["height_in"]
    height_m = height_in * 0.0254
    height_m2 = height_m**2

    # Load LMS data for ALMI calculations
    gender_code = 0 if gender == "male" else 1
    almi_L, almi_M, almi_S = load_lms_data("appendicular_LMI", gender_code)

    # Generate a realistic starting age (25-45)
    from dateutil import parser

    if isinstance(user_info.get("birth_date"), str):
        birth_date = parser.parse(user_info["birth_date"])
        current_age = (datetime.now() - birth_date).days / 365.25
    else:
        current_age = random.uniform(25, 45)

    # Choose a progression profile type to create variety
    profile_types = [
        "steady_improver",  # 20% - consistent moderate gains
        "plateau_profile",  # 25% - good start then plateaus
        "ups_and_downs",  # 25% - variable progress with setbacks
        "slow_starter",  # 15% - poor start then improvement
        "inconsistent",  # 15% - yo-yo pattern
    ]
    weights = [0.20, 0.25, 0.25, 0.15, 0.15]
    profile_type = random.choices(profile_types, weights=weights)[0]

    # Target ALMI percentile ranges based on profile type (more realistic starting points)
    if profile_type == "steady_improver":
        target_percentile = random.uniform(
            0.25, 0.45
        )  # Start low-average, end high-average
        random.uniform(0.60, 0.80)
    elif profile_type == "plateau_profile":
        target_percentile = random.uniform(0.30, 0.50)  # Start average, plateau
        random.uniform(0.40, 0.70)
    elif profile_type == "ups_and_downs":
        target_percentile = random.uniform(0.20, 0.45)  # Variable journey
        random.uniform(0.30, 0.65)
    elif profile_type == "slow_starter":
        target_percentile = random.uniform(0.15, 0.35)  # Start low
        random.uniform(0.35, 0.60)
    else:  # inconsistent
        target_percentile = random.uniform(0.20, 0.40)  # Yo-yo pattern
        random.uniform(0.25, 0.55)

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
    alm_ratio = (
        random.uniform(0.42, 0.46) if gender == "male" else random.uniform(0.38, 0.42)
    )
    target_lean_mass = target_alm_lbs / alm_ratio

    # Set initial body fat ranges based on profile type
    if gender == "male":
        if profile_type in ["steady_improver", "slow_starter"]:
            initial_bf_pct = random.uniform(
                18, 28
            )  # Higher starting BF for improvement potential
        else:
            initial_bf_pct = random.uniform(15, 25)
    else:
        if profile_type in ["steady_improver", "slow_starter"]:
            initial_bf_pct = random.uniform(
                25, 35
            )  # Higher starting BF for improvement potential
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
    initial_alm * random.uniform(0.33, 0.37)  # ~35% of ALM in arms
    initial_alm * random.uniform(0.63, 0.67)  # ~65% of ALM in legs

    # More conservative progression rates (reduced by 40-60% from original)
    if training_level == "novice":
        base_lean_gain = random.uniform(2.0, 6.0)  # lbs per scan period (12-18 months)
        base_fat_loss = random.uniform(1.5, 5.0)
    elif training_level == "intermediate":
        base_lean_gain = random.uniform(1.0, 4.0)
        base_fat_loss = random.uniform(1.0, 3.5)
    else:  # advanced
        base_lean_gain = random.uniform(0.5, 2.0)
        base_fat_loss = random.uniform(0.5, 2.0)

    # Generate scans over 36-60 months (3-5 years) for more realistic long-term tracking
    total_period_days = random.randint(1095, 1825)  # 3-5 years
    start_date = datetime.now() - timedelta(days=total_period_days)

    for i in range(num_scans):
        # Date progression (12-18 months between scans)
        if i == 0:
            scan_date = start_date
        else:
            days_gap = random.randint(365, 550)  # 12-18 months between scans
            scan_date = scans[-1]["scan_date"] + timedelta(days=days_gap)

        # Progressive body composition changes based on profile type
        if i == 0:
            weight = initial_weight
            lean_mass = initial_lean_mass
            fat_mass = initial_fat_mass
            alm = initial_alm
        else:
            # Calculate progress multiplier based on profile type and scan number
            progress_multiplier = _get_progress_multiplier(profile_type, i, num_scans)

            # Apply profile-specific progression logic
            lean_change, fat_change = _calculate_profile_changes(
                profile_type,
                i,
                num_scans,
                base_lean_gain,
                base_fat_loss,
                progress_multiplier,
            )

            # Apply changes with realistic bounds
            lean_mass = max(
                scans[-1]["total_lean_mass_lbs"] + lean_change, initial_lean_mass * 0.85
            )
            fat_mass = max(
                scans[-1]["fat_mass_lbs"] + fat_change, initial_fat_mass * 0.3
            )
            fat_mass = min(
                fat_mass, initial_fat_mass * 1.3
            )  # Prevent excessive fat gain
            weight = lean_mass + fat_mass

            # ALM should progress proportionally with lean mass, but maintain realistic ratio
            prev_alm = scans[-1]["arms_lean_lbs"] + scans[-1]["legs_lean_lbs"]
            alm_change = lean_change * alm_ratio  # ALM changes proportionally
            alm = max(prev_alm + alm_change, initial_alm * 0.90)

        # Calculate derived values
        body_fat_percentage = (fat_mass / weight) * 100
        arms_lean = alm * random.uniform(
            0.33, 0.37
        )  # Maintain consistent arm/leg distribution
        legs_lean = alm * random.uniform(0.63, 0.67)

        scan = {
            "date": scan_date.strftime("%m/%d/%Y"),
            "scan_date": scan_date,  # Keep for sorting, will be removed
            "total_weight_lbs": round(weight, 1),
            "total_lean_mass_lbs": round(lean_mass, 1),
            "fat_mass_lbs": round(fat_mass, 1),
            "body_fat_percentage": round(body_fat_percentage, 1),
            "arms_lean_lbs": round(arms_lean, 1),
            "legs_lean_lbs": round(legs_lean, 1),
        }
        scans.append(scan)

    # Remove the helper scan_date field
    for scan in scans:
        del scan["scan_date"]

    return scans


def _get_progress_multiplier(profile_type, scan_index, total_scans):
    """
    Calculate progress multiplier based on profile type and position in timeline.

    Returns:
        float: Multiplier for base progression rates
    """
    import random

    progress_ratio = scan_index / (total_scans - 1) if total_scans > 1 else 0

    if profile_type == "steady_improver":
        # Consistent moderate progress with slight decline over time
        return random.uniform(0.8, 1.2) * (1 - progress_ratio * 0.3)

    elif profile_type == "plateau_profile":
        # Good progress early, then plateau
        if progress_ratio < 0.4:
            return random.uniform(1.0, 1.5)  # Good early progress
        else:
            return random.uniform(-0.2, 0.3)  # Plateau/minimal progress

    elif profile_type == "ups_and_downs":
        # Variable progress with setbacks
        return random.uniform(-0.8, 1.5)  # Wide range including setbacks

    elif profile_type == "slow_starter":
        # Poor early progress, then improvement
        if progress_ratio < 0.5:
            return random.uniform(-0.3, 0.4)  # Poor early progress
        else:
            return random.uniform(0.8, 1.4)  # Better later progress

    else:  # inconsistent
        # Yo-yo pattern
        if scan_index % 2 == 0:
            return random.uniform(0.6, 1.3)  # Good periods
        else:
            return random.uniform(-0.6, 0.4)  # Regression periods


def _calculate_profile_changes(
    profile_type,
    scan_index,
    total_scans,
    base_lean_gain,
    base_fat_loss,
    progress_multiplier,
):
    """
    Calculate lean mass and fat mass changes based on profile type and multiplier.

    Returns:
        tuple: (lean_change, fat_change) in lbs
    """
    import random

    # Base changes with profile multiplier
    lean_change = base_lean_gain * progress_multiplier * random.uniform(0.7, 1.3)
    fat_change = -base_fat_loss * abs(progress_multiplier) * random.uniform(0.5, 1.2)

    # Add profile-specific adjustments
    if profile_type == "ups_and_downs":
        # 30% chance of a significant setback
        if random.random() < 0.3:
            lean_change *= -0.5  # Lose some lean mass
            fat_change *= -0.8  # Gain some fat back

    elif profile_type == "inconsistent":
        # More extreme swings
        if progress_multiplier < 0:
            lean_change = abs(lean_change) * -0.7  # Lose lean mass
            fat_change = abs(fat_change) * 0.6  # Gain fat

    elif profile_type == "plateau_profile":
        # Later scans show minimal change
        progress_ratio = scan_index / (total_scans - 1) if total_scans > 1 else 0
        if progress_ratio > 0.4:
            lean_change *= 0.2  # Minimal lean changes
            fat_change *= 0.3  # Minimal fat changes

    return lean_change, fat_change


def get_metric_explanations():
    """
    Returns explanatory text for metrics and tooltips.

    Returns:
        dict: Dictionary with explanations for different metrics
    """
    return {
        "header_info": {
            "title": "RecompTracker",
            "subtitle": "Analyze and your body composition using scientifically validated percentile curves. Set your body recomposition goals and track your progress.",
            "almi_explanation": """
            **ALMI (Appendicular Lean Mass Index)** measures the lean muscle mass in your arms and legs
            relative to your height. It's calculated as (Arms Lean Mass + Legs Lean Mass) ÷ Height².
            This metric is important for assessing functional muscle mass and overall strength potential.
            """,
            "ffmi_explanation": """
            **FFMI (Fat-Free Mass Index)** measures your total lean body mass relative to your height.
            It's calculated as Total Lean Mass ÷ Height². This gives a normalized measure of your overall
            muscle mass that accounts for differences in height.
            """,
            "percentiles_explanation": """
            **Percentiles** show how you compare to a reference population. For example, the 75th percentile
            means you have more muscle mass than 75% of people your age and gender. The reference data comes
            from the [LEAD cohort study](https://www.nature.com/articles/s41430-020-0596-5) of healthy adults.
            """,
        },
        "tooltips": {
            "z_score": "Z-score: How many standard deviations you are from the population median. Positive values are above average.",
            "percentile": "Percentile: The percentage of the population with lower values than yours.",
            "almi": "ALMI (Appendicular Lean Mass Index): Measures lean muscle mass in your arms and legs relative to height (kg/m²). Higher values indicate more functional muscle mass.",
            "ffmi": "FFMI (Fat-Free Mass Index): Measures your total lean body mass relative to height (kg/m²). Higher values indicate more overall muscle mass.",
            "training_level": "Training level affects goal suggestions and muscle gain rate estimates.",
            "goal_age": 'Target age to reach your goal. Use "?" for automatic calculation based on realistic progression rates.',
            "target_percentile": "The percentile you want to reach (e.g., 0.75 = 75th percentile).",
            "body_fat_percentage": "Percentage of total body weight that is fat mass, measured by body composition scan.",
            "lean_mass": "Total muscle, bone, and organ mass excluding fat.",
            "arms_lean": "Lean mass in both arms combined.",
            "legs_lean": "Lean mass in both legs combined.",
        },
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

    if field_name == "birth_date":
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

    elif field_name == "height_in":
        try:
            height = float(value)
            if height < 12:
                return False, "Height must be at least 12 inches"
            if height > 120:
                return False, "Height must be less than 120 inches"
            return True, ""
        except (ValueError, TypeError):
            return False, "Please enter a valid number"

    elif field_name == "scan_date":
        try:
            scan_date = datetime.strptime(value, "%m/%d/%Y")
            if user_data and "birth_date" in user_data:
                birth_date = datetime.strptime(user_data["birth_date"], "%m/%d/%Y")
                if scan_date <= birth_date:
                    return False, "Scan date must be after birth date"
            return True, ""
        except ValueError:
            return False, "Please use MM/DD/YYYY format"

    elif field_name in [
        "total_weight_lbs",
        "total_lean_mass_lbs",
        "fat_mass_lbs",
        "arms_lean_lbs",
        "legs_lean_lbs",
    ]:
        try:
            weight = float(value)
            if weight <= 0:
                return False, "Value must be greater than 0"
            if weight > 1000:
                return False, "Value seems unreasonably high"
            return True, ""
        except (ValueError, TypeError):
            return False, "Please enter a valid number"

    elif field_name == "body_fat_percentage":
        try:
            bf_pct = float(value)
            if bf_pct <= 0:
                return False, "Body fat percentage must be greater than 0"
            if bf_pct >= 100:
                return False, "Body fat percentage must be less than 100"
            return True, ""
        except (ValueError, TypeError):
            return False, "Please enter a valid number"

    elif field_name == "target_percentile":
        try:
            percentile = float(value)
            if percentile <= 0:
                return False, "Percentile must be greater than 0"
            if percentile >= 1:
                return (
                    False,
                    "Percentile must be less than 1 (e.g., 0.75 for 75th percentile)",
                )
            return True, ""
        except (ValueError, TypeError):
            return False, "Please enter a decimal between 0 and 1"

    elif field_name == "target_age":
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
        "almi_L": None,
        "almi_M": None,
        "almi_S": None,
        "lmi_L": None,
        "lmi_M": None,
        "lmi_S": None,
    }
    lms_functions["almi_L"], lms_functions["almi_M"], lms_functions["almi_S"] = (
        load_lms_data(metric="appendicular_LMI", gender_code=user_info["gender_code"])
    )
    lms_functions["lmi_L"], lms_functions["lmi_M"], lms_functions["lmi_S"] = (
        load_lms_data(metric="LMI", gender_code=user_info["gender_code"])
    )

    if not all(lms_functions.values()):
        raise ValueError("Failed to load all necessary LMS data")

    # Process data and generate results DataFrame
    df_results, goal_calculations = process_scans_and_goal(
        user_info, scan_history, almi_goal, ffmi_goal, lms_functions
    )

    # Get T-score reference values for plotting
    almi_mu_peak, almi_sigma_peak = calculate_tscore_reference_values(
        user_info["gender_code"]
    )

    # Generate plots
    almi_fig = create_metric_plot(
        df_results, "ALMI", lms_functions, goal_calculations, return_figure=True
    )
    ffmi_fig = create_metric_plot(
        df_results, "FFMI", lms_functions, goal_calculations, return_figure=True
    )
    # Generate T-score plots
    almi_tscore_fig = create_tscore_plot(
        df_results, "ALMI", almi_mu_peak, almi_sigma_peak, return_figure=True
    )
    ffmi_tscore_fig = create_tscore_plot(
        df_results, "FFMI", almi_mu_peak, almi_sigma_peak, return_figure=True
    )
    bf_fig = create_body_fat_plot(df_results, user_info, return_figure=True)
    figures = {
        "ALMI": almi_fig,
        "FFMI": ffmi_fig,
        "ALMI_TSCORE": almi_tscore_fig,
        "FFMI_TSCORE": ffmi_tscore_fig,
        "BODY_FAT": bf_fig,
    }

    # Generate comparison table for web interface
    comparison_table_html = create_scan_comparison_table(df_results, return_html=True)

    return df_results, goal_calculations, figures, comparison_table_html


def run_analysis(
    config_path="example_config.json",
    suggest_goals=False,
    target_percentile=0.90,
    training_level_override=None,
    return_results=False,
):
    """
    Main analysis function that orchestrates the entire RecompTracker analysis workflow.

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
        print("RecompTracker Body Composition Analysis with Intelligent TLM Estimation")
        print("=" * 65)
        print(
            "Note: Run 'python test_zscore_calculations.py' for comprehensive testing\n"
        )

    try:
        # Step 1: Load Configuration
        config = load_config_json(config_path, quiet=return_results)
        user_info, scan_history, almi_goal, ffmi_goal = extract_data_from_config(config)

        # Apply training level override if provided
        if training_level_override:
            user_info["training_level"] = training_level_override.lower()

        # Display loaded configuration
        print("User Info:")
        print(f"  - Birth Date: {config['user_info']['birth_date']}")
        print(f"  - Height: {config['user_info']['height_in']} inches")
        print(f"  - Gender: {config['user_info']['gender']}")
        if user_info.get("training_level"):
            print(f"  - Training Level: {user_info['training_level']}")

        print("\nGoals:")
        if almi_goal:
            target_age_str = (
                str(almi_goal["target_age"])
                if almi_goal.get("target_age") not in [None, "?"]
                else "auto-calculated"
            )
            print(
                f"  - ALMI Goal: {almi_goal['target_percentile'] * 100:.0f}th percentile at age {target_age_str}"
            )
            if "description" in almi_goal:
                print(f"    Description: {almi_goal['description']}")
        if ffmi_goal:
            target_age_str = (
                str(ffmi_goal["target_age"])
                if ffmi_goal.get("target_age") not in [None, "?"]
                else "auto-calculated"
            )
            print(
                f"  - FFMI Goal: {ffmi_goal['target_percentile'] * 100:.0f}th percentile at age {target_age_str}"
            )
            if "description" in ffmi_goal:
                print(f"    Description: {ffmi_goal['description']}")
        if not almi_goal and not ffmi_goal:
            print("  - No goals specified (scan history analysis only)")
        print()

        # Step 2: Load LMS Data
        lms_functions = {
            "almi_L": None,
            "almi_M": None,
            "almi_S": None,
            "lmi_L": None,
            "lmi_M": None,
            "lmi_S": None,
        }
        lms_functions["almi_L"], lms_functions["almi_M"], lms_functions["almi_S"] = (
            load_lms_data(
                metric="appendicular_LMI", gender_code=user_info["gender_code"]
            )
        )
        lms_functions["lmi_L"], lms_functions["lmi_M"], lms_functions["lmi_S"] = (
            load_lms_data(metric="LMI", gender_code=user_info["gender_code"])
        )

        if not all(lms_functions.values()):
            print("Failed to load all necessary LMS data. Aborting analysis.")
            return 1

        # Step 3: Process data and generate results DataFrame
        df_results, goal_calculations = process_scans_and_goal(
            user_info, scan_history, almi_goal, ffmi_goal, lms_functions
        )

        # Step 4: Generate Plots
        if return_results:
            # Return figure objects for web interface
            almi_fig = create_metric_plot(
                df_results, "ALMI", lms_functions, goal_calculations, return_figure=True
            )
            ffmi_fig = create_metric_plot(
                df_results, "FFMI", lms_functions, goal_calculations, return_figure=True
            )
            bf_fig = create_body_fat_plot(df_results, user_info, return_figure=True)
            figures = {"ALMI": almi_fig, "FFMI": ffmi_fig, "BODY_FAT": bf_fig}
            comparison_table_html = create_scan_comparison_table(
                df_results, return_html=True
            )
            return df_results, goal_calculations, figures, comparison_table_html
        else:
            # Save plots to disk for CLI interface
            plot_metric_with_table(df_results, "ALMI", lms_functions, goal_calculations)
            plot_metric_with_table(df_results, "FFMI", lms_functions, goal_calculations)
            create_body_fat_plot(df_results, user_info, return_figure=False)

            # Step 5: Print Comparison Table
            print("\n--- Changes So Far ---")
            comparison_table = create_scan_comparison_table(
                df_results, return_html=False
            )
            print(comparison_table)

            # Step 6: Print Final Table
            print("\n--- Final Comprehensive Data Table ---")

            # Split columns into main data and changes
            main_columns = [
                "date_str",
                "age_at_scan",
                "total_weight_lbs",
                "total_lean_mass_lbs",
                "fat_mass_lbs",
                "body_fat_percentage",
                "almi_kg_m2",
                "ffmi_kg_m2",
            ]
            main_names = ["Date", "Age", "Weight", "Lean", "Fat", "BF%", "ALMI", "FFMI"]

            # Change columns for the bottom row

            # Create main data table
            df_display = df_results[main_columns].copy()
            df_display.columns = main_names

            # Format main data columns
            for col in df_display.columns:
                if col in ["Date"]:
                    continue  # Skip string columns
                if df_display[col].dtype == "float64":
                    df_display[col] = df_display[col].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                    )

            # Create changes row - use the last scan's changes
            last_scan_idx = len(df_results) - 1
            changes_data = {}
            changes_data["Date"] = "**Changes**"  # Bold text for date column
            changes_data["Age"] = ""  # Empty for non-applicable columns

            # Map change columns to display positions
            change_mapping = {
                "Weight": "weight_change_last",
                "Lean": "lean_change_last",
                "Fat": "fat_change_last",
                "BF%": "bf_change_last",
                "ALMI": "almi_z_change_last",
                "FFMI": "ffmi_z_change_last",
            }

            for display_col, change_col in change_mapping.items():
                if change_col in df_results.columns:
                    change_val = df_results.loc[last_scan_idx, change_col]
                    if pd.notna(change_val):
                        if "z_change" in change_col:
                            changes_data[display_col] = f"{change_val:+.2f}"
                        else:
                            changes_data[display_col] = f"{change_val:+.1f}"
                    else:
                        changes_data[display_col] = "N/A"
                else:
                    changes_data[display_col] = ""

            # Convert changes to DataFrame and append
            changes_row = pd.DataFrame([changes_data])
            df_display_with_changes = pd.concat(
                [df_display, changes_row], ignore_index=True
            )

            # Use tabulate for better formatting
            try:
                from tabulate import tabulate

                table_output = tabulate(
                    df_display_with_changes,
                    headers="keys",
                    tablefmt="pipe",
                    showindex=False,
                )

                # Split table into lines and add thick border before Changes row
                lines = table_output.split("\n")
                # Find the line with "Changes" (should be second to last line before final border)
                changes_line_idx = None
                for i, line in enumerate(lines):
                    if "**Changes**" in line:
                        changes_line_idx = i
                        break

                if changes_line_idx:
                    # Insert thick border line before Changes row
                    separator_line = lines[1]  # Use header separator format
                    thick_border = separator_line.replace("-", "=")  # Make it thicker
                    lines.insert(changes_line_idx, thick_border)

                print("\n".join(lines))

            except ImportError:
                # Fallback to pandas display if tabulate not available
                print(df_display_with_changes.to_string(index=False))

            return 0

    except (
        FileNotFoundError,
        json.JSONDecodeError,
        ValidationError,
        KeyError,
        ValueError,
    ) as e:
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
