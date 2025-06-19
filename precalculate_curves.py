"""
================================================================================
README: Percentile Curve Pre-calculation Script
================================================================================

PURPOSE:
This script reads the raw LMS reference data from CSV files and pre-calculates
the (age, value) points for a standard set of percentile curves. It saves this
data into JSON files. The primary goal is to optimize a front-end application
by offloading the computationally intensive task of generating these curves.
The resulting JSON files are small and can be fetched quickly by a web browser,
which will only need to render the data, not calculate it.

HOW TO USE:
1.  Place this script in a directory containing the required LMS data files:
    - `adults_LMS_appendicular_LMI_gender0.csv`
    - `adults_LMS_appendicular_LMI_gender1.csv`
    - `adults_LMS_LMI_gender0.csv`
    - `adults_LMS_LMI_gender1.csv`
2.  Run the script from your terminal: `python <script_name>.py`
3.  The script will generate four JSON files in the same directory:
    - `almi_male_curves.json`
    - `almi_female_curves.json`
    - `ffmi_male_curves.json`
    - `ffmi_female_curves.json`
4.  These JSON files can then be uploaded to a web host or CDN to be fetched
    by the front-end JavaScript application.

"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import scipy.stats as stats
import json
import os

def get_value_from_zscore(z, l_val, m_val, s_val, eps=1e-5):
    """Calculates a metric value from a Z-score (inverse LMS transformation)."""
    if pd.isna(z) or pd.isna(l_val) or pd.isna(m_val) or pd.isna(s_val):
        return None
    if abs(l_val) < eps:
        return m_val * np.exp(s_val * z)
    else:
        term = l_val * s_val * z + 1
        if term < 0 and (1.0 / l_val) % 1 != 0:
            return None  # Avoids complex numbers
        return m_val * np.power(term, (1.0 / l_val))

def precalculate_curves():
    """Main function to generate and save all percentile curve JSON files."""
    metrics = {'ALMI': 'appendicular_LMI', 'FFMI': 'LMI'}
    genders = {'male': 0, 'female': 1}
    percentiles_to_plot = {
        '3%': stats.norm.ppf(0.03), '10%': stats.norm.ppf(0.10), 
        '25%': stats.norm.ppf(0.25), '50%': stats.norm.ppf(0.50), 
        '75%': stats.norm.ppf(0.75), '90%': stats.norm.ppf(0.90), 
        '97%': stats.norm.ppf(0.97)
    }
    ages_plot = np.linspace(18, 81, (81 - 18) * 4 + 1)  # 0.25 year increments

    print("Starting pre-calculation of percentile curves...")

    for metric_key, metric_file_part in metrics.items():
        for gender_key, gender_code in genders.items():
            lms_filename = f"data/adults_LMS_{metric_file_part}_gender{gender_code}.csv"
            
            if not os.path.exists(lms_filename):
                print(f"Skipping: Data file '{lms_filename}' not found.")
                continue

            print(f"  Processing: {lms_filename}")
            df_lms = pd.read_csv(lms_filename)
            df_lms.rename(columns={'lambda': 'L', 'mu': 'M', 'sigma': 'S'}, inplace=True)

            L_func = interp1d(df_lms['age'], df_lms['L'], kind='cubic', fill_value="extrapolate")
            M_func = interp1d(df_lms['age'], df_lms['M'], kind='cubic', fill_value="extrapolate")
            S_func = interp1d(df_lms['age'], df_lms['S'], kind='cubic', fill_value="extrapolate")

            output_data = {}
            for name, z_val in percentiles_to_plot.items():
                l_at_ages = L_func(ages_plot)
                m_at_ages = M_func(ages_plot)
                s_at_ages = S_func(ages_plot)
                
                metric_values = np.vectorize(get_value_from_zscore)(z_val, l_at_ages, m_at_ages, s_at_ages)
                
                # Structure for Chart.js: {x, y}
                # Filter out any null/NaN values that might result from extrapolation
                output_data[name] = [
                    {'x': round(age, 2), 'y': round(val, 3)} 
                    for age, val in zip(ages_plot, metric_values) if val is not None
                ]
            
            output_filename = f"{metric_key.lower()}_{gender_key}_curves.json"
            with open(output_filename, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"    -> Generated '{output_filename}'")

    print("\nPre-calculation complete.")

if __name__ == '__main__':
    precalculate_curves()

