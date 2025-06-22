#!/usr/bin/env python3
"""
RecompTracker - Streamlit Web Application

This web interface provides an intuitive way to analyze DEXA scan data using
the same core analysis engine as the command-line tool. Features include:
- Interactive data input forms with validation
- Real-time analysis updates
- Comprehensive visualizations
- Goal setting with intelligent suggestions
- Downloadable results

Run with: streamlit run webapp.py
"""

import base64
import json
import os
import urllib.parse
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import scipy.stats as stats
import streamlit as st
import streamlit.components.v1 as components

# Import core analysis functions
from core import (
    calculate_tscore_reference_values,
    create_plotly_dual_mode_plot,
    detect_training_level_from_scans,
    extract_data_from_config,
    generate_fake_profile,
    generate_fake_scans,
    get_metric_explanations,
    get_value_from_zscore,
    load_config_json,
    parse_gender,
    run_analysis_from_data,
    validate_user_input,
)

# Configure page
st.set_page_config(
    page_title="RecompTracker - ALMI & FFMI Progress Tracker & Goal Calculator",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_css():
    """Load custom CSS from external file."""
    try:
        css_path = os.path.join(os.path.dirname(__file__), "static", "webapp.css")
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()

        # Include LZ-string library
        st.markdown(
            """
        <script src="https://cdn.jsdelivr.net/npm/lz-string@1.5.0/libs/lz-string.min.js"></script>
        """,
            unsafe_allow_html=True,
        )

        # Inject custom CSS
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("CSS file not found. Using default styling.")
    except Exception as e:
        st.error(f"Error loading CSS: {e}")


# Load custom CSS
load_css()


def get_birth_date_range():
    """Calculate valid birth date range based on LMS dataset age limits."""
    from datetime import date, timedelta

    today = date.today()

    # LMS dataset covers ages 18-82, add buffer for 15-85 years
    max_age = 85  # oldest allowed user
    min_age = 15  # youngest allowed user

    min_birth_date = today - timedelta(days=max_age * 365.25)  # ~85 years ago
    max_birth_date = today - timedelta(days=min_age * 365.25)  # ~15 years ago

    return min_birth_date, max_birth_date


def get_compact_config():
    """Convert current session state to compact JSON format."""
    compact = {
        "u": {
            "bd": st.session_state.user_info.get("birth_date", ""),
            "h": st.session_state.user_info.get("height_in"),
            "hd": st.session_state.get("height_display", ""),  # height display format
            "g": st.session_state.user_info.get("gender", "male")[0],  # 'm' or 'f'
        }
    }

    # Add training level if set
    if st.session_state.user_info.get("training_level"):
        compact["u"]["tl"] = st.session_state.user_info["training_level"]

    # Convert scan history to array format
    compact["s"] = []
    for scan in st.session_state.scan_history:
        if scan.get("date"):  # Only include scans with dates
            compact["s"].append(
                [
                    scan.get("date", ""),
                    scan.get("total_weight_lbs", 0.0),
                    scan.get("total_lean_mass_lbs", 0.0),
                    scan.get("fat_mass_lbs", 0.0),
                    scan.get("body_fat_percentage", 0.0),
                    scan.get("arms_lean_lbs", 0.0),
                    scan.get("legs_lean_lbs", 0.0),
                ]
            )

    # Add goals if set
    if st.session_state.almi_goal.get("target_percentile"):
        compact["ag"] = {"tp": st.session_state.almi_goal["target_percentile"]}
        if (
            st.session_state.almi_goal.get("target_age")
            and st.session_state.almi_goal["target_age"] != "?"
        ):
            compact["ag"]["ta"] = st.session_state.almi_goal["target_age"]

    if st.session_state.ffmi_goal.get("target_percentile"):
        compact["fg"] = {"tp": st.session_state.ffmi_goal["target_percentile"]}
        if (
            st.session_state.ffmi_goal.get("target_age")
            and st.session_state.ffmi_goal["target_age"] != "?"
        ):
            compact["fg"]["ta"] = st.session_state.ffmi_goal["target_age"]

    return compact


def expand_compact_config(compact_config):
    """Convert compact JSON format back to full session state format."""
    # User info
    user_info = {
        "birth_date": compact_config.get("u", {}).get("bd", ""),
        "height_in": compact_config.get("u", {}).get("h", 66.0),
        "gender": "male"
        if compact_config.get("u", {}).get("g", "m") == "m"
        else "female",
        "training_level": compact_config.get("u", {}).get("tl", ""),
    }

    # Height display format
    height_display = compact_config.get("u", {}).get("hd", "")

    # Scan history
    scan_history = []
    for scan_array in compact_config.get("s", []):
        if len(scan_array) >= 7:
            scan_history.append(
                {
                    "date": scan_array[0],
                    "total_weight_lbs": scan_array[1],
                    "total_lean_mass_lbs": scan_array[2],
                    "fat_mass_lbs": scan_array[3],
                    "body_fat_percentage": scan_array[4],
                    "arms_lean_lbs": scan_array[5],
                    "legs_lean_lbs": scan_array[6],
                }
            )

    # Goals
    almi_goal = {"target_percentile": 0.75, "target_age": "?"}
    if "ag" in compact_config:
        almi_goal["target_percentile"] = compact_config["ag"].get("tp", 0.75)
        almi_goal["target_age"] = compact_config["ag"].get("ta", "?")

    ffmi_goal = {"target_percentile": 0.75, "target_age": "?"}
    if "fg" in compact_config:
        ffmi_goal["target_percentile"] = compact_config["fg"].get("tp", 0.75)
        ffmi_goal["target_age"] = compact_config["fg"].get("ta", "?")

    return user_info, scan_history, almi_goal, ffmi_goal, height_display


def get_base_url():
    """Get the base URL based on environment with smart detection."""
    # Priority 1: Custom override via environment variable
    custom_url = os.getenv("STREAMLIT_BASE_URL")
    if custom_url:
        return custom_url.rstrip("/")

    # Priority 2: Check for explicit development environment
    streamlit_env = os.getenv("STREAMLIT_ENV", "").lower()
    if streamlit_env == "development":
        return "http://localhost:8501"

    # Priority 3: Check for production environment indicators
    # Streamlit Cloud sets these environment variables
    if (
        os.getenv("STREAMLIT_SHARING_MODE")
        or os.getenv("STREAMLIT_SERVER_PORT")
        or "streamlit.app" in os.getenv("HOSTNAME", "")
    ):
        return "https://recomptracker.streamlit.app"

    # Priority 4: Check if we're running locally but STREAMLIT_ENV wasn't set
    # This handles cases where someone runs locally without setting the env var
    hostname = os.getenv("HOSTNAME", "").lower()
    if (
        "localhost" in hostname
        or hostname.startswith("127.0.0.1")
        or hostname == ""
        or os.getenv("USER") in ["runner", "github"]
    ):  # CI environments
        # If no clear production indicators, and we might be local, default to production
        # This is safer than assuming localhost
        pass

    # Priority 5: Default to production (safer assumption)
    return "https://recomptracker.streamlit.app"


def encode_state_to_url():
    """Generate a shareable URL with the current state encoded."""
    try:
        # Get compact config
        compact_config = get_compact_config()

        # Convert to JSON string and encode as base64 (simple approach)
        json_str = json.dumps(compact_config, separators=(",", ":"))
        encoded_data = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")

        # Create the shareable URL with environment-appropriate base URL
        base_url = get_base_url()
        share_url = f"{base_url}?data={urllib.parse.quote(encoded_data)}"

        return share_url

    except Exception as e:
        st.error(f"Failed to generate share URL: {e}")
        return None


def decode_state_from_url():
    """Decode state from URL parameters and update session state."""
    try:
        # Get URL parameters using the newer API
        query_params = st.query_params

        if "data" not in query_params:
            return False

        compressed_data = query_params["data"]

        # For now, we'll use a simplified approach without compression
        # In a real implementation, we'd need a different approach for JavaScript integration
        try:
            # Try to decode as base64 first (fallback)
            decoded_bytes = base64.b64decode(compressed_data.encode("utf-8"))
            json_str = decoded_bytes.decode("utf-8")
            compact_config = json.loads(json_str)

            # Update session state
            user_info, scan_history, almi_goal, ffmi_goal, height_display = (
                expand_compact_config(compact_config)
            )

            st.session_state.user_info = user_info
            st.session_state.scan_history = scan_history
            st.session_state.almi_goal = almi_goal
            st.session_state.ffmi_goal = ffmi_goal
            st.session_state.height_display = height_display

            st.success("Configuration loaded from URL!")
            return True

        except Exception:
            # If base64 fails, we'd need JavaScript decompression
            st.warning(
                "URL contains compressed data. Full compression support requires additional setup."
            )
            return False

    except Exception as e:
        st.error(f"Failed to decode URL data: {e}")
        return False


@st.dialog("💪 The Philosophy Behind RecompTracker", width="large")
def display_philosophy_modal():
    """Display the Philosophy section in a modal dialog."""
    philosophy_content = extract_philosophy_section()

    if philosophy_content:
        # Parse the content to make the ALMI vs BMI section expandable
        lines = philosophy_content.split("\n")

        # Find the start and end of the ALMI vs BMI section
        almi_section_start = -1
        almi_section_end = -1

        for i, line in enumerate(lines):
            if "### Why ALMI Over BMI: A Better Metric for Body Composition" in line:
                almi_section_start = i
            elif (
                almi_section_start != -1
                and line.startswith("### ")
                and "Why ALMI Over BMI" not in line
            ):
                almi_section_end = i
                break

        if almi_section_start != -1:
            # Display content before ALMI section
            before_almi = "\n".join(lines[:almi_section_start])
            if before_almi.strip():
                st.markdown(before_almi)

            # Display ALMI section in an expander
            almi_section_end = (
                almi_section_end if almi_section_end != -1 else len(lines)
            )
            almi_content = "\n".join(lines[almi_section_start + 1 : almi_section_end])

            with st.expander(
                "📊 Why ALMI Over BMI: A Better Metric for Body Composition",
                expanded=False,
            ):
                st.markdown(almi_content)

            # Display content after ALMI section
            if almi_section_end < len(lines):
                after_almi = "\n".join(lines[almi_section_end:])
                if after_almi.strip():
                    st.markdown(after_almi)
        else:
            # Fallback: display all content normally if parsing fails
            st.markdown(philosophy_content)

        # Close button
        if st.button("Close", key="close_philosophy_modal"):
            st.rerun()
    else:
        st.error("Could not load Philosophy section from README.md")


@st.dialog("What are T-scores?", width="large")
def display_tscore_modal():
    """Display T-score explanation in a modal dialog."""
    # TL;DR section
    st.info(
        "**TL;DR**: T-scores compare your muscle mass to peak young adults (ages 20-30) rather than your age group. Click 'Add T-score Overlay' in the chart above to see the colored zones and ranges. This is experimental and for fun - stick with percentiles for actual goal-setting."
    )

    # Main explanation
    st.markdown("### 📊 What are T-scores?")
    st.markdown("""
    T-scores are a standardized measurement that compares your value to a "peak reference population" - typically healthy young adults aged 20-30. T-scores tell you how many standard deviations you are from the peak values typically seen in young adulthood.
    """)

    st.markdown("### 🦴 T-scores in Bone Density vs. Muscle Mass")
    st.markdown("""
    T-scores are **commonly used** for bone density analysis, where they help diagnose osteoporosis and fracture risk. The World Health Organization officially defines osteoporosis as a bone density T-score of -2.5 or lower.

    For muscle mass and ALMI, T-scores are **much less standard**. While some research has explored T-score approaches for sarcopenia (muscle loss) assessment, there are no official clinical guidelines or widely accepted thresholds like there are for bone density.
    """)

    st.markdown("### 💪 What T-scores Mean for Your Muscle Mass")
    st.markdown("""
    In simple terms: **T-scores show how your current muscle mass compares to what you might have had in your physical prime.**

    - **T-score of 0**: Your muscle mass matches the average healthy 25-year-old
    - **T-score of +2**: You have exceptional muscle mass - better than 97% of young adults at their peak
    - **T-score of -2**: Your muscle mass is significantly below typical young adult levels
    """)

    with st.expander("🎯 Our T-score Zones (Experimental)"):
        st.markdown("""
        We've created 5 experimental zones that are **not based on clinical standards**:

        - **Elite Zone** (T ≥ +2.0): Exceptional muscle mass
        - **Peak Zone** (0 ≤ T < +2.0): Excellent muscle mass
        - **Approaching Peak** (-1.0 ≤ T < 0): Good muscle mass
        - **Below Peak** (-2.0 ≤ T < -1.0): Below optimal
        - **Well Below Peak** (T < -2.0): Significantly below optimal
        """)

    with st.expander("⚠️ Important Disclaimer"):
        st.markdown("""
        **Age-appropriate percentiles remain the recommended approach** for actual goal-setting and health assessment. T-scores are provided as an experimental feature for those interested in comparing against peak young adult muscle mass.

        Think of T-scores as a "fun fact" overlay rather than clinical guidance - your primary focus should remain on improving within your age and gender demographic using the standard percentile system.
        """)

    with st.expander("🔬 Technical Implementation"):
        st.markdown("""
        For the curious, detailed information about how we calculate T-score reference values using Monte Carlo sampling from LMS distributions can be found in our [T-Score Methodology Documentation](https://github.com/bmabey/recomptracker/blob/master/docs/t-score-methodology.md).

        This covers the technical challenges we solved, including why naive approaches fail and how we generate realistic population statistics from LMS parameters.
        """)

    # Close button
    if st.button("Close", key="close_tscore_modal"):
        st.rerun()


@st.dialog("Goal Setting Guide", width="large")
def display_goals_modal():
    """Display Goals explanation in a modal dialog."""
    # TL;DR section
    st.info(
        "**TL;DR:** Set an ALMI percentile goal (75th-90th recommended). FFMI goals are available if you're curious, but ALMI is the primary metric Attia focuses on for longevity planning."
    )

    # Attia's Recommendations
    st.markdown("### 🎯 Attia's Recommendations")
    st.markdown("""
    - **Baseline Goal:** 75th percentile ALMI (supported by mortality data showing significant longevity benefits)
    - **Aspirational Goal:** 90th-97th percentile ALMI (Attia's personal standard for optimal healthspan)
    """)

    # Expandable sections
    with st.expander("🏋️ Training Level Detection", expanded=False):
        st.markdown("""
        If you leave training level blank, the app automatically infers it from your scan progression:
        - **Novice:** Rapid lean mass gains (>0.5 kg/month) typical of early training
        - **Intermediate:** Moderate gains (0.25-0.5 kg/month) showing consistent progress
        - **Advanced:** Slow gains (<0.25 kg/month) reflecting training maturity
        - Single scan defaults to novice for conservative goal-setting
        """)

    with st.expander("📅 Target Age Calculation", expanded=False):
        st.markdown("""
        When you set target age to "?" or leave it blank, the app calculates a realistic timeframe based on:
        - Your current ALMI percentile and progression rate
        - Your training level (detected or specified) and associated lean mass gain rates
        - Conservative age-adjusted estimates to ensure achievable goals
        """)

    with st.expander("⚖️ Total Lean Mass Estimation", expanded=False):
        st.markdown("""
        The app calculates how much total lean mass you need by:
        - Using your personal ALM/TLM (Appendicular/Total Lean Mass) ratio from your scan history
        - If you have only one scan, it uses population-based ratios from the reference data
        - This accounts for the fact that not all lean mass gain goes to your arms and legs
        """)

    # Close button
    if st.button("Close", key="close_goals_modal"):
        st.rerun()


def extract_philosophy_section():
    """
    Extract the Philosophy section from README.md.

    Returns:
        str: The Philosophy section content in markdown format, or None if not found
    """
    try:
        readme_path = os.path.join(os.path.dirname(__file__), "README.md")

        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find the Philosophy section
        start_marker = "## Operationalizing Peter Attia's Medicine 3.0 Philosophy"
        end_marker = "## Features"

        start_idx = content.find(start_marker)
        if start_idx == -1:
            return None

        end_idx = content.find(end_marker, start_idx)
        if end_idx == -1:
            # If no Features section found, take rest of the content
            philosophy_content = content[start_idx:]
        else:
            philosophy_content = content[start_idx:end_idx]

        # Clean up the content - remove the main heading since we'll add our own
        lines = philosophy_content.split("\n")
        # Remove the main heading line and any empty lines after it
        filtered_lines = []
        skip_empty = True
        for i, line in enumerate(lines):
            if i == 0:  # Skip the main heading
                continue
            if skip_empty and line.strip() == "":
                continue
            skip_empty = False
            filtered_lines.append(line)

        return "\n".join(filtered_lines).strip()

    except Exception as e:
        st.error(f"Error reading Philosophy section: {e}")
        return None


def shorten_url_with_tinyurl(long_url):
    """
    Shorten a URL using the TinyURL API.

    Args:
        long_url (str): The long URL to shorten

    Returns:
        tuple: (success: bool, result: str) where result is either the shortened URL or error message
    """
    try:
        # TinyURL API endpoint
        api_url = "http://tinyurl.com/api-create.php"

        # Make request with timeout
        response = requests.get(
            api_url,
            params={"url": long_url},
            timeout=10,  # 10 second timeout
        )

        # Check if request was successful
        if response.status_code == 200:
            shortened_url = response.text.strip()

            # Validate that we got a proper TinyURL back
            if shortened_url.startswith("http") and "tinyurl.com" in shortened_url:
                return True, shortened_url
            else:
                return False, f"Invalid response from TinyURL: {shortened_url}"
        else:
            return False, f"TinyURL API error: HTTP {response.status_code}"

    except requests.exceptions.Timeout:
        return False, "TinyURL request timed out"
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to TinyURL service"
    except requests.exceptions.RequestException as e:
        return False, f"TinyURL request failed: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def format_inference_message(inference_msg):
    """
    Format inference messages by converting kg/month to lb/month.

    Args:
        inference_msg (str): Raw inference message (e.g., 'moderate progression 0.44 kg/month')

    Returns:
        str: Formatted message with lb/month
    """
    import re

    # Convert kg/month to lb/month in the message
    def kg_to_lb_replacement(match):
        kg_value = float(match.group(1))
        lb_value = kg_value * 2.20462
        return f"{lb_value:.2f} lb/month"

    # Replace kg/month with lb/month
    formatted_msg = re.sub(
        r"(\d+\.?\d*)\s*kg/month", kg_to_lb_replacement, inference_msg
    )
    return formatted_msg


def format_training_level_suggestion(level, inference_msg):
    """
    Format a clean training level suggestion message with lb/month progression rate.

    Args:
        level (str): Training level (e.g., 'intermediate')
        inference_msg (str): Raw inference message (e.g., 'Detected intermediate level: moderate progression 0.44 kg/month')

    Returns:
        str: Formatted suggestion message
    """
    import re

    # Extract progression description and rate from message
    progression_match = re.search(
        r"(rapid|moderate|slow)\s+progression\s+(\d+\.?\d*)\s*kg/month", inference_msg
    )

    if progression_match:
        progression_type = progression_match.group(1)
        kg_per_month = float(progression_match.group(2))
        # Convert kg/month to lb/month (1 kg = 2.20462 lbs)
        lb_per_month = kg_per_month * 2.20462

        # Format the suggestion with progression rate
        return f"💡 **Suggestion**: Based on your scans, you appear to be {level}, with {progression_type} lean mass progression of {lb_per_month:.2f} lb/month."
    else:
        # Fallback if no rate found
        return (
            f"💡 **Suggestion**: Based on your scans, you appear to be {level} level."
        )


def get_inferred_training_level():
    """Infer training level from current scan data."""
    try:
        # Need at least user info with gender and multiple scans
        if (
            not st.session_state.user_info.get("gender")
            or len(st.session_state.scan_history) < 2
        ):
            return None, "Need at least 2 scans to infer training level"

        # Filter scans that have meaningful data
        valid_scans = []
        for scan in st.session_state.scan_history:
            if (
                scan.get("date")
                and scan.get("date").strip()
                and scan.get("total_lean_mass_lbs", 0) > 0
            ):
                valid_scans.append(scan)

        if len(valid_scans) < 2:
            return None, "Need at least 2 complete scans to infer training level"

        # Create user_info compatible with core functions
        user_info = {"gender_code": parse_gender(st.session_state.user_info["gender"])}

        # Infer training level using core logic
        inferred_level, inference_explanation = detect_training_level_from_scans(
            valid_scans, user_info
        )

        if (
            inferred_level == "intermediate"
            and "defaulting to intermediate" in inference_explanation
        ):
            return None, "Insufficient data for reliable inference"

        return inferred_level, inference_explanation

    except Exception as e:
        return None, f"Error inferring training level: {str(e)}"


def create_plotly_metric_plot(
    df_results, metric_to_plot, lms_functions, goal_calculations
):
    """
    Creates interactive Plotly plots with hover tooltips for RecompTracker analysis.

    Args:
        df_results (pd.DataFrame): Complete results DataFrame with scan history and goals
        metric_to_plot (str): Either 'ALMI' or 'FFMI'
        lms_functions (dict): Dictionary containing LMS interpolation functions
        goal_calculations (dict): Goal calculation results

    Returns:
        plotly.graph_objects.Figure: Interactive plotly figure
    """
    # Define age range for percentile curves
    age_range = np.linspace(18, 80, 100)

    # Select appropriate LMS functions and labels
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

    # Create figure
    fig = go.Figure()

    # Define percentiles and colors (matching matplotlib version)
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

    # Add percentile curves
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
                curve_values.append(None)

        fig.add_trace(
            go.Scatter(
                x=age_range,
                y=curve_values,
                mode="lines",
                name=f"{label} percentile",
                line={"color": color, "width": 2},
                hovertemplate=f"{label} percentile<br>Age: %{{x:.1f}} years<br>{y_label}: %{{y:.2f}}<extra></extra>",
            )
        )

    # Filter data for actual scans (not goal rows)
    scan_data = df_results[~df_results["date_str"].str.contains("Goal", na=False)]

    # Add actual data points with detailed hover information
    if len(scan_data) > 0:
        # Create hover text with comprehensive information
        hover_text = []
        for _, scan in scan_data.iterrows():
            percentile_col = f"{metric_to_plot.lower()}_percentile"
            z_score_col = f"{metric_to_plot.lower()}_z_score"

            hover_info = [
                f"<b>Scan Date:</b> {scan['date_str']}",
                f"<b>Age:</b> {scan['age_at_scan']:.1f} years",
                f"<b>{y_label}:</b> {scan[y_column]:.2f}",
                f"<b>Percentile:</b> {scan[percentile_col]:.1f}%",
                f"<b>Z-Score:</b> {scan[z_score_col]:.2f}",
                "",
                f"<b>Weight:</b> {scan['total_weight_lbs']:.1f} lbs",
                f"<b>Lean Mass:</b> {scan['total_lean_mass_lbs']:.1f} lbs",
                f"<b>Fat Mass:</b> {scan['fat_mass_lbs']:.1f} lbs",
                f"<b>Body Fat:</b> {scan['body_fat_percentage']:.1f}%",
            ]
            hover_text.append("<br>".join(hover_info))

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
                    line={"color": "red", "width": 2, "dash": "solid"},
                    opacity=0.7,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Add goal if available
    goal_key = metric_to_plot.lower()
    if goal_key in goal_calculations:
        goal_calc = goal_calculations[goal_key]
        goal_age = goal_calc["target_age"]
        goal_value = goal_calc["target_metric_value"]
        goal_percentile = goal_calc["target_percentile"]

        # Build comprehensive goal hover information
        goal_z_score = goal_calc.get("target_z_score", 0)
        target_body_comp = goal_calc.get("target_body_composition", {})

        goal_hover_text = "<br>".join(
            [
                f"<b>🎯 {metric_to_plot} Goal</b>",
                f"<b>Target Age:</b> {goal_age:.1f} years",
                f"<b>Target {y_label}:</b> {goal_value:.2f}",
                f"<b>Target Percentile:</b> {goal_percentile * 100:.0f}%",
                f"<b>Target Z-Score:</b> {goal_z_score:.2f}",
                "",
                "<b>Target Body Composition:</b>",
                f"<b>Weight:</b> {target_body_comp.get('weight_lbs', 0):.1f} lbs",
                f"<b>Lean Mass:</b> {target_body_comp.get('lean_mass_lbs', 0):.1f} lbs",
                f"<b>Fat Mass:</b> {target_body_comp.get('fat_mass_lbs', 0):.1f} lbs",
                f"<b>Body Fat:</b> {target_body_comp.get('body_fat_percentage', 0):.1f}%",
                "",
                "<b>Changes Needed:</b>",
                f"<b>Weight:</b> {goal_calc.get('weight_change', 0):+.1f} lbs",
                f"<b>Lean Mass:</b> {goal_calc.get('lean_change', 0):+.1f} lbs",
                f"<b>Fat Mass:</b> {goal_calc.get('fat_change', 0):+.1f} lbs",
                f"<b>Body Fat:</b> {goal_calc.get('bf_change', 0):+.1f}%",
                f"<b>Percentile:</b> {goal_calc.get('percentile_change', 0):+.1f} points",
            ]
        )

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
            )
        )

        # Draw line from last scan to goal
        if len(scan_data) > 0:
            last_scan = scan_data.iloc[-1]
            fig.add_trace(
                go.Scatter(
                    x=[last_scan["age_at_scan"], goal_age],
                    y=[last_scan[y_column], goal_value],
                    mode="lines",
                    name="Goal Projection",
                    line={"color": "gold", "width": 3, "dash": "dash"},
                    opacity=0.8,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Customize layout
    fig.update_layout(
        title={
            "text": plot_title,
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
            "title": {"text": y_label, "font": {"size": 14}},
            "tickfont": {"size": 12},
            "gridcolor": "lightgray",
            "gridwidth": 0.5,
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
        height=600,
        margin={"r": 150},  # Right margin for legend
    )

    return fig


def auto_update_url():
    """Automatically update the URL when state changes."""
    try:
        # Only update if we have meaningful data
        if not (
            st.session_state.user_info.get("birth_date")
            or st.session_state.user_info.get("height_in") is not None
            or len(st.session_state.scan_history) > 0
        ):
            return

        # Generate current state hash to detect changes
        current_compact = get_compact_config()
        current_hash = hash(json.dumps(current_compact, sort_keys=True))

        # Check if state has changed
        if "last_state_hash" not in st.session_state:
            st.session_state.last_state_hash = current_hash
            st.session_state.share_url = encode_state_to_url()
        elif st.session_state.last_state_hash != current_hash:
            # State has changed, update URL
            st.session_state.last_state_hash = current_hash
            st.session_state.share_url = encode_state_to_url()

            # Hide URL input and reset shortening state when app state changes
            # This prevents users from copying stale URLs that don't reflect current state
            if "show_share_url" in st.session_state and st.session_state.show_share_url:
                st.session_state.show_share_url = False
                st.session_state.shortened_url = None
                st.session_state.shortening_error = None
                st.session_state.is_shortening = False

            # Update browser URL using JavaScript (debounced)
            if (
                st.session_state.share_url
                and "url_update_count" not in st.session_state
            ):
                st.session_state.url_update_count = 0

            # Only update browser URL every few changes to avoid excessive updates
            if (
                st.session_state.share_url
                and st.session_state.url_update_count % 3 == 0
            ):
                # Extract just the query parameter part
                url_parts = st.session_state.share_url.split("?", 1)
                if len(url_parts) > 1:
                    query_part = url_parts[1]
                    update_url_js = f"""
                    <script>
                    if (window.history && window.history.replaceState) {{
                        const newUrl = window.location.pathname + '?' + '{query_part}';
                        window.history.replaceState(null, '', newUrl);
                    }}
                    </script>
                    """
                    components.html(update_url_js, height=0)

            if "url_update_count" in st.session_state:
                st.session_state.url_update_count += 1

    except Exception:
        # Silently handle errors to avoid disrupting the UI
        pass


def initialize_session_state():
    """Initialize session state variables."""
    # Check if we need to load from URL first
    if "url_loaded" not in st.session_state:
        st.session_state.url_loaded = False

        # Try to load state from URL
        if decode_state_from_url():
            st.session_state.url_loaded = True
            # Mark that we should trigger analysis after initialization
            st.session_state.url_loaded_needs_analysis = True
            # Don't return early - we still need to initialize other attributes

    # Always initialize essential attributes if not already set
    # This ensures all required session state variables exist, even after URL loading
    if "user_info" not in st.session_state:
        st.session_state.user_info = {
            "birth_date": "",
            "height_in": 66.0,
            "gender": "male",
            "training_level": "",
        }

    # Initialize height display value separately
    if "height_display" not in st.session_state:
        st.session_state.height_display = ""

    if "scan_history" not in st.session_state:
        st.session_state.scan_history = []

    if "almi_goal" not in st.session_state:
        st.session_state.almi_goal = {"target_percentile": 0.75, "target_age": "?"}

    if "ffmi_goal" not in st.session_state:
        st.session_state.ffmi_goal = {"target_percentile": 0.75, "target_age": "?"}

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

    # Initialize URL shortening workflow state
    if "shortened_url" not in st.session_state:
        st.session_state.shortened_url = None

    if "is_shortening" not in st.session_state:
        st.session_state.is_shortening = False

    if "shortening_error" not in st.session_state:
        st.session_state.shortening_error = None

    if "url_loaded_needs_analysis" not in st.session_state:
        st.session_state.url_loaded_needs_analysis = False

    # Initialize editing state
    if "editing_scan_index" not in st.session_state:
        st.session_state.editing_scan_index = None


def validate_form_data():
    """Validate all form data and return validation results."""
    errors = {}

    # Validate user info
    for field, value in st.session_state.user_info.items():
        if field in ["birth_date", "height_in"] and value:
            is_valid, error_msg = validate_user_input(field, value)
            if not is_valid:
                errors[f"user_{field}"] = error_msg

    # Validate scan history
    for i, scan in enumerate(st.session_state.scan_history):
        for field, value in scan.items():
            if value:  # Only validate non-empty values
                is_valid, error_msg = validate_user_input(
                    field, value, st.session_state.user_info
                )
                if not is_valid:
                    errors[f"scan_{i}_{field}"] = error_msg

    # Check minimum requirements
    if not st.session_state.user_info.get("birth_date"):
        errors["user_birth_date"] = "Birth date is required"
    if not st.session_state.user_info.get("gender"):
        errors["user_gender"] = "Gender is required"
    if len(st.session_state.scan_history) == 0:
        errors["scan_history"] = "At least one scan is required"

    return errors


def run_analysis():
    """Run the analysis if data is valid."""
    errors = validate_form_data()

    if errors:
        st.error("Please fix the following errors before running analysis:")
        for error in errors.values():
            st.error(f"• {error}")
        return

    try:
        # Prepare user info
        user_info = st.session_state.user_info.copy()
        user_info["gender_code"] = parse_gender(user_info["gender"])

        # Prepare goals
        almi_goal = None
        ffmi_goal = None

        if st.session_state.almi_goal.get("target_percentile"):
            almi_goal = st.session_state.almi_goal.copy()
            target_age = almi_goal.get("target_age")
            if target_age == "?":
                almi_goal["target_age"] = None
            elif isinstance(target_age, str) and target_age.strip():
                try:
                    almi_goal["target_age"] = float(target_age)
                except (ValueError, TypeError):
                    # Invalid string, set to None for auto-calculation
                    almi_goal["target_age"] = None

        if st.session_state.ffmi_goal.get("target_percentile"):
            ffmi_goal = st.session_state.ffmi_goal.copy()
            target_age = ffmi_goal.get("target_age")
            if target_age == "?":
                ffmi_goal["target_age"] = None
            elif isinstance(target_age, str) and target_age.strip():
                try:
                    ffmi_goal["target_age"] = float(target_age)
                except (ValueError, TypeError):
                    # Invalid string, set to None for auto-calculation
                    ffmi_goal["target_age"] = None

        # Run analysis
        with st.spinner("Running analysis..."):
            df_results, goal_calculations, figures, comparison_table_html = (
                run_analysis_from_data(
                    user_info, st.session_state.scan_history, almi_goal, ffmi_goal
                )
            )

        # Store results
        st.session_state.analysis_results = {
            "df_results": df_results,
            "goal_calculations": goal_calculations,
            "figures": figures,
            "comparison_table_html": comparison_table_html,
        }

        # Update session state goals with calculated values (only auto-calculated fields)
        if goal_calculations:
            # Update ALMI goal with calculated values
            if "almi" in goal_calculations and st.session_state.almi_goal.get("target_percentile"):
                almi_calc = goal_calculations["almi"]
                
                # Update target_age if it was auto-calculated ("?" or None)
                if st.session_state.almi_goal.get("target_age") in ["?", None]:
                    calculated_age = almi_calc.get("target_age")
                    if calculated_age is not None:
                        st.session_state.almi_goal["target_age"] = str(calculated_age)
                
                # Update target_body_fat_percentage if not already set by user
                if "target_body_fat_percentage" not in st.session_state.almi_goal:
                    target_bf = almi_calc.get("target_body_composition", {}).get("body_fat_percentage")
                    if target_bf is not None:
                        st.session_state.almi_goal["target_body_fat_percentage"] = target_bf

            # Update FFMI goal with calculated values  
            if "ffmi" in goal_calculations and st.session_state.ffmi_goal.get("target_percentile"):
                ffmi_calc = goal_calculations["ffmi"]
                
                # Update target_age if it was auto-calculated ("?" or None)
                if st.session_state.ffmi_goal.get("target_age") in ["?", None]:
                    calculated_age = ffmi_calc.get("target_age")
                    if calculated_age is not None:
                        st.session_state.ffmi_goal["target_age"] = str(calculated_age)
                
                # Update target_body_fat_percentage if not already set by user
                if "target_body_fat_percentage" not in st.session_state.ffmi_goal:
                    target_bf = ffmi_calc.get("target_body_composition", {}).get("body_fat_percentage")
                    if target_bf is not None:
                        st.session_state.ffmi_goal["target_body_fat_percentage"] = target_bf

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")


def format_goal_info(goal_calc, metric):
    """Format concise goal information for display in the webapp."""
    if not goal_calc:
        return None

    # Extract values for the concise format
    alm_lbs = goal_calc.get("alm_change_needed_lbs", 0)
    tlm_lbs = goal_calc.get("tlm_change_needed_lbs", 0)
    target_bf = goal_calc.get("target_body_composition", {}).get(
        "body_fat_percentage", 0
    )
    fat_change = goal_calc.get("fat_change", 0)

    # Ensure target_age is numeric for formatting
    target_age = goal_calc.get("target_age")
    if isinstance(target_age, str):
        try:
            target_age = float(target_age)
        except (ValueError, TypeError):
            target_age = 0.0  # fallback value if conversion fails

    # Determine lean mass action
    if abs(tlm_lbs) < 0.1:  # Minimal change
        lean_action = "maintain your current lean mass"
        alm_part = ""
    elif tlm_lbs > 0:
        lean_action = f"add {tlm_lbs:.1f} lbs of total lean mass"
        alm_part = f", assuming {alm_lbs:.1f} lbs of that goes towards your ALM"
    else:
        lean_action = f"lose {abs(tlm_lbs):.1f} lbs of total lean mass"
        alm_part = f" (including {abs(alm_lbs):.1f} lbs from ALM)"

    # Determine fat mass action
    if abs(fat_change) < 0.1:  # Minimal change
        fat_action = (
            f"while maintaining your current body fat percentage of {target_bf:.1f}%"
        )
    elif fat_change < 0:  # Need to lose fat
        fat_action = f"To hit a BF% of {target_bf:.1f}% you will also need to drop {abs(fat_change):.1f} lbs of fat"
    else:  # Can gain fat
        fat_action = (
            f"To hit a BF% of {target_bf:.1f}% you can gain {fat_change:.1f} lbs of fat"
        )

    # Construct the message
    if abs(tlm_lbs) < 0.1:
        # Just maintain lean mass
        message = f"{lean_action} {fat_action}."
    elif fat_action.startswith("while"):
        # Maintain fat case
        message = f"Try to {lean_action}{alm_part} {fat_action}."
    else:
        # Change both lean and fat
        if tlm_lbs > 0:
            message = f"Try to {lean_action}{alm_part}. {fat_action}."
        else:
            message = f"You need to {lean_action}{alm_part} and {fat_action.lower()}."

    # Create concise goal information
    goal_info = f"""**🎯 {metric.upper()} Goal: {goal_calc["target_percentile"] * 100:.0f}th percentile by age {target_age:.1f}**

{message}"""

    return goal_info


def display_share_button():
    """Display share button with URL shortening functionality."""
    if "share_url" in st.session_state and st.session_state.share_url:
        # Initialize show_url state if not exists
        if "show_share_url" not in st.session_state:
            st.session_state.show_share_url = False

        # ROW 1: Link button and Shorten button (side by side)
        col1, col2 = st.columns([0.5, 0.5])

        with col1:
            # Share button that toggles URL display
            if st.button("🔗", help="Show shareable URL", key="share_button"):
                st.session_state.show_share_url = not st.session_state.show_share_url
                # Reset shortening state when toggling visibility
                if st.session_state.show_share_url:
                    st.session_state.shortened_url = None
                    st.session_state.shortening_error = None

        with col2:
            # Shorten URL button (only shows after link button is clicked)
            if st.session_state.show_share_url:
                if (
                    not st.session_state.shortened_url
                    and not st.session_state.is_shortening
                ):
                    if st.button(
                        "🔗 Shorten URL",
                        key="shorten_button",
                        help="Create a shorter TinyURL",
                    ):
                        st.session_state.is_shortening = True
                        st.session_state.shortening_error = None

                        # Call TinyURL API
                        success, result = shorten_url_with_tinyurl(
                            st.session_state.share_url
                        )

                        if success:
                            st.session_state.shortened_url = result
                            st.session_state.shortening_error = None
                        else:
                            st.session_state.shortening_error = result
                            st.session_state.shortened_url = None

                        st.session_state.is_shortening = False
                        st.rerun()

                elif st.session_state.is_shortening:
                    # Show loading state
                    st.button(
                        "⏳ Shortening...", disabled=True, key="shortening_loading"
                    )

                elif st.session_state.shortened_url:
                    # Show option to get original URL back
                    if st.button(
                        "↩️ Original", key="show_original", help="Show original URL"
                    ):
                        st.session_state.shortened_url = None
                        st.session_state.shortening_error = None
                        st.rerun()

        # ROW 2: Status messages (left) and URL input (right) - only when URL is shown
        if st.session_state.show_share_url:
            col1, col2 = st.columns([0.2, 0.8])

            with col1:
                # Show status messages (only after shortening is attempted) - very compact
                if st.session_state.shortening_error:
                    st.markdown(
                        '<div style="color: #ff4b4b; font-size: 12px; padding: 4px;">⚠️ Error</div>',
                        unsafe_allow_html=True,
                    )
                elif st.session_state.shortened_url:
                    st.markdown(
                        '<div style="color: #00c851; font-size: 12px; padding: 4px;">✅ Done</div>',
                        unsafe_allow_html=True,
                    )

            with col2:
                # Determine which URL to display
                if st.session_state.shortened_url:
                    display_url = st.session_state.shortened_url
                    url_label = "Shortened URL:"
                    url_help = "This is your shortened TinyURL - copy and share it!"
                else:
                    display_url = st.session_state.share_url
                    url_label = "Full URL:"
                    url_help = "Click in the field and press Ctrl+A then Ctrl+C (or Cmd+A, Cmd+C on Mac) to copy"

                # Display the URL input
                st.text_input(
                    url_label,
                    value=display_url,
                    key="share_url_display",
                    help=url_help,
                    label_visibility="collapsed",
                )


def reset_all_data():
    """Reset all form data and analysis results."""
    st.session_state.user_info = {
        "birth_date": "",
        "height_in": None,
        "gender": "male",
        "training_level": "",
    }
    st.session_state.height_display = ""  # Reset height display value too
    st.session_state.scan_history = []
    st.session_state.almi_goal = {"target_percentile": 0.75, "target_age": "?"}
    st.session_state.ffmi_goal = {"target_percentile": 0.75, "target_age": "?"}
    st.session_state.analysis_results = None

    # Clear confirmation state
    if "show_reset_confirmation" in st.session_state:
        del st.session_state.show_reset_confirmation

    # Clear URL state
    if "share_url" in st.session_state:
        del st.session_state.share_url
    if "last_state_hash" in st.session_state:
        del st.session_state.last_state_hash

    # Clear URL shortening state
    if "show_share_url" in st.session_state:
        st.session_state.show_share_url = False
    if "shortened_url" in st.session_state:
        st.session_state.shortened_url = None
    if "is_shortening" in st.session_state:
        st.session_state.is_shortening = False
    if "shortening_error" in st.session_state:
        st.session_state.shortening_error = None


def display_header():
    """Display the application header with explanations."""
    explanations = get_metric_explanations()

    # Compact three-column header layout
    col1, col2, col3 = st.columns([0.6, 0.2, 0.2])

    with col1:
        st.title(explanations["header_info"]["title"])

        # Philosophy teaser without subtitle
        st.markdown(
            """**Operationalizing Peter Attia's Medicine 3.0 philosophy** — Build your muscle buffer against inevitable decline."""
        )

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)  # Align with title
        # Philosophy button
        if st.button(
            "🧠 Learn why this matters →",
            key="show_philosophy",
            help="Learn about the philosophy behind RecompTracker",
            type="secondary",
            use_container_width=True,
        ):
            display_philosophy_modal()

    with col3:
        st.markdown("<br><br>", unsafe_allow_html=True)  # Align with title
        display_share_button()

    # Metric explanations in expandable sections
    with st.expander("📖 Understanding the Metrics", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(explanations["header_info"]["almi_explanation"])

        with col2:
            st.markdown(explanations["header_info"]["ffmi_explanation"])

        st.markdown(explanations["header_info"]["percentiles_explanation"])

        st.markdown(
            "**💡 Want to learn more?** If you're curious about why we use ALMI instead of BMI, see the **Learn why this matters** section above for the full rationale."
        )


def inches_to_feet_inches_str(inches):
    """
    Convert inches to feet'inches string format.

    Args:
        inches (float): Height in inches

    Returns:
        str: Height in x'y" format (e.g., "5'10\"")
    """
    if inches is None:
        return ""

    feet = int(inches // 12)
    remaining_inches = inches % 12

    # Round to nearest 0.25 inch for cleaner display (more precise than 0.5)
    remaining_inches = round(remaining_inches * 4) / 4

    # Handle case where rounding pushes us to 12 inches
    if remaining_inches >= 12:
        feet += 1
        remaining_inches = 0

    if remaining_inches == int(remaining_inches):
        return f"{feet}'{int(remaining_inches)}\""
    else:
        return f"{feet}'{remaining_inches}\""


def parse_height_input(height_str):
    """
    Parse height input in either x'y" format or inches format.

    Args:
        height_str (str): Height input string (e.g., "5'10\"", "5'10", "70", "70.5")

    Returns:
        float or None: Height in inches, or None if invalid
    """
    if not height_str:
        return None

    height_str = str(height_str).strip()

    # Try to parse as feet and inches format (e.g., "5'10\"", "5'10", "5' 10\"")
    import re

    feet_inches_pattern = r"(\d+)'?\s*(\d*\.?\d*)\"?"
    match = re.match(feet_inches_pattern, height_str)

    if match and "'" in height_str:
        feet_str = match.group(1)
        inches_str = match.group(2) if match.group(2) else "0"

        try:
            feet = float(feet_str)
            inches = float(inches_str) if inches_str else 0

            # Validate reasonable ranges
            if 0 <= feet <= 8 and 0 <= inches < 12:
                total_inches = feet * 12 + inches
                if 12 <= total_inches <= 120:  # Reasonable height range
                    return total_inches
        except ValueError:
            pass

    # Try to parse as direct inches (e.g., "70", "70.5")
    try:
        inches = float(height_str)
        if 12 <= inches <= 120:  # Reasonable height range
            return inches
    except ValueError:
        pass

    return None


def display_user_profile_form():
    """Display the user profile input form."""
    st.subheader("👤 User Profile")

    col1, col2 = st.columns(2)

    with col1:
        # Handle birth date conversion between date object and string
        birth_date_str = st.session_state.user_info.get("birth_date", "")
        birth_date_obj = None

        # Convert string to date object for the date input
        if (
            birth_date_str
            and isinstance(birth_date_str, str)
            and birth_date_str.strip()
        ):
            try:
                birth_date_obj = pd.to_datetime(
                    birth_date_str, format="%m/%d/%Y"
                ).date()
            except (ValueError, TypeError):
                birth_date_obj = None

        # Get valid birth date range
        min_date, max_date = get_birth_date_range()

        birth_date_input = st.date_input(
            "Birth Date",
            value=birth_date_obj,
            min_value=min_date,
            max_value=max_date,
            help="Select your birth date to calculate age for percentile comparisons",
            format="MM/DD/YYYY",
        )

        # Convert date object back to string for storage
        if birth_date_input:
            st.session_state.user_info["birth_date"] = birth_date_input.strftime(
                "%m/%d/%Y"
            )
        else:
            st.session_state.user_info["birth_date"] = ""

        height_input = st.text_input(
            "Height (x'y\" or inches)",
            value=st.session_state.height_display,
            help="Enter height as feet and inches (e.g., 5'10\") or just inches (e.g., 70)",
            placeholder="e.g., 5'10\" or 70",
        )

        # Update display value and parse height
        st.session_state.height_display = height_input
        parsed_height = parse_height_input(height_input)

        if height_input.strip() and parsed_height is None:
            st.error("Invalid height format. Use formats like 5'10\" or 70 inches")

        st.session_state.user_info["height_in"] = parsed_height

    with col2:
        gender = st.selectbox(
            "Gender",
            options=["male", "female"],
            index=0
            if st.session_state.user_info.get("gender", "male") == "male"
            else 1,
            help="Gender affects percentile comparisons and goal suggestions",
        )
        st.session_state.user_info["gender"] = gender

        # Get inferred training level
        inferred_level, inference_msg = get_inferred_training_level()

        # Set up options and default index
        options = ["", "novice", "intermediate", "advanced"]
        current_level = st.session_state.user_info.get("training_level", "")

        # If user hasn't set a level and we have an inference, suggest it
        if not current_level and inferred_level:
            suggested_index = (
                options.index(inferred_level) if inferred_level in options else 0
            )
        else:
            suggested_index = (
                options.index(current_level) if current_level in options else 0
            )

        training_level = st.selectbox(
            "Training Level",
            options=options,
            index=suggested_index,
            help=get_metric_explanations()["tooltips"]["training_level"],
        )
        st.session_state.user_info["training_level"] = training_level

    # Fixed-height reserved space for training level inference messages
    inference_container = st.container()
    with inference_container:
        # Always reserve space for inference messages with fixed height
        if inferred_level:
            if training_level == inferred_level:
                # Format the confirmed inference message
                formatted_msg = format_inference_message(inference_msg)
                st.markdown(
                    f'<div class="inference-success">✅ <strong>Inferred:</strong> {inferred_level.title()} - {formatted_msg}</div>',
                    unsafe_allow_html=True,
                )
            elif training_level and training_level != inferred_level:
                st.markdown(
                    f"<div class=\"inference-override\">🔄 <strong>Manual Override:</strong> Using '{training_level}' instead of inferred '{inferred_level}'</div>",
                    unsafe_allow_html=True,
                )
            else:
                # Parse and reformat the inference message
                suggestion_text = format_training_level_suggestion(
                    inferred_level, inference_msg
                )
                st.markdown(
                    f'<div class="inference-info">{suggestion_text}</div>',
                    unsafe_allow_html=True,
                )
        elif inference_msg and len(st.session_state.scan_history) > 0:
            # Format any standalone inference messages
            formatted_msg = format_inference_message(inference_msg)
            st.markdown(
                f'<div class="inference-info">ℹ️ {formatted_msg}</div>',
                unsafe_allow_html=True,
            )
        else:
            # Always reserve space even when no inference is available
            st.markdown('<div style="height: 60px;"></div>', unsafe_allow_html=True)

    # Action buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        if st.button(
            "🎲 Fake Data",
            help="Generate a fake user profile and scan history for testing",
        ):
            fake_profile = generate_fake_profile()
            # Generate fake scans first (needs numeric height_in)
            fake_scans = generate_fake_scans(fake_profile)

            # Convert height from inches to feet'inches format for display
            if fake_profile.get("height_in"):
                height_inches = fake_profile["height_in"]
                height_str = inches_to_feet_inches_str(height_inches)
                st.session_state.height_display = (
                    height_str  # Store display format separately
                )
                # Keep numeric value in fake_profile for user_info

            # Generate realistic fake goals
            import random
            fake_almi_goal = {
                "target_percentile": random.choice([0.70, 0.75, 0.80, 0.85, 0.90]),
                "target_age": "?",  # Let analysis auto-calculate
            }
            fake_ffmi_goal = {
                "target_percentile": random.choice([0.70, 0.75, 0.80, 0.85, 0.90]),
                "target_age": "?",  # Let analysis auto-calculate
            }

            st.session_state.user_info.update(fake_profile)
            st.session_state.scan_history = fake_scans
            st.session_state.almi_goal = fake_almi_goal
            st.session_state.ffmi_goal = fake_ffmi_goal
            run_analysis()
            st.rerun()

    with col2:
        # Only show Load Example button in development environment
        if os.getenv("STREAMLIT_ENV") == "development":
            if st.button("📋 Load Example"):
                try:
                    config = load_config_json("example_config.json", quiet=True)
                    user_info, scan_history, almi_goal, ffmi_goal = (
                        extract_data_from_config(config)
                    )

                    st.session_state.user_info = {
                        "birth_date": config["user_info"]["birth_date"],
                        "height_in": config["user_info"]["height_in"],
                        "gender": config["user_info"]["gender"],
                        "training_level": user_info.get("training_level", ""),
                    }

                    # Set height display format for example data
                    height_inches = config["user_info"]["height_in"]
                    st.session_state.height_display = inches_to_feet_inches_str(
                        height_inches
                    )

                    # Clean up scan history - remove date_str fields that are not needed for UI
                    cleaned_scan_history = []
                    for scan in scan_history:
                        clean_scan = scan.copy()
                        if "date_str" in clean_scan:
                            del clean_scan["date_str"]
                        cleaned_scan_history.append(clean_scan)

                    st.session_state.scan_history = cleaned_scan_history
                    if almi_goal:
                        st.session_state.almi_goal = almi_goal
                    if ffmi_goal:
                        st.session_state.ffmi_goal = ffmi_goal
                    run_analysis()
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load example config: {e}")

    with col3:
        # Initialize confirmation state if not exists
        if "show_reset_confirmation" not in st.session_state:
            st.session_state.show_reset_confirmation = False

        if not st.session_state.show_reset_confirmation:
            if st.button("🗑️ Reset Data"):
                st.session_state.show_reset_confirmation = True
                st.rerun()
        else:
            st.warning("Are you sure you want to clear the form data?")
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                if st.button("✅ Yes, Clear", type="primary"):
                    reset_all_data()
                    st.rerun()
            with subcol2:
                if st.button("❌ Cancel"):
                    st.session_state.show_reset_confirmation = False
                    st.rerun()


def add_scan_form():
    """Display a form for adding a new DEXA scan."""
    # Get tooltips for help text
    tooltips = get_metric_explanations()["tooltips"]

    with st.form("add_scan_form", clear_on_submit=True):
        st.markdown("#### ➕ Add New DEXA Scan")

        col1, col2 = st.columns(2)

        with col1:
            # Get reasonable scan date range (past 10 years, up to 1 year in future)
            from datetime import date, timedelta

            today = date.today()
            min_scan_date = today - timedelta(days=10 * 365)  # 10 years ago
            max_scan_date = today + timedelta(days=365)  # 1 year in future

            scan_date = st.date_input(
                "Scan Date *",
                min_value=min_scan_date,
                max_value=max_scan_date,
                help="Date when the DEXA scan was performed",
                format="MM/DD/YYYY",
            )

            total_weight = st.number_input(
                "Total Weight (lbs) *",
                min_value=0.0,
                step=0.1,
                format="%.1f",
                help="Total body weight from DEXA scan",
            )

            total_lean_mass = st.number_input(
                "Total Lean Mass (lbs) *",
                min_value=0.0,
                step=0.1,
                format="%.1f",
                help=tooltips["lean_mass"],
            )

            fat_mass = st.number_input(
                "Fat Mass (lbs) *",
                min_value=0.0,
                step=0.1,
                format="%.1f",
                help="Total fat mass from DEXA scan",
            )

        with col2:
            body_fat_percentage = st.number_input(
                "Body Fat % *",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                format="%.1f",
                help=tooltips["body_fat_percentage"],
            )

            arms_lean = st.number_input(
                "Arms Lean Mass (lbs) *",
                min_value=0.0,
                step=0.1,
                format="%.1f",
                help=tooltips["arms_lean"],
            )

            legs_lean = st.number_input(
                "Legs Lean Mass (lbs) *",
                min_value=0.0,
                step=0.1,
                format="%.1f",
                help=tooltips["legs_lean"],
            )

        st.markdown("*Required fields")

        submitted = st.form_submit_button(
            "➕ Add Scan", type="primary", use_container_width=True
        )

        if submitted:
            # Validate required fields
            errors = []
            if not scan_date:
                errors.append("Scan date is required")
            if total_weight <= 0:
                errors.append("Total weight must be greater than 0")
            if total_lean_mass <= 0:
                errors.append("Total lean mass must be greater than 0")
            if fat_mass <= 0:
                errors.append("Fat mass must be greater than 0")
            if body_fat_percentage <= 0:
                errors.append("Body fat percentage must be greater than 0")
            if arms_lean <= 0:
                errors.append("Arms lean mass must be greater than 0")
            if legs_lean <= 0:
                errors.append("Legs lean mass must be greater than 0")

            if errors:
                for error in errors:
                    st.error(f"• {error}")
                return

            # Check scan limit
            current_scans = [
                scan
                for scan in st.session_state.scan_history
                if scan.get("date", "").strip()
            ]
            if len(current_scans) >= 20:
                st.error(
                    "⚠️ Cannot add more scans. Maximum of 20 scans supported for URL sharing."
                )
                return

            # Create new scan entry
            new_scan = {
                "date": scan_date.strftime("%m/%d/%Y"),
                "total_weight_lbs": float(total_weight),
                "total_lean_mass_lbs": float(total_lean_mass),
                "fat_mass_lbs": float(fat_mass),
                "body_fat_percentage": float(body_fat_percentage),
                "arms_lean_lbs": float(arms_lean),
                "legs_lean_lbs": float(legs_lean),
            }

            # Remove any empty placeholder scans
            st.session_state.scan_history = [
                scan
                for scan in st.session_state.scan_history
                if scan.get("date", "").strip()
                or any(
                    scan.get(field, 0) > 0
                    for field in [
                        "total_weight_lbs",
                        "total_lean_mass_lbs",
                        "fat_mass_lbs",
                        "body_fat_percentage",
                        "arms_lean_lbs",
                        "legs_lean_lbs",
                    ]
                )
            ]

            # Add new scan and sort by date
            st.session_state.scan_history.append(new_scan)
            st.session_state.scan_history.sort(
                key=lambda x: pd.to_datetime(
                    x.get("date", "01/01/1900"), format="%m/%d/%Y"
                )
            )

            st.success(
                f"✅ Scan from {scan_date.strftime('%m/%d/%Y')} added successfully!"
            )
            st.rerun()


def edit_scan_form(scan_index, scan_data):
    """Display a form for editing an existing DEXA scan."""
    # Get tooltips for help text
    tooltips = get_metric_explanations()["tooltips"]

    original_date = scan_data.get("date", "Unknown")

    with st.form(f"edit_scan_form_{scan_index}", clear_on_submit=False):
        st.markdown(f"#### ✏️ Edit DEXA Scan from {original_date}")

        col1, col2 = st.columns(2)

        with col1:
            # Parse existing date for display
            existing_date = None
            if scan_data.get("date"):
                try:
                    existing_date = pd.to_datetime(
                        scan_data["date"], format="%m/%d/%Y"
                    ).date()
                except (ValueError, TypeError):
                    pass

            # Get reasonable scan date range (past 10 years, up to 1 year in future)
            from datetime import date, timedelta

            today = date.today()
            min_scan_date = today - timedelta(days=10 * 365)  # 10 years ago
            max_scan_date = today + timedelta(days=365)  # 1 year in future

            scan_date = st.date_input(
                "Scan Date *",
                value=existing_date,
                min_value=min_scan_date,
                max_value=max_scan_date,
                help="Date when the DEXA scan was performed",
                format="MM/DD/YYYY",
            )

            total_weight = st.number_input(
                "Total Weight (lbs) *",
                value=float(scan_data.get("total_weight_lbs", 0)),
                min_value=0.0,
                step=0.1,
                format="%.1f",
                help="Total body weight from DEXA scan",
            )

            total_lean_mass = st.number_input(
                "Total Lean Mass (lbs) *",
                value=float(scan_data.get("total_lean_mass_lbs", 0)),
                min_value=0.0,
                step=0.1,
                format="%.1f",
                help=tooltips["lean_mass"],
            )

            fat_mass = st.number_input(
                "Fat Mass (lbs) *",
                value=float(scan_data.get("fat_mass_lbs", 0)),
                min_value=0.0,
                step=0.1,
                format="%.1f",
                help="Total fat mass from DEXA scan",
            )

        with col2:
            body_fat_percentage = st.number_input(
                "Body Fat % *",
                value=float(scan_data.get("body_fat_percentage", 0)),
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                format="%.1f",
                help=tooltips["body_fat_percentage"],
            )

            arms_lean = st.number_input(
                "Arms Lean Mass (lbs) *",
                value=float(scan_data.get("arms_lean_lbs", 0)),
                min_value=0.0,
                step=0.1,
                format="%.1f",
                help=tooltips["arms_lean"],
            )

            legs_lean = st.number_input(
                "Legs Lean Mass (lbs) *",
                value=float(scan_data.get("legs_lean_lbs", 0)),
                min_value=0.0,
                step=0.1,
                format="%.1f",
                help=tooltips["legs_lean"],
            )

        st.markdown("*Required fields")

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button(
                "💾 Update Scan", type="primary", use_container_width=True
            )
        with col2:
            cancelled = st.form_submit_button("❌ Cancel", use_container_width=True)

        if cancelled:
            st.session_state.editing_scan_index = None
            st.rerun()

        if submitted:
            # Validate required fields
            errors = []
            if not scan_date:
                errors.append("Scan date is required")
            if total_weight <= 0:
                errors.append("Total weight must be greater than 0")
            if total_lean_mass <= 0:
                errors.append("Total lean mass must be greater than 0")
            if fat_mass <= 0:
                errors.append("Fat mass must be greater than 0")
            if body_fat_percentage <= 0:
                errors.append("Body fat percentage must be greater than 0")
            if arms_lean <= 0:
                errors.append("Arms lean mass must be greater than 0")
            if legs_lean <= 0:
                errors.append("Legs lean mass must be greater than 0")

            if errors:
                for error in errors:
                    st.error(f"• {error}")
                return

            # Check for date conflicts (excluding the current scan being edited)
            new_date_str = scan_date.strftime("%m/%d/%Y")
            existing_dates = [
                scan.get("date", "")
                for i, scan in enumerate(st.session_state.scan_history)
                if i != scan_index and scan.get("date", "").strip()
            ]

            if new_date_str in existing_dates:
                st.error(
                    f"⚠️ A scan already exists for {new_date_str}. Please choose a different date."
                )
                return

            # Update the scan data
            updated_scan = {
                "date": new_date_str,
                "total_weight_lbs": float(total_weight),
                "total_lean_mass_lbs": float(total_lean_mass),
                "fat_mass_lbs": float(fat_mass),
                "body_fat_percentage": float(body_fat_percentage),
                "arms_lean_lbs": float(arms_lean),
                "legs_lean_lbs": float(legs_lean),
            }

            # Update scan in place
            st.session_state.scan_history[scan_index] = updated_scan

            # Re-sort by date
            st.session_state.scan_history.sort(
                key=lambda x: pd.to_datetime(
                    x.get("date", "01/01/1900"), format="%m/%d/%Y"
                )
            )

            # Clear editing state
            st.session_state.editing_scan_index = None

            st.success(f"✅ Scan from {original_date} updated successfully!")
            st.rerun()


def display_scan_history_form():
    """Display the DEXA scan history with add form and read-only table."""
    st.subheader("🔬 DEXA Scan History")

    # Check scan limit for display warnings
    meaningful_scans = [
        scan for scan in st.session_state.scan_history if scan.get("date", "").strip()
    ]
    num_scans = len(meaningful_scans)

    if num_scans >= 20:
        st.error(
            "⚠️ Maximum of 20 scans supported for URL sharing. Please remove some scans before adding more."
        )
    elif num_scans >= 15:
        st.warning(
            f"📊 You have {num_scans} scans. URL sharing supports up to 20 scans."
        )

    # Initialize session state if empty
    if len(st.session_state.scan_history) == 0:
        st.session_state.scan_history = []

    # Display existing scans table if any
    if meaningful_scans:
        st.markdown("*All values in pounds except Body Fat (shown as %)*")

        # Create custom table layout with integrated edit and remove buttons
        # Header row - cleaner headers without repetitive (lbs)
        cols = st.columns([0.12, 0.12, 0.12, 0.12, 0.11, 0.13, 0.13, 0.17])

        with cols[0]:
            st.markdown("**Date**")
        with cols[1]:
            st.markdown("**Weight**")
        with cols[2]:
            st.markdown("**Lean Mass**")
        with cols[3]:
            st.markdown("**Fat Mass**")
        with cols[4]:
            st.markdown("**Body Fat**")
        with cols[5]:
            st.markdown("**Arms Lean**")
        with cols[6]:
            st.markdown("**Legs Lean**")
        with cols[7]:
            st.markdown("")

        # Data rows
        for i, scan in enumerate(meaningful_scans):
            # Find the actual index in the full scan_history list
            actual_index = None
            for j, full_scan in enumerate(st.session_state.scan_history):
                if full_scan.get("date", "") == scan.get("date", ""):
                    actual_index = j
                    break

            row_cols = st.columns([0.12, 0.12, 0.12, 0.12, 0.11, 0.13, 0.13, 0.17])

            with row_cols[0]:
                st.text(scan.get("date", ""))
            with row_cols[1]:
                st.text(f"{scan.get('total_weight_lbs', 0):.1f}")
            with row_cols[2]:
                st.text(f"{scan.get('total_lean_mass_lbs', 0):.1f}")
            with row_cols[3]:
                st.text(f"{scan.get('fat_mass_lbs', 0):.1f}")
            with row_cols[4]:
                st.text(f"{scan.get('body_fat_percentage', 0):.1f}%")
            with row_cols[5]:
                st.text(f"{scan.get('arms_lean_lbs', 0):.1f}")
            with row_cols[6]:
                st.text(f"{scan.get('legs_lean_lbs', 0):.1f}")
            with row_cols[7]:
                # Create sub-columns for edit and delete buttons
                action_col1, action_col2 = st.columns([0.5, 0.5])

                with action_col1:
                    # Disable edit button if another scan is being edited
                    edit_disabled = (
                        st.session_state.editing_scan_index is not None
                        and st.session_state.editing_scan_index != actual_index
                    )

                    if st.button(
                        "✏️",
                        key=f"edit_scan_{i}",
                        help=f"Edit scan from {scan.get('date', 'Unknown')}",
                        type="secondary",
                        disabled=edit_disabled,
                    ):
                        st.session_state.editing_scan_index = actual_index
                        st.rerun()

                with action_col2:
                    # Disable delete button if this scan is being edited
                    delete_disabled = (
                        st.session_state.editing_scan_index == actual_index
                    )

                    if st.button(
                        "🗑️",
                        key=f"delete_scan_{i}",
                        help=f"Delete scan from {scan.get('date', 'Unknown')}",
                        type="secondary",
                        disabled=delete_disabled,
                    ):
                        # Find and remove the specific scan by date
                        scan_date = scan.get("date", "")
                        st.session_state.scan_history = [
                            s
                            for s in st.session_state.scan_history
                            if s.get("date", "") != scan_date
                        ]
                        # Clear editing state if we deleted the scan being edited
                        if st.session_state.editing_scan_index == actual_index:
                            st.session_state.editing_scan_index = None
                        st.success(f"✅ Scan from {scan_date} removed!")
                        st.rerun()

        st.divider()

    # Display edit form if a scan is being edited, otherwise show add form
    if st.session_state.editing_scan_index is not None:
        # Get the scan being edited by index in scan_history
        if (
            0
            <= st.session_state.editing_scan_index
            < len(st.session_state.scan_history)
        ):
            edit_scan_form(
                st.session_state.editing_scan_index,
                st.session_state.scan_history[st.session_state.editing_scan_index],
            )
        else:
            # Invalid index, clear editing state
            st.session_state.editing_scan_index = None
            st.rerun()
    elif num_scans < 20:
        # Display add new scan form if under limit and not editing
        add_scan_form()

    # Show helpful message if no scans exist
    if num_scans == 0:
        st.info("👆 Add your first DEXA scan using the form above to get started!")

    # Clean up any empty placeholder scans that might exist
    st.session_state.scan_history = [
        scan
        for scan in st.session_state.scan_history
        if scan.get("date", "").strip()
        or any(
            scan.get(field, 0) > 0
            for field in [
                "total_weight_lbs",
                "total_lean_mass_lbs",
                "fat_mass_lbs",
                "body_fat_percentage",
                "arms_lean_lbs",
                "legs_lean_lbs",
            ]
        )
    ]


def display_goals_form():
    """Display the goals input form."""
    st.subheader("🎯 Goals")

    # Add Goals explanation modal button
    if st.button(
        "ℹ️ Learn about Goals and Recommendations →",
        key="goals_info_modal",
        help="Understand goal setting methodology and recommendations",
        type="secondary",
        use_container_width=True,
    ):
        display_goals_modal()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ALMI Goal**")
        almi_percentile = st.number_input(
            "Target ALMI Percentile",
            min_value=0.01,
            max_value=0.99,
            value=st.session_state.almi_goal.get("target_percentile", 0.75),
            step=0.05,
            help=get_metric_explanations()["tooltips"]["target_percentile"],
        )
        st.session_state.almi_goal["target_percentile"] = almi_percentile

        almi_age = st.text_input(
            "Target Age (or '?' for auto)",
            value=str(st.session_state.almi_goal.get("target_age", "?")),
            key="almi_target_age",
            help=get_metric_explanations()["tooltips"]["goal_age"],
        )
        st.session_state.almi_goal["target_age"] = almi_age

        almi_bf = st.number_input(
            "Target Body Fat % (optional)",
            min_value=1.0,
            max_value=50.0,
            value=st.session_state.almi_goal.get("target_body_fat_percentage", None),
            step=0.1,
            help="Leave empty for intelligent targeting based on health guidelines and feasibility",
            format="%.1f",
        )
        if almi_bf is not None and almi_bf > 0:
            st.session_state.almi_goal["target_body_fat_percentage"] = almi_bf
        else:
            st.session_state.almi_goal.pop("target_body_fat_percentage", None)

    with col2:
        st.markdown("**FFMI Goal**")
        ffmi_percentile = st.number_input(
            "Target FFMI Percentile",
            min_value=0.01,
            max_value=0.99,
            value=st.session_state.ffmi_goal.get("target_percentile", 0.75),
            step=0.05,
            help=get_metric_explanations()["tooltips"]["target_percentile"],
        )
        st.session_state.ffmi_goal["target_percentile"] = ffmi_percentile

        ffmi_age = st.text_input(
            "Target Age (or '?' for auto)",
            value=str(st.session_state.ffmi_goal.get("target_age", "?")),
            key="ffmi_target_age",
            help=get_metric_explanations()["tooltips"]["goal_age"],
        )
        st.session_state.ffmi_goal["target_age"] = ffmi_age

        ffmi_bf = st.number_input(
            "Target Body Fat % (optional)",
            min_value=1.0,
            max_value=50.0,
            value=st.session_state.ffmi_goal.get("target_body_fat_percentage", None),
            step=0.1,
            help="Leave empty for intelligent targeting based on health guidelines and feasibility",
            format="%.1f",
            key="ffmi_bf",
        )
        if ffmi_bf is not None and ffmi_bf > 0:
            st.session_state.ffmi_goal["target_body_fat_percentage"] = ffmi_bf
        else:
            st.session_state.ffmi_goal.pop("target_body_fat_percentage", None)


def display_results():
    """Display analysis results."""
    if st.session_state.analysis_results is None:
        # st.info("👈 Enter your data to the left and run the analysis")
        return

    results = st.session_state.analysis_results
    df_results = results["df_results"]
    goal_calculations = results["goal_calculations"]
    figures = results["figures"]
    comparison_table_html = results.get("comparison_table_html", "")

    st.subheader("📊 Analysis Results")

    # Display results in tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "💪 ALMI Analysis",
            "🔥 FFMI Analysis",
            "📈 Body Fat Analysis",
            "📊 Changes Summary",
        ]
    )

    with tab1:
        # ALMI summary metrics
        scan_data = df_results[~df_results["date_str"].str.contains("Goal", na=False)]
        if len(scan_data) > 0:
            latest_scan = scan_data.iloc[-1]
            first_scan = scan_data.iloc[0]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Current ALMI",
                    f"{latest_scan['almi_kg_m2']:.2f} kg/m²",
                    help="ALMI (Appendicular Lean Mass Index) measures lean muscle mass in arms and legs relative to height",
                )

            with col2:
                st.metric(
                    "ALMI Percentile",
                    f"{latest_scan['almi_percentile']:.1f}%",
                    help=get_metric_explanations()["tooltips"]["percentile"],
                )

            with col3:
                # Calculate progress since start
                if len(scan_data) > 1:
                    almi_progress = (
                        latest_scan["almi_percentile"] - first_scan["almi_percentile"]
                    )
                    progress_display = f"{almi_progress:+.1f} points"
                else:
                    progress_display = "N/A"

                st.metric(
                    "Progress Since Start",
                    progress_display,
                    help="Change in ALMI percentile since first scan",
                )

        # ALMI plot - full width

        # Create plotly figure with hover tooltips
        # Need to get LMS functions for plotting
        try:
            from core import load_lms_data, parse_gender

            user_info = st.session_state.user_info.copy()
            user_info["gender_code"] = parse_gender(user_info["gender"])

            # Load LMS functions
            lms_functions_local = {}
            (
                lms_functions_local["almi_L"],
                lms_functions_local["almi_M"],
                lms_functions_local["almi_S"],
            ) = load_lms_data("appendicular_LMI", user_info["gender_code"])
            (
                lms_functions_local["lmi_L"],
                lms_functions_local["lmi_M"],
                lms_functions_local["lmi_S"],
            ) = load_lms_data("LMI", user_info["gender_code"])

            if all(lms_functions_local.values()):
                # Calculate T-score reference values
                almi_mu_peak, almi_sigma_peak = calculate_tscore_reference_values(
                    user_info["gender_code"]
                )

                # Add T-score explanation modal button above the plot
                if almi_mu_peak is not None and almi_sigma_peak is not None:
                    if st.button(
                        "ℹ️ Learn about T-scores and peak zones →",
                        key="tscore_info_modal",
                        help="Understand T-score methodology and zone classifications",
                        type="secondary",
                        use_container_width=True,
                    ):
                        display_tscore_modal()

                # Create dual-mode plot with toggle functionality
                almi_fig = create_plotly_dual_mode_plot(
                    df_results,
                    "ALMI",
                    lms_functions_local,
                    goal_calculations,
                    almi_mu_peak,
                    almi_sigma_peak,
                )
                st.plotly_chart(almi_fig, use_container_width=True)

            else:
                st.error("Could not load LMS data for plotting")
        except Exception as e:
            st.error(f"Error creating interactive plot: {e}")
            # Fallback to matplotlib if plotly fails
            st.pyplot(figures["ALMI"])

        # Show ALMI goal information if available - below the plot
        if "almi" in goal_calculations:
            goal_info = format_goal_info(goal_calculations["almi"], "almi")
            if goal_info:
                st.markdown(goal_info)

        # ALMI-focused data table
        st.subheader("📋 ALMI Results Table")

        # Column explanations
        with st.expander("📖 Column Explanations", expanded=False):
            st.markdown("""
            **Basic Metrics:**
            - **Age**: Age at time of scan
            - **Weight/Lean/Fat Mass**: Body composition values in pounds
            - **Body Fat %**: Percentage of total weight that is fat
            - **ALMI**: Appendicular Lean Mass Index (kg/m²) - lean mass in arms and legs
            - **ALMI Percentile**: Your ranking compared to others your age and gender
            - **ALMI Z-Score**: Standard deviations from population average

            **Change Tracking:**
            - **Change (Last)**: Difference from your previous scan
            - **Change (First)**: Total change since your first scan
            """)

        almi_columns = [
            "age_at_scan",
            "total_weight_lbs",
            "total_lean_mass_lbs",
            "fat_mass_lbs",
            "body_fat_percentage",
            "almi_kg_m2",
            "almi_percentile",
            "almi_z_score",
        ]

        almi_names = [
            "Age",
            "Weight (lbs)",
            "Lean Mass (lbs)",
            "Fat Mass (lbs)",
            "Body Fat %",
            "ALMI (kg/m²)",
            "ALMI Percentile",
            "ALMI Z-Score",
        ]

        # Check which columns exist in the dataframe
        available_almi_columns = [
            col for col in almi_columns if col in scan_data.columns
        ]
        available_almi_names = [
            almi_names[i]
            for i, col in enumerate(almi_columns)
            if col in scan_data.columns
        ]

        df_almi = scan_data[available_almi_columns].copy()
        df_almi.columns = available_almi_names

        # Format numeric columns
        for col in df_almi.columns:
            if df_almi[col].dtype in ["float64", "int64"]:
                if "Percentile" in col:
                    df_almi[col] = df_almi[col].apply(
                        lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
                    )
                else:
                    df_almi[col] = df_almi[col].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                    )

        st.dataframe(df_almi, use_container_width=True)

    with tab2:
        # FFMI summary metrics
        scan_data = df_results[~df_results["date_str"].str.contains("Goal", na=False)]
        if len(scan_data) > 0:
            latest_scan = scan_data.iloc[-1]
            first_scan = scan_data.iloc[0]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Current FFMI",
                    f"{latest_scan['ffmi_kg_m2']:.2f} kg/m²",
                    help="FFMI (Fat-Free Mass Index) measures total lean body mass relative to height",
                )

            with col2:
                st.metric(
                    "FFMI Percentile",
                    f"{latest_scan['ffmi_percentile']:.1f}%",
                    help=get_metric_explanations()["tooltips"]["percentile"],
                )

            with col3:
                # Calculate progress since start
                if len(scan_data) > 1:
                    ffmi_progress = (
                        latest_scan["ffmi_percentile"] - first_scan["ffmi_percentile"]
                    )
                    progress_display = f"{ffmi_progress:+.1f} points"
                else:
                    progress_display = "N/A"

                st.metric(
                    "Progress Since Start",
                    progress_display,
                    help="Change in FFMI percentile since first scan",
                )

        # FFMI plot - full width

        # Create plotly figure with hover tooltips
        try:
            from core import load_lms_data, parse_gender

            user_info = st.session_state.user_info.copy()
            user_info["gender_code"] = parse_gender(user_info["gender"])

            # Load LMS functions (reuse from tab1 if needed, but reload for clarity)
            lms_functions_local = {}
            (
                lms_functions_local["almi_L"],
                lms_functions_local["almi_M"],
                lms_functions_local["almi_S"],
            ) = load_lms_data("appendicular_LMI", user_info["gender_code"])
            (
                lms_functions_local["lmi_L"],
                lms_functions_local["lmi_M"],
                lms_functions_local["lmi_S"],
            ) = load_lms_data("LMI", user_info["gender_code"])

            if all(lms_functions_local.values()):
                # Calculate T-score reference values
                almi_mu_peak, almi_sigma_peak = calculate_tscore_reference_values(
                    user_info["gender_code"]
                )

                # Create dual-mode plot with toggle functionality
                ffmi_fig = create_plotly_dual_mode_plot(
                    df_results,
                    "FFMI",
                    lms_functions_local,
                    goal_calculations,
                    almi_mu_peak,
                    almi_sigma_peak,
                )
                st.plotly_chart(ffmi_fig, use_container_width=True)
            else:
                st.error("Could not load LMS data for plotting")
        except Exception as e:
            st.error(f"Error creating interactive plot: {e}")
            # Fallback to matplotlib if plotly fails
            st.pyplot(figures["FFMI"])

        # Show FFMI goal information if available - below the plot
        if "ffmi" in goal_calculations:
            goal_info = format_goal_info(goal_calculations["ffmi"], "ffmi")
            if goal_info:
                st.markdown(goal_info)

        # FFMI-focused data table
        st.subheader("📋 FFMI Results Table")

        # Column explanations
        with st.expander("📖 Column Explanations", expanded=False):
            st.markdown("""
            **Basic Metrics:**
            - **Age**: Age at time of scan
            - **Weight/Lean/Fat Mass**: Body composition values in pounds
            - **Body Fat %**: Percentage of total weight that is fat
            - **FFMI**: Fat-Free Mass Index (kg/m²) - total lean body mass relative to height
            - **FFMI Percentile**: Your ranking compared to others your age and gender
            - **FFMI Z-Score**: Standard deviations from population average

            **Change Tracking:**
            - **Change (Last)**: Difference from your previous scan
            - **Change (First)**: Total change since your first scan
            """)

        ffmi_columns = [
            "age_at_scan",
            "total_weight_lbs",
            "total_lean_mass_lbs",
            "fat_mass_lbs",
            "body_fat_percentage",
            "ffmi_kg_m2",
            "ffmi_percentile",
            "ffmi_z_score",
        ]

        ffmi_names = [
            "Age",
            "Weight (lbs)",
            "Lean Mass (lbs)",
            "Fat Mass (lbs)",
            "Body Fat %",
            "FFMI (kg/m²)",
            "FFMI Percentile",
            "FFMI Z-Score",
        ]

        # Check which columns exist in the dataframe
        available_ffmi_columns = [
            col for col in ffmi_columns if col in scan_data.columns
        ]
        available_ffmi_names = [
            ffmi_names[i]
            for i, col in enumerate(ffmi_columns)
            if col in scan_data.columns
        ]

        df_ffmi = scan_data[available_ffmi_columns].copy()
        df_ffmi.columns = available_ffmi_names

        # Format numeric columns
        for col in df_ffmi.columns:
            if df_ffmi[col].dtype in ["float64", "int64"]:
                if "Percentile" in col:
                    df_ffmi[col] = df_ffmi[col].apply(
                        lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
                    )
                else:
                    df_ffmi[col] = df_ffmi[col].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                    )

        st.dataframe(df_ffmi, use_container_width=True)

    with tab3:
        # Body Fat Analysis tab
        st.subheader("📈 Body Fat Percentage Analysis")

        # Body fat summary metrics for latest scan
        if len(scan_data) > 0:
            latest_scan = scan_data.iloc[-1]
            bf_pct = latest_scan["body_fat_percentage"]

            # Get user's gender for health range assessment
            user_info = st.session_state.user_info.copy()
            user_info["gender_code"] = parse_gender(user_info["gender"])
            gender = user_info["gender"]

            # Import the function from core
            from core import get_bf_category

            bf_category = get_bf_category(bf_pct, gender)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Current Body Fat",
                    f"{bf_pct:.1f}%",
                    help="Current body fat percentage from latest DEXA scan",
                )

            with col2:
                # Calculate change from previous scan if available
                bf_change = (
                    latest_scan.get("bf_change_last", 0)
                    if pd.notna(latest_scan.get("bf_change_last"))
                    else None
                )
                change_display = f"{bf_change:+.1f}%" if bf_change is not None else None
                st.metric(
                    "Change (Last Scan)",
                    change_display if change_display else "N/A",
                    help="Change in body fat percentage from previous scan",
                )

            with col3:
                # Show health category with color coding
                category_colors = {
                    "athletic": "🟢",
                    "fitness": "🔵",
                    "acceptable": "🟡",
                    "overweight": "🔴",
                }
                category_icon = category_colors.get(bf_category, "⚪")
                st.metric(
                    "Health Category",
                    f"{category_icon} {bf_category.title()}",
                    help=f"Health category based on body fat percentage for {gender}s",
                )

        # Body fat plot - full width
        try:
            # Create plotly body fat plot
            from core import create_plotly_body_fat_plot

            user_info = st.session_state.user_info.copy()
            user_info["gender_code"] = parse_gender(user_info["gender"])

            bf_fig = create_plotly_body_fat_plot(df_results, user_info)
            st.plotly_chart(bf_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating body fat plot: {e}")
            # Fallback to matplotlib if plotly fails
            if "BODY_FAT" in figures:
                st.pyplot(figures["BODY_FAT"])

        # Body fat-focused data table
        st.subheader("📋 Body Fat Results Table")

        # Column explanations
        with st.expander("📖 Column Explanations", expanded=False):
            st.markdown("""
            **Basic Metrics:**
            - **Age**: Age at time of scan
            - **Weight**: Total body weight in pounds
            - **Lean Mass**: Muscle, bone, and organ mass (excludes fat)
            - **Fat Mass**: Total fat tissue in pounds
            - **Body Fat %**: Percentage of total weight that is fat tissue

            **Change Tracking:**
            - **Change (Last)**: Change in body fat % from your previous scan
            - **Change (First)**: Total change in body fat % since your first scan

            *Positive changes mean BF% increased; negative changes mean BF% decreased (fat loss)*
            """)

        bf_columns = [
            "age_at_scan",
            "total_weight_lbs",
            "total_lean_mass_lbs",
            "fat_mass_lbs",
            "body_fat_percentage",
        ]

        bf_names = [
            "Age",
            "Weight (lbs)",
            "Lean Mass (lbs)",
            "Fat Mass (lbs)",
            "Body Fat %",
        ]

        # Add change columns if they exist
        if "bf_change_last" in scan_data.columns:
            bf_columns.extend(["bf_change_last", "bf_change_first"])
            bf_names.extend(["Change (Last)", "Change (First)"])

        # Check which columns exist in the dataframe
        available_bf_columns = [col for col in bf_columns if col in scan_data.columns]
        available_bf_names = [
            bf_names[i] for i, col in enumerate(bf_columns) if col in scan_data.columns
        ]

        df_bf = scan_data[available_bf_columns].copy()
        df_bf.columns = available_bf_names

        # Format numeric columns
        for col in df_bf.columns:
            if df_bf[col].dtype in ["float64", "int64"]:
                if "Change" in col:
                    # Changes - show with +/- sign
                    df_bf[col] = df_bf[col].apply(
                        lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
                    )
                elif "Body Fat %" in col:
                    # Body fat percentage
                    df_bf[col] = df_bf[col].apply(
                        lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
                    )
                else:
                    # Regular values
                    df_bf[col] = df_bf[col].apply(
                        lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
                    )

        st.dataframe(df_bf, use_container_width=True)

    with tab4:
        # Changes Summary tab
        st.subheader("📊 Changes Summary")

        # Column explanations
        with st.expander("📖 Column Explanations", expanded=False):
            st.markdown("""
            **Basic Metrics:**
            - **Date**: Date of DEXA scan
            - **Age**: Age at time of scan
            - **Weight**: Total body weight in pounds
            - **Lean**: Total lean mass (muscle, bone, organs) in pounds
            - **Fat**: Total fat mass in pounds
            - **BF%**: Body fat percentage
            - **ALMI**: Appendicular Lean Mass Index (kg/m²) - lean mass in arms and legs
            - **FFMI**: Fat-Free Mass Index (kg/m²) - total lean mass normalized for height

            **Changes Row:**
            - **Changes**: Shows the change from your first scan to most recent scan
            - **Age**: Years elapsed since first scan (neutral - no color coding)
            - **Weight/Lean/Fat/BF%**: Physical changes from baseline
            - **ALMI/FFMI**: Z-score changes (performance improvements/declines)

            **Color Coding:**
            - 🟢 **Green**: Positive changes (lean mass gain, fat loss, BF% reduction, Z-score improvements)
            - 🔴 **Red**: Negative changes (lean mass loss, fat gain, BF% increase, Z-score declines)
            - ⚪ **White**: Neutral changes (weight, age)
            - **Intensity**: Color intensity reflects the magnitude of change
            """)

        # Display comparison table if available
        if comparison_table_html:
            st.markdown(comparison_table_html, unsafe_allow_html=True)

    # Download button for CSV
    csv_buffer = BytesIO()
    df_results.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    st.download_button(
        label="📥 Download Complete Results (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"recomptracker_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Download the complete analysis results including all calculated metrics",
    )


def main():
    """Main application function."""
    initialize_session_state()

    # Auto-run analysis if URL was loaded with valid data
    if st.session_state.get("url_loaded_needs_analysis", False):
        # Check if we have valid data for analysis
        if (
            st.session_state.user_info.get("birth_date")
            and st.session_state.user_info.get("gender")
            and len(st.session_state.scan_history) > 0
            and any(scan.get("date") for scan in st.session_state.scan_history)
        ):
            errors = validate_form_data()
            if not errors:
                run_analysis()

        # Clear the flag so we don't run analysis again on subsequent reruns
        st.session_state.url_loaded_needs_analysis = False

    # Auto-update URL when state changes
    auto_update_url()

    # Header
    display_header()

    # Main layout
    col1, col2 = st.columns([0.4, 0.6])

    with col1:
        # Left panel - Input forms
        display_user_profile_form()
        st.divider()
        display_scan_history_form()
        st.divider()
        display_goals_form()

        # Analyze button with custom light blue styling
        st.markdown(
            """
        <style>
        div.stButton > button:first-child {
            background-color: #8BB9F7;
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            font-weight: 600;
        }
        div.stButton > button:first-child:hover {
            background-color: #6BA3F0;
            border: none;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        if st.button("🔬 Run Analysis", use_container_width=True):
            run_analysis()

    with col2:
        # Right panel - Results
        display_results()

    # Auto-run analysis when data changes (if valid)
    if (
        st.session_state.user_info.get("birth_date")
        and st.session_state.user_info.get("gender")
        and len(st.session_state.scan_history) > 0
        and any(scan.get("date") for scan in st.session_state.scan_history)
    ):
        errors = validate_form_data()
        if not errors:
            run_analysis()


if __name__ == "__main__":
    main()
