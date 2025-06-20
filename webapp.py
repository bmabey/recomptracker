#!/usr/bin/env python3
"""
DEXA Body Composition Analysis - Streamlit Web Application

This web interface provides an intuitive way to analyze DEXA scan data using 
the same core analysis engine as the command-line tool. Features include:
- Interactive data input forms with validation
- Real-time analysis updates
- Comprehensive visualizations
- Goal setting with intelligent suggestions
- Downloadable results

Run with: streamlit run webapp.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json
import tempfile
import os
import base64
import urllib.parse
from datetime import datetime
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import core analysis functions
from core import (
    run_analysis_from_data, 
    generate_fake_profile, 
    generate_fake_scans,
    get_metric_explanations,
    validate_user_input,
    parse_gender,
    load_config_json,
    extract_data_from_config,
    detect_training_level_from_scans,
    get_value_from_zscore
)
import scipy.stats as stats

# Configure page
st.set_page_config(
    page_title="DEXA Body Composition Analysis",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling and add LZ-string library
st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/lz-string@1.5.0/libs/lz-string.min.js"></script>
<style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .error-message {
        color: #ff4b4b;
        font-size: 0.875rem;
        margin-top: 0.25rem;
    }
    .success-message {
        color: #00c851;
        font-size: 0.875rem;
        margin-top: 0.25rem;
    }
    .explanation-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def get_compact_config():
    """Convert current session state to compact JSON format."""
    compact = {
        "u": {
            "bd": st.session_state.user_info.get('birth_date', ''),
            "h": st.session_state.user_info.get('height_in', 66.0),
            "g": st.session_state.user_info.get('gender', 'male')[0],  # 'm' or 'f'
        }
    }
    
    # Add training level if set
    if st.session_state.user_info.get('training_level'):
        compact["u"]["tl"] = st.session_state.user_info['training_level']
    
    # Convert scan history to array format
    compact["s"] = []
    for scan in st.session_state.scan_history:
        if scan.get('date'):  # Only include scans with dates
            compact["s"].append([
                scan.get('date', ''),
                scan.get('total_weight_lbs', 0.0),
                scan.get('total_lean_mass_lbs', 0.0),
                scan.get('fat_mass_lbs', 0.0),
                scan.get('body_fat_percentage', 0.0),
                scan.get('arms_lean_lbs', 0.0),
                scan.get('legs_lean_lbs', 0.0)
            ])
    
    # Add goals if set
    if st.session_state.almi_goal.get('target_percentile'):
        compact["ag"] = {"tp": st.session_state.almi_goal['target_percentile']}
        if st.session_state.almi_goal.get('target_age') and st.session_state.almi_goal['target_age'] != '?':
            compact["ag"]["ta"] = st.session_state.almi_goal['target_age']
    
    if st.session_state.ffmi_goal.get('target_percentile'):
        compact["fg"] = {"tp": st.session_state.ffmi_goal['target_percentile']}
        if st.session_state.ffmi_goal.get('target_age') and st.session_state.ffmi_goal['target_age'] != '?':
            compact["fg"]["ta"] = st.session_state.ffmi_goal['target_age']
    
    return compact


def expand_compact_config(compact_config):
    """Convert compact JSON format back to full session state format."""
    # User info
    user_info = {
        'birth_date': compact_config.get("u", {}).get("bd", ''),
        'height_in': compact_config.get("u", {}).get("h", 66.0),
        'gender': 'male' if compact_config.get("u", {}).get("g", 'm') == 'm' else 'female',
        'training_level': compact_config.get("u", {}).get("tl", '')
    }
    
    # Scan history
    scan_history = []
    for scan_array in compact_config.get("s", []):
        if len(scan_array) >= 7:
            scan_history.append({
                'date': scan_array[0],
                'total_weight_lbs': scan_array[1],
                'total_lean_mass_lbs': scan_array[2],
                'fat_mass_lbs': scan_array[3],
                'body_fat_percentage': scan_array[4],
                'arms_lean_lbs': scan_array[5],
                'legs_lean_lbs': scan_array[6]
            })
    
    # Goals
    almi_goal = {'target_percentile': 0.75, 'target_age': '?'}
    if "ag" in compact_config:
        almi_goal['target_percentile'] = compact_config["ag"].get("tp", 0.75)
        almi_goal['target_age'] = compact_config["ag"].get("ta", '?')
    
    ffmi_goal = {'target_percentile': 0.75, 'target_age': '?'}
    if "fg" in compact_config:
        ffmi_goal['target_percentile'] = compact_config["fg"].get("tp", 0.75)
        ffmi_goal['target_age'] = compact_config["fg"].get("ta", '?')
    
    return user_info, scan_history, almi_goal, ffmi_goal


def encode_state_to_url():
    """Generate a shareable URL with the current state encoded."""
    try:
        # Get compact config
        compact_config = get_compact_config()
        
        # Convert to JSON string and encode as base64 (simple approach)
        json_str = json.dumps(compact_config, separators=(',', ':'))
        encoded_data = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
        
        # Create the shareable URL
        base_url = "http://localhost:8501"  # This would be the actual domain in production
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
        
        if 'data' not in query_params:
            return False
        
        compressed_data = query_params['data']
        
        # For now, we'll use a simplified approach without compression
        # In a real implementation, we'd need a different approach for JavaScript integration
        try:
            # Try to decode as base64 first (fallback)
            decoded_bytes = base64.b64decode(compressed_data.encode('utf-8'))
            json_str = decoded_bytes.decode('utf-8')
            compact_config = json.loads(json_str)
            
            # Update session state
            user_info, scan_history, almi_goal, ffmi_goal = expand_compact_config(compact_config)
            
            st.session_state.user_info = user_info
            st.session_state.scan_history = scan_history
            st.session_state.almi_goal = almi_goal
            st.session_state.ffmi_goal = ffmi_goal
            
            st.success("Configuration loaded from URL!")
            return True
            
        except Exception:
            # If base64 fails, we'd need JavaScript decompression
            st.warning("URL contains compressed data. Full compression support requires additional setup.")
            return False
        
    except Exception as e:
        st.error(f"Failed to decode URL data: {e}")
        return False


def copy_url_to_clipboard(url):
    """Create JavaScript to copy URL to clipboard."""
    copy_js = f"""
    <script>
    function copyToClipboard() {{
        const url = `{url}`;
        navigator.clipboard.writeText(url).then(function() {{
            // Show success message
            const successDiv = document.createElement('div');
            successDiv.innerHTML = '<div style="background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin: 10px 0;">✅ URL copied to clipboard!</div>';
            document.body.appendChild(successDiv);
            
            // Remove after 3 seconds
            setTimeout(() => {{
                if (successDiv.parentNode) {{
                    successDiv.parentNode.removeChild(successDiv);
                }}
            }}, 3000);
        }}, function(err) {{
            console.error('Could not copy text: ', err);
        }});
    }}
    
    // Auto-execute
    copyToClipboard();
    </script>
    """
    return copy_js


def get_inferred_training_level():
    """Infer training level from current scan data."""
    try:
        # Need at least user info with gender and multiple scans
        if (not st.session_state.user_info.get('gender') or 
            len(st.session_state.scan_history) < 2):
            return None, "Need at least 2 scans to infer training level"
        
        # Filter scans that have meaningful data
        valid_scans = []
        for scan in st.session_state.scan_history:
            if (scan.get('date') and scan.get('date').strip() and 
                scan.get('total_lean_mass_lbs', 0) > 0):
                valid_scans.append(scan)
        
        if len(valid_scans) < 2:
            return None, "Need at least 2 complete scans to infer training level"
        
        # Create user_info compatible with core functions
        user_info = {
            'gender_code': parse_gender(st.session_state.user_info['gender'])
        }
        
        # Infer training level using core logic
        inferred_level = detect_training_level_from_scans(valid_scans, user_info)
        
        if inferred_level == 'intermediate' and "defaulting to intermediate" in str(inferred_level):
            return None, "Insufficient data for reliable inference"
        
        return inferred_level, f"Based on lean mass progression across {len(valid_scans)} scans"
        
    except Exception as e:
        return None, f"Error inferring training level: {str(e)}"


def create_plotly_metric_plot(df_results, metric_to_plot, lms_functions, goal_calculations):
    """
    Creates interactive Plotly plots with hover tooltips for DEXA analysis.
    
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
    
    # Create figure
    fig = go.Figure()
    
    # Define percentiles and colors (matching matplotlib version)
    percentiles = [0.03, 0.10, 0.25, 0.50, 0.75, 0.90, 0.97]
    percentile_labels = ['3rd', '10th', '25th', '50th', '75th', '90th', '97th']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECCA7', '#DDA0DD', '#FFB347']
    
    # Add percentile curves
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
                curve_values.append(None)
        
        fig.add_trace(go.Scatter(
            x=age_range,
            y=curve_values,
            mode='lines',
            name=f'{label} percentile',
            line=dict(color=color, width=2),
            hovertemplate=f'{label} percentile<br>Age: %{{x:.1f}} years<br>{y_label}: %{{y:.2f}}<extra></extra>'
        ))
    
    # Filter data for actual scans (not goal rows)
    scan_data = df_results[~df_results['date_str'].str.contains('Goal', na=False)]
    
    # Add actual data points with detailed hover information
    if len(scan_data) > 0:
        # Create hover text with comprehensive information
        hover_text = []
        for _, scan in scan_data.iterrows():
            percentile_col = f'{metric_to_plot.lower()}_percentile'
            z_score_col = f'{metric_to_plot.lower()}_z_score'
            
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
                f"<b>Body Fat:</b> {scan['body_fat_percentage']:.1f}%"
            ]
            hover_text.append("<br>".join(hover_info))
        
        fig.add_trace(go.Scatter(
            x=scan_data['age_at_scan'],
            y=scan_data[y_column],
            mode='markers',
            name='Your Scans',
            marker=dict(
                color='red',
                size=12,
                line=dict(color='black', width=1)
            ),
            hovertemplate='%{text}<extra></extra>',
            text=hover_text
        ))
        
        # Connect points with lines if multiple scans
        if len(scan_data) > 1:
            fig.add_trace(go.Scatter(
                x=scan_data['age_at_scan'],
                y=scan_data[y_column],
                mode='lines',
                name='Progression',
                line=dict(color='red', width=2, dash='solid'),
                opacity=0.7,
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add goal if available
    goal_key = metric_to_plot.lower()
    if goal_key in goal_calculations:
        goal_calc = goal_calculations[goal_key]
        goal_age = goal_calc['target_age']
        goal_value = goal_calc['target_metric_value']
        goal_percentile = goal_calc['target_percentile']
        
        # Build comprehensive goal hover information
        goal_z_score = goal_calc.get('target_z_score', 0)
        target_body_comp = goal_calc.get('target_body_composition', {})
        
        goal_hover_text = "<br>".join([
            f"<b>🎯 {metric_to_plot} Goal</b>",
            f"<b>Target Age:</b> {goal_age:.1f} years",
            f"<b>Target {y_label}:</b> {goal_value:.2f}",
            f"<b>Target Percentile:</b> {goal_percentile*100:.0f}%",
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
            f"<b>Percentile:</b> {goal_calc.get('percentile_change', 0):+.1f} points"
        ])
        
        fig.add_trace(go.Scatter(
            x=[goal_age],
            y=[goal_value],
            mode='markers',
            name='Goal',
            marker=dict(
                color='gold',
                size=15,
                symbol='star',
                line=dict(color='black', width=1)
            ),
            hovertemplate='%{text}<extra></extra>',
            text=[goal_hover_text]
        ))
        
        # Draw line from last scan to goal
        if len(scan_data) > 0:
            last_scan = scan_data.iloc[-1]
            fig.add_trace(go.Scatter(
                x=[last_scan['age_at_scan'], goal_age],
                y=[last_scan[y_column], goal_value],
                mode='lines',
                name='Goal Projection',
                line=dict(color='gold', width=3, dash='dash'),
                opacity=0.8,
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Customize layout
    fig.update_layout(
        title=dict(
            text=plot_title,
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
            title=dict(text=y_label, font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
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
        margin=dict(r=150)  # Right margin for legend
    )
    
    return fig


def auto_update_url():
    """Automatically update the URL when state changes."""
    try:
        # Only update if we have meaningful data
        if not (st.session_state.user_info.get('birth_date') or 
                st.session_state.user_info.get('height_in', 66.0) != 66.0 or
                len(st.session_state.scan_history) > 0):
            return
        
        # Generate current state hash to detect changes
        current_compact = get_compact_config()
        current_hash = hash(json.dumps(current_compact, sort_keys=True))
        
        # Check if state has changed
        if 'last_state_hash' not in st.session_state:
            st.session_state.last_state_hash = current_hash
            st.session_state.share_url = encode_state_to_url()
        elif st.session_state.last_state_hash != current_hash:
            # State has changed, update URL
            st.session_state.last_state_hash = current_hash
            st.session_state.share_url = encode_state_to_url()
            
            # Update browser URL using JavaScript (debounced)
            if st.session_state.share_url and 'url_update_count' not in st.session_state:
                st.session_state.url_update_count = 0
            
            # Only update browser URL every few changes to avoid excessive updates
            if st.session_state.share_url and st.session_state.url_update_count % 3 == 0:
                # Extract just the query parameter part
                url_parts = st.session_state.share_url.split('?', 1)
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
            
            if 'url_update_count' in st.session_state:
                st.session_state.url_update_count += 1
        
    except Exception as e:
        # Silently handle errors to avoid disrupting the UI
        pass


def initialize_session_state():
    """Initialize session state variables."""
    # Check if we need to load from URL first
    if 'url_loaded' not in st.session_state:
        st.session_state.url_loaded = False
        
        # Try to load state from URL
        if decode_state_from_url():
            st.session_state.url_loaded = True
            # Don't return early - we still need to initialize other attributes
    
    # Always initialize essential attributes if not already set
    # This ensures all required session state variables exist, even after URL loading
    if 'user_info' not in st.session_state:
        st.session_state.user_info = {
            'birth_date': '',
            'height_in': 66.0,
            'gender': 'male',
            'training_level': ''
        }
    
    if 'scan_history' not in st.session_state:
        st.session_state.scan_history = []
    
    if 'almi_goal' not in st.session_state:
        st.session_state.almi_goal = {'target_percentile': 0.75, 'target_age': '?'}
    
    if 'ffmi_goal' not in st.session_state:
        st.session_state.ffmi_goal = {'target_percentile': 0.75, 'target_age': '?'}
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None


def validate_form_data():
    """Validate all form data and return validation results."""
    errors = {}
    
    # Validate user info
    for field, value in st.session_state.user_info.items():
        if field in ['birth_date', 'height_in'] and value:
            is_valid, error_msg = validate_user_input(field, value)
            if not is_valid:
                errors[f"user_{field}"] = error_msg
    
    # Validate scan history
    for i, scan in enumerate(st.session_state.scan_history):
        for field, value in scan.items():
            if value:  # Only validate non-empty values
                is_valid, error_msg = validate_user_input(field, value, st.session_state.user_info)
                if not is_valid:
                    errors[f"scan_{i}_{field}"] = error_msg
    
    # Check minimum requirements
    if not st.session_state.user_info.get('birth_date'):
        errors['user_birth_date'] = "Birth date is required"
    if not st.session_state.user_info.get('gender'):
        errors['user_gender'] = "Gender is required"
    if len(st.session_state.scan_history) == 0:
        errors['scan_history'] = "At least one scan is required"
    
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
        user_info['gender_code'] = parse_gender(user_info['gender'])
        
        # Prepare goals
        almi_goal = None
        ffmi_goal = None
        
        if st.session_state.almi_goal.get('target_percentile'):
            almi_goal = st.session_state.almi_goal.copy()
            if almi_goal.get('target_age') == '?':
                almi_goal['target_age'] = None
        
        if st.session_state.ffmi_goal.get('target_percentile'):
            ffmi_goal = st.session_state.ffmi_goal.copy()
            if ffmi_goal.get('target_age') == '?':
                ffmi_goal['target_age'] = None
        
        # Run analysis
        with st.spinner("Running analysis..."):
            df_results, goal_calculations, figures = run_analysis_from_data(
                user_info, st.session_state.scan_history, almi_goal, ffmi_goal
            )
        
        # Store results
        st.session_state.analysis_results = {
            'df_results': df_results,
            'goal_calculations': goal_calculations,
            'figures': figures
        }
        
        st.success("Analysis completed successfully!")
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")


def format_goal_info(goal_calc, metric):
    """Format concise goal information for display in the webapp."""
    if not goal_calc:
        return None
    
    # Extract values for the concise format
    alm_lbs = goal_calc.get('alm_change_needed_lbs', 0)
    tlm_lbs = goal_calc.get('tlm_change_needed_lbs', 0)
    target_bf = goal_calc.get('target_body_composition', {}).get('body_fat_percentage', 0)
    fat_change = goal_calc.get('fat_change', 0)
    
    # Determine lean mass action
    if abs(tlm_lbs) < 0.1:  # Minimal change
        lean_action = f"maintain your current lean mass"
        alm_part = ""
    elif tlm_lbs > 0:
        lean_action = f"add {tlm_lbs:.1f} lbs of total lean mass"
        alm_part = f", assuming {alm_lbs:.1f} lbs of that goes towards your ALM"
    else:
        lean_action = f"lose {abs(tlm_lbs):.1f} lbs of total lean mass"
        alm_part = f" (including {abs(alm_lbs):.1f} lbs from ALM)"
    
    # Determine fat mass action
    if abs(fat_change) < 0.1:  # Minimal change
        fat_action = f"while maintaining your current body fat percentage of {target_bf:.1f}%"
    elif fat_change < 0:  # Need to lose fat
        fat_action = f"To hit a BF% of {target_bf:.1f}% you will also need to drop {abs(fat_change):.1f} lbs of fat"
    else:  # Can gain fat
        fat_action = f"To hit a BF% of {target_bf:.1f}% you can gain {fat_change:.1f} lbs of fat"
    
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
    goal_info = f"""**🎯 {metric.upper()} Goal: {goal_calc['target_percentile']*100:.0f}th percentile by age {goal_calc['target_age']:.1f}**

{message}"""
    
    return goal_info


def display_header():
    """Display the application header with explanations."""
    explanations = get_metric_explanations()
    
    st.title(explanations['header_info']['title'])
    st.markdown(explanations['header_info']['subtitle'])
    
    # Metric explanations in expandable sections
    with st.expander("📖 Understanding the Metrics", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(explanations['header_info']['almi_explanation'])
        
        with col2:
            st.markdown(explanations['header_info']['ffmi_explanation'])
        
        st.markdown(explanations['header_info']['percentiles_explanation'])
        st.info(explanations['header_info']['population_source'])


def display_user_profile_form():
    """Display the user profile input form."""
    st.subheader("👤 User Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        birth_date = st.text_input(
            "Birth Date (MM/DD/YYYY)",
            value=st.session_state.user_info.get('birth_date', ''),
            help="Enter your birth date to calculate age for percentile comparisons"
        )
        st.session_state.user_info['birth_date'] = birth_date
        
        height_in = st.number_input(
            "Height (inches)",
            min_value=12.0,
            max_value=120.0,
            value=st.session_state.user_info.get('height_in', 66.0),
            step=0.1,
            help="Your height in inches (used to calculate ALMI and FFMI)"
        )
        st.session_state.user_info['height_in'] = height_in
    
    with col2:
        gender = st.selectbox(
            "Gender",
            options=['male', 'female'],
            index=0 if st.session_state.user_info.get('gender', 'male') == 'male' else 1,
            help="Gender affects percentile comparisons and goal suggestions"
        )
        st.session_state.user_info['gender'] = gender
        
        # Get inferred training level
        inferred_level, inference_msg = get_inferred_training_level()
        
        # Set up options and default index
        options = ['', 'novice', 'intermediate', 'advanced']
        current_level = st.session_state.user_info.get('training_level', '')
        
        # If user hasn't set a level and we have an inference, suggest it
        if not current_level and inferred_level:
            suggested_index = options.index(inferred_level) if inferred_level in options else 0
        else:
            suggested_index = options.index(current_level) if current_level in options else 0
        
        training_level = st.selectbox(
            "Training Level",
            options=options,
            index=suggested_index,
            help=get_metric_explanations()['tooltips']['training_level']
        )
        st.session_state.user_info['training_level'] = training_level
        
        # Show inference information
        if inferred_level:
            if training_level == inferred_level:
                st.success(f"✅ **Inferred**: {inferred_level.title()} - {inference_msg}")
            elif training_level and training_level != inferred_level:
                st.info(f"🔄 **Manual Override**: Using '{training_level}' instead of inferred '{inferred_level}'")
            else:
                st.info(f"💡 **Suggestion**: Based on your scans, you appear to be '{inferred_level}' level")
        elif inference_msg and len(st.session_state.scan_history) > 0:
            st.info(f"ℹ️ {inference_msg}")
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🎲 Random Profile"):
            fake_profile = generate_fake_profile()
            st.session_state.user_info.update(fake_profile)
            fake_scans = generate_fake_scans(st.session_state.user_info)
            st.session_state.scan_history = fake_scans
            run_analysis()
            st.rerun()
    
    with col2:
        if st.button("📋 Load Example"):
            try:
                config = load_config_json('example_config.json', quiet=True)
                user_info, scan_history, almi_goal, ffmi_goal = extract_data_from_config(config)
                
                st.session_state.user_info = {
                    'birth_date': config['user_info']['birth_date'],
                    'height_in': config['user_info']['height_in'],
                    'gender': config['user_info']['gender'],
                    'training_level': user_info.get('training_level', '')
                }
                st.session_state.scan_history = scan_history
                if almi_goal:
                    st.session_state.almi_goal = almi_goal
                if ffmi_goal:
                    st.session_state.ffmi_goal = ffmi_goal
                run_analysis()
                st.rerun()
            except Exception as e:
                st.error(f"Could not load example config: {e}")


def display_scan_history_form():
    """Display the DEXA scan history form using an editable data table."""
    st.subheader("🔬 DEXA Scan History")
    
    # Check scan limit
    num_scans = len([scan for scan in st.session_state.scan_history if scan.get('date', '').strip()])
    if num_scans >= 20:
        st.error("⚠️ Maximum of 20 scans supported for URL sharing. Please remove some scans before adding more.")
    elif num_scans >= 15:
        st.warning(f"📊 You have {num_scans} scans. URL sharing supports up to 20 scans.")
    
    # Initialize with empty scan if none exist
    if len(st.session_state.scan_history) == 0:
        st.session_state.scan_history = [{
            'date': '',
            'total_weight_lbs': 0.0,
            'total_lean_mass_lbs': 0.0,
            'fat_mass_lbs': 0.0,
            'body_fat_percentage': 0.0,
            'arms_lean_lbs': 0.0,
            'legs_lean_lbs': 0.0
        }]
    
    # Convert session state to DataFrame for data editor
    df = pd.DataFrame(st.session_state.scan_history)
    
    # Get tooltips for help text
    tooltips = get_metric_explanations()['tooltips']
    
    # Configure column types and validation
    column_config = {
        "date": st.column_config.TextColumn(
            "Date (MM/DD/YYYY)",
            help="Date of the DEXA scan",
            required=True,
            validate=r"^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}$"
        ),
        "total_weight_lbs": st.column_config.NumberColumn(
            "Total Weight (lbs)",
            help="Total body weight from DEXA scan",
            min_value=0.0,
            step=0.1,
            format="%.1f",
            required=True
        ),
        "total_lean_mass_lbs": st.column_config.NumberColumn(
            "Total Lean Mass (lbs)",
            help=tooltips['lean_mass'],
            min_value=0.0,
            step=0.1,
            format="%.1f",
            required=True
        ),
        "fat_mass_lbs": st.column_config.NumberColumn(
            "Fat Mass (lbs)",
            help="Total fat mass from DEXA scan",
            min_value=0.0,
            step=0.1,
            format="%.1f",
            required=True
        ),
        "body_fat_percentage": st.column_config.NumberColumn(
            "Body Fat %",
            help=tooltips['body_fat_percentage'],
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            format="%.1f",
            required=True
        ),
        "arms_lean_lbs": st.column_config.NumberColumn(
            "Arms Lean Mass (lbs)",
            help=tooltips['arms_lean'],
            min_value=0.0,
            step=0.1,
            format="%.1f",
            required=True
        ),
        "legs_lean_lbs": st.column_config.NumberColumn(
            "Legs Lean Mass (lbs)",
            help=tooltips['legs_lean'],
            min_value=0.0,
            step=0.1,
            format="%.1f",
            required=True
        )
    }
    
    # Display editable data table
    edited_df = st.data_editor(
        df,
        column_config=column_config,
        num_rows="dynamic",
        use_container_width=True,
        key="scan_hisory_editor",
        height=min(75 + (len(df) * 35), 400)  # Dynamic height based on number of rows
    )
    
    # Update session state with edited data, but limit to 20 scans
    new_scan_history = edited_df.to_dict('records')
    
    # Count meaningful scans (those with actual data)
    meaningful_scans = [scan for scan in new_scan_history if 
                       scan.get('date', '').strip() or 
                       any(scan.get(field, 0) > 0 for field in ['total_weight_lbs', 'total_lean_mass_lbs', 
                           'fat_mass_lbs', 'body_fat_percentage', 'arms_lean_lbs', 'legs_lean_lbs'])]
    
    if len(meaningful_scans) > 20:
        st.error("⚠️ Cannot add more than 20 scans. URL sharing limit exceeded.")
        # Keep only the first 20 meaningful scans plus any empty rows
        empty_scans = [scan for scan in new_scan_history if scan not in meaningful_scans]
        st.session_state.scan_history = meaningful_scans[:20] + empty_scans
    else:
        st.session_state.scan_history = new_scan_history
    
    # Validate scan data and show errors
    scan_errors = []
    for i, scan in enumerate(st.session_state.scan_history):
        # Check if this row has been meaningfully edited (not just default empty values)
        has_any_data = (
            (scan.get('date', '').strip() != '') or
            any(scan.get(field, 0) > 0 for field in ['total_weight_lbs', 'total_lean_mass_lbs', 
                'fat_mass_lbs', 'body_fat_percentage', 'arms_lean_lbs', 'legs_lean_lbs'])
        )
        
        # Only validate rows that have some data entered
        if has_any_data:
            # Check for empty required fields
            if not scan.get('date') or scan.get('date', '').strip() == '':
                scan_errors.append(f"Row {i+1}: Date is required")
            elif scan.get('date'):
                # Validate date format
                is_valid, error_msg = validate_user_input('scan_date', scan['date'])
                if not is_valid:
                    scan_errors.append(f"Row {i+1}: {error_msg}")
            
            # Check for zero values in required numeric fields
            required_numeric = ['total_weight_lbs', 'total_lean_mass_lbs', 'fat_mass_lbs', 
                              'body_fat_percentage', 'arms_lean_lbs', 'legs_lean_lbs']
            for field in required_numeric:
                if not scan.get(field) or scan.get(field, 0) <= 0:
                    field_display = field.replace('_', ' ').replace('lbs', '(lbs)').replace('percentage', '%').title()
                    scan_errors.append(f"Row {i+1}: {field_display} must be greater than 0")
    
    if scan_errors:
        st.error("Please fix the following scan data issues:")
        for error in scan_errors:
            st.error(f"• {error}")
    


def display_goals_form():
    """Display the goals input form."""
    st.subheader("🎯 Goals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**ALMI Goal**")
        almi_percentile = st.number_input(
            "Target ALMI Percentile",
            min_value=0.01,
            max_value=0.99,
            value=st.session_state.almi_goal.get('target_percentile', 0.75),
            step=0.05,
            help=get_metric_explanations()['tooltips']['target_percentile']
        )
        st.session_state.almi_goal['target_percentile'] = almi_percentile
        
        almi_age = st.text_input(
            "Target Age (or '?' for auto)",
            value=str(st.session_state.almi_goal.get('target_age', '?')),
            key="almi_target_age",
            help=get_metric_explanations()['tooltips']['goal_age']
        )
        st.session_state.almi_goal['target_age'] = almi_age
        
        almi_bf = st.number_input(
            "Target Body Fat % (optional)",
            min_value=1.0,
            max_value=50.0,
            value=st.session_state.almi_goal.get('target_body_fat_percentage', None),
            step=0.1,
            help="Leave empty for intelligent targeting based on health guidelines and feasibility",
            format="%.1f"
        )
        if almi_bf is not None and almi_bf > 0:
            st.session_state.almi_goal['target_body_fat_percentage'] = almi_bf
        else:
            st.session_state.almi_goal.pop('target_body_fat_percentage', None)
    
    with col2:
        st.markdown("**FFMI Goal**")
        ffmi_percentile = st.number_input(
            "Target FFMI Percentile",
            min_value=0.01,
            max_value=0.99,
            value=st.session_state.ffmi_goal.get('target_percentile', 0.75),
            step=0.05,
            help=get_metric_explanations()['tooltips']['target_percentile']
        )
        st.session_state.ffmi_goal['target_percentile'] = ffmi_percentile
        
        ffmi_age = st.text_input(
            "Target Age (or '?' for auto)",
            value=str(st.session_state.ffmi_goal.get('target_age', '?')),
            key="ffmi_target_age",
            help=get_metric_explanations()['tooltips']['goal_age']
        )
        st.session_state.ffmi_goal['target_age'] = ffmi_age
        
        ffmi_bf = st.number_input(
            "Target Body Fat % (optional)",
            min_value=1.0,
            max_value=50.0,
            value=st.session_state.ffmi_goal.get('target_body_fat_percentage', None),
            step=0.1,
            help="Leave empty for intelligent targeting based on health guidelines and feasibility",
            format="%.1f",
            key="ffmi_bf"
        )
        if ffmi_bf is not None and ffmi_bf > 0:
            st.session_state.ffmi_goal['target_body_fat_percentage'] = ffmi_bf
        else:
            st.session_state.ffmi_goal.pop('target_body_fat_percentage', None)


def display_results():
    """Display analysis results."""
    if st.session_state.analysis_results is None:
        st.info("👈 Enter your data to the left and run the analysis")
        return
    
    results = st.session_state.analysis_results
    df_results = results['df_results']
    goal_calculations = results['goal_calculations']
    figures = results['figures']
    
    st.subheader("📊 Analysis Results")
    
    # Display summary metrics for latest scan
    scan_data = df_results[~df_results['date_str'].str.contains('Goal', na=False)]
    if len(scan_data) > 0:
        latest_scan = scan_data.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current ALMI",
                f"{latest_scan['almi_kg_m2']:.2f} kg/m²",
                help="ALMI (Appendicular Lean Mass Index) measures lean muscle mass in arms and legs relative to height"
            )
        
        with col2:
            st.metric(
                "ALMI Percentile",
                f"{latest_scan['almi_percentile']:.1f}%",
                help=get_metric_explanations()['tooltips']['percentile']
            )
        
        with col3:
            st.metric(
                "Current FFMI",
                f"{latest_scan['ffmi_kg_m2']:.2f} kg/m²",
                help="FFMI (Fat-Free Mass Index) measures total lean body mass relative to height"
            )
        
        with col4:
            st.metric(
                "FFMI Percentile",
                f"{latest_scan['ffmi_percentile']:.1f}%",
                help=get_metric_explanations()['tooltips']['percentile']
            )
    
    # Display results in tabs
    tab1, tab2 = st.tabs(["🔥 ALMI Analysis", "💪 FFMI Analysis"])
    
    with tab1:
        # Show ALMI goal information if available - above the plot
        if 'almi' in goal_calculations:
            goal_info = format_goal_info(goal_calculations['almi'], 'almi')
            if goal_info:
                st.markdown(goal_info)
        
        # ALMI plot - full width
        
        # Create plotly figure with hover tooltips
        # Need to get LMS functions for plotting
        try:
            from core import load_lms_data, parse_gender
            user_info = st.session_state.user_info.copy()
            user_info['gender_code'] = parse_gender(user_info['gender'])
            
            # Load LMS functions
            lms_functions_local = {}
            lms_functions_local['almi_L'], lms_functions_local['almi_M'], lms_functions_local['almi_S'] = load_lms_data('appendicular_LMI', user_info['gender_code'])
            lms_functions_local['lmi_L'], lms_functions_local['lmi_M'], lms_functions_local['lmi_S'] = load_lms_data('LMI', user_info['gender_code'])
            
            if all(lms_functions_local.values()):
                almi_fig = create_plotly_metric_plot(df_results, 'ALMI', lms_functions_local, goal_calculations)
                st.plotly_chart(almi_fig, use_container_width=True)
            else:
                st.error("Could not load LMS data for plotting")
        except Exception as e:
            st.error(f"Error creating interactive plot: {e}")
            # Fallback to matplotlib if plotly fails
            st.pyplot(figures['ALMI'])
        
        # ALMI-focused data table
        st.subheader("📋 ALMI Results Table")
        
        almi_columns = [
            'date_str', 'age_at_scan', 
            'total_weight_lbs', 'total_lean_mass_lbs', 'fat_mass_lbs', 'body_fat_percentage',
            'almi_kg_m2', 'almi_percentile', 'almi_z_score'
        ]
        
        almi_names = [
            'Date', 'Age', 'Weight (lbs)', 'Lean Mass (lbs)', 'Fat Mass (lbs)', 'Body Fat %',
            'ALMI (kg/m²)', 'ALMI Percentile', 'ALMI Z-Score'
        ]
        
        # Check which columns exist in the dataframe
        available_almi_columns = [col for col in almi_columns if col in df_results.columns]
        available_almi_names = [almi_names[i] for i, col in enumerate(almi_columns) if col in df_results.columns]
        
        df_almi = df_results[available_almi_columns].copy()
        df_almi.columns = available_almi_names
        
        # Format numeric columns
        for col in df_almi.columns:
            if df_almi[col].dtype in ['float64', 'int64']:
                if 'Percentile' in col:
                    df_almi[col] = df_almi[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
                else:
                    df_almi[col] = df_almi[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        st.dataframe(df_almi, use_container_width=True)
    
    with tab2:
        # Show FFMI goal information if available - above the plot
        if 'ffmi' in goal_calculations:
            goal_info = format_goal_info(goal_calculations['ffmi'], 'ffmi')
            if goal_info:
                st.markdown(goal_info)
        
        # FFMI plot - full width
        
        # Create plotly figure with hover tooltips
        try:
            from core import load_lms_data, parse_gender
            user_info = st.session_state.user_info.copy()
            user_info['gender_code'] = parse_gender(user_info['gender'])
            
            # Load LMS functions (reuse from tab1 if needed, but reload for clarity)
            lms_functions_local = {}
            lms_functions_local['almi_L'], lms_functions_local['almi_M'], lms_functions_local['almi_S'] = load_lms_data('appendicular_LMI', user_info['gender_code'])
            lms_functions_local['lmi_L'], lms_functions_local['lmi_M'], lms_functions_local['lmi_S'] = load_lms_data('LMI', user_info['gender_code'])
            
            if all(lms_functions_local.values()):
                ffmi_fig = create_plotly_metric_plot(df_results, 'FFMI', lms_functions_local, goal_calculations)
                st.plotly_chart(ffmi_fig, use_container_width=True)
            else:
                st.error("Could not load LMS data for plotting")
        except Exception as e:
            st.error(f"Error creating interactive plot: {e}")
            # Fallback to matplotlib if plotly fails
            st.pyplot(figures['FFMI'])
        
        # FFMI-focused data table
        st.subheader("📋 FFMI Results Table")
        
        ffmi_columns = [
            'date_str', 'age_at_scan', 
            'total_weight_lbs', 'total_lean_mass_lbs', 'fat_mass_lbs', 'body_fat_percentage',
            'ffmi_kg_m2', 'ffmi_percentile', 'ffmi_z_score'
        ]
        
        ffmi_names = [
            'Date', 'Age', 'Weight (lbs)', 'Lean Mass (lbs)', 'Fat Mass (lbs)', 'Body Fat %',
            'FFMI (kg/m²)', 'FFMI Percentile', 'FFMI Z-Score'
        ]
        
        # Check which columns exist in the dataframe
        available_ffmi_columns = [col for col in ffmi_columns if col in df_results.columns]
        available_ffmi_names = [ffmi_names[i] for i, col in enumerate(ffmi_columns) if col in df_results.columns]
        
        df_ffmi = df_results[available_ffmi_columns].copy()
        df_ffmi.columns = available_ffmi_names
        
        # Format numeric columns
        for col in df_ffmi.columns:
            if df_ffmi[col].dtype in ['float64', 'int64']:
                if 'Percentile' in col:
                    df_ffmi[col] = df_ffmi[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
                else:
                    df_ffmi[col] = df_ffmi[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        st.dataframe(df_ffmi, use_container_width=True)
    
    # Download button for CSV
    csv_buffer = BytesIO()
    df_results.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    st.download_button(
        label="📥 Download Complete Results (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"dexa_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Download the complete analysis results including all calculated metrics"
    )


def main():
    """Main application function."""
    initialize_session_state()
    
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
        
        # Analyze button
        if st.button("🔬 Run Analysis", type="primary", use_container_width=True):
            run_analysis()
        
        # Share section
        st.divider()
        st.markdown("### 🔗 Share Configuration")
        
        # Show current URL status
        if 'share_url' in st.session_state and st.session_state.share_url:
            st.success("🔗 URL automatically updated!")
            st.text_area(
                "Current Share URL", 
                st.session_state.share_url, 
                height=80, 
                help="This URL updates automatically as you edit your configuration. Copy it to share with others."
            )
            
            # Show URL length info
            url_length = len(st.session_state.share_url)
            if url_length < 1500:
                st.info(f"📊 URL length: {url_length} characters (Excellent)")
            elif url_length < 2000:
                st.warning(f"📊 URL length: {url_length} characters (Good)")
            else:
                st.error(f"📊 URL length: {url_length} characters (May be too long)")
        else:
            st.info("📝 Enter some data above and the shareable URL will appear here automatically")
    
    with col2:
        # Right panel - Results
        display_results()
    
    # Auto-run analysis when data changes (if valid)
    if (st.session_state.user_info.get('birth_date') and 
        st.session_state.user_info.get('gender') and 
        len(st.session_state.scan_history) > 0 and
        any(scan.get('date') for scan in st.session_state.scan_history)):
        
        errors = validate_form_data()
        if not errors:
            run_analysis()


if __name__ == "__main__":
    main()