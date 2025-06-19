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
import pandas as pd
import numpy as np
import json
import tempfile
import os
from datetime import datetime
from io import BytesIO

# Import core analysis functions
from core import (
    run_analysis_from_data, 
    generate_fake_profile, 
    generate_fake_scans,
    get_metric_explanations,
    validate_user_input,
    parse_gender,
    load_config_json,
    extract_data_from_config
)

# Configure page
st.set_page_config(
    page_title="DEXA Body Composition Analysis",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
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


def initialize_session_state():
    """Initialize session state variables."""
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
        
        training_level = st.selectbox(
            "Training Level",
            options=['', 'novice', 'intermediate', 'advanced'],
            index=0,
            help=get_metric_explanations()['tooltips']['training_level']
        )
        st.session_state.user_info['training_level'] = training_level
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🎲 Random Profile"):
            fake_profile = generate_fake_profile()
            st.session_state.user_info.update(fake_profile)
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
                st.rerun()
            except Exception as e:
                st.error(f"Could not load example config: {e}")


def display_scan_history_form():
    """Display the DEXA scan history form using an editable data table."""
    st.subheader("🔬 DEXA Scan History")
    
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
        key="scan_history_editor",
        height=min(200 + (len(df) * 35), 400)  # Dynamic height based on number of rows
    )
    
    # Update session state with edited data
    st.session_state.scan_history = edited_df.to_dict('records')
    
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
    
    # Helper buttons
    col1, col2 = st.columns([1, 3])
    with col2:
        if st.button("🎲 Generate Fake Scans"):
            if st.session_state.user_info.get('gender') and st.session_state.user_info.get('height_in'):
                fake_scans = generate_fake_scans(st.session_state.user_info)
                st.session_state.scan_history = fake_scans
                st.rerun()
            else:
                st.error("Please set gender and height first")


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


def display_results():
    """Display analysis results."""
    if st.session_state.analysis_results is None:
        st.info("👆 Enter your data above and the analysis will appear here automatically")
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
                help=get_metric_explanations()['tooltips']['percentile']
            )
        
        with col2:
            st.metric(
                "ALMI Percentile",
                f"{latest_scan['almi_percentile']*100:.1f}%",
                help=get_metric_explanations()['tooltips']['percentile']
            )
        
        with col3:
            st.metric(
                "Current FFMI",
                f"{latest_scan['ffmi_kg_m2']:.2f} kg/m²",
                help=get_metric_explanations()['tooltips']['percentile']
            )
        
        with col4:
            st.metric(
                "FFMI Percentile",
                f"{latest_scan['ffmi_percentile']*100:.1f}%",
                help=get_metric_explanations()['tooltips']['percentile']
            )
    
    # Display plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ALMI Percentile Curves")
        st.pyplot(figures['ALMI'])
    
    with col2:
        st.subheader("FFMI Percentile Curves")
        st.pyplot(figures['FFMI'])
    
    # Display data table
    st.subheader("📋 Detailed Results Table")
    
    # Prepare display table
    display_columns = [
        'date_str', 'age_at_scan', 
        'total_weight_lbs', 'total_lean_mass_lbs', 'fat_mass_lbs', 'body_fat_percentage',
        'almi_kg_m2', 'ffmi_kg_m2'
    ]
    
    display_names = [
        'Date', 'Age', 'Weight (lbs)', 'Lean Mass (lbs)', 'Fat Mass (lbs)', 'Body Fat %',
        'ALMI (kg/m²)', 'FFMI (kg/m²)'
    ]
    
    df_display = df_results[display_columns].copy()
    df_display.columns = display_names
    
    # Format numeric columns
    for col in df_display.columns:
        if df_display[col].dtype in ['float64', 'int64']:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    st.dataframe(df_display, use_container_width=True)
    
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