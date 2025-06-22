# Part 7: Advanced Mode UI Integration

## Overview
Create a sophisticated, multi-step wizard interface for power users who want full control over forecast parameters. Provide comprehensive customization options while maintaining usability through progressive disclosure and intelligent defaults.

## Core Requirements

### Multi-Step Wizard Flow
**Guided progression through complex configuration**:
1. **Current State Review**: Validate and augment user profile data
2. **Goal Refinement**: Advanced goal setting with multiple scenarios  
3. **Training Overrides**: Custom training progression and variance factors
4. **Simulation Settings**: Monte Carlo parameters and computational options
5. **Preview & Customize**: Final review with real-time parameter adjustment

### Comprehensive Parameter Control
**Full access to simulation internals**:
- **P-ratio overrides**: Custom lean mass partitioning assumptions
- **Rate customization**: Training-level-specific gain/loss rates
- **Template selection**: Manual override of automatic template choice
- **Variance adjustment**: Personal variance factors from scan history
- **Timeline constraints**: Custom phase duration limits

### Real-Time Parameter Sensitivity
**Immediate feedback on parameter changes**:
- **Preview simulations**: Fast 500-run previews during parameter adjustment
- **Sensitivity analysis**: Show impact of parameter changes on outcomes
- **Range validation**: Real-time bounds checking with helpful guidance
- **Conflict resolution**: Automatic detection and resolution of conflicting settings

### Export and Analysis Tools
**Professional-grade data access**:
- **Raw data export**: CSV/JSON download of complete simulation results
- **Report generation**: PDF reports with charts and analysis
- **Scenario comparison**: Side-by-side comparison of different parameter sets
- **API access**: Programmatic access to simulation engine for custom analysis

## Wizard Step Implementation

### Step 1: Current State Review
```python
def render_current_state_step() -> UserProfile:
    """Review and enhance user profile data"""
    
    st.markdown("## 👤 Current State Review")
    st.markdown("Let's review and enhance your profile data for advanced forecasting.")
    
    # Load current profile
    current_profile = st.session_state.user_profile
    
    # Enhanced user information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Basic Information")
        birth_date = st.date_input(
            "Birth Date",
            value=datetime.strptime(current_profile.birth_date, "%m/%d/%Y").date(),
            help="Used for age-adjusted percentile calculations"
        )
        
        height_in = st.number_input(
            "Height (inches)",
            value=current_profile.height_in,
            min_value=36.0,
            max_value=96.0,
            step=0.5,
            help="Height in inches for ALMI/FFMI calculations"
        )
        
        gender = st.selectbox(
            "Gender",
            options=["male", "female"],
            index=0 if current_profile.gender == "male" else 1,
            help="Affects LMS reference curves and rate calculations"
        )
    
    with col2:
        st.markdown("### Training Information") 
        training_level = st.selectbox(
            "Training Level",
            options=["novice", "intermediate", "advanced"],
            index=["novice", "intermediate", "advanced"].index(current_profile.training_level),
            help="Determines muscle gain rates and outcome variance"
        )
        
        training_years = st.number_input(
            "Years of Serious Training",
            value=get_estimated_training_years(current_profile),
            min_value=0.0,
            max_value=50.0,
            step=0.5,
            help="Used for variance calculations and rate adjustments"
        )
        
        # Advanced training metrics
        with st.expander("⚙️ Advanced Training Metrics"):
            strength_progression = st.selectbox(
                "Recent Strength Progression",
                options=["rapid", "steady", "slow", "plateau"],
                index=1,  # Default to steady
                help="Affects variance and rate estimates"
            )
            
            consistency_rating = st.slider(
                "Training Consistency (1-10)",
                min_value=1,
                max_value=10,
                value=7,
                help="Higher consistency = lower outcome variance"
            )
    
    # Scan history review
    st.markdown("### 📊 DEXA Scan History")
    render_scan_history_editor(current_profile.scan_history)
    
    # Enhanced profile creation
    enhanced_profile = create_enhanced_profile(
        birth_date, height_in, gender, training_level,
        training_years, strength_progression, consistency_rating,
        current_profile.scan_history
    )
    
    return enhanced_profile

def render_scan_history_editor(scan_history: List[DexaScan]) -> List[DexaScan]:
    """Interactive scan history editor with validation"""
    
    # Display current scans in editable table
    scan_df = pd.DataFrame([
        {
            'Date': scan.date,
            'Weight (lbs)': scan.total_weight_lbs,
            'Lean Mass (lbs)': scan.total_lean_mass_lbs,
            'Fat Mass (lbs)': scan.fat_mass_lbs,
            'Body Fat %': scan.body_fat_percentage,
            'Arms Lean (lbs)': scan.arms_lean_lbs,
            'Legs Lean (lbs)': scan.legs_lean_lbs
        }
        for scan in scan_history
    ])
    
    edited_df = st.data_editor(
        scan_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Date": st.column_config.DateColumn("Date", format="MM/DD/YYYY"),
            "Weight (lbs)": st.column_config.NumberColumn("Weight", min_value=50, max_value=500, step=0.1),
            "Body Fat %": st.column_config.NumberColumn("Body Fat %", min_value=3, max_value=60, step=0.1),
        },
        key="scan_editor"
    )
    
    # Validation feedback
    validation_issues = validate_scan_data(edited_df)
    if validation_issues:
        st.warning("⚠️ Scan data issues detected:")
        for issue in validation_issues:
            st.markdown(f"• {issue}")
    
    return convert_df_to_scans(edited_df)
```

### Step 2: Goal Refinement
```python
def render_goal_refinement_step(user_profile: UserProfile) -> List[GoalScenario]:
    """Advanced goal setting with multiple scenarios"""
    
    st.markdown("## 🎯 Goal Refinement")
    st.markdown("Define detailed goals and explore multiple scenarios.")
    
    # Multiple scenario support
    scenario_count = st.number_input(
        "Number of scenarios to analyze",
        min_value=1,
        max_value=5,
        value=1,
        help="Compare different goal targets or timelines"
    )
    
    scenarios = []
    
    for i in range(scenario_count):
        with st.expander(f"📋 Scenario {i+1}", expanded=i==0):
            scenario = render_single_goal_scenario(user_profile, i)
            scenarios.append(scenario)
    
    return scenarios

def render_single_goal_scenario(user_profile: UserProfile, scenario_index: int) -> GoalScenario:
    """Configure individual goal scenario"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        metric_type = st.selectbox(
            "Primary Metric",
            options=["almi", "ffmi", "both"],
            index=0,
            key=f"metric_{scenario_index}",
            help="Which metric to optimize for"
        )
        
        if metric_type in ["almi", "both"]:
            almi_target = st.slider(
                "ALMI Target Percentile",
                min_value=50,
                max_value=99,
                value=75,
                key=f"almi_target_{scenario_index}",
                help="Target ALMI percentile for your age/gender"
            )
        else:
            almi_target = None
            
        if metric_type in ["ffmi", "both"]:
            ffmi_target = st.slider(
                "FFMI Target Percentile", 
                min_value=50,
                max_value=99,
                value=75,
                key=f"ffmi_target_{scenario_index}",
                help="Target FFMI percentile for your age/gender"
            )
        else:
            ffmi_target = None
    
    with col2:
        timeline_constraint = st.selectbox(
            "Timeline Preference",
            options=["flexible", "target_date", "max_duration"],
            key=f"timeline_{scenario_index}",
            help="How to handle timeline constraints"
        )
        
        if timeline_constraint == "target_date":
            target_date = st.date_input(
                "Target Achievement Date",
                value=datetime.now().date() + timedelta(days=365),
                key=f"target_date_{scenario_index}"
            )
            max_weeks = (target_date - datetime.now().date()).days // 7
        elif timeline_constraint == "max_duration":
            max_weeks = st.number_input(
                "Maximum Duration (weeks)",
                min_value=12,
                max_value=260,  # 5 years
                value=104,  # 2 years
                key=f"max_weeks_{scenario_index}"
            )
        else:
            max_weeks = None
        
        # Goal feasibility analysis
        feasibility = analyze_goal_feasibility(user_profile, almi_target, ffmi_target, max_weeks)
        render_feasibility_feedback(feasibility)
    
    return GoalScenario(
        name=f"Scenario {scenario_index + 1}",
        almi_target=almi_target,
        ffmi_target=ffmi_target,
        max_weeks=max_weeks,
        feasibility=feasibility
    )
```

### Step 3: Training Overrides  
```python
def render_training_overrides_step(user_profile: UserProfile) -> TrainingConfig:
    """Advanced training parameter customization"""
    
    st.markdown("## 💪 Training Overrides")
    st.markdown("Customize training progression and variance assumptions.")
    
    # Rate customization
    st.markdown("### 📈 Muscle Gain Rates")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Beginner Phase**")
        beginner_rate = st.number_input(
            "Monthly lean gain (lbs)",
            value=get_default_rate(user_profile, "novice"),
            min_value=0.5,
            max_value=8.0,
            step=0.1,
            key="beginner_rate",
            help="Expected monthly muscle gain for beginners"
        )
        
        beginner_duration = st.number_input(
            "Duration (months)",
            value=6,
            min_value=1,
            max_value=24,
            key="beginner_duration",
            help="How long beginner gains last"
        )
    
    with col2:
        st.markdown("**Intermediate Phase**")
        intermediate_rate = st.number_input(
            "Monthly lean gain (lbs)",
            value=get_default_rate(user_profile, "intermediate"),
            min_value=0.2,
            max_value=4.0,
            step=0.1,
            key="intermediate_rate"
        )
        
        intermediate_duration = st.number_input(
            "Duration (months)",
            value=24,
            min_value=6,
            max_value=60,
            key="intermediate_duration"
        )
    
    with col3:
        st.markdown("**Advanced Phase**")
        advanced_rate = st.number_input(
            "Monthly lean gain (lbs)",
            value=get_default_rate(user_profile, "advanced"),
            min_value=0.1,
            max_value=2.0,
            step=0.05,
            key="advanced_rate"
        )
    
    # P-ratio customization
    st.markdown("### ⚖️ P-Ratio Overrides")
    
    use_custom_pratios = st.checkbox(
        "Override default P-ratio assumptions",
        help="Customize lean mass partitioning during weight changes"
    )
    
    if use_custom_pratios:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Bulking P-Ratios**")
            bulk_lean_ratio = st.slider(
                "Lean mass ratio (bulking)",
                min_value=0.30,
                max_value=0.70,
                value=0.50,
                step=0.05,
                help="Proportion of weight gain as lean mass"
            )
        
        with col2:
            st.markdown("**Cutting P-Ratios**") 
            cut_lean_loss_high_bf = st.slider(
                "Lean loss ratio (high BF cut)",
                min_value=0.10,
                max_value=0.40,
                value=0.25,
                step=0.05,
                help="Proportion of weight loss as lean mass (high body fat)"
            )
            
            cut_lean_loss_mod_bf = st.slider(
                "Lean loss ratio (moderate BF cut)",
                min_value=0.20,
                max_value=0.50,
                value=0.35,
                step=0.05,
                help="Proportion of weight loss as lean mass (moderate body fat)"
            )
    else:
        # Use research defaults
        bulk_lean_ratio = 0.50
        cut_lean_loss_high_bf = 0.25
        cut_lean_loss_mod_bf = 0.35
    
    # Variance customization
    st.markdown("### 📊 Outcome Variance")
    
    variance_source = st.radio(
        "Variance estimation method",
        options=["scan_history", "training_level", "custom"],
        help="How to estimate outcome variability"
    )
    
    if variance_source == "custom":
        custom_variance = st.slider(
            "Variance factor",
            min_value=0.05,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Higher values = more variable outcomes"
        )
    else:
        custom_variance = None
    
    return TrainingConfig(
        rates={
            "beginner": (beginner_rate, beginner_duration),
            "intermediate": (intermediate_rate, intermediate_duration), 
            "advanced": (advanced_rate, None)
        },
        p_ratios={
            "bulk_lean": bulk_lean_ratio,
            "cut_high_bf": cut_lean_loss_high_bf,
            "cut_mod_bf": cut_lean_loss_mod_bf
        },
        variance_method=variance_source,
        custom_variance=custom_variance
    )
```

### Step 4: Simulation Settings
```python
def render_simulation_settings_step() -> SimulationConfig:
    """Configure Monte Carlo simulation parameters"""
    
    st.markdown("## ⚙️ Simulation Settings")
    st.markdown("Fine-tune computational parameters for your analysis.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎲 Monte Carlo Parameters")
        
        run_count = st.selectbox(
            "Number of simulation runs", 
            options=[500, 1000, 2000, 5000],
            index=2,  # Default to 2000
            help="More runs = better statistics but slower computation"
        )
        
        convergence_threshold = st.number_input(
            "Convergence threshold",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.3f",
            help="Stop early if results converge (experimental)"
        )
        
        random_seed = st.number_input(
            "Random seed (optional)",
            min_value=0,
            max_value=999999,
            value=42,
            help="For reproducible results"
        )
        
        use_seed = st.checkbox("Use fixed seed", value=False)
    
    with col2:
        st.markdown("### 🚀 Performance Options")
        
        parallel_processing = st.checkbox(
            "Enable parallel processing",
            value=True,
            help="Use multiple CPU cores for faster computation"
        )
        
        if parallel_processing:
            max_workers = st.slider(
                "Maximum worker threads",
                min_value=1,
                max_value=os.cpu_count(),
                value=min(4, os.cpu_count()),
                help="More workers = faster but more CPU usage"
            )
        else:
            max_workers = 1
        
        memory_limit = st.selectbox(
            "Memory usage limit",
            options=["conservative", "moderate", "aggressive"],
            index=1,
            help="Trade memory usage for computation speed"
        )
        
        # Advanced options
        with st.expander("🔬 Advanced Simulation Options"):
            time_step = st.selectbox(
                "Simulation time step",
                options=["daily", "weekly", "biweekly"],
                index=1,  # Weekly default
                help="Finer steps = more accuracy but slower computation"
            )
            
            adaptive_stepping = st.checkbox(
                "Adaptive time stepping",
                value=False,
                help="Automatically adjust step size during phases (experimental)"
            )
            
            early_termination = st.checkbox(
                "Early goal termination",
                value=True,
                help="Stop simulation when goal is reached"
            )
    
    # Performance preview
    estimated_time = estimate_computation_time(run_count, max_workers, time_step)
    estimated_memory = estimate_memory_usage(run_count, time_step)
    
    st.info(f"⏱️ **Estimated computation time:** {estimated_time:.1f} seconds\n\n"
            f"💾 **Estimated memory usage:** {estimated_memory:.0f} MB")
    
    return SimulationConfig(
        run_count=run_count,
        convergence_threshold=convergence_threshold,
        random_seed=random_seed if use_seed else None,
        parallel_processing=parallel_processing,
        max_workers=max_workers,
        memory_limit=memory_limit,
        time_step=time_step,
        adaptive_stepping=adaptive_stepping,
        early_termination=early_termination
    )
```

### Step 5: Preview & Customize
```python
def render_preview_step(
    user_profile: UserProfile,
    goal_scenarios: List[GoalScenario],
    training_config: TrainingConfig,
    simulation_config: SimulationConfig
) -> None:
    """Final preview with real-time customization"""
    
    st.markdown("## 👀 Preview & Customize")
    st.markdown("Review your configuration and fine-tune with live preview.")
    
    # Configuration summary
    with st.expander("📋 Configuration Summary", expanded=False):
        render_configuration_summary(user_profile, goal_scenarios, training_config, simulation_config)
    
    # Run preview simulation
    if st.button("🔍 Generate Preview", type="primary"):
        preview_config = simulation_config.copy()
        preview_config.run_count = 500  # Fast preview
        
        with st.spinner("Running preview simulation..."):
            preview_results = run_preview_simulation(
                user_profile, goal_scenarios[0], training_config, preview_config
            )
        
        st.session_state.preview_results = preview_results
    
    # Display preview results with real-time controls
    if "preview_results" in st.session_state:
        render_live_preview_interface(st.session_state.preview_results)
    
    # Final action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬅️ Back to Edit", type="secondary"):
            st.session_state.wizard_step = 4
            st.rerun()
    
    with col2:
        if st.button("🎯 Run Full Analysis", type="primary"):
            run_full_advanced_analysis(user_profile, goal_scenarios, training_config, simulation_config)
    
    with col3:
        if st.button("💾 Save Configuration"):
            save_advanced_configuration(user_profile, goal_scenarios, training_config, simulation_config)

def render_live_preview_interface(preview_results: ForecastPlan) -> None:
    """Interactive preview with real-time parameter adjustment"""
    
    # Quick parameter adjustments
    st.markdown("### ⚡ Quick Adjustments")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rate_multiplier = st.slider(
            "Rate multiplier",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Adjust all rates proportionally"
        )
    
    with col2:
        variance_multiplier = st.slider(
            "Variance multiplier",
            min_value=0.2,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Adjust outcome uncertainty"
        )
    
    with col3:
        if st.button("🔄 Update Preview"):
            # Re-run preview with adjusted parameters
            updated_preview = update_preview_with_adjustments(
                preview_results, rate_multiplier, variance_multiplier
            )
            st.session_state.preview_results = updated_preview
            st.rerun()
    
    # Preview visualization
    st.markdown("### 📊 Preview Results")
    
    # Use advanced visualization with preview data
    display_config = AdvancedDisplayConfig(
        confidence_bands=[80],
        show_spaghetti=True,
        spaghetti_count=20,
        panels=['weight_bf', 'almi']
    )
    
    preview_chart = create_advanced_mode_chart(preview_results, display_config)
    st.plotly_chart(preview_chart, use_container_width=True)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Timeline", f"{preview_results.total_duration_weeks} weeks")
    
    with col2:
        st.metric("Phases", f"{preview_results.total_phases}")
    
    with col3:
        final_percentile = preview_results.median_checkpoints[-1].percentile_progress if preview_results.median_checkpoints else 0
        st.metric("Final Progress", f"{final_percentile:.0%}")
```

## Wizard Navigation System

### State Management
```python
@dataclass
class WizardState:
    step: int = 1
    total_steps: int = 5
    user_profile: Optional[UserProfile] = None
    goal_scenarios: List[GoalScenario] = field(default_factory=list)
    training_config: Optional[TrainingConfig] = None
    simulation_config: Optional[SimulationConfig] = None
    preview_results: Optional[ForecastPlan] = None
    
    def can_proceed_to_step(self, target_step: int) -> bool:
        """Check if wizard can proceed to target step"""
        if target_step <= self.step:
            return True  # Can always go back
        
        # Check prerequisites for forward navigation
        if target_step >= 2 and not self.user_profile:
            return False
        if target_step >= 3 and not self.goal_scenarios:
            return False
        if target_step >= 4 and not self.training_config:
            return False
        if target_step >= 5 and not self.simulation_config:
            return False
        
        return True

def render_wizard_navigation(wizard_state: WizardState) -> None:
    """Render wizard navigation with progress indication"""
    
    steps = [
        "👤 Current State",
        "🎯 Goals", 
        "💪 Training",
        "⚙️ Simulation",
        "👀 Preview"
    ]
    
    # Progress bar
    progress = wizard_state.step / wizard_state.total_steps
    st.progress(progress)
    
    # Step indicators
    cols = st.columns(wizard_state.total_steps)
    
    for i, (col, step_name) in enumerate(zip(cols, steps)):
        with col:
            step_num = i + 1
            
            if step_num == wizard_state.step:
                # Current step
                st.markdown(f"**{step_name}**")
            elif step_num < wizard_state.step:
                # Completed step
                if wizard_state.can_proceed_to_step(step_num):
                    if st.button(f"✅ {step_name}", key=f"nav_{step_num}"):
                        wizard_state.step = step_num
                        st.rerun()
                else:
                    st.markdown(f"✅ {step_name}")
            else:
                # Future step
                if wizard_state.can_proceed_to_step(step_num):
                    if st.button(f"⭕ {step_name}", key=f"nav_{step_num}"):
                        wizard_state.step = step_num
                        st.rerun()
                else:
                    st.markdown(f"⭕ {step_name}")
    
    st.markdown("---")
```

## Export and Analysis Tools

### Data Export Interface
```python
def render_export_interface(forecast_plan: ForecastPlan) -> None:
    """Comprehensive data export options"""
    
    st.markdown("### 📤 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Raw Data Export**")
        
        if st.button("📊 Download CSV"):
            csv_data = export_to_csv(forecast_plan)
            st.download_button(
                "💾 Download Complete Dataset",
                data=csv_data,
                file_name=f"forecast_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        if st.button("📋 Download JSON"):
            json_data = export_to_json(forecast_plan)
            st.download_button(
                "💾 Download JSON Data",
                data=json_data,
                file_name=f"forecast_config_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
    
    with col2:
        st.markdown("**Report Generation**")
        
        report_format = st.selectbox(
            "Report format",
            options=["pdf", "html", "docx"],
            help="Choose format for detailed analysis report"
        )
        
        include_charts = st.checkbox("Include charts", value=True)
        include_raw_data = st.checkbox("Include raw data", value=False)
        
        if st.button("📄 Generate Report"):
            report_data = generate_analysis_report(
                forecast_plan, report_format, include_charts, include_raw_data
            )
            
            st.download_button(
                f"💾 Download {report_format.upper()} Report",
                data=report_data,
                file_name=f"forecast_report_{datetime.now().strftime('%Y%m%d_%H%M')}.{report_format}",
                mime=get_mime_type(report_format)
            )

def export_to_csv(forecast_plan: ForecastPlan) -> str:
    """Export complete forecast data to CSV format"""
    
    # Prepare comprehensive dataset
    data_rows = []
    
    # Representative path
    for state in forecast_plan.representative_path:
        data_rows.append({
            'Type': 'Representative',
            'Week': state.week,
            'Weight_lbs': state.weight_lbs,
            'Lean_Mass_lbs': state.lean_mass_lbs,
            'Fat_Mass_lbs': state.fat_mass_lbs,
            'Body_Fat_Pct': state.body_fat_pct,
            'ALMI': state.almi,
            'FFMI': state.ffmi,
            'Phase': state.phase.value
        })
    
    # Percentile bands
    for percentile, trajectory in forecast_plan.percentile_bands.items():
        for state in trajectory:
            data_rows.append({
                'Type': f'P{percentile}',
                'Week': state.week,
                'Weight_lbs': state.weight_lbs,
                'Lean_Mass_lbs': state.lean_mass_lbs,
                'Fat_Mass_lbs': state.fat_mass_lbs,
                'Body_Fat_Pct': state.body_fat_pct,
                'ALMI': state.almi,
                'FFMI': state.ffmi,
                'Phase': state.phase.value
            })
    
    # Checkpoints
    for checkpoint in forecast_plan.median_checkpoints:
        data_rows.append({
            'Type': 'Checkpoint',
            'Week': checkpoint.week,
            'Weight_lbs': checkpoint.weight_lbs,
            'Lean_Mass_lbs': checkpoint.lean_mass_lbs,
            'Fat_Mass_lbs': checkpoint.fat_mass_lbs,
            'Body_Fat_Pct': checkpoint.body_fat_pct,
            'ALMI': checkpoint.almi,
            'FFMI': checkpoint.ffmi,
            'Phase': checkpoint.phase.value,
            'Percentile_Progress': checkpoint.percentile_progress
        })
    
    # Convert to CSV
    df = pd.DataFrame(data_rows)
    return df.to_csv(index=False)
```

## Testing Strategy

### Wizard Flow Testing
- **Step navigation**: Test forward/backward navigation with various data states
- **Validation handling**: Verify proper error display and prevention of invalid progression
- **State persistence**: Ensure data survives page reloads and navigation
- **Performance**: Test wizard responsiveness with large datasets

### Parameter Validation Testing  
- **Range checking**: Verify all numeric inputs respect bounds
- **Conflict detection**: Test automatic resolution of conflicting parameters
- **Sensitivity analysis**: Validate parameter impact calculations
- **Export functionality**: Test all export formats with various configurations

## Implementation Tools

### Streamlit Advanced Features
- **Multi-page apps**: Complex wizard navigation
- **Session state**: Preserve wizard state across interactions
- **File downloads**: Export functionality
- **Custom components**: Advanced form controls

### Data Processing
- **Pandas**: Data manipulation for export and analysis
- **ReportLab**: PDF report generation
- **Jinja2**: HTML template rendering for reports
- **openpyxl**: Excel export functionality

## Success Criteria

### User Experience
- ✅ Wizard guides users through complex configuration without confusion
- ✅ Parameter changes provide immediate feedback on impact
- ✅ Export functionality produces professional-quality outputs
- ✅ Advanced features don't overwhelm but provide value to power users

### Functionality
- ✅ All configuration options work correctly and interact properly
- ✅ Preview simulations update responsively to parameter changes
- ✅ Export formats contain complete and accurate data
- ✅ Wizard state management prevents data loss and invalid configurations

### Technical Requirements
- ✅ Performance remains acceptable even with maximum parameter complexity
- ✅ Memory usage scales appropriately with configuration complexity
- ✅ Export files are well-formatted and machine-readable
- ✅ Integration with visualization system is seamless

## Integration Points

This Advanced Mode UI:
- **Consumes**: All previous parts (1-6) for complete functionality
- **Extends**: Quick Mode experience with sophisticated controls
- **Provides**: Professional-grade analysis tools for coaches and researchers
- **Enables**: Future API access and programmatic simulation control