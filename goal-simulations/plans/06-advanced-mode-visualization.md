# Part 6: Advanced Mode Visualization

## Overview
Create sophisticated, customizable visualizations for power users who want full control over forecast analysis. Build upon Quick Mode foundations with enhanced features, detailed controls, and professional-grade analysis tools.

## Core Requirements

### Configurable Confidence Bands
**Multiple confidence interval options**:
- **Preset options**: 50%, 80%, 90%, 95% confidence levels
- **Custom ranges**: User-defined percentile bands (e.g., 5th-95th)
- **Multiple bands**: Display 2-3 bands simultaneously for comparison
- **Band styling**: Distinct colors, transparency, and line styles

### Enhanced Phase Markers
**Detailed phase transition analysis**:
- **Drill-down capability**: Click markers for detailed phase information
- **Phase duration indicators**: Visual bars showing phase lengths
- **Rate annotations**: Display actual vs. target rates during phases
- **Custom marker styles**: Different shapes/colors for different phase types

### Spaghetti Plot Option
**Individual trajectory visualization**:
- **Random sampling**: Show 30-50 random trajectories from Monte Carlo runs
- **Trajectory highlighting**: Hover to highlight individual paths
- **Statistical overlay**: Percentile bands with individual runs visible
- **Toggle control**: Easy on/off for visual clarity

### Advanced Metrics Display
**Comprehensive data visualization**:
- **Secondary Y-axis**: Body fat percentage alongside weight
- **ALMI/FFMI tracking**: Dedicated plots for muscle-specific metrics
- **Percentile progress**: Show progression toward goal percentile over time
- **Rate indicators**: Visual display of actual gain/loss rates

## Visualization Components

### Multi-Panel Layout
```python
def create_advanced_mode_chart(
    forecast_plan: ForecastPlan,
    display_config: AdvancedDisplayConfig
) -> go.Figure:
    """Create comprehensive advanced visualization"""
    
    # Create subplots for multiple metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Weight & Body Fat %', 'ALMI Progress', 'FFMI Progress', 'Phase Analysis'),
        specs=[
            [{"secondary_y": True}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Panel 1: Weight with body fat overlay
    add_weight_bf_panel(fig, forecast_plan, display_config, row=1, col=1)
    
    # Panel 2: ALMI progression
    add_almi_panel(fig, forecast_plan, display_config, row=1, col=2)
    
    # Panel 3: FFMI progression  
    add_ffmi_panel(fig, forecast_plan, display_config, row=2, col=1)
    
    # Panel 4: Phase analysis
    add_phase_analysis_panel(fig, forecast_plan, display_config, row=2, col=2)
    
    # Configure overall layout
    configure_advanced_layout(fig, display_config)
    
    return fig
```

### Configurable Confidence Bands
```python
def add_confidence_bands(
    fig: go.Figure,
    forecast_plan: ForecastPlan,
    display_config: AdvancedDisplayConfig,
    row: int,
    col: int,
    metric: str
) -> None:
    """Add customizable confidence bands to visualization"""
    
    bands = display_config.confidence_bands  # e.g., [50, 80, 95]
    colors = ADVANCED_COLOR_PALETTE['confidence_bands']
    
    for i, confidence_level in enumerate(bands):
        lower_percentile = (100 - confidence_level) / 2
        upper_percentile = 100 - lower_percentile
        
        # Get percentile data
        lower_data = get_percentile_data(forecast_plan, lower_percentile, metric)
        upper_data = get_percentile_data(forecast_plan, upper_percentile, metric)
        
        # Add filled area
        fig.add_trace(
            go.Scatter(
                x=upper_data['weeks'],
                y=upper_data['values'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=lower_data['weeks'],
                y=lower_data['values'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=colors[i],
                name=f'{confidence_level}% Confidence',
                opacity=0.3 - (i * 0.1),  # Fade outer bands
                hovertemplate=f'<b>{confidence_level}% Band</b><br>Week: %{{x}}<br>{metric}: %{{y:.1f}}<extra></extra>'
            ),
            row=row, col=col
        )
```

### Spaghetti Plot Implementation
```python
def add_spaghetti_plot(
    fig: go.Figure,
    forecast_plan: ForecastPlan,
    display_config: AdvancedDisplayConfig,
    row: int,
    col: int,
    metric: str
) -> None:
    """Add individual trajectory overlay"""
    
    if not display_config.show_spaghetti:
        return
    
    # Sample random trajectories
    sample_size = min(display_config.spaghetti_count, len(forecast_plan.all_trajectories))
    sampled_trajectories = random.sample(forecast_plan.all_trajectories, sample_size)
    
    for i, trajectory in enumerate(sampled_trajectories):
        trajectory_data = extract_metric_from_trajectory(trajectory, metric)
        
        fig.add_trace(
            go.Scatter(
                x=trajectory_data['weeks'],
                y=trajectory_data['values'],
                mode='lines',
                line=dict(
                    color=ADVANCED_COLOR_PALETTE['spaghetti'],
                    width=1,
                    opacity=0.15
                ),
                showlegend=False,
                hoverinfo='skip',
                name=f'Trajectory {i+1}'
            ),
            row=row, col=col
        )
```

### Interactive Phase Markers
```python
def add_interactive_phase_markers(
    fig: go.Figure,
    forecast_plan: ForecastPlan,
    row: int,
    col: int,
    metric: str
) -> None:
    """Add clickable phase transition markers with detailed information"""
    
    for checkpoint in forecast_plan.median_checkpoints:
        marker_data = get_checkpoint_metric_value(checkpoint, metric)
        
        # Create detailed hover information
        hover_text = create_checkpoint_hover_text(checkpoint)
        
        # Determine marker style based on phase
        marker_config = get_phase_marker_config(checkpoint.phase)
        
        fig.add_trace(
            go.Scatter(
                x=[checkpoint.week],
                y=[marker_data],
                mode='markers',
                marker=dict(
                    symbol=marker_config['symbol'],
                    size=marker_config['size'],
                    color=marker_config['color'],
                    line=dict(width=2, color='white'),
                    opacity=0.9
                ),
                name=f'{checkpoint.phase.value.title()} Transition',
                showlegend=False,
                hovertemplate=hover_text,
                customdata=[checkpoint]
            ),
            row=row, col=col
        )

def create_checkpoint_hover_text(checkpoint: CheckpointData) -> str:
    """Create rich hover information for phase markers"""
    
    phase_emoji = "🔥" if checkpoint.phase == PhaseType.CUT else "💪"
    phase_name = "Cut Complete" if checkpoint.phase == PhaseType.CUT else "Bulk Peak"
    
    return (
        f'<b>{phase_emoji} {phase_name}</b><br>'
        f'<b>Timeline:</b> Week {checkpoint.week}<br>'
        f'<b>Weight:</b> {checkpoint.weight_lbs:.1f} lbs<br>'
        f'<b>Body Fat:</b> {checkpoint.body_fat_pct:.1f}%<br>'
        f'<b>Lean Mass:</b> {checkpoint.lean_mass_lbs:.1f} lbs<br>'
        f'<b>ALMI:</b> {checkpoint.almi:.2f} kg/m²<br>'
        f'<b>FFMI:</b> {checkpoint.ffmi:.2f} kg/m²<br>'
        f'<b>Goal Progress:</b> {checkpoint.percentile_progress:.0%}<br>'
        '<extra></extra>'
    )
```

### Advanced Metrics Panels
```python
def add_almi_panel(
    fig: go.Figure,
    forecast_plan: ForecastPlan,
    display_config: AdvancedDisplayConfig,
    row: int,
    col: int
) -> None:
    """Add ALMI-specific visualization panel"""
    
    # Add confidence bands for ALMI
    add_confidence_bands(fig, forecast_plan, display_config, row, col, 'almi')
    
    # Representative path
    rep_path_data = get_representative_path_data(forecast_plan, 'almi')
    fig.add_trace(
        go.Scatter(
            x=rep_path_data['weeks'],
            y=rep_path_data['values'],
            mode='lines',
            line=dict(
                color=ADVANCED_COLOR_PALETTE['representative_path'],
                width=3
            ),
            name='Most Likely ALMI',
            hovertemplate='<b>Week %{x}</b><br>ALMI: %{y:.2f} kg/m²<extra></extra>'
        ),
        row=row, col=col
    )
    
    # Goal line
    goal_almi = calculate_goal_almi(forecast_plan.user_profile, forecast_plan.goal_config)
    fig.add_hline(
        y=goal_almi,
        line=dict(
            color=ADVANCED_COLOR_PALETTE['goal_line'],
            width=2,
            dash='dash'
        ),
        annotation_text=f"Goal: {goal_almi:.2f} kg/m²",
        row=row, col=col
    )
    
    # Phase markers
    add_interactive_phase_markers(fig, forecast_plan, row, col, 'almi')
    
    # Percentile reference bands
    if display_config.show_percentile_references:
        add_percentile_reference_bands(fig, forecast_plan, row, col, 'almi')
```

### Color Palette System
```python
ADVANCED_COLOR_PALETTE = {
    'representative_path': '#2E86C1',
    'confidence_bands': [
        'rgba(46, 134, 193, 0.2)',  # 50% - darkest
        'rgba(46, 134, 193, 0.15)', # 80% - medium  
        'rgba(46, 134, 193, 0.1)'   # 95% - lightest
    ],
    'spaghetti': 'rgba(52, 73, 94, 0.15)',
    'goal_line': '#E74C3C',
    'phase_markers': {
        'cut': '#27AE60',
        'bulk': '#F39C12',
        'maintenance': '#8E44AD'
    },
    'percentile_references': {
        'p50': 'rgba(149, 165, 166, 0.3)',
        'p75': 'rgba(149, 165, 166, 0.2)',
        'p90': 'rgba(149, 165, 166, 0.1)'
    }
}
```

## Advanced Display Configuration

### Configuration Data Structure
```python
@dataclass
class AdvancedDisplayConfig:
    # Confidence band settings
    confidence_bands: List[int] = field(default_factory=lambda: [80])
    custom_percentiles: Optional[Tuple[int, int]] = None
    
    # Spaghetti plot settings
    show_spaghetti: bool = False
    spaghetti_count: int = 30
    spaghetti_opacity: float = 0.15
    
    # Panel configuration
    panels: List[str] = field(default_factory=lambda: ['weight_bf', 'almi', 'ffmi'])
    panel_layout: str = 'grid'  # 'grid', 'stacked', 'tabs'
    
    # Reference lines
    show_percentile_references: bool = True
    show_goal_lines: bool = True
    show_current_percentile: bool = True
    
    # Interaction settings
    enable_drill_down: bool = True
    show_rate_annotations: bool = False
    highlight_phase_durations: bool = False
    
    # Export settings
    chart_height: int = 600
    chart_width: Optional[int] = None
    export_format: str = 'png'  # 'png', 'svg', 'html'
```

### Control Panel Integration
```python
def create_visualization_controls(
    current_config: AdvancedDisplayConfig
) -> AdvancedDisplayConfig:
    """Create UI controls for visualization customization"""
    
    st.markdown("### 📊 Visualization Controls")
    
    # Confidence band controls
    with st.expander("📈 Confidence Bands", expanded=True):
        band_options = st.multiselect(
            "Select confidence levels",
            options=[50, 80, 90, 95],
            default=current_config.confidence_bands,
            help="Choose which confidence intervals to display"
        )
        
        custom_bands = st.checkbox(
            "Custom percentile range",
            value=current_config.custom_percentiles is not None
        )
        
        if custom_bands:
            col1, col2 = st.columns(2)
            with col1:
                lower = st.slider("Lower percentile", 5, 45, 10)
            with col2:
                upper = st.slider("Upper percentile", 55, 95, 90)
            custom_percentiles = (lower, upper)
        else:
            custom_percentiles = None
    
    # Spaghetti plot controls
    with st.expander("🍝 Individual Trajectories"):
        show_spaghetti = st.checkbox(
            "Show individual simulation runs",
            value=current_config.show_spaghetti,
            help="Display sample trajectories from Monte Carlo simulation"
        )
        
        if show_spaghetti:
            spaghetti_count = st.slider(
                "Number of trajectories",
                min_value=10,
                max_value=100,
                value=current_config.spaghetti_count,
                step=10
            )
            
            spaghetti_opacity = st.slider(
                "Trajectory opacity",
                min_value=0.05,
                max_value=0.5,
                value=current_config.spaghetti_opacity,
                step=0.05
            )
        else:
            spaghetti_count = current_config.spaghetti_count
            spaghetti_opacity = current_config.spaghetti_opacity
    
    # Panel configuration
    with st.expander("📋 Panel Layout"):
        panels = st.multiselect(
            "Select panels to display",
            options=['weight_bf', 'almi', 'ffmi', 'phase_analysis'],
            default=current_config.panels,
            format_func=lambda x: {
                'weight_bf': 'Weight & Body Fat %',
                'almi': 'ALMI Progress',
                'ffmi': 'FFMI Progress', 
                'phase_analysis': 'Phase Analysis'
            }[x]
        )
        
        layout = st.radio(
            "Panel arrangement",
            options=['grid', 'stacked', 'tabs'],
            index=['grid', 'stacked', 'tabs'].index(current_config.panel_layout),
            horizontal=True
        )
    
    # Reference line controls
    with st.expander("📏 Reference Lines"):
        show_percentile_refs = st.checkbox(
            "Population percentile references",
            value=current_config.show_percentile_references
        )
        
        show_goal_lines = st.checkbox(
            "Goal target lines",
            value=current_config.show_goal_lines
        )
        
        show_current_pct = st.checkbox(
            "Current percentile indicator",
            value=current_config.show_current_percentile
        )
    
    # Build updated configuration
    return AdvancedDisplayConfig(
        confidence_bands=band_options,
        custom_percentiles=custom_percentiles,
        show_spaghetti=show_spaghetti,
        spaghetti_count=spaghetti_count,
        spaghetti_opacity=spaghetti_opacity,
        panels=panels,
        panel_layout=layout,
        show_percentile_references=show_percentile_refs,
        show_goal_lines=show_goal_lines,
        show_current_percentile=show_current_pct
    )
```

## Performance Optimization

### Efficient Data Processing
```python
def optimize_trajectory_rendering(
    trajectories: List[List[SimulationState]],
    display_config: AdvancedDisplayConfig
) -> Dict[str, Any]:
    """Optimize trajectory data for visualization performance"""
    
    # Downsample for spaghetti plots if too many points
    if display_config.show_spaghetti and len(trajectories[0]) > 200:
        downsampled_trajectories = [
            downsample_trajectory(traj, target_points=200) 
            for traj in trajectories[:display_config.spaghetti_count]
        ]
    else:
        downsampled_trajectories = trajectories[:display_config.spaghetti_count]
    
    # Pre-calculate percentile bands for all requested confidence levels
    percentile_data = {}
    for confidence in display_config.confidence_bands:
        lower = (100 - confidence) / 2
        upper = 100 - lower
        percentile_data[confidence] = {
            'lower': calculate_percentile_trajectory(trajectories, lower),
            'upper': calculate_percentile_trajectory(trajectories, upper)
        }
    
    return {
        'spaghetti_trajectories': downsampled_trajectories,
        'percentile_bands': percentile_data,
        'representative_path': find_representative_trajectory(trajectories)
    }
```

### Lazy Loading for Large Datasets
```python
def create_progressive_chart(
    forecast_plan: ForecastPlan,
    display_config: AdvancedDisplayConfig
) -> go.Figure:
    """Create chart with progressive loading for better performance"""
    
    # Start with essential elements
    fig = create_base_chart(forecast_plan, display_config)
    
    # Add core elements immediately
    add_representative_path(fig, forecast_plan)
    add_primary_confidence_band(fig, forecast_plan, display_config)
    
    # Progressive enhancement
    if display_config.show_spaghetti:
        # Load spaghetti plot in background
        st.session_state.pending_spaghetti = True
    
    if len(display_config.confidence_bands) > 1:
        # Load additional bands progressively
        st.session_state.pending_additional_bands = True
    
    return fig
```

## Testing Strategy

### Visual Regression Testing
- **Chart appearance**: Automated screenshot comparisons across configurations
- **Interactive elements**: Verify hover, click, and zoom functionality
- **Performance testing**: Measure rendering time with large datasets
- **Cross-browser compatibility**: Test advanced Plotly features

### Configuration Testing
- **All combinations**: Test major configuration option combinations
- **Edge cases**: Verify behavior with extreme settings
- **State persistence**: Ensure configuration survives page reloads
- **Default handling**: Test graceful fallbacks for invalid configurations

## Implementation Tools

### Visualization Libraries
- **Plotly advanced features**: Subplots, secondary axes, custom interactions
- **NumPy**: Efficient percentile calculations and data processing
- **SciPy**: Advanced interpolation and statistical functions

### Performance Libraries
- **Pandas**: Efficient data manipulation for large datasets
- **Numba**: JIT compilation for computationally intensive operations
- **Dask**: Parallel processing for very large simulation datasets

## Success Criteria

### Functionality Requirements
- ✅ All configuration options work correctly
- ✅ Interactive elements respond smoothly
- ✅ Charts render correctly across different panel layouts
- ✅ Performance remains acceptable with maximum settings

### User Experience Requirements
- ✅ Configuration changes update visualizations in real-time
- ✅ Complex charts remain readable and professional
- ✅ Hover information is comprehensive but not overwhelming
- ✅ Export functionality produces publication-quality outputs

### Technical Requirements
- ✅ Memory efficient with large datasets (2000+ trajectories)
- ✅ Responsive design works across screen sizes
- ✅ State management preserves user preferences
- ✅ Extensible for future visualization enhancements

## Integration Points

This visualization system:
- **Extends**: Quick Mode Visualization (Part 4) with advanced features
- **Integrates with**: Advanced Mode UI (Part 7) for control panel
- **Consumes**: Forecast API Layer (Part 3) for rich data structures
- **Provides**: Export capabilities and detailed analysis for power users