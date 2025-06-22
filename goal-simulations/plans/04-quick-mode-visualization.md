# Part 4: Quick Mode Visualization

## Overview
Create clean, intuitive visualizations for the Quick Forecast mode that provide immediate value to mainstream users without overwhelming complexity. Focus on essential insights with professional presentation.

## Core Requirements

### Simplified Fan Chart
**Fixed 80% confidence band** (10th-90th percentile):
- Single shaded region showing realistic outcome range
- Clean color scheme aligned with RecompTracker branding
- No overwhelming statistics or complex options
- Clear visual hierarchy: representative path > confidence band > background

### Essential Phase Markers
**Triangle markers at key transitions**:
- ▲ **Bulk peaks**: Maximum safe body fat reached
- ▼ **Cut troughs**: Target lean body fat achieved
- **Hover tooltips**: Week, weight, body fat %, lean mass gained/lost
- **Visual clarity**: Markers don't clutter the main trajectory

### Representative Path Highlighting
**Bold trajectory line showing "most likely" outcome**:
- Distinctive color and weight for primary path
- Smooth line interpolation between weekly data points
- Clear differentiation from confidence band
- Visual prominence without overwhelming other elements

### Minimal, Clean Design
**Streamlined for quick comprehension**:
- Essential axes labels only (Time, Weight, Body Fat %)
- Clean grid lines for reference
- Readable font sizes for all screen sizes
- Professional color palette with good contrast

## Visual Design Specifications

### Color Palette
```python
QUICK_MODE_COLORS = {
    'representative_path': '#2E86C1',      # Professional blue
    'confidence_band': '#AED6F1',         # Light blue (80% opacity)
    'bulk_markers': '#E74C3C',            # Red triangles (▲)
    'cut_markers': '#27AE60',             # Green triangles (▼)
    'background': '#FFFFFF',              # Clean white
    'grid_lines': '#ECF0F1',              # Subtle gray
    'text': '#2C3E50'                     # Dark gray for readability
}
```

### Typography
- **Title**: 16px, bold, dark gray
- **Axis labels**: 12px, regular, medium gray  
- **Tooltip text**: 11px, regular, black on white background
- **Marker labels**: 10px, bold, colored to match markers

### Layout Dimensions
- **Chart height**: 400px (mobile-friendly)
- **Margins**: 60px left, 40px right, 50px top, 60px bottom
- **Marker size**: 8px triangles for clear visibility
- **Line width**: 3px for representative path, 2px for bands

## Implementation Components

### Plotly Configuration
```python
def create_quick_mode_chart(forecast_plan: ForecastPlan) -> go.Figure:
    """Create simplified fan chart for Quick Mode"""
    
    fig = go.Figure()
    
    # Add confidence band (80%)
    fig.add_trace(go.Scatter(
        x=weeks_from_bands(forecast_plan.percentile_bands.p90),
        y=weights_from_bands(forecast_plan.percentile_bands.p90),
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=weeks_from_bands(forecast_plan.percentile_bands.p10),
        y=weights_from_bands(forecast_plan.percentile_bands.p10),
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor=QUICK_MODE_COLORS['confidence_band'],
        name='80% Confidence Range',
        hovertemplate='<b>Week %{x}</b><br>Weight: %{y:.1f} lbs<extra></extra>'
    ))
    
    # Add representative path
    fig.add_trace(go.Scatter(
        x=[state.week for state in forecast_plan.representative_path],
        y=[state.weight_lbs for state in forecast_plan.representative_path],
        mode='lines',
        line=dict(
            color=QUICK_MODE_COLORS['representative_path'],
            width=3
        ),
        name='Most Likely Path',
        hovertemplate='<b>Week %{x}</b><br>Weight: %{y:.1f} lbs<br>Body Fat: %{customdata:.1f}%<extra></extra>',
        customdata=[state.body_fat_pct for state in forecast_plan.representative_path]
    ))
    
    # Add phase markers
    add_phase_markers(fig, forecast_plan.median_checkpoints)
    
    # Configure layout
    fig.update_layout(
        title=dict(
            text="Your Body Composition Journey",
            font=dict(size=16, color=QUICK_MODE_COLORS['text']),
            x=0.5
        ),
        xaxis=dict(
            title="Weeks from Now",
            gridcolor=QUICK_MODE_COLORS['grid_lines'],
            titlefont=dict(size=12),
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title="Body Weight (lbs)",
            gridcolor=QUICK_MODE_COLORS['grid_lines'],
            titlefont=dict(size=12),
            tickfont=dict(size=11)
        ),
        plot_bgcolor=QUICK_MODE_COLORS['background'],
        paper_bgcolor=QUICK_MODE_COLORS['background'],
        height=400,
        margin=dict(l=60, r=40, t=50, b=60),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig
```

### Phase Marker Implementation
```python
def add_phase_markers(fig: go.Figure, checkpoints: List[CheckpointData]) -> None:
    """Add triangle markers for phase transitions"""
    
    for checkpoint in checkpoints:
        if checkpoint.phase == PhaseType.BULK:
            # Bulk peak marker (▲)
            fig.add_trace(go.Scatter(
                x=[checkpoint.week],
                y=[checkpoint.weight_lbs],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=8,
                    color=QUICK_MODE_COLORS['bulk_markers'],
                    line=dict(width=1, color='white')
                ),
                name='Bulk Peaks',
                showlegend=False,
                hovertemplate=(
                    '<b>Bulk Peak</b><br>'
                    'Week: %{x}<br>'
                    'Weight: %{y:.1f} lbs<br>'
                    'Body Fat: %{customdata[0]:.1f}%<br>'
                    'Lean Mass: %{customdata[1]:.1f} lbs'
                    '<extra></extra>'
                ),
                customdata=[[checkpoint.body_fat_pct, checkpoint.lean_mass_lbs]]
            ))
        elif checkpoint.phase == PhaseType.CUT:
            # Cut trough marker (▼)
            fig.add_trace(go.Scatter(
                x=[checkpoint.week],
                y=[checkpoint.weight_lbs],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=8,
                    color=QUICK_MODE_COLORS['cut_markers'],
                    line=dict(width=1, color='white')
                ),
                name='Cut Completions',
                showlegend=False,
                hovertemplate=(
                    '<b>Cut Complete</b><br>'
                    'Week: %{x}<br>'
                    'Weight: %{y:.1f} lbs<br>'
                    'Body Fat: %{customdata[0]:.1f}%<br>'
                    'Fat Lost: %{customdata[1]:.1f} lbs'
                    '<extra></extra>'
                ),
                customdata=[[checkpoint.body_fat_pct, checkpoint.fat_mass_lbs]]
            ))
```

### Responsive Design
```python
def configure_responsive_layout(fig: go.Figure, container_width: int) -> go.Figure:
    """Adjust chart layout based on container width"""
    
    if container_width < 600:  # Mobile
        fig.update_layout(
            height=350,
            margin=dict(l=50, r=30, t=40, b=50),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(size=10)
            ),
            title=dict(font=dict(size=14))
        )
    elif container_width < 900:  # Tablet
        fig.update_layout(
            height=380,
            margin=dict(l=55, r=35, t=45, b=55)
        )
    # Desktop uses default layout
    
    return fig
```

## Data Processing for Visualization

### Time Series Alignment
```python
def prepare_visualization_data(forecast_plan: ForecastPlan) -> Dict[str, List]:
    """Process simulation data for clean visualization"""
    
    # Ensure consistent time grid
    weeks = list(range(0, forecast_plan.total_duration_weeks + 1))
    
    # Interpolate representative path to weekly intervals
    rep_path_data = interpolate_to_weekly_grid(
        forecast_plan.representative_path, 
        weeks
    )
    
    # Prepare confidence band data
    band_data = {
        'weeks': weeks,
        'p10_weight': interpolate_percentile_band(forecast_plan.percentile_bands.p10, weeks, 'weight_lbs'),
        'p90_weight': interpolate_percentile_band(forecast_plan.percentile_bands.p90, weeks, 'weight_lbs'),
        'p10_bf': interpolate_percentile_band(forecast_plan.percentile_bands.p10, weeks, 'body_fat_pct'),
        'p90_bf': interpolate_percentile_band(forecast_plan.percentile_bands.p90, weeks, 'body_fat_pct')
    }
    
    return {
        'representative': rep_path_data,
        'confidence_bands': band_data,
        'checkpoints': forecast_plan.median_checkpoints
    }
```

### Smooth Interpolation
```python
def interpolate_to_weekly_grid(
    trajectory: List[SimulationState], 
    target_weeks: List[int]
) -> Dict[str, List[float]]:
    """Create smooth weekly interpolation of trajectory data"""
    
    from scipy import interpolate
    
    # Extract source data
    source_weeks = [state.week for state in trajectory]
    weights = [state.weight_lbs for state in trajectory]
    body_fats = [state.body_fat_pct for state in trajectory]
    lean_masses = [state.lean_mass_lbs for state in trajectory]
    
    # Create interpolation functions
    weight_interp = interpolate.interp1d(source_weeks, weights, kind='cubic')
    bf_interp = interpolate.interp1d(source_weeks, body_fats, kind='cubic')
    lean_interp = interpolate.interp1d(source_weeks, lean_masses, kind='cubic')
    
    # Generate smooth weekly data
    return {
        'weeks': target_weeks,
        'weights': [float(weight_interp(w)) for w in target_weeks],
        'body_fats': [float(bf_interp(w)) for w in target_weeks],
        'lean_masses': [float(lean_interp(w)) for w in target_weeks]
    }
```

## Testing Strategy

### Visual Regression Tests
- **Screenshot comparisons**: Automated visual diff testing
- **Layout consistency**: Verify responsive behavior across screen sizes
- **Color accuracy**: Validate color palette implementation
- **Marker positioning**: Ensure phase markers align correctly

### Interactive Tests
- **Hover functionality**: Verify tooltip content and positioning
- **Legend behavior**: Test show/hide functionality
- **Zoom/pan**: Ensure smooth interaction without performance issues

### Data Accuracy Tests
- **Interpolation quality**: Verify smooth curves without artifacts
- **Marker alignment**: Ensure checkpoints match simulation data
- **Band calculations**: Validate confidence interval accuracy

## Implementation Tools

### Visualization Libraries
- **Plotly**: Primary charting with interactive features
- **SciPy**: Smooth interpolation for clean curves
- **NumPy**: Efficient data processing for large datasets

### Styling & Layout
- **CSS customization**: Fine-tune appearance beyond Plotly defaults
- **Responsive utilities**: Screen size detection and adaptation
- **Color management**: Centralized palette with accessibility compliance

### Testing Tools
- **Playwright**: Automated visual regression testing
- **pytest-mpl**: Matplotlib-style visual comparisons
- **Streamlit testing**: Framework-specific UI testing

## Success Criteria

### User Experience
- ✅ Chart loads and renders in <2 seconds
- ✅ Clear visual hierarchy guides attention to key insights
- ✅ Hover tooltips provide useful context without clutter
- ✅ Responsive design works well on mobile and desktop

### Visual Quality
- ✅ Professional appearance suitable for health/fitness context
- ✅ High contrast and accessibility compliance
- ✅ Clean, uncluttered layout focuses on essential information
- ✅ Consistent with existing RecompTracker design language

### Technical Requirements
- ✅ Smooth performance with 2000-point datasets
- ✅ Memory efficient rendering and updates
- ✅ No visual artifacts or rendering glitches
- ✅ Graceful handling of extreme data scenarios

## Integration Points

This visualization component serves:
- **Part 5**: Quick Mode UI Integration (embedded chart display)
- **Part 6**: Advanced Mode Visualization (shared components and styling)
- **Future enhancements**: Export functionality and report generation