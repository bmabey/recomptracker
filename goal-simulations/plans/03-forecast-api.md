# Part 3: Forecast API Layer

## Overview
Create a clean, cacheable API layer that bridges the Monte Carlo engine with the UI components. This layer handles expensive computation caching, data transformation, and provides a stable interface for both Quick and Advanced modes.

## Core Requirements

### Clean API Interface
Single entry point that encapsulates all complexity:
```python
def get_plan(
    user_profile: UserProfile,
    goal_config: GoalConfig,
    simulation_config: Optional[SimulationConfig] = None
) -> ForecastPlan
```

### Intelligent Caching Strategy
Streamlit's `@st.cache_data` with smart invalidation:
- **Cache key**: Hash of core inputs (profile, goals, phase rules)
- **Cache miss triggers**: Changes to user data, goal percentiles, template selection
- **Cache hit optimization**: UI parameter changes (visualization options) don't invalidate
- **Memory management**: LRU eviction for large simulation results

### Data Structure Design
Rich, typed data structures for downstream consumption:

```python
@dataclass
class CheckpointData:
    week: int
    phase: PhaseType
    weight_lbs: float
    body_fat_pct: float
    lean_mass_lbs: float
    fat_mass_lbs: float
    almi: float
    ffmi: float
    percentile_progress: float

@dataclass
class PercentileBands:
    p10: List[SimulationState]
    p25: List[SimulationState] 
    p50: List[SimulationState]  # median
    p75: List[SimulationState]
    p90: List[SimulationState]

@dataclass
class ForecastPlan:
    # Core simulation results
    representative_path: List[SimulationState]
    percentile_bands: PercentileBands
    median_checkpoints: List[CheckpointData]
    
    # Metadata
    total_duration_weeks: int
    total_phases: int
    template_used: TemplateType
    convergence_quality: float
    
    # Performance metrics
    simulation_time_ms: int
    cache_hit: bool
    run_count: int
```

## Caching Implementation

### Cache Key Generation
```python
def generate_cache_key(
    user_profile: UserProfile,
    goal_config: GoalConfig, 
    simulation_config: SimulationConfig
) -> str:
    """Generate stable hash for cache invalidation"""
    key_data = {
        'birth_date': user_profile.birth_date,
        'height_in': user_profile.height_in,
        'gender': user_profile.gender,
        'training_level': user_profile.training_level,
        'latest_scan': user_profile.scan_history[-1],
        'goal_percentile': goal_config.target_percentile,
        'goal_metric': goal_config.metric_type,
        'template': simulation_config.template,
        'variance_factor': simulation_config.variance_factor,
        'run_count': simulation_config.run_count
    }
    return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
```

### Streamlit Integration
```python
@st.cache_data(
    ttl=3600,  # 1 hour cache lifetime
    max_entries=50,  # LRU eviction
    show_spinner=False  # Custom loading UI
)
def _cached_simulation(cache_key: str, config: SimulationConfig) -> ForecastPlan:
    """Cached wrapper around expensive Monte Carlo simulation"""
    start_time = time.time()
    
    # Run the simulation
    engine = MonteCarloEngine()
    results = engine.run_simulation(config)
    
    # Process results
    plan = _build_forecast_plan(results, config, start_time)
    plan.cache_hit = False
    
    return plan
```

## Data Processing Pipeline

### Representative Path Selection
Find the trajectory closest to median checkpoints:
```python
def select_representative_path(
    trajectories: List[List[SimulationState]],
    median_checkpoints: List[CheckpointData]
) -> List[SimulationState]:
    """Select trajectory with minimum RMS distance to median checkpoints"""
    
    best_trajectory = None
    min_distance = float('inf')
    
    for trajectory in trajectories:
        distance = calculate_rms_distance(trajectory, median_checkpoints)
        if distance < min_distance:
            min_distance = distance
            best_trajectory = trajectory
    
    return best_trajectory
```

### Percentile Band Calculation
```python
def calculate_percentile_bands(
    trajectories: List[List[SimulationState]]
) -> PercentileBands:
    """Calculate confidence bands across all simulation runs"""
    
    # Align all trajectories to common time grid
    max_weeks = max(len(traj) for traj in trajectories)
    aligned_data = align_trajectories(trajectories, max_weeks)
    
    # Calculate percentiles at each time step
    bands = PercentileBands(
        p10=calculate_percentile(aligned_data, 10),
        p25=calculate_percentile(aligned_data, 25),
        p50=calculate_percentile(aligned_data, 50),
        p75=calculate_percentile(aligned_data, 75),
        p90=calculate_percentile(aligned_data, 90)
    )
    
    return bands
```

### Checkpoint Extraction
```python
def extract_median_checkpoints(
    trajectories: List[List[SimulationState]]
) -> List[CheckpointData]:
    """Extract phase transition points and key milestones"""
    
    checkpoints = []
    
    # Find phase transitions across all runs
    for phase_type in [PhaseType.CUT, PhaseType.BULK]:
        transition_points = find_phase_transitions(trajectories, phase_type)
        if transition_points:
            median_checkpoint = calculate_median_checkpoint(transition_points)
            checkpoints.append(median_checkpoint)
    
    # Sort by timeline
    return sorted(checkpoints, key=lambda x: x.week)
```

## Error Handling

### Simulation Failures
```python
class SimulationError(Exception):
    """Base class for simulation-related errors"""

class ConvergenceError(SimulationError):
    """Raised when simulation fails to converge to goal"""

class InvalidInputError(SimulationError):
    """Raised when user input is invalid for simulation"""

class CacheError(SimulationError):
    """Raised when caching operations fail"""
```

### Graceful Degradation
- **Simulation timeout**: Fall back to fewer iterations (1000 → 500)
- **Convergence failure**: Extend timeline with warning
- **Cache corruption**: Invalidate and regenerate
- **Memory pressure**: Reduce run count dynamically

## Performance Optimization

### Memory Efficiency
- **Lazy evaluation**: Only calculate needed percentiles
- **Data compression**: Store compressed trajectory data
- **Garbage collection**: Explicit cleanup of large arrays
- **Memory monitoring**: Track usage and warn on excessive consumption

### Computational Efficiency
- **Vectorized operations**: NumPy for all heavy calculations
- **Parallel processing**: Thread pool for independent calculations
- **Early termination**: Stop simulation when convergence detected
- **Progressive refinement**: Show preview with fewer runs, refine in background

## Validation & Testing

### Input Validation
```python
def validate_simulation_inputs(
    user_profile: UserProfile,
    goal_config: GoalConfig,
    simulation_config: SimulationConfig
) -> None:
    """Comprehensive input validation with helpful error messages"""
    
    # User profile validation
    validate_scan_history(user_profile.scan_history)
    validate_training_level(user_profile.training_level)
    
    # Goal validation
    validate_percentile_target(goal_config.target_percentile)
    validate_goal_feasibility(user_profile, goal_config)
    
    # Simulation config validation
    validate_run_count(simulation_config.run_count)
    validate_template_compatibility(user_profile, simulation_config.template)
```

### Integration Tests
- **Cache behavior**: Verify hits/misses under different scenarios
- **Data consistency**: Ensure processed results match raw simulation
- **Performance bounds**: Guarantee response times under load
- **Memory limits**: Validate memory usage stays within bounds

## Implementation Tools

### Core Libraries
- **Streamlit**: `@st.cache_data` for intelligent caching
- **NumPy**: Efficient array operations for percentile calculations
- **Pandas**: Data manipulation for trajectory alignment
- **Hashlib**: Stable cache key generation

### Data Validation
- **Pydantic**: Runtime validation with automatic documentation
- **Marshmallow**: Complex data transformation and validation
- **Custom validators**: Domain-specific business rule validation

### Performance Monitoring
- **cProfile**: Performance profiling for optimization
- **Memory profiler**: Track memory usage patterns
- **Time measurement**: Detailed timing for cache performance

## Success Criteria

### Performance Requirements
- ✅ Cache hit rate >80% for typical usage patterns
- ✅ Cold simulation completes in <10 seconds
- ✅ Cached results return in <100ms
- ✅ Memory usage <1GB for largest scenarios

### Quality Requirements
- ✅ 100% input validation coverage
- ✅ Graceful handling of all error conditions
- ✅ Comprehensive logging for debugging
- ✅ Clear error messages for user-facing issues

### Integration Requirements
- ✅ Clean interface for both Quick and Advanced modes
- ✅ Backward compatibility with existing goal system
- ✅ Extensible for future simulation enhancements
- ✅ Thread-safe for concurrent Streamlit sessions

## Integration Points

This API layer serves:
- **Quick Mode**: Simple, cached forecasts with sensible defaults
- **Advanced Mode**: Full customization with real-time parameter updates
- **Future enhancements**: Extensible for new simulation strategies
- **External tools**: Clean data structures for export/analysis