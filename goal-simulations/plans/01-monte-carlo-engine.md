# Part 1: Core Simulation Engine with Tests

## Overview
Build the foundational Monte Carlo simulation engine that powers multi-phase body composition forecasting. This is the computational heart of the system that must be robust, well-tested, and performant.

## Core Requirements

### Monte Carlo Simulation Loop
- **2000 iterations** for statistical robustness
- **Weekly time steps** for granular progression tracking
- **Vectorized NumPy operations** for performance
- **Reproducible results** with seeded random number generation

### P-Ratio Modeling
Based on research from Forbes/Hall and Stronger By Science:
- **Bulk**: 0.45-0.50 lean mass ratio regardless of body fat
- **Cut (high BF >25%M/>30%F)**: 0.20-0.25 lean mass ratio  
- **Cut (moderate BF)**: 0.30-0.40 lean mass ratio
- **Normal distributions** around mean rates with variance

### Training-Level Variance
- **Novice**: σ = 0.50 (high variability in outcomes)
- **Intermediate**: σ = 0.25 (moderate variability)
- **Advanced**: σ = 0.10 (low variability, consistent results)
- **Empirical blending**: 50% from user's scan history variance

### State Progression
- **Body composition tracking**: weight, lean mass, fat mass, body fat %
- **Phase transitions**: automatic switching based on BF thresholds
- **Duration guardrails**: minimum phase durations (8wk cut, 12wk bulk)
- **Goal detection**: stop when ALMI/FFMI percentile target reached (age is calculated result)

## Data Structures

```python
@dataclass
class SimulationState:
    week: int
    weight_lbs: float
    lean_mass_lbs: float
    fat_mass_lbs: float
    body_fat_pct: float
    phase: PhaseType
    almi: float
    ffmi: float

@dataclass
class SimulationConfig:
    user_profile: UserProfile
    goal_percentile: float
    training_level: TrainingLevel
    template: TemplateType  # CUT_FIRST or BULK_FIRST
    variance_factor: float
    random_seed: Optional[int]

@dataclass
class SimulationResults:
    trajectories: List[List[SimulationState]]  # 2000 runs
    median_checkpoints: Dict[str, CheckpointData]
    representative_path: List[SimulationState]
    percentile_bands: Dict[str, List[SimulationState]]  # 10th, 25th, 75th, 90th
    goal_achievement_week: int  # Week when target percentile reached
    goal_achievement_age: float  # User's age when goal achieved
```

## Test Framework (Integrated)

### Unit Tests
- **Rate calculations**: Verify P-ratio formulas for different BF levels
- **Phase transitions**: Test automatic switching logic
- **Variance modeling**: Validate training-level adjustments
- **Goal detection**: Ensure simulation stops at correct percentile

### Canned User Profiles
Create realistic test scenarios:

1. **Novice + Overweight Male**: 28% BF, 180 lbs, targeting 75th percentile ALMI
2. **Novice + Lean Female**: 22% BF, 140 lbs, targeting 90th percentile FFMI  
3. **Intermediate Male**: 15% BF, 170 lbs, targeting 85th percentile ALMI
4. **Advanced Female**: 20% BF, 130 lbs, targeting 95th percentile FFMI

### Statistical Validation
- **Convergence testing**: Verify 2000 runs provide stable statistics
- **Distribution checks**: Ensure normal distributions around expected means
- **Boundary conditions**: Test extreme scenarios (very lean, very heavy)
- **Performance benchmarks**: <5 seconds for 2000 runs

## Implementation Tools

### Core Libraries
- **NumPy**: Vectorized operations, random number generation
- **SciPy**: Statistical distributions, percentile calculations
- **Dataclasses**: Clean data structures with type hints
- **Typing**: Full type annotations for maintainability

### Testing Stack
- **pytest**: Test framework with fixtures and parametrization
- **hypothesis**: Property-based testing for edge cases
- **numpy.testing**: Numerical assertion helpers
- **pytest-benchmark**: Performance regression testing

### Development Practices
- **Type hints**: Full typing for all functions and classes
- **Docstrings**: Comprehensive documentation with examples
- **Error handling**: Robust validation with custom exceptions
- **Logging**: Structured logging for debugging complex simulations

## Success Criteria

### Functional Requirements
- ✅ All unit tests pass with >95% code coverage
- ✅ Canned profiles generate sensible median timelines
- ✅ Statistical properties match theoretical expectations
- ✅ Phase transitions occur at correct BF thresholds

### Performance Requirements  
- ✅ 2000 simulations complete in <5 seconds
- ✅ Memory usage <500MB for largest scenarios
- ✅ Reproducible results with same random seed
- ✅ Stable statistics across multiple runs

### Quality Requirements
- ✅ No runtime errors or warnings
- ✅ Clean separation of concerns
- ✅ Extensible design for future enhancements
- ✅ Comprehensive error messages for invalid inputs

## Integration Points

This engine will be consumed by:
- **Part 3**: Forecast API Layer (caching and interface)
- **Part 4**: Quick Mode Visualization (basic fan charts)
- **Part 6**: Advanced Mode Visualization (detailed analysis)

The engine must be **completely independent** of UI concerns and **thoroughly tested** before any visualization work begins.