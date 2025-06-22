# Part 2: Phase Planning System

## Overview
Implement the intelligent phase planning logic that determines bulk/cut sequences based on user's current body composition, goals, and scientific evidence from MacroFactor research.

## Core Requirements

### Template Selection Logic
Automatic template choice based on current body fat percentage:

**Cut-First Template (Default)**
- Triggered when BF > healthy threshold (25% men, 35% women)
- Prioritizes getting to sustainable body fat range first
- Follows evidence that cutting from high BF is safe and effective

**Bulk-First Template (Optional)**
- Available for users already in healthy BF range
- Allows muscle-first approach for those prioritizing size
- Mirrors Stronger By Science flexible philosophy

### Phase Transition Rules

#### Cut-First Template Sequence
1. **Initial Cut**: Reach healthy BF range (<20% men, <30% women)
2. **First Bulk**: Stop at max acceptable BF (15-18% men, 25-30% women)  
3. **Maintenance Cut**: Return to desired BF (12-15% men, 22-25% women)
4. **Cycle Repeat**: Continue bulk/cut until ALMI/FFMI goal reached

#### Bulk-First Template Sequence  
1. **Initial Bulk**: +3-5% BF from current or cap (18-22% men, 28-32% women)
2. **First Cut**: Drop to preferred BF range (12-15% men, 22-25% women)
3. **Second Bulk**: Resume surplus, respecting BF caps
4. **Cycle Repeat**: Alternate until percentile goal achieved

### Duration Guardrails
Prevent unrealistic phase lengths:
- **Minimum cut duration**: 8 weeks (sustainable fat loss)
- **Minimum bulk duration**: 12 weeks (meaningful muscle growth)
- **Maximum phase duration**: 52 weeks (practical limit)

### Rate Integration (MacroFactor Evidence)

#### Bulking Rates by Training Level
| Level | Conservative | Happy Medium | Aggressive |
|-------|-------------|-------------|------------|
| **Beginner** | 0.2% BW/wk | 0.5% BW/wk | 0.8% BW/wk |
| **Intermediate** | 0.15% BW/wk | 0.325% BW/wk | 0.575% BW/wk |
| **Experienced** | 0.1% BW/wk | 0.15% BW/wk | 0.35% BW/wk |

#### Cutting Rates (Universal)
- **Conservative**: 0.25% BW/week (minimal muscle loss)
- **Moderate**: 0.5-0.75% BW/week (optimal balance) 
- **Aggressive**: 1.0% BW/week (muscle loss risk)

## Data Structures

```python
class PhaseType(Enum):
    CUT = "cut"
    BULK = "bulk"
    MAINTENANCE = "maintenance"

class TemplateType(Enum):
    CUT_FIRST = "cut_first"
    BULK_FIRST = "bulk_first"

class TrainingLevel(Enum):
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

@dataclass
class PhaseConfig:
    phase_type: PhaseType
    target_bf_pct: float
    min_duration_weeks: int
    max_duration_weeks: int
    rate_pct_per_week: float

@dataclass
class PhaseSequence:
    template: TemplateType
    phases: List[PhaseConfig]
    rationale: str

@dataclass
class PhaseTransition:
    from_phase: PhaseType
    to_phase: PhaseType
    trigger_condition: str
    target_metrics: Dict[str, float]
```

## Implementation Components

### Template Engine
```python
class PhaseTemplateEngine:
    def select_template(self, user_profile: UserProfile) -> TemplateType
    def generate_sequence(self, template: TemplateType, user_profile: UserProfile) -> PhaseSequence
    def get_phase_config(self, phase_type: PhaseType, user_profile: UserProfile) -> PhaseConfig
```

### Transition Logic
```python
class PhaseTransitionManager:
    def should_transition(self, current_state: SimulationState, phase_config: PhaseConfig) -> bool
    def get_next_phase(self, current_phase: PhaseType, sequence: PhaseSequence) -> Optional[PhaseConfig]
    def validate_transition(self, transition: PhaseTransition) -> bool
```

### Rate Calculator
```python
class RateCalculator:
    def get_bulk_rate(self, training_level: TrainingLevel, aggressiveness: str) -> float
    def get_cut_rate(self, aggressiveness: str) -> float
    def apply_body_weight_scaling(self, base_rate: float, weight_lbs: float) -> float
    def get_p_ratio(self, phase_type: PhaseType, body_fat_pct: float) -> float
```

## Validation Rules

### Body Fat Thresholds
- **Men**: Healthy <25%, Acceptable <20%, Preferred <15%
- **Women**: Healthy <35%, Acceptable <30%, Preferred <25%
- **Safety bounds**: No cuts below 8% men / 16% women
- **Upper bounds**: No bulks above 30% men / 40% women

### Duration Constraints
- **Minimum viable phases**: 6 weeks (practical minimum)
- **Realistic maximums**: 1 year per phase (lifestyle sustainability)
- **Total timeline**: Warn if >5 years to goal (unrealistic)

### Rate Boundaries
- **Conservative lower bound**: 0.1% BW/week (progress detection)
- **Aggressive upper bound**: 1.5% BW/week (sustainability limit)
- **Absolute maximums**: 2 lbs/week regardless of body weight

## Error Handling

### Configuration Validation
```python
class PhaseConfigError(Exception):
    """Raised when phase configuration is invalid"""

class TransitionError(Exception):
    """Raised when phase transition logic fails"""

class RateCalculationError(Exception):
    """Raised when rate calculation produces invalid results"""
```

### Graceful Degradation
- **Invalid BF targets**: Fall back to population averages
- **Extreme training levels**: Use intermediate as default
- **Conflicting preferences**: Prioritize safety over speed

## Testing Strategy

### Unit Tests
- **Template selection**: Test all BF threshold combinations
- **Phase sequences**: Verify correct ordering and durations
- **Rate calculations**: Validate all training level combinations
- **Transition logic**: Test boundary conditions and edge cases

### Integration Tests  
- **End-to-end sequences**: Complete template execution
- **Cross-validation**: Ensure rates match MacroFactor research
- **Edge case handling**: Extreme user profiles and goals

### Property-Based Tests
- **Rate bounds**: All calculated rates within valid ranges
- **Sequence validity**: All generated sequences are executable
- **Conservation laws**: Phase transitions preserve key constraints

## Implementation Tools

### Core Libraries
- **Enum**: Type-safe phase and template definitions
- **Dataclasses**: Clean configuration structures
- **Typing**: Comprehensive type annotations
- **Decimal**: Precise percentage calculations

### Validation
- **Pydantic**: Runtime validation with clear error messages
- **Custom validators**: Domain-specific business rules
- **Range checking**: Bounds validation for all numeric inputs

### Configuration Management
- **YAML/JSON**: External configuration for easy tuning
- **Environment variables**: Runtime behavior modification
- **Version control**: Track changes to phase rules over time

## Success Criteria

### Functional Requirements
- ✅ Template selection matches MacroFactor decision tree
- ✅ All phase sequences reach goals within reasonable timeframes
- ✅ Rate calculations match published research values
- ✅ Transition logic handles all edge cases gracefully

### Quality Requirements
- ✅ 100% test coverage for business logic
- ✅ Clear error messages for invalid configurations
- ✅ Extensible design for new templates/strategies
- ✅ Performance suitable for real-time UI interaction

### Integration Requirements
- ✅ Clean interface with Monte Carlo engine
- ✅ Support for both Quick and Advanced mode requirements
- ✅ Backward compatibility with existing goal system
- ✅ Future-proof for additional templates and strategies

## Integration Points

This system provides:
- **Phase rules** for Monte Carlo simulation engine
- **Template options** for Advanced mode customization
- **Rate defaults** for Quick mode simplicity
- **Validation logic** for user input forms