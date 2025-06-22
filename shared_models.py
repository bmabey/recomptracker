"""
Shared Data Models for RecompTracker

This module contains all shared dataclasses and enums used throughout the
RecompTracker application, including the core analysis engine, Monte Carlo
simulation system, and web interface.

Unified data models provide:
- Type safety and validation
- Consistent data structures across modules
- Better IDE support and auto-completion
- Single source of truth for core data types
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


# ============================================================================
# ENUMS
# ============================================================================

class PhaseType(Enum):
    """Body composition phase types"""
    CUT = "cut"
    BULK = "bulk"
    MAINTENANCE = "maintenance"


class TemplateType(Enum):
    """Multi-phase strategy templates"""
    CUT_FIRST = "cut_first"
    BULK_FIRST = "bulk_first"


class TrainingLevel(Enum):
    """Training experience levels affecting variance and rates"""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class ScanData:
    """Individual DEXA scan data structure"""
    date: str  # MM/DD/YYYY format
    total_weight_lbs: float
    total_lean_mass_lbs: float
    fat_mass_lbs: float
    body_fat_percentage: float
    arms_lean_lbs: float
    legs_lean_lbs: float
    
    # Optional fields for backward compatibility
    date_str: Optional[str] = None  # Alternative date field name
    
    def __post_init__(self):
        """Set date_str for backward compatibility if not provided"""
        if self.date_str is None:
            self.date_str = self.date


@dataclass
class BodyComposition:
    """Calculated body composition metrics"""
    age_at_scan: float
    almi_kg_m2: float  # Appendicular Lean Mass Index
    ffmi_kg_m2: float  # Fat-Free Mass Index
    almi_z_score: float
    almi_percentile: float  # 0-100 scale
    almi_t_score: float
    ffmi_z_score: float
    ffmi_percentile: float  # 0-100 scale
    ffmi_t_score: float
    alm_kg: float  # Appendicular lean mass in kg
    
    # Change tracking (optional, calculated when multiple scans exist)
    weight_change_last: Optional[float] = None
    lean_change_last: Optional[float] = None
    fat_change_last: Optional[float] = None
    bf_change_last: Optional[float] = None
    almi_z_change_last: Optional[float] = None
    ffmi_z_change_last: Optional[float] = None
    almi_t_change_last: Optional[float] = None
    ffmi_t_change_last: Optional[float] = None
    almi_pct_change_last: Optional[float] = None
    ffmi_pct_change_last: Optional[float] = None
    
    # Changes from first scan
    weight_change_first: Optional[float] = None
    lean_change_first: Optional[float] = None
    fat_change_first: Optional[float] = None
    bf_change_first: Optional[float] = None
    almi_t_change_first: Optional[float] = None
    ffmi_t_change_first: Optional[float] = None


@dataclass
class UserProfile:
    """User demographic and anthropometric data"""
    birth_date: str  # MM/DD/YYYY format
    height_in: float
    gender: str  # 'male' or 'female'
    training_level: TrainingLevel
    scan_history: List[ScanData]
    
    # Optional fields for backward compatibility
    birth_date_str: Optional[str] = None
    gender_code: Optional[int] = None  # 0=male, 1=female
    
    def __post_init__(self):
        """Set derived fields for backward compatibility"""
        if self.birth_date_str is None:
            self.birth_date_str = self.birth_date
        if self.gender_code is None:
            self.gender_code = 0 if self.gender.lower() in ['m', 'male'] else 1
        
        # Convert scan_history dicts to ScanData objects if needed
        if self.scan_history and isinstance(self.scan_history[0], dict):
            self.scan_history = [
                ScanData(**scan) if isinstance(scan, dict) else scan
                for scan in self.scan_history
            ]


@dataclass
class GoalConfig:
    """Goal configuration - simplified without target_age"""
    metric_type: str  # 'almi' or 'ffmi'
    target_percentile: float  # 0.01 to 0.99
    description: Optional[str] = None
    suggested: bool = False  # Whether this was auto-suggested
    target_body_fat_percentage: Optional[float] = None


@dataclass
class SimulationState:
    """State of a single simulation at a specific time point"""
    week: int
    weight_lbs: float
    lean_mass_lbs: float
    fat_mass_lbs: float
    body_fat_pct: float
    phase: PhaseType
    almi: float
    ffmi: float


@dataclass
class CheckpointData:
    """Key milestone in the forecast timeline"""
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
class SimulationConfig:
    """Configuration for Monte Carlo simulation"""
    user_profile: UserProfile
    goal_config: GoalConfig
    training_level: TrainingLevel
    template: TemplateType
    variance_factor: float
    random_seed: Optional[int] = None
    run_count: int = 2000
    max_duration_weeks: Optional[int] = None  # Override age-based default


@dataclass
class SimulationResults:
    """Results from Monte Carlo simulation"""
    trajectories: List[List[SimulationState]]  # All simulation runs
    median_checkpoints: List[CheckpointData]  # Key phase transitions
    representative_path: List[SimulationState]  # Most likely trajectory
    percentile_bands: Dict[str, List[SimulationState]]  # Confidence intervals
    goal_achievement_week: int  # Week when target percentile reached
    goal_achievement_age: float  # User's age when goal achieved
    convergence_quality: float  # Statistical confidence measure
    total_phases: int  # Number of bulk/cut cycles


@dataclass
class GoalResults:
    """Results from goal calculation and body composition planning"""
    target_age: float  # Auto-calculated target age
    target_percentile: float
    target_metric_value: float
    target_z_score: float
    metric_change_needed: float
    lean_change_needed_lbs: float
    alm_change_needed_lbs: float
    alm_change_needed_kg: float
    tlm_change_needed_lbs: float
    tlm_change_needed_kg: float
    weight_change: float
    lean_change: float
    fat_change: float
    bf_change: float
    percentile_change: float
    z_change: float
    target_body_composition: Dict[str, float]
    
    # Backward compatibility fields
    alm_to_add_kg: Optional[float] = None
    estimated_tlm_gain_kg: Optional[float] = None
    tlm_to_add_kg: Optional[float] = None
    target_z: Optional[float] = None
    suggested: bool = False
    target_almi: Optional[float] = None
    target_ffmi: Optional[float] = None
    
    def __post_init__(self):
        """Set backward compatibility fields"""
        self.alm_to_add_kg = self.alm_change_needed_kg
        self.estimated_tlm_gain_kg = self.tlm_change_needed_kg
        self.tlm_to_add_kg = self.tlm_change_needed_kg
        self.target_z = self.target_z_score


@dataclass
class AnalysisResults:
    """Complete analysis results including scans and goals"""
    user_profile: UserProfile
    scan_results: List[BodyComposition]
    goal_results: Dict[str, GoalResults]  # 'almi' and/or 'ffmi' keys
    messages: List[str] = field(default_factory=list)  # Calculation explanations


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def convert_dict_to_user_profile(user_info: dict, scan_history: List[dict]) -> UserProfile:
    """Convert legacy dict format to UserProfile dataclass"""
    # Convert training level string to enum
    training_level_str = user_info.get("training_level", "intermediate").lower()
    training_level = TrainingLevel(training_level_str)
    
    # Convert scan history dicts to ScanData objects
    scan_data_list = []
    for scan in scan_history:
        scan_copy = scan.copy()
        # Handle both 'date' and 'date_str' field names
        if 'date' not in scan_copy and 'date_str' in scan_copy:
            scan_copy['date'] = scan_copy['date_str']
        elif 'date_str' not in scan_copy and 'date' in scan_copy:
            scan_copy['date_str'] = scan_copy['date']
        scan_data_list.append(ScanData(**scan_copy))
    
    return UserProfile(
        birth_date=user_info["birth_date"],
        height_in=user_info["height_in"],
        gender=user_info["gender"],
        training_level=training_level,
        scan_history=scan_data_list
    )


def convert_dict_to_goal_config(goal_dict: dict) -> GoalConfig:
    """Convert legacy dict format to GoalConfig dataclass"""
    return GoalConfig(
        metric_type=goal_dict.get("metric_type", "almi"),
        target_percentile=goal_dict["target_percentile"],
        description=goal_dict.get("description"),
        suggested=goal_dict.get("suggested", False),
        target_body_fat_percentage=goal_dict.get("target_body_fat_percentage")
    )


# ============================================================================
# CONSTANTS AND CONFIGURATIONS
# ============================================================================

# P-Ratio constants based on research
P_RATIO_DEFAULTS = {
    "bulk_any_bf": (0.45, 0.50),  # Lean mass ratio range for bulking
    "cut_high_bf": (0.20, 0.25),  # High BF (>25%M, >30%F) cutting
    "cut_moderate_bf": (0.30, 0.40),  # Moderate BF cutting
}

# Training-level variance factors (sigma)
TRAINING_VARIANCE = {
    TrainingLevel.NOVICE: 0.50,  # High variability
    TrainingLevel.INTERMEDIATE: 0.25,  # Moderate variability
    TrainingLevel.ADVANCED: 0.10,  # Low variability
}

# MacroFactor-based rate defaults (% body weight per week)
RATE_DEFAULTS = {
    "bulk": {
        TrainingLevel.NOVICE: 0.5,  # Happy medium for beginners
        TrainingLevel.INTERMEDIATE: 0.325,  # Happy medium for intermediate
        TrainingLevel.ADVANCED: 0.15,  # Happy medium for advanced
    },
    "cut": {
        "conservative": 0.25,  # Minimal muscle loss
        "moderate": 0.625,  # 0.5-0.75% average
        "aggressive": 1.0,  # Higher muscle loss risk
    },
}

# Body fat thresholds for phase transitions
BF_THRESHOLDS = {
    "male": {
        "healthy_max": 25,  # Cut recommended above this
        "acceptable_max": 20,  # Reasonable bulk stopping point
        "preferred_max": 15,  # Lean maintenance range
        "minimum": 8,  # Safety floor
    },
    "female": {
        "healthy_max": 35,
        "acceptable_max": 30,
        "preferred_max": 25,
        "minimum": 16,
    },
}