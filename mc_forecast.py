"""
Monte Carlo Simulation Engine for Multi-Phase Body Composition Forecasting

This module contains the core Monte Carlo simulation engine that powers
RecompTracker's multi-phase goal planning. It implements evidence-based
P-ratio calculations, training-level variance modeling, and phase transitions.

Key Features:
- 2000-iteration Monte Carlo simulations for statistical robustness
- P-ratio modeling based on Forbes/Hall and Stronger By Science research
- Training-level-specific variance factors
- Automatic phase transitions (cut/bulk cycles)
- Goal detection when target percentiles are reached
"""

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
class UserProfile:
    """User profile data for simulation"""
    birth_date: str  # MM/DD/YYYY format
    height_in: float
    gender: str  # 'male' or 'female'
    training_level: TrainingLevel
    scan_history: List[Dict]  # DEXA scan data


@dataclass
class GoalConfig:
    """Goal configuration (simplified - no target age)"""
    metric_type: str  # 'almi' or 'ffmi'
    target_percentile: float  # 0.01 to 0.99
    description: Optional[str] = None


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


# P-Ratio constants based on research
P_RATIO_DEFAULTS = {
    "bulk_any_bf": (0.45, 0.50),  # Lean mass ratio range for bulking
    "cut_high_bf": (0.20, 0.25),  # High BF (>25%M, >30%F) cutting
    "cut_moderate_bf": (0.30, 0.40),  # Moderate BF cutting
}

# Training-level variance factors (sigma)
TRAINING_VARIANCE = {
    TrainingLevel.NOVICE: 0.50,     # High variability
    TrainingLevel.INTERMEDIATE: 0.25, # Moderate variability  
    TrainingLevel.ADVANCED: 0.10,   # Low variability
}

# MacroFactor-based rate defaults (% body weight per week)
RATE_DEFAULTS = {
    "bulk": {
        TrainingLevel.NOVICE: 0.5,      # Happy medium for beginners
        TrainingLevel.INTERMEDIATE: 0.325, # Happy medium for intermediate
        TrainingLevel.ADVANCED: 0.15,   # Happy medium for advanced
    },
    "cut": {
        "conservative": 0.25,  # Minimal muscle loss
        "moderate": 0.625,     # 0.5-0.75% average
        "aggressive": 1.0,     # Higher muscle loss risk
    }
}

# Body fat thresholds for phase transitions
BF_THRESHOLDS = {
    "male": {
        "healthy_max": 25,     # Cut recommended above this
        "acceptable_max": 20,  # Reasonable bulk stopping point
        "preferred_max": 15,   # Lean maintenance range
        "minimum": 8,          # Safety floor
    },
    "female": {
        "healthy_max": 35,
        "acceptable_max": 30,
        "preferred_max": 25,
        "minimum": 16,
    }
}


class MonteCarloEngine:
    """Core Monte Carlo simulation engine"""
    
    def __init__(self, config: SimulationConfig):
        """Initialize engine with configuration"""
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)
        
        # Calculate user's current age and height in meters
        self.current_age = self._calculate_current_age()
        self.height_m = config.user_profile.height_in * 0.0254
        
        # Determine variance factor
        self.variance_factor = self._calculate_variance_factor()
        
        logger.info(f"Initialized Monte Carlo engine for {config.run_count} runs")
    
    def run_simulation(self) -> SimulationResults:
        """Execute full Monte Carlo simulation"""
        logger.info("Starting Monte Carlo simulation...")
        
        # Run all trajectories
        trajectories = []
        for run_idx in range(self.config.run_count):
            if run_idx % 500 == 0:
                logger.info(f"Completed {run_idx}/{self.config.run_count} runs")
            
            trajectory = self._run_single_trajectory(run_idx)
            trajectories.append(trajectory)
        
        logger.info(f"Completed all {self.config.run_count} simulation runs")
        
        # Process results
        results = self._process_simulation_results(trajectories)
        
        logger.info(f"Goal achieved at week {results.goal_achievement_week}, "
                   f"age {results.goal_achievement_age:.1f}")
        
        return results
    
    def _run_single_trajectory(self, run_idx: int) -> List[SimulationState]:
        """Run a single Monte Carlo trajectory"""
        
        # Initialize starting state from latest scan
        latest_scan = self.config.user_profile.scan_history[-1]
        current_state = self._create_initial_state(latest_scan)
        
        trajectory = [current_state]
        current_phase = self._determine_initial_phase(current_state)
        
        max_weeks = 260  # 5 year safety limit
        
        for week in range(1, max_weeks + 1):
            # Check if goal achieved
            if self._goal_achieved(current_state):
                break
            
            # Check for phase transition
            if self._should_transition_phase(current_state, current_phase, week):
                current_phase = self._get_next_phase(current_phase, current_state)
            
            # Simulate one week of progress
            next_state = self._simulate_week_progress(
                current_state, current_phase, week, run_idx
            )
            
            trajectory.append(next_state)
            current_state = next_state
        
        return trajectory
    
    def _create_initial_state(self, scan: Dict) -> SimulationState:
        """Create initial simulation state from DEXA scan"""
        
        # Calculate ALMI and FFMI
        alm_kg = (scan['arms_lean_lbs'] + scan['legs_lean_lbs']) * 0.453592
        tlm_kg = scan['total_lean_mass_lbs'] * 0.453592
        almi = alm_kg / (self.height_m ** 2)
        ffmi = tlm_kg / (self.height_m ** 2)
        
        return SimulationState(
            week=0,
            weight_lbs=scan['total_weight_lbs'],
            lean_mass_lbs=scan['total_lean_mass_lbs'],
            fat_mass_lbs=scan['fat_mass_lbs'],
            body_fat_pct=scan['body_fat_percentage'],
            phase=PhaseType.MAINTENANCE,  # Will be set by phase logic
            almi=almi,
            ffmi=ffmi
        )
    
    def _determine_initial_phase(self, state: SimulationState) -> PhaseType:
        """Determine initial phase based on template and current BF%"""
        
        gender = self.config.user_profile.gender
        thresholds = BF_THRESHOLDS[gender]
        
        if self.config.template == TemplateType.CUT_FIRST:
            # Cut first if above healthy threshold
            if state.body_fat_pct > thresholds["healthy_max"]:
                return PhaseType.CUT
            else:
                return PhaseType.BULK
        else:  # BULK_FIRST
            # Always start with bulk unless extremely high BF
            if state.body_fat_pct > thresholds["healthy_max"] + 5:
                return PhaseType.CUT
            else:
                return PhaseType.BULK
    
    def _should_transition_phase(self, state: SimulationState, current_phase: PhaseType, week: int) -> bool:
        """Check if phase transition should occur"""
        
        gender = self.config.user_profile.gender
        thresholds = BF_THRESHOLDS[gender]
        
        # Minimum phase duration constraints
        min_cut_weeks = 8
        min_bulk_weeks = 12
        
        # Check duration minimums (simplified - assumes phase started at week 0)
        if current_phase == PhaseType.CUT and week < min_cut_weeks:
            return False
        if current_phase == PhaseType.BULK and week < min_bulk_weeks:
            return False
        
        # Check BF thresholds for transitions
        if current_phase == PhaseType.CUT:
            # Transition from cut to bulk when reaching acceptable BF
            return state.body_fat_pct <= thresholds["acceptable_max"]
        elif current_phase == PhaseType.BULK:
            # Transition from bulk to cut when reaching upper limit
            return state.body_fat_pct >= thresholds["acceptable_max"] + 3
        
        return False
    
    def _get_next_phase(self, current_phase: PhaseType, state: SimulationState) -> PhaseType:
        """Determine next phase after transition"""
        
        if current_phase == PhaseType.CUT:
            return PhaseType.BULK
        elif current_phase == PhaseType.BULK:
            return PhaseType.CUT
        else:
            # From maintenance, choose based on BF level
            return self._determine_initial_phase(state)
    
    def _simulate_week_progress(
        self, 
        current_state: SimulationState, 
        phase: PhaseType, 
        week: int, 
        run_idx: int
    ) -> SimulationState:
        """Simulate one week of body composition changes"""
        
        # Get base rates for this phase and training level
        if phase == PhaseType.BULK:
            rate_pct_per_week = RATE_DEFAULTS["bulk"][self.config.training_level]
            weight_change_lbs = current_state.weight_lbs * (rate_pct_per_week / 100)
        elif phase == PhaseType.CUT:
            rate_pct_per_week = RATE_DEFAULTS["cut"]["moderate"]
            weight_change_lbs = -current_state.weight_lbs * (rate_pct_per_week / 100)
        else:  # MAINTENANCE
            weight_change_lbs = 0
        
        # Add variance
        weight_noise = self.rng.normal(0, abs(weight_change_lbs) * self.variance_factor)
        weight_change_lbs += weight_noise
        
        # Calculate P-ratio for this change
        p_ratio = self._get_p_ratio(phase, current_state.body_fat_pct)
        
        # Apply P-ratio with variance
        p_ratio_noise = self.rng.normal(0, 0.05)  # ±5% P-ratio variance
        actual_p_ratio = np.clip(p_ratio + p_ratio_noise, 0.1, 0.8)
        
        # Calculate lean vs fat changes
        if weight_change_lbs >= 0:  # Weight gain
            lean_change_lbs = weight_change_lbs * actual_p_ratio
            fat_change_lbs = weight_change_lbs * (1 - actual_p_ratio)
        else:  # Weight loss
            lean_change_lbs = weight_change_lbs * actual_p_ratio  # Negative (loss)
            fat_change_lbs = weight_change_lbs * (1 - actual_p_ratio)  # Negative (loss)
        
        # Calculate new state
        new_weight = current_state.weight_lbs + weight_change_lbs
        new_lean = current_state.lean_mass_lbs + lean_change_lbs
        new_fat = current_state.fat_mass_lbs + fat_change_lbs
        new_bf_pct = (new_fat / new_weight) * 100
        
        # Recalculate ALMI/FFMI (simplified - maintain ALM/TLM ratio)
        # Use actual ALM/TLM ratio from initial scan
        latest_scan = self.config.user_profile.scan_history[-1]
        initial_alm = latest_scan['arms_lean_lbs'] + latest_scan['legs_lean_lbs']
        initial_tlm = latest_scan['total_lean_mass_lbs']
        alm_ratio = initial_alm / initial_tlm
        
        new_alm_kg = new_lean * 0.453592 * alm_ratio
        new_almi = new_alm_kg / (self.height_m ** 2)
        new_ffmi = (new_lean * 0.453592) / (self.height_m ** 2)
        
        return SimulationState(
            week=week,
            weight_lbs=new_weight,
            lean_mass_lbs=new_lean,
            fat_mass_lbs=new_fat,
            body_fat_pct=new_bf_pct,
            phase=phase,
            almi=new_almi,
            ffmi=new_ffmi
        )
    
    def _get_p_ratio(self, phase: PhaseType, body_fat_pct: float) -> float:
        """Calculate P-ratio based on phase and body fat percentage"""
        
        if phase == PhaseType.BULK:
            # Bulking P-ratio independent of BF (Stronger by Science finding)
            return np.mean(P_RATIO_DEFAULTS["bulk_any_bf"])
        
        elif phase == PhaseType.CUT:
            # Cutting P-ratio depends on body fat level
            gender = self.config.user_profile.gender
            high_bf_threshold = 25 if gender == "male" else 30
            
            if body_fat_pct > high_bf_threshold:
                return np.mean(P_RATIO_DEFAULTS["cut_high_bf"])
            else:
                return np.mean(P_RATIO_DEFAULTS["cut_moderate_bf"])
        
        else:  # MAINTENANCE
            return 0.6  # Slight lean mass bias at maintenance
    
    def _goal_achieved(self, state: SimulationState) -> bool:
        """Check if target percentile goal has been achieved"""
        
        # Calculate current percentile (simplified - would use LMS curves in reality)
        if self.config.goal_config.metric_type == "almi":
            current_percentile = self._estimate_almi_percentile(state.almi)
        else:  # ffmi
            current_percentile = self._estimate_ffmi_percentile(state.ffmi)
        
        return current_percentile >= self.config.goal_config.target_percentile
    
    def _estimate_almi_percentile(self, almi: float) -> float:
        """Estimate ALMI percentile (simplified for testing)"""
        # This is a placeholder - would use actual LMS curves
        # Very conservative approximation to ensure realistic testing goals
        gender = self.config.user_profile.gender
        
        if gender == "male":
            # Male ALMI rough percentiles (very conservative ranges for testing)
            if almi < 6.0:
                return 0.05
            elif almi < 7.0:
                return 0.15
            elif almi < 8.0:
                return 0.25
            elif almi < 9.0:
                return 0.35
            elif almi < 10.0:
                return 0.50
            elif almi < 11.0:
                return 0.65
            elif almi < 12.0:
                return 0.75
            elif almi < 13.0:
                return 0.85
            elif almi < 14.0:
                return 0.92
            elif almi < 15.0:
                return 0.96
            else:
                return 0.98
        else:  # female
            # Female ALMI rough percentiles (very conservative ranges for testing)
            if almi < 4.5:
                return 0.05
            elif almi < 5.5:
                return 0.15
            elif almi < 6.0:
                return 0.25
            elif almi < 6.5:
                return 0.35
            elif almi < 7.0:
                return 0.50
            elif almi < 7.5:
                return 0.65
            elif almi < 8.0:
                return 0.75
            elif almi < 8.5:
                return 0.85
            elif almi < 9.0:
                return 0.92
            elif almi < 9.5:
                return 0.96
            else:
                return 0.98
    
    def _estimate_ffmi_percentile(self, ffmi: float) -> float:
        """Estimate FFMI percentile (simplified for testing)"""
        # This is a placeholder - would use actual LMS curves
        gender = self.config.user_profile.gender
        
        if gender == "male":
            # Male FFMI rough percentiles
            if ffmi < 17:
                return 0.10
            elif ffmi < 18:
                return 0.25
            elif ffmi < 19:
                return 0.50
            elif ffmi < 20:
                return 0.75
            elif ffmi < 21:
                return 0.90
            else:
                return 0.95
        else:  # female
            # Female FFMI rough percentiles
            if ffmi < 14:
                return 0.10
            elif ffmi < 15:
                return 0.25
            elif ffmi < 16:
                return 0.50
            elif ffmi < 17:
                return 0.75
            elif ffmi < 18:
                return 0.90
            else:
                return 0.95
    
    def _calculate_current_age(self) -> float:
        """Calculate user's current age from birth date"""
        birth_date = datetime.strptime(self.config.user_profile.birth_date, "%m/%d/%Y")
        age_days = (datetime.now() - birth_date).days
        return age_days / 365.25
    
    def _calculate_variance_factor(self) -> float:
        """Calculate overall variance factor"""
        base_variance = TRAINING_VARIANCE[self.config.training_level]
        
        # TODO: Blend with empirical variance from scan history (50/50)
        # For now, just use training level variance
        
        return base_variance
    
    def _process_simulation_results(self, trajectories: List[List[SimulationState]]) -> SimulationResults:
        """Process raw trajectories into structured results"""
        
        # Find median goal achievement time
        goal_weeks = []
        for trajectory in trajectories:
            # Find when goal was achieved in this run
            for state in trajectory:
                if self._goal_achieved(state):
                    goal_weeks.append(state.week)
                    break
            else:
                # Goal not achieved - use final week
                goal_weeks.append(trajectory[-1].week)
        
        median_goal_week = int(np.median(goal_weeks))
        goal_achievement_age = self.current_age + (median_goal_week / 52)
        
        # Calculate percentile bands (simplified)
        percentile_bands = self._calculate_percentile_bands(trajectories)
        
        # Find representative path
        representative_path = self._find_representative_path(trajectories, median_goal_week)
        
        # Extract checkpoints (simplified)
        median_checkpoints = self._extract_median_checkpoints(trajectories)
        
        # Calculate convergence quality
        convergence_quality = min(1.0, len([w for w in goal_weeks if w <= median_goal_week * 1.2]) / len(goal_weeks))
        
        return SimulationResults(
            trajectories=trajectories,
            median_checkpoints=median_checkpoints,
            representative_path=representative_path,
            percentile_bands=percentile_bands,
            goal_achievement_week=median_goal_week,
            goal_achievement_age=goal_achievement_age,
            convergence_quality=convergence_quality,
            total_phases=len(median_checkpoints)
        )
    
    def _calculate_percentile_bands(self, trajectories: List[List[SimulationState]]) -> Dict[str, List[SimulationState]]:
        """Calculate percentile bands across all trajectories"""
        
        # Find maximum trajectory length
        max_length = max(len(traj) for traj in trajectories)
        
        # Align all trajectories to same length (pad with final state)
        aligned_trajectories = []
        for traj in trajectories:
            if len(traj) < max_length:
                # Pad with final state
                padded = traj + [traj[-1]] * (max_length - len(traj))
                aligned_trajectories.append(padded)
            else:
                aligned_trajectories.append(traj[:max_length])
        
        # Calculate percentiles at each time step
        bands = {}
        for percentile in [10, 25, 50, 75, 90]:
            band_states = []
            for week_idx in range(max_length):
                # Get all states at this week
                week_states = [traj[week_idx] for traj in aligned_trajectories]
                
                # Calculate percentile for each metric
                weights = [s.weight_lbs for s in week_states]
                bfs = [s.body_fat_pct for s in week_states]
                almis = [s.almi for s in week_states]
                
                percentile_weight = np.percentile(weights, percentile)
                percentile_bf = np.percentile(bfs, percentile)
                percentile_almi = np.percentile(almis, percentile)
                
                # Create representative state
                band_state = SimulationState(
                    week=week_idx,
                    weight_lbs=percentile_weight,
                    lean_mass_lbs=percentile_weight * (100 - percentile_bf) / 100,
                    fat_mass_lbs=percentile_weight * percentile_bf / 100,
                    body_fat_pct=percentile_bf,
                    phase=PhaseType.MAINTENANCE,  # Simplified
                    almi=percentile_almi,
                    ffmi=percentile_almi * 1.2  # Rough approximation
                )
                band_states.append(band_state)
            
            bands[f"p{percentile}"] = band_states
        
        return bands
    
    def _find_representative_path(self, trajectories: List[List[SimulationState]], target_weeks: int) -> List[SimulationState]:
        """Find trajectory closest to median at key time points"""
        
        # For simplicity, return the median trajectory
        if not trajectories:
            return []
        
        # Return first trajectory that reaches approximately the target time
        for traj in trajectories:
            if abs(len(traj) - target_weeks) <= 5:
                return traj
        
        # Fallback to median-length trajectory
        lengths = [len(traj) for traj in trajectories]
        median_length = int(np.median(lengths))
        
        for traj in trajectories:
            if len(traj) == median_length:
                return traj
        
        return trajectories[0]  # Final fallback
    
    def _extract_median_checkpoints(self, trajectories: List[List[SimulationState]]) -> List[CheckpointData]:
        """Extract key phase transition checkpoints"""
        
        # Simplified checkpoint extraction - just return milestones
        checkpoints = []
        
        # For now, create sample checkpoints based on median trajectory
        if trajectories:
            sample_traj = trajectories[len(trajectories) // 2]
            
            # Look for phase changes (simplified)
            current_phase = None
            for state in sample_traj:
                if state.phase != current_phase:
                    current_phase = state.phase
                    
                    # Calculate percentile progress
                    if self.config.goal_config.metric_type == "almi":
                        progress = self._estimate_almi_percentile(state.almi)
                    else:
                        progress = self._estimate_ffmi_percentile(state.ffmi)
                    
                    checkpoint = CheckpointData(
                        week=state.week,
                        phase=state.phase,
                        weight_lbs=state.weight_lbs,
                        body_fat_pct=state.body_fat_pct,
                        lean_mass_lbs=state.lean_mass_lbs,
                        fat_mass_lbs=state.fat_mass_lbs,
                        almi=state.almi,
                        ffmi=state.ffmi,
                        percentile_progress=progress
                    )
                    checkpoints.append(checkpoint)
        
        return checkpoints[:5]  # Limit to 5 major checkpoints


# Factory function for easy instantiation
def create_simulation_engine(
    user_profile: UserProfile,
    goal_config: GoalConfig,
    template: TemplateType = TemplateType.CUT_FIRST,
    run_count: int = 2000,
    random_seed: Optional[int] = None
) -> MonteCarloEngine:
    """Create configured Monte Carlo simulation engine"""
    
    # Determine training level if not specified
    training_level = user_profile.training_level
    
    # Calculate variance factor
    variance_factor = TRAINING_VARIANCE[training_level]
    
    config = SimulationConfig(
        user_profile=user_profile,
        goal_config=goal_config,
        training_level=training_level,
        template=template,
        variance_factor=variance_factor,
        random_seed=random_seed,
        run_count=run_count
    )
    
    return MonteCarloEngine(config)