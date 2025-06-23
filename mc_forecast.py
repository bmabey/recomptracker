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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Import cached percentile calculation from core
from core import calculate_percentile_cached

# Import phase planning components
from phase_planning import (
    PhaseTemplateEngine,
    PhaseTransitionManager,
    PhaseValidationEngine,
    RateCalculator,
)

# Import shared dataclasses and enums
from shared_models import (
    BF_THRESHOLDS,
    P_RATIO_DEFAULTS,
    RATE_DEFAULTS,
    TRAINING_VARIANCE,
    CheckpointData,
    GoalConfig,
    PhaseConfig,
    PhaseSequence,
    PhaseType,
    SimulationConfig,
    SimulationResults,
    SimulationState,
    TemplateType,
    TrainingLevel,
    UserProfile,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonteCarloEngine:
    """Core Monte Carlo simulation engine"""

    def __init__(self, config: SimulationConfig):
        """Initialize engine with configuration and phase planning components"""
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)

        # Calculate user's current age and height in meters
        self.current_age = self._calculate_current_age()
        self.height_m = config.user_profile.height_in * 0.0254

        # Determine variance factor
        self.variance_factor = self._calculate_variance_factor()

        # Calculate maximum duration (age-based or override)
        self.max_duration_weeks = self._calculate_max_duration_weeks()

        # Initialize phase planning components
        self.rate_calculator = RateCalculator()
        self.template_engine = PhaseTemplateEngine(self.rate_calculator)
        self.transition_manager = PhaseTransitionManager(self.rate_calculator)

        # Generate phase sequence if not provided
        if config.phase_sequence is None:
            self.phase_sequence = self.template_engine.generate_sequence(
                config.template,
                config.user_profile,
                bf_range_config=config.bf_range_config,
            )
            # Validate the generated sequence
            PhaseValidationEngine.validate_sequence(
                self.phase_sequence, config.user_profile.gender
            )
        else:
            self.phase_sequence = config.phase_sequence

        logger.info(
            f"Initialized Monte Carlo engine for {config.run_count} runs with "
            f"{self.phase_sequence.template.value} template ({len(self.phase_sequence.phases)} phases)"
        )

    def run_simulation(self) -> SimulationResults:
        """Execute full Monte Carlo simulation"""
        logger.info("Starting Monte Carlo simulation...")

        # Validate goal feasibility with weight constraints if provided
        if (
            self.config.bf_range_config is not None
            and self.config.bf_range_config.max_weight_lbs is not None
        ):
            from core import validate_goal_feasibility_with_weight_constraint

            is_feasible, error_msg, min_weight = (
                validate_goal_feasibility_with_weight_constraint(
                    self.config.user_profile,
                    self.config.goal_config,
                    self.config.bf_range_config,
                    self.current_age,
                )
            )

            if not is_feasible:
                raise ValueError(
                    f"Goal not achievable with weight constraint: {error_msg}"
                )

            if min_weight is not None:
                logger.info(
                    f"Weight constraint validation passed. Minimum weight needed: {min_weight:.1f} lbs, "
                    f"maximum allowed: {self.config.bf_range_config.max_weight_lbs:.1f} lbs"
                )

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

        logger.info(
            f"Goal achieved at week {results.goal_achievement_week}, "
            f"age {results.goal_achievement_age:.1f}"
        )

        return results

    def _run_single_trajectory(self, run_idx: int) -> List[SimulationState]:
        """Run a single Monte Carlo trajectory using sophisticated phase planning"""

        # Initialize starting state from latest scan
        latest_scan = self.config.user_profile.scan_history[-1]
        current_state = self._create_initial_state(latest_scan)

        trajectory = [current_state]

        # Initialize phase tracking using sophisticated system
        current_phase_idx = 0
        current_phase_config = self.phase_sequence.phases[current_phase_idx]
        weeks_in_current_phase = 0
        sequence_completed = False

        # Use age-based maximum duration
        max_weeks = self.max_duration_weeks

        if run_idx == 0:  # Debug first trajectory only
            print(
                f"Debug trajectory {run_idx}: initial_state.week={current_state.week}, max_weeks={max_weeks}"
            )
            print(
                f"Debug trajectory {run_idx}: initial_state.almi={current_state.almi:.2f}"
            )
            print(
                f"Debug trajectory {run_idx}: Starting phase: {current_phase_config.phase_type.value} "
                f"(target BF: {current_phase_config.target_bf_pct}%)"
            )

        for week in range(1, max_weeks + 1):
            # Check if goal achieved
            if self._goal_achieved(current_state):
                if run_idx == 0:
                    print(
                        f"Debug trajectory {run_idx}: Goal achieved at week {week - 1}"
                    )
                break

            # Increment phase duration counter
            weeks_in_current_phase += 1

            # Check for sophisticated phase transition only if sequence not completed
            if not sequence_completed:
                should_transition = self.transition_manager.should_transition(
                    current_state.body_fat_pct,
                    current_phase_config.phase_type,
                    current_phase_config,
                    weeks_in_current_phase,
                    self.config.user_profile.gender,
                    current_weight_lbs=current_state.weight_lbs,
                    bf_range_config=self.config.bf_range_config,
                )

                if should_transition:
                    next_phase_config = self.transition_manager.get_next_phase_config(
                        current_phase_config,
                        self.config.user_profile,
                        self.phase_sequence,
                    )

                    if next_phase_config is not None:
                        if run_idx == 0:
                            print(
                                f"Debug trajectory {run_idx}: Phase transition at week {week}: "
                                f"{current_phase_config.phase_type.value} -> {next_phase_config.phase_type.value}"
                            )
                        current_phase_config = next_phase_config
                        current_phase_idx += 1
                        weeks_in_current_phase = 0
                    else:
                        if run_idx == 0:
                            print(
                                f"Debug trajectory {run_idx}: Phase sequence completed at week {week}"
                            )

                        # Mark sequence as completed and check if goal achieved
                        sequence_completed = True
                        from shared_models import PhaseConfig

                        # If goal not achieved, continue with moderate bulk to make progress
                        if not self._goal_achieved(current_state):
                            # Continue bulking at moderate rate to achieve goal more efficiently
                            moderate_bulk_rate = self.rate_calculator.get_bulk_rate(
                                self.config.training_level, "moderate"
                            )
                            current_phase_config = PhaseConfig(
                                phase_type=PhaseType.BULK,
                                target_bf_pct=current_state.body_fat_pct
                                + 10,  # Allow reasonable BF increase
                                min_duration_weeks=1,
                                max_duration_weeks=9999,  # No limit for goal pursuit
                                rate_pct_per_week=moderate_bulk_rate,
                                rationale="Moderate bulk to achieve goal after sequence completion",
                            )
                        else:
                            # Goal achieved, true maintenance
                            current_phase_config = PhaseConfig(
                                phase_type=PhaseType.MAINTENANCE,
                                target_bf_pct=current_state.body_fat_pct,  # Stay at current BF
                                min_duration_weeks=1,
                                max_duration_weeks=9999,  # No limit for maintenance
                                rate_pct_per_week=0.0,  # No weight change
                                rationale="Maintenance after goal achievement",
                            )
                        weeks_in_current_phase = 0
                        if run_idx == 0:
                            if current_phase_config.phase_type == PhaseType.MAINTENANCE:
                                print(
                                    f"Debug trajectory {run_idx}: Switched to maintenance mode (goal achieved)"
                                )
                            else:
                                print(
                                    f"Debug trajectory {run_idx}: Switched to moderate bulk mode (goal not achieved)"
                                )

            # Simulate one week of progress using sophisticated rate calculation
            next_state = self._simulate_week_progress_advanced(
                current_state, current_phase_config, week, run_idx
            )

            trajectory.append(next_state)
            current_state = next_state

        if run_idx == 0:
            print(
                f"Debug trajectory {run_idx}: Final trajectory length: {len(trajectory)}"
            )

        return trajectory

    def _create_initial_state(self, scan) -> SimulationState:
        """Create initial simulation state from DEXA scan (dict or ScanData)"""

        # Handle both dict and ScanData object formats
        if hasattr(scan, "arms_lean_lbs"):  # ScanData object
            arms_lean = scan.arms_lean_lbs
            legs_lean = scan.legs_lean_lbs
            total_weight = scan.total_weight_lbs
            total_lean = scan.total_lean_mass_lbs
            fat_mass = scan.fat_mass_lbs
            body_fat_pct = scan.body_fat_percentage
        else:  # dict format
            arms_lean = scan["arms_lean_lbs"]
            legs_lean = scan["legs_lean_lbs"]
            total_weight = scan["total_weight_lbs"]
            total_lean = scan["total_lean_mass_lbs"]
            fat_mass = scan["fat_mass_lbs"]
            body_fat_pct = scan["body_fat_percentage"]

        # Calculate ALMI and FFMI
        alm_kg = (arms_lean + legs_lean) * 0.453592
        tlm_kg = total_lean * 0.453592
        almi = alm_kg / (self.height_m**2)
        ffmi = tlm_kg / (self.height_m**2)

        return SimulationState(
            week=0,
            weight_lbs=total_weight,
            lean_mass_lbs=total_lean,
            fat_mass_lbs=fat_mass,
            body_fat_pct=body_fat_pct,
            phase=PhaseType.MAINTENANCE,  # Will be set by phase logic
            almi=almi,
            ffmi=ffmi,
            # Initialize training progression tracking
            weeks_training=0,
            current_training_level=self.config.training_level,
            training_level_transition_weeks=[],
            simulation_age=self.current_age,
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

    def _should_transition_phase(
        self, state: SimulationState, current_phase: PhaseType, week: int
    ) -> bool:
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

    def _get_next_phase(
        self, current_phase: PhaseType, state: SimulationState
    ) -> PhaseType:
        """Determine next phase after transition"""

        if current_phase == PhaseType.CUT:
            return PhaseType.BULK
        elif current_phase == PhaseType.BULK:
            return PhaseType.CUT
        else:
            # From maintenance, choose based on BF level
            return self._determine_initial_phase(state)

    def _simulate_week_progress(
        self, current_state: SimulationState, phase: PhaseType, week: int, run_idx: int
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

        # Handle both dict and ScanData object formats
        if hasattr(latest_scan, "arms_lean_lbs"):  # ScanData object
            initial_alm = latest_scan.arms_lean_lbs + latest_scan.legs_lean_lbs
            initial_tlm = latest_scan.total_lean_mass_lbs
        else:  # dict format
            initial_alm = latest_scan["arms_lean_lbs"] + latest_scan["legs_lean_lbs"]
            initial_tlm = latest_scan["total_lean_mass_lbs"]

        alm_ratio = initial_alm / initial_tlm

        new_alm_kg = new_lean * 0.453592 * alm_ratio
        new_almi = new_alm_kg / (self.height_m**2)
        new_ffmi = (new_lean * 0.453592) / (self.height_m**2)

        return SimulationState(
            week=week,
            weight_lbs=new_weight,
            lean_mass_lbs=new_lean,
            fat_mass_lbs=new_fat,
            body_fat_pct=new_bf_pct,
            phase=phase,
            almi=new_almi,
            ffmi=new_ffmi,
        )

    def _simulate_week_progress_advanced(
        self,
        current_state: SimulationState,
        phase_config: PhaseConfig,
        week: int,
        run_idx: int,
    ) -> SimulationState:
        """
        Simulate one week of body composition changes using sophisticated rate calculator.

        This method uses the research-backed rate calculator and phase configuration
        to provide more accurate and evidence-based body composition changes with
        dynamic training level progression and age tracking.
        """

        # Update training progression tracking
        updated_state = current_state
        updated_state.weeks_training += 1
        updated_state.simulation_age = self.current_age + (
            week / 52.0
        )  # Update age based on simulation time

        # Update training level progression
        updated_state = self._update_training_progression(updated_state)

        # Get sophisticated rate calculation using CURRENT training level (not initial)
        current_training_level = updated_state.current_training_level
        if phase_config.phase_type == PhaseType.BULK:
            base_rate = self.rate_calculator.get_bulk_rate(
                current_training_level, "moderate"
            )
        elif phase_config.phase_type == PhaseType.CUT:
            base_rate = self.rate_calculator.get_cut_rate("moderate")
        else:  # MAINTENANCE
            base_rate = 0

        # Apply age adjustment using CURRENT simulation age
        age_adjusted_rate = self.rate_calculator.apply_age_adjustment(
            base_rate, updated_state.simulation_age
        )

        # Calculate absolute weight change using body weight scaling
        if phase_config.phase_type != PhaseType.MAINTENANCE:
            abs_weight_change = self.rate_calculator.apply_body_weight_scaling(
                age_adjusted_rate, current_state.weight_lbs
            )

            # Apply direction (negative for cutting)
            if phase_config.phase_type == PhaseType.CUT:
                abs_weight_change = -abs_weight_change
        else:
            abs_weight_change = 0

        # Add variance using dynamic training level variance factor
        current_variance_factor = TRAINING_VARIANCE[current_training_level]
        weight_noise = self.rng.normal(
            0, abs(abs_weight_change) * current_variance_factor
        )
        weight_change_lbs = abs_weight_change + weight_noise

        # Calculate P-ratio using sophisticated calculator
        p_ratio = self.rate_calculator.get_p_ratio(
            phase_config.phase_type,
            current_state.body_fat_pct,
            self.config.user_profile.gender,
        )

        # Apply P-ratio variance
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

        # Recalculate ALMI/FFMI maintaining ALM/TLM ratio
        latest_scan = self.config.user_profile.scan_history[-1]

        # Handle both dict and ScanData object formats
        if hasattr(latest_scan, "arms_lean_lbs"):  # ScanData object
            initial_alm = latest_scan.arms_lean_lbs + latest_scan.legs_lean_lbs
            initial_tlm = latest_scan.total_lean_mass_lbs
        else:  # dict format
            initial_alm = latest_scan["arms_lean_lbs"] + latest_scan["legs_lean_lbs"]
            initial_tlm = latest_scan["total_lean_mass_lbs"]

        alm_ratio = initial_alm / initial_tlm

        new_alm_kg = new_lean * 0.453592 * alm_ratio
        new_almi = new_alm_kg / (self.height_m**2)
        new_ffmi = (new_lean * 0.453592) / (self.height_m**2)

        return SimulationState(
            week=week,
            weight_lbs=new_weight,
            lean_mass_lbs=new_lean,
            fat_mass_lbs=new_fat,
            body_fat_pct=new_bf_pct,
            phase=phase_config.phase_type,
            almi=new_almi,
            ffmi=new_ffmi,
            # Carry forward training progression tracking
            weeks_training=updated_state.weeks_training,
            current_training_level=updated_state.current_training_level,
            training_level_transition_weeks=updated_state.training_level_transition_weeks.copy(),
            simulation_age=updated_state.simulation_age,
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

        # Debug: Check if percentile calculation is working
        if np.isnan(current_percentile):
            logger.warning(
                f"Got NaN percentile for {self.config.goal_config.metric_type}={state.almi if self.config.goal_config.metric_type == 'almi' else state.ffmi} at age {self.current_age}"
            )
            return False

        achieved = current_percentile >= self.config.goal_config.target_percentile
        if achieved and state.week == 0:  # Debug initial goal achievement
            print(
                f"Debug: Goal achieved immediately! ALMI {state.almi:.2f} = {current_percentile:.3f} percentile >= {self.config.goal_config.target_percentile}"
            )

        return achieved

    def _estimate_almi_percentile(self, almi: float) -> float:
        """Calculate ALMI percentile using real LMS curves with caching"""
        gender = self.config.user_profile.gender
        gender_code = 0 if gender == "male" else 1

        # Use cached percentile calculation from core
        percentile = calculate_percentile_cached(
            value=almi,
            age=self.current_age,
            metric="appendicular_LMI",
            gender_code=gender_code,
        )

        # Return 0.5 as fallback if calculation fails
        return percentile if not np.isnan(percentile) else 0.5

    def _estimate_ffmi_percentile(self, ffmi: float) -> float:
        """Calculate FFMI percentile using real LMS curves with caching"""
        gender = self.config.user_profile.gender
        gender_code = 0 if gender == "male" else 1

        # Use cached percentile calculation from core
        percentile = calculate_percentile_cached(
            value=ffmi, age=self.current_age, metric="LMI", gender_code=gender_code
        )

        # Return 0.5 as fallback if calculation fails
        return percentile if not np.isnan(percentile) else 0.5

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

    def _calculate_dynamic_training_level(
        self, current_state: SimulationState, initial_training_level: TrainingLevel
    ) -> TrainingLevel:
        """
        Calculate current training level based on time progression and age factors.

        Research-based transition thresholds:
        - Novice phase: 6-18 months (varies by age/gender)
        - Intermediate phase: 18 months - 3 years
        - Advanced phase: 3+ years

        Age factors affect progression speed:
        - Younger users (18-25): Standard progression timelines
        - Adult users (25-40): Slightly faster initial progression
        - Older users (40+): Faster progression through novice phase
        """
        weeks_training = current_state.weeks_training
        simulation_age = current_state.simulation_age
        gender = self.config.user_profile.gender.lower()

        # Age-based progression modifiers (research shows older beginners progress faster initially)
        if simulation_age < 25:
            age_modifier = 1.0  # Standard progression
        elif simulation_age < 40:
            age_modifier = 0.9  # 10% faster progression
        else:
            age_modifier = 0.75  # 25% faster progression (older adults adapt quicker to initial training)

        # Gender-based slight adjustments (research shows minimal difference in progression timeline)
        gender_modifier = 0.95 if gender == "female" else 1.0

        # Calculate adjusted thresholds
        base_novice_weeks = 52  # 1 year base
        base_intermediate_weeks = 156  # 3 years base

        novice_to_intermediate_weeks = int(
            base_novice_weeks * age_modifier * gender_modifier
        )
        intermediate_to_advanced_weeks = int(
            base_intermediate_weeks * age_modifier * gender_modifier
        )

        # Determine current training level (transition happens after the threshold week)
        if weeks_training < novice_to_intermediate_weeks:
            return TrainingLevel.NOVICE
        elif weeks_training < intermediate_to_advanced_weeks:
            return TrainingLevel.INTERMEDIATE
        else:
            return TrainingLevel.ADVANCED

    def _update_training_progression(
        self, current_state: SimulationState
    ) -> SimulationState:
        """
        Update training progression tracking in simulation state.

        Args:
            current_state: Current simulation state

        Returns:
            Updated simulation state with progression tracking
        """
        # Calculate new training level
        new_training_level = self._calculate_dynamic_training_level(
            current_state, self.config.training_level
        )

        # Check for training level transition
        if new_training_level != current_state.current_training_level:
            # Record transition
            current_state.training_level_transition_weeks.append(current_state.week)
            logger.info(
                f"Training level transition at week {current_state.week}: "
                f"{current_state.current_training_level.value} -> {new_training_level.value}"
            )
            current_state.current_training_level = new_training_level

        return current_state

    def _calculate_max_duration_weeks(self) -> int:
        """Calculate maximum simulation duration based on user age"""

        # Use override if specified
        if self.config.max_duration_weeks is not None:
            return self.config.max_duration_weeks

        # Age-based default limits
        current_age = self.current_age

        if current_age < 41:
            return 520  # 10 years for younger users
        elif current_age < 56:
            return 416  # 8 years for middle-aged users
        elif current_age < 71:
            return 260  # 5 years for older users
        else:
            return 156  # 3 years for elderly users

    def _process_simulation_results(
        self, trajectories: List[List[SimulationState]]
    ) -> SimulationResults:
        """Process raw trajectories into structured results"""

        # Find median goal achievement time
        goal_weeks = []
        for i, trajectory in enumerate(trajectories):
            # Find when goal was achieved in this run
            for state in trajectory:
                if self._goal_achieved(state):
                    goal_weeks.append(state.week)
                    break
            else:
                # Goal not achieved - use final week
                if trajectory:
                    final_week = trajectory[-1].week
                    goal_weeks.append(final_week)
                    if i == 0:  # Log details for first trajectory
                        logger.warning(
                            f"Goal not achieved in trajectory {i}, final week: {final_week}, final state: ALMI={trajectory[-1].almi:.2f}"
                        )
                else:
                    logger.error(f"Empty trajectory {i}")
                    goal_weeks.append(0)  # This might be causing the issue

        print(f"Debug: goal_weeks = {goal_weeks[:10]}...")  # Show first 10 values
        print(
            f"Debug: len(goal_weeks) = {len(goal_weeks)}, min = {min(goal_weeks) if goal_weeks else 'N/A'}, max = {max(goal_weeks) if goal_weeks else 'N/A'}"
        )

        if not goal_weeks or all(w == 0 for w in goal_weeks):
            print("Debug: All goal_weeks are 0 or empty - simulation failed")
            median_goal_week = 0
        else:
            median_goal_week = int(np.median(goal_weeks))

        goal_achievement_age = self.current_age + (median_goal_week / 52)

        # Calculate percentile bands (simplified)
        percentile_bands = self._calculate_percentile_bands(trajectories)

        # Find representative path
        representative_path = self._find_representative_path(
            trajectories, median_goal_week
        )

        # Extract checkpoints (simplified)
        median_checkpoints = self._extract_median_checkpoints(trajectories)

        # Calculate convergence quality
        convergence_quality = min(
            1.0,
            len([w for w in goal_weeks if w <= median_goal_week * 1.2])
            / len(goal_weeks),
        )

        return SimulationResults(
            trajectories=trajectories,
            median_checkpoints=median_checkpoints,
            representative_path=representative_path,
            percentile_bands=percentile_bands,
            goal_achievement_week=median_goal_week,
            goal_achievement_age=goal_achievement_age,
            convergence_quality=convergence_quality,
            total_phases=len(median_checkpoints),
        )

    def _calculate_percentile_bands(
        self, trajectories: List[List[SimulationState]]
    ) -> Dict[str, List[SimulationState]]:
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
                    ffmi=percentile_almi * 1.2,  # Rough approximation
                )
                band_states.append(band_state)

            bands[f"p{percentile}"] = band_states

        return bands

    def _find_representative_path(
        self, trajectories: List[List[SimulationState]], target_weeks: int
    ) -> List[SimulationState]:
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

    def _extract_median_checkpoints(
        self, trajectories: List[List[SimulationState]]
    ) -> List[CheckpointData]:
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
                        percentile_progress=progress,
                    )
                    checkpoints.append(checkpoint)

        return checkpoints[:5]  # Limit to 5 major checkpoints


# Factory function for easy instantiation
def create_simulation_engine(
    user_profile: UserProfile,
    goal_config: GoalConfig,
    template: TemplateType = TemplateType.CUT_FIRST,
    bf_range_config=None,
    run_count: int = 2000,
    random_seed: Optional[int] = None,
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
        bf_range_config=bf_range_config,
        random_seed=random_seed,
        run_count=run_count,
    )

    return MonteCarloEngine(config)
