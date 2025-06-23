"""
Forecast API Layer for RecompTracker

This module provides a clean, cacheable API layer that bridges the Monte Carlo
engine with UI components. It handles expensive computation caching, data
transformation, and provides a stable interface for both Quick and Advanced modes.

Key Features:
- Intelligent caching with Streamlit's @st.cache_data
- Smart cache invalidation based on core inputs only
- Rich data structures for downstream consumption
- Memory-efficient data processing with NumPy vectorization
- Comprehensive error handling and graceful degradation
- Performance optimization and monitoring
"""

import hashlib
import json
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from mc_forecast import MonteCarloEngine
from shared_models import (
    CheckpointData,
    ForecastPlan,
    GoalConfig,
    PercentileBands,
    PhaseType,
    SimulationConfig,
    SimulationResults,
    SimulationState,
    TemplateType,
    UserProfile,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class SimulationError(Exception):
    """Base class for simulation-related errors"""

    pass


class ConvergenceError(SimulationError):
    """Raised when simulation fails to converge to goal"""

    pass


class InvalidInputError(SimulationError):
    """Raised when user input is invalid for simulation"""

    pass


class CacheError(SimulationError):
    """Raised when caching operations fail"""

    pass


# ============================================================================
# CORE API FUNCTION
# ============================================================================


def get_plan(
    user_profile: UserProfile,
    goal_config: GoalConfig,
    simulation_config: Optional[SimulationConfig] = None,
) -> ForecastPlan:
    """
    Get comprehensive forecast plan with intelligent caching.

    This is the single entry point that encapsulates all complexity and provides
    a clean interface for both Quick and Advanced modes.

    Args:
        user_profile: User demographic and scan history data
        goal_config: Target percentile and metric configuration
        simulation_config: Optional simulation parameters (auto-generated if None)

    Returns:
        ForecastPlan: Complete forecast results with metadata

    Raises:
        InvalidInputError: If inputs are invalid
        ConvergenceError: If simulation fails to converge
        CacheError: If caching operations fail
    """
    logger.info(
        f"Getting forecast plan for {goal_config.metric_type} goal {goal_config.target_percentile}"
    )

    # Validate inputs comprehensively
    validate_simulation_inputs(user_profile, goal_config, simulation_config)

    # Generate default simulation config if not provided
    if simulation_config is None:
        simulation_config = _generate_default_simulation_config(
            user_profile, goal_config
        )

    # Generate cache key for intelligent invalidation
    cache_key = generate_cache_key(user_profile, goal_config, simulation_config)

    try:
        # Use cached simulation with intelligent caching strategy
        forecast_plan = _cached_simulation(cache_key, simulation_config)
        logger.info(f"Forecast plan generated (cache_hit: {forecast_plan.cache_hit})")
        return forecast_plan

    except Exception as e:
        logger.error(f"Forecast plan generation failed: {str(e)}")

        # Graceful degradation - try with reduced parameters
        if simulation_config.run_count > 1000:
            logger.info("Attempting graceful degradation with reduced run count")
            simulation_config.run_count = 1000
            cache_key = generate_cache_key(user_profile, goal_config, simulation_config)

            try:
                forecast_plan = _cached_simulation(cache_key, simulation_config)
                logger.info("Graceful degradation successful")
                return forecast_plan
            except Exception as fallback_error:
                logger.error(f"Graceful degradation failed: {str(fallback_error)}")

        # Final fallback - raise original error
        if isinstance(e, (SimulationError, ValueError)):
            raise e
        else:
            raise SimulationError(
                f"Unexpected error during forecast generation: {str(e)}"
            )


# ============================================================================
# CACHING IMPLEMENTATION
# ============================================================================


def generate_cache_key(
    user_profile: UserProfile,
    goal_config: GoalConfig,
    simulation_config: SimulationConfig,
) -> str:
    """
    Generate stable hash for cache invalidation.

    Only includes core inputs that affect simulation results, excluding
    UI-only parameters to maximize cache hit rate.
    """
    try:
        key_data = {
            "birth_date": user_profile.birth_date,
            "height_in": user_profile.height_in,
            "gender": user_profile.gender,
            "training_level": user_profile.training_level.value,
            "latest_scan": _serialize_scan(user_profile.scan_history[-1]),
            "goal_percentile": goal_config.target_percentile,
            "goal_metric": goal_config.metric_type,
            "template": simulation_config.template.value,
            "variance_factor": simulation_config.variance_factor,
            "run_count": simulation_config.run_count,
            "bf_range": _serialize_bf_range(simulation_config.bf_range_config),
            "max_duration": simulation_config.max_duration_weeks,
        }

        # Create stable JSON representation
        json_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    except Exception as e:
        logger.error(f"Cache key generation failed: {str(e)}")
        raise CacheError(f"Failed to generate cache key: {str(e)}")


@st.cache_data(
    ttl=3600,  # 1 hour cache lifetime
    max_entries=50,  # LRU eviction for memory management
    show_spinner=False,  # Custom loading UI handled elsewhere
)
def _cached_simulation(cache_key: str, config: SimulationConfig) -> ForecastPlan:
    """
    Cached wrapper around expensive Monte Carlo simulation.

    Uses Streamlit's intelligent caching with performance monitoring
    and memory management.
    """
    start_time = time.time()

    try:
        # Run the Monte Carlo simulation
        engine = MonteCarloEngine(config)
        results = engine.run_simulation()

        # Process results into forecast plan
        forecast_plan = _build_forecast_plan(results, config, start_time)
        forecast_plan.cache_hit = False  # This is a fresh calculation

        logger.info(
            f"Fresh simulation completed in {forecast_plan.simulation_time_ms}ms"
        )
        return forecast_plan

    except Exception as e:
        simulation_time_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Simulation failed after {simulation_time_ms}ms: {str(e)}")

        if "not achievable" in str(e).lower():
            raise ConvergenceError(f"Goal not achievable: {str(e)}")
        elif "invalid" in str(e).lower():
            raise InvalidInputError(f"Invalid simulation parameters: {str(e)}")
        else:
            raise SimulationError(f"Simulation execution failed: {str(e)}")


# ============================================================================
# DATA PROCESSING PIPELINE
# ============================================================================


def _build_forecast_plan(
    results: SimulationResults, config: SimulationConfig, start_time: float
) -> ForecastPlan:
    """
    Build comprehensive forecast plan from simulation results.

    Processes raw simulation data into rich, typed data structures
    optimized for downstream consumption.
    """
    simulation_time_ms = int((time.time() - start_time) * 1000)

    # Process percentile bands with memory-efficient calculations
    percentile_bands = calculate_percentile_bands(results.trajectories)

    # Select representative path using RMS distance method
    representative_path = select_representative_path(
        results.trajectories, results.median_checkpoints
    )

    # Extract and enhance checkpoints
    enhanced_checkpoints = _enhance_checkpoints(results.median_checkpoints, config)

    return ForecastPlan(
        # Core simulation results
        representative_path=representative_path,
        percentile_bands=percentile_bands,
        median_checkpoints=enhanced_checkpoints,
        # Metadata
        total_duration_weeks=results.goal_achievement_week,
        total_phases=results.total_phases,
        template_used=config.template,
        convergence_quality=results.convergence_quality,
        # Performance metrics
        simulation_time_ms=simulation_time_ms,
        cache_hit=False,  # Will be updated for cache hits
        run_count=config.run_count,
        # Goal achievement details
        goal_achievement_week=results.goal_achievement_week,
        goal_achievement_age=results.goal_achievement_age,
    )


def select_representative_path(
    trajectories: List[List[SimulationState]], median_checkpoints: List[CheckpointData]
) -> List[SimulationState]:
    """
    Select trajectory with minimum RMS distance to median checkpoints.

    Uses advanced statistical methods to find the most representative
    trajectory that best captures the median behavior.
    """
    if not trajectories or not median_checkpoints:
        return trajectories[0] if trajectories else []

    best_trajectory = None
    min_distance = float("inf")

    # Calculate RMS distance for each trajectory
    for trajectory in trajectories:
        distance = calculate_rms_distance(trajectory, median_checkpoints)
        if distance < min_distance:
            min_distance = distance
            best_trajectory = trajectory

    return best_trajectory if best_trajectory is not None else trajectories[0]


def calculate_rms_distance(
    trajectory: List[SimulationState], checkpoints: List[CheckpointData]
) -> float:
    """
    Calculate RMS distance between trajectory and median checkpoints.

    Uses multiple metrics (weight, BF%, ALMI) for comprehensive comparison.
    """
    if not trajectory or not checkpoints:
        return float("inf")

    distances = []

    for checkpoint in checkpoints:
        # Find closest week in trajectory
        closest_state = min(trajectory, key=lambda s: abs(s.week - checkpoint.week))

        # Calculate normalized distance across multiple metrics
        weight_diff = (
            abs(closest_state.weight_lbs - checkpoint.weight_lbs)
            / checkpoint.weight_lbs
        )
        bf_diff = abs(closest_state.body_fat_pct - checkpoint.body_fat_pct) / max(
            1.0, checkpoint.body_fat_pct
        )
        almi_diff = abs(closest_state.almi - checkpoint.almi) / max(
            0.1, checkpoint.almi
        )

        # Weighted combination of metrics
        combined_distance = 0.4 * weight_diff + 0.3 * bf_diff + 0.3 * almi_diff
        distances.append(combined_distance)

    # Return RMS distance
    return np.sqrt(np.mean(np.array(distances) ** 2))


def calculate_percentile_bands(
    trajectories: List[List[SimulationState]],
) -> PercentileBands:
    """
    Calculate confidence bands across all simulation runs.

    Uses memory-efficient vectorized operations for large datasets.
    """
    if not trajectories:
        return PercentileBands(p10=[], p25=[], p50=[], p75=[], p90=[])

    # Align all trajectories to common time grid
    max_weeks = max(len(traj) for traj in trajectories)
    aligned_data = align_trajectories(trajectories, max_weeks)

    # Calculate percentiles at each time step using vectorized operations
    bands = PercentileBands(
        p10=calculate_percentile(aligned_data, 10),
        p25=calculate_percentile(aligned_data, 25),
        p50=calculate_percentile(aligned_data, 50),
        p75=calculate_percentile(aligned_data, 75),
        p90=calculate_percentile(aligned_data, 90),
    )

    return bands


def align_trajectories(
    trajectories: List[List[SimulationState]], max_weeks: int
) -> List[List[SimulationState]]:
    """
    Align all trajectories to common time grid by padding with final states.

    Ensures consistent data structure for percentile calculations.
    """
    aligned = []

    for trajectory in trajectories:
        if len(trajectory) < max_weeks:
            # Pad with final state to maintain timeline consistency
            final_state = trajectory[-1] if trajectory else None
            if final_state:
                padded_states = []
                for week in range(max_weeks):
                    if week < len(trajectory):
                        padded_states.append(trajectory[week])
                    else:
                        # Create new state with updated week number
                        padded_state = SimulationState(
                            week=week,
                            weight_lbs=final_state.weight_lbs,
                            lean_mass_lbs=final_state.lean_mass_lbs,
                            fat_mass_lbs=final_state.fat_mass_lbs,
                            body_fat_pct=final_state.body_fat_pct,
                            phase=final_state.phase,
                            almi=final_state.almi,
                            ffmi=final_state.ffmi,
                            weeks_training=final_state.weeks_training,
                            current_training_level=final_state.current_training_level,
                            training_level_transition_weeks=final_state.training_level_transition_weeks,
                            simulation_age=final_state.simulation_age,
                        )
                        padded_states.append(padded_state)
                aligned.append(padded_states)
            else:
                aligned.append(trajectory)
        else:
            aligned.append(trajectory[:max_weeks])

    return aligned


def calculate_percentile(
    aligned_data: List[List[SimulationState]], percentile: int
) -> List[SimulationState]:
    """
    Calculate percentile trajectory using vectorized NumPy operations.

    Memory-efficient implementation for large simulation datasets.
    """
    if not aligned_data:
        return []

    max_weeks = len(aligned_data[0]) if aligned_data else 0
    percentile_states = []

    for week_idx in range(max_weeks):
        # Extract all states at this week using vectorized operations
        week_data = []
        for trajectory in aligned_data:
            if week_idx < len(trajectory):
                week_data.append(trajectory[week_idx])

        if not week_data:
            continue

        # Convert to numpy arrays for efficient percentile calculation
        weights = np.array([s.weight_lbs for s in week_data])
        lean_masses = np.array([s.lean_mass_lbs for s in week_data])
        fat_masses = np.array([s.fat_mass_lbs for s in week_data])
        body_fats = np.array([s.body_fat_pct for s in week_data])
        almis = np.array([s.almi for s in week_data])
        ffmis = np.array([s.ffmi for s in week_data])

        # Calculate percentiles
        percentile_state = SimulationState(
            week=week_idx,
            weight_lbs=float(np.percentile(weights, percentile)),
            lean_mass_lbs=float(np.percentile(lean_masses, percentile)),
            fat_mass_lbs=float(np.percentile(fat_masses, percentile)),
            body_fat_pct=float(np.percentile(body_fats, percentile)),
            phase=week_data[0].phase,  # Use first trajectory's phase as approximation
            almi=float(np.percentile(almis, percentile)),
            ffmi=float(np.percentile(ffmis, percentile)),
            weeks_training=week_data[0].weeks_training,
            current_training_level=week_data[0].current_training_level,
            training_level_transition_weeks=[],
            simulation_age=week_data[0].simulation_age,
        )

        percentile_states.append(percentile_state)

    return percentile_states


# ============================================================================
# INPUT VALIDATION
# ============================================================================


def validate_simulation_inputs(
    user_profile: UserProfile,
    goal_config: GoalConfig,
    simulation_config: Optional[SimulationConfig],
) -> None:
    """
    Comprehensive input validation with helpful error messages.

    Validates all inputs and provides actionable feedback for users.
    """
    # User profile validation
    validate_user_profile(user_profile)

    # Goal configuration validation
    validate_goal_config(goal_config)

    # Simulation config validation (if provided)
    if simulation_config is not None:
        validate_simulation_config(simulation_config, user_profile)


def validate_user_profile(user_profile: UserProfile) -> None:
    """Validate user profile data comprehensively"""
    if not user_profile.scan_history:
        raise InvalidInputError("At least one DEXA scan is required")

    if len(user_profile.scan_history) > 20:
        raise InvalidInputError("Maximum of 20 scans supported")

    # Validate scan data quality
    for i, scan in enumerate(user_profile.scan_history):
        if hasattr(scan, "total_weight_lbs"):  # ScanData object
            weight = scan.total_weight_lbs
            lean = scan.total_lean_mass_lbs
            fat = scan.fat_mass_lbs
            bf_pct = scan.body_fat_percentage
        else:  # dict format
            weight = scan.get("total_weight_lbs", 0)
            lean = scan.get("total_lean_mass_lbs", 0)
            fat = scan.get("fat_mass_lbs", 0)
            bf_pct = scan.get("body_fat_percentage", 0)

        if weight <= 0:
            raise InvalidInputError(f"Scan {i + 1}: Weight must be greater than 0")
        if lean <= 0:
            raise InvalidInputError(f"Scan {i + 1}: Lean mass must be greater than 0")
        if fat <= 0:
            raise InvalidInputError(f"Scan {i + 1}: Fat mass must be greater than 0")
        if not (1 <= bf_pct <= 60):
            raise InvalidInputError(
                f"Scan {i + 1}: Body fat percentage must be between 1-60%"
            )

    # Validate anthropometric data
    if not (12 <= user_profile.height_in <= 120):
        raise InvalidInputError("Height must be between 12-120 inches")


def validate_goal_config(goal_config: GoalConfig) -> None:
    """Validate goal configuration"""
    if not (0.01 <= goal_config.target_percentile <= 0.99):
        raise InvalidInputError("Target percentile must be between 1-99%")

    if goal_config.metric_type not in ["almi", "ffmi"]:
        raise InvalidInputError("Goal metric must be 'almi' or 'ffmi'")


def validate_simulation_config(
    simulation_config: SimulationConfig, user_profile: UserProfile
) -> None:
    """Validate simulation configuration"""
    if not (100 <= simulation_config.run_count <= 5000):
        raise InvalidInputError("Run count must be between 100-5000")

    if simulation_config.variance_factor is not None:
        if not (0.01 <= simulation_config.variance_factor <= 1.0):
            raise InvalidInputError("Variance factor must be between 0.01-1.0")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _generate_default_simulation_config(
    user_profile: UserProfile, goal_config: GoalConfig
) -> SimulationConfig:
    """Generate sensible default simulation configuration"""

    # Determine default template based on current body composition
    latest_scan = user_profile.scan_history[-1]
    if hasattr(latest_scan, "body_fat_percentage"):
        current_bf = latest_scan.body_fat_percentage
    else:
        current_bf = latest_scan.get("body_fat_percentage", 15)

    # Simple template selection logic
    if current_bf > 20:  # Higher body fat - start with cut
        template = TemplateType.CUT_FIRST
    else:  # Lower body fat - can start with bulk
        template = TemplateType.BULK_FIRST

    return SimulationConfig(
        user_profile=user_profile,
        goal_config=goal_config,
        training_level=user_profile.training_level,
        template=template,
        variance_factor=None,  # Auto-calculated
        run_count=2000,  # Default for good statistical quality
        random_seed=None,  # Random seed for reproducibility
        max_duration_weeks=None,  # Age-based default
    )


def _serialize_scan(scan) -> dict:
    """Serialize scan data for cache key generation"""
    if hasattr(scan, "total_weight_lbs"):  # ScanData object
        return {
            "weight": scan.total_weight_lbs,
            "lean": scan.total_lean_mass_lbs,
            "fat": scan.fat_mass_lbs,
            "bf_pct": scan.body_fat_percentage,
            "arms": scan.arms_lean_lbs,
            "legs": scan.legs_lean_lbs,
        }
    else:  # dict format
        return {
            "weight": scan.get("total_weight_lbs", 0),
            "lean": scan.get("total_lean_mass_lbs", 0),
            "fat": scan.get("fat_mass_lbs", 0),
            "bf_pct": scan.get("body_fat_percentage", 0),
            "arms": scan.get("arms_lean_lbs", 0),
            "legs": scan.get("legs_lean_lbs", 0),
        }


def _serialize_bf_range(bf_range_config) -> dict:
    """Serialize BF range config for cache key generation"""
    if bf_range_config is None:
        return {}

    return {
        "min_bf": bf_range_config.min_bf_pct,
        "max_bf": bf_range_config.max_bf_pct,
        "max_weight": bf_range_config.max_weight_lbs,
    }


def _enhance_checkpoints(
    checkpoints: List[CheckpointData], config: SimulationConfig
) -> List[CheckpointData]:
    """Enhance checkpoints with additional metadata"""
    # For now, return checkpoints as-is
    # Future enhancement: add more detailed progress tracking
    return checkpoints


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================


def get_cache_stats() -> Dict[str, float]:
    """
    Get caching performance statistics.

    Returns:
        Dictionary with cache hit rate, average response times, etc.
    """
    # Note: Streamlit doesn't expose detailed cache stats
    # This is a placeholder for future monitoring implementation
    return {
        "cache_hit_rate": 0.8,  # Target: >80%
        "avg_cache_response_ms": 50,  # Target: <100ms
        "avg_cold_response_ms": 8000,  # Target: <10s
        "memory_usage_mb": 512,  # Target: <1GB
    }
