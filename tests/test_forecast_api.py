"""
Integration tests for the Forecast API Layer

Tests caching behavior, data consistency, performance bounds,
and memory limits for the forecast API implementation.
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from forecast_api import (
    CacheError,
    ConvergenceError,
    InvalidInputError,
    SimulationError,
    calculate_percentile_bands,
    calculate_rms_distance,
    generate_cache_key,
    get_plan,
    select_representative_path,
    validate_simulation_inputs,
)
from shared_models import (
    CheckpointData,
    ForecastPlan,
    GoalConfig,
    PercentileBands,
    PhaseType,
    ScanData,
    SimulationConfig,
    SimulationState,
    TemplateType,
    TrainingLevel,
    UserProfile,
)


class TestCacheBehavior:
    """Test caching behavior under different scenarios"""

    def test_cache_key_generation_consistency(
        self, sample_user_profile, sample_goal_config
    ):
        """Test that cache keys are consistent for identical inputs"""
        config = SimulationConfig(
            user_profile=sample_user_profile,
            goal_config=sample_goal_config,
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.CUT_FIRST,
            run_count=2000,
        )

        # Generate cache key multiple times
        key1 = generate_cache_key(sample_user_profile, sample_goal_config, config)
        key2 = generate_cache_key(sample_user_profile, sample_goal_config, config)

        assert key1 == key2
        assert len(key1) == 64  # SHA256 hex length

    def test_cache_key_different_for_different_inputs(
        self, sample_user_profile, sample_goal_config
    ):
        """Test that cache keys differ for different inputs"""
        config1 = SimulationConfig(
            user_profile=sample_user_profile,
            goal_config=sample_goal_config,
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.CUT_FIRST,
            run_count=2000,
        )

        config2 = SimulationConfig(
            user_profile=sample_user_profile,
            goal_config=sample_goal_config,
            training_level=TrainingLevel.ADVANCED,  # Different training level
            template=TemplateType.CUT_FIRST,
            run_count=2000,
        )

        key1 = generate_cache_key(sample_user_profile, sample_goal_config, config1)
        key2 = generate_cache_key(sample_user_profile, sample_goal_config, config2)

        assert key1 != key2

    def test_cache_key_handles_none_values(
        self, sample_user_profile, sample_goal_config
    ):
        """Test cache key generation with None values"""
        config = SimulationConfig(
            user_profile=sample_user_profile,
            goal_config=sample_goal_config,
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.CUT_FIRST,
            variance_factor=None,  # None value
            run_count=2000,
        )

        # Should not raise an exception
        key = generate_cache_key(sample_user_profile, sample_goal_config, config)
        assert isinstance(key, str)
        assert len(key) == 64


class TestDataConsistency:
    """Test that processed results match raw simulation data"""

    def test_percentile_bands_calculation(self, sample_trajectories):
        """Test percentile band calculation consistency"""
        bands = calculate_percentile_bands(sample_trajectories)

        # Verify structure
        assert isinstance(bands, PercentileBands)
        assert len(bands.p10) > 0
        assert len(bands.p25) > 0
        assert len(bands.p50) > 0
        assert len(bands.p75) > 0
        assert len(bands.p90) > 0

        # Verify ordering (p10 <= p25 <= p50 <= p75 <= p90)
        for i in range(len(bands.p10)):
            if (
                i < len(bands.p25)
                and i < len(bands.p50)
                and i < len(bands.p75)
                and i < len(bands.p90)
            ):
                assert bands.p10[i].weight_lbs <= bands.p25[i].weight_lbs
                assert bands.p25[i].weight_lbs <= bands.p50[i].weight_lbs
                assert bands.p50[i].weight_lbs <= bands.p75[i].weight_lbs
                assert bands.p75[i].weight_lbs <= bands.p90[i].weight_lbs

    def test_representative_path_selection(
        self, sample_trajectories, sample_checkpoints
    ):
        """Test representative path selection algorithm"""
        path = select_representative_path(sample_trajectories, sample_checkpoints)

        # Should return one of the original trajectories
        assert path in sample_trajectories
        assert len(path) > 0

        # Should be the one with minimum RMS distance
        distances = []
        for traj in sample_trajectories:
            distance = calculate_rms_distance(traj, sample_checkpoints)
            distances.append(distance)

        min_distance_idx = distances.index(min(distances))
        expected_path = sample_trajectories[min_distance_idx]
        assert path == expected_path

    def test_rms_distance_calculation(self, sample_trajectory, sample_checkpoints):
        """Test RMS distance calculation"""
        distance = calculate_rms_distance(sample_trajectory, sample_checkpoints)

        assert isinstance(distance, float)
        assert distance >= 0.0
        assert not np.isnan(distance)
        assert not np.isinf(distance)

    def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        # Empty trajectories
        bands = calculate_percentile_bands([])
        assert len(bands.p10) == 0

        # Empty checkpoints
        path = select_representative_path([[]], [])
        assert len(path) == 1  # Should return the single empty trajectory

        # Empty trajectory for RMS distance
        distance = calculate_rms_distance([], [])
        assert distance == float("inf")


class TestPerformanceBounds:
    """Test that performance stays within acceptable bounds"""

    @patch("forecast_api._cached_simulation")
    def test_response_time_bounds(
        self, mock_cached_sim, sample_user_profile, sample_goal_config
    ):
        """Test that API response times stay within bounds"""
        # Mock a fast cache hit
        mock_plan = Mock(spec=ForecastPlan)
        mock_plan.cache_hit = True
        mock_plan.simulation_time_ms = 50
        mock_cached_sim.return_value = mock_plan

        start_time = time.time()
        result = get_plan(sample_user_profile, sample_goal_config)
        end_time = time.time()

        response_time_ms = (end_time - start_time) * 1000

        # Cache hits should be very fast (< 200ms including overhead)
        assert response_time_ms < 200
        assert result.cache_hit is True

    def test_memory_efficiency_large_trajectories(self):
        """Test memory efficiency with large trajectory datasets"""
        # Create large dataset
        large_trajectories = []
        for i in range(100):  # 100 trajectories
            trajectory = []
            for week in range(200):  # 200 weeks each
                state = SimulationState(
                    week=week,
                    weight_lbs=180.0 + np.random.normal(0, 5),
                    lean_mass_lbs=140.0 + np.random.normal(0, 3),
                    fat_mass_lbs=40.0 + np.random.normal(0, 2),
                    body_fat_pct=22.0 + np.random.normal(0, 1),
                    phase=PhaseType.BULK,
                    almi=8.5 + np.random.normal(0, 0.5),
                    ffmi=19.0 + np.random.normal(0, 0.5),
                    weeks_training=week,
                    current_training_level=TrainingLevel.INTERMEDIATE,
                    training_level_transition_weeks=[],
                    simulation_age=25.0 + week / 52.0,
                )
                trajectory.append(state)
            large_trajectories.append(trajectory)

        # Test percentile calculation with large dataset
        start_time = time.time()
        bands = calculate_percentile_bands(large_trajectories)
        end_time = time.time()

        # Should complete in reasonable time (< 5 seconds)
        processing_time = end_time - start_time
        assert processing_time < 5.0

        # Should produce valid results
        assert len(bands.p50) == 200  # All weeks present
        assert all(isinstance(state, SimulationState) for state in bands.p50)


class TestInputValidation:
    """Test comprehensive input validation"""

    def test_valid_inputs_pass_validation(
        self, sample_user_profile, sample_goal_config
    ):
        """Test that valid inputs pass validation without errors"""
        config = SimulationConfig(
            user_profile=sample_user_profile,
            goal_config=sample_goal_config,
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.CUT_FIRST,
            run_count=2000,
        )

        # Should not raise any exceptions
        validate_simulation_inputs(sample_user_profile, sample_goal_config, config)

    def test_invalid_goal_percentile_raises_error(self, sample_user_profile):
        """Test that invalid goal percentiles raise appropriate errors"""
        invalid_goal = GoalConfig(
            metric_type="almi",
            target_percentile=1.5,  # Invalid: > 1.0
        )

        with pytest.raises(
            InvalidInputError, match="Target percentile must be between"
        ):
            validate_simulation_inputs(sample_user_profile, invalid_goal, None)

    def test_invalid_run_count_raises_error(
        self, sample_user_profile, sample_goal_config
    ):
        """Test that invalid run counts raise appropriate errors"""
        invalid_config = SimulationConfig(
            user_profile=sample_user_profile,
            goal_config=sample_goal_config,
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.CUT_FIRST,
            run_count=50,  # Invalid: < 100
        )

        with pytest.raises(InvalidInputError, match="Run count must be between"):
            validate_simulation_inputs(
                sample_user_profile, sample_goal_config, invalid_config
            )

    def test_empty_scan_history_raises_error(self, sample_goal_config):
        """Test that empty scan history raises appropriate error"""
        invalid_profile = UserProfile(
            birth_date="01/01/1990",
            height_in=70.0,
            gender="male",
            training_level=TrainingLevel.INTERMEDIATE,
            scan_history=[],  # Empty scan history
        )

        with pytest.raises(
            InvalidInputError, match="At least one DEXA scan is required"
        ):
            validate_simulation_inputs(invalid_profile, sample_goal_config, None)

    def test_invalid_scan_data_raises_error(self, sample_goal_config):
        """Test that invalid scan data raises appropriate errors"""
        invalid_scan = ScanData(
            date="01/01/2023",
            total_weight_lbs=0.0,  # Invalid: <= 0
            total_lean_mass_lbs=140.0,
            fat_mass_lbs=40.0,
            body_fat_percentage=22.0,
            arms_lean_lbs=25.0,
            legs_lean_lbs=45.0,
        )

        invalid_profile = UserProfile(
            birth_date="01/01/1990",
            height_in=70.0,
            gender="male",
            training_level=TrainingLevel.INTERMEDIATE,
            scan_history=[invalid_scan],
        )

        with pytest.raises(InvalidInputError, match="Weight must be greater than 0"):
            validate_simulation_inputs(invalid_profile, sample_goal_config, None)


class TestErrorHandling:
    """Test error handling and graceful degradation"""

    @patch("forecast_api.MonteCarloEngine")
    def test_simulation_error_handling(
        self, mock_engine_class, sample_user_profile, sample_goal_config
    ):
        """Test that simulation errors are properly handled"""
        # Mock engine to raise an error
        mock_engine = Mock()
        mock_engine.run_simulation.side_effect = ValueError("Simulation failed")
        mock_engine_class.return_value = mock_engine

        with pytest.raises(SimulationError, match="Simulation execution failed"):
            get_plan(sample_user_profile, sample_goal_config)

    @patch("forecast_api.MonteCarloEngine")
    def test_graceful_degradation(
        self, mock_engine_class, sample_user_profile, sample_goal_config
    ):
        """Test graceful degradation with reduced parameters"""
        # Mock engine to fail on first attempt, succeed on second
        mock_engine = Mock()
        mock_engine.run_simulation.side_effect = [
            ValueError("Memory error"),  # First attempt fails
            Mock(  # Second attempt succeeds
                goal_achievement_week=52,
                goal_achievement_age=26.0,
                convergence_quality=0.85,
                total_phases=3,
                trajectories=[[]],
                median_checkpoints=[],
                percentile_bands={},
            ),
        ]
        mock_engine_class.return_value = mock_engine

        # Should attempt graceful degradation and succeed
        config = SimulationConfig(
            user_profile=sample_user_profile,
            goal_config=sample_goal_config,
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.CUT_FIRST,
            run_count=2000,  # Will be reduced to 1000 on retry
        )

        get_plan(sample_user_profile, sample_goal_config, config)

        # Should have called engine twice (original + degraded attempt)
        assert mock_engine.run_simulation.call_count == 2


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_user_profile():
    """Create sample user profile for testing"""
    scan = ScanData(
        date="01/01/2023",
        total_weight_lbs=180.0,
        total_lean_mass_lbs=140.0,
        fat_mass_lbs=40.0,
        body_fat_percentage=22.0,
        arms_lean_lbs=25.0,
        legs_lean_lbs=45.0,
    )

    return UserProfile(
        birth_date="01/01/1990",
        height_in=70.0,
        gender="male",
        training_level=TrainingLevel.INTERMEDIATE,
        scan_history=[scan],
    )


@pytest.fixture
def sample_goal_config():
    """Create sample goal configuration for testing"""
    return GoalConfig(metric_type="almi", target_percentile=0.75)


@pytest.fixture
def sample_trajectories():
    """Create sample trajectories for testing"""
    trajectories = []

    for i in range(5):  # 5 trajectories
        trajectory = []
        for week in range(20):  # 20 weeks each
            state = SimulationState(
                week=week,
                weight_lbs=180.0 + i * 2 + week * 0.1,  # Slight variation
                lean_mass_lbs=140.0 + i + week * 0.05,
                fat_mass_lbs=40.0 + i - week * 0.05,
                body_fat_pct=22.0 + i * 0.5 - week * 0.1,
                phase=PhaseType.BULK if week < 10 else PhaseType.CUT,
                almi=8.5 + i * 0.1 + week * 0.01,
                ffmi=19.0 + i * 0.1 + week * 0.01,
                weeks_training=week,
                current_training_level=TrainingLevel.INTERMEDIATE,
                training_level_transition_weeks=[],
                simulation_age=25.0 + week / 52.0,
            )
            trajectory.append(state)
        trajectories.append(trajectory)

    return trajectories


@pytest.fixture
def sample_trajectory():
    """Create single sample trajectory for testing"""
    trajectory = []
    for week in range(10):
        state = SimulationState(
            week=week,
            weight_lbs=180.0 + week * 0.1,
            lean_mass_lbs=140.0 + week * 0.05,
            fat_mass_lbs=40.0 - week * 0.05,
            body_fat_pct=22.0 - week * 0.1,
            phase=PhaseType.BULK,
            almi=8.5 + week * 0.01,
            ffmi=19.0 + week * 0.01,
            weeks_training=week,
            current_training_level=TrainingLevel.INTERMEDIATE,
            training_level_transition_weeks=[],
            simulation_age=25.0 + week / 52.0,
        )
        trajectory.append(state)
    return trajectory


@pytest.fixture
def sample_checkpoints():
    """Create sample checkpoints for testing"""
    return [
        CheckpointData(
            week=5,
            phase=PhaseType.BULK,
            weight_lbs=182.0,
            body_fat_pct=21.5,
            lean_mass_lbs=141.0,
            fat_mass_lbs=41.0,
            almi=8.6,
            ffmi=19.1,
            percentile_progress=0.72,
        ),
        CheckpointData(
            week=15,
            phase=PhaseType.CUT,
            weight_lbs=178.0,
            body_fat_pct=20.0,
            lean_mass_lbs=142.0,
            fat_mass_lbs=36.0,
            almi=8.7,
            ffmi=19.2,
            percentile_progress=0.75,
        ),
    ]
