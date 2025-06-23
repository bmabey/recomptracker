"""
Test suite for weight constraint features in Monte Carlo simulation and phase planning.

This module tests the weight constraint functionality that allows users to specify
maximum weight limits that act as dynamic body fat percentage constraints.
"""

import unittest
from unittest.mock import patch

import numpy as np

from core import (
    calculate_value_from_percentile_cached,
    validate_goal_feasibility_with_weight_constraint,
)
from mc_forecast import MonteCarloEngine, create_simulation_engine
from phase_planning import PhaseTemplateEngine, PhaseTransitionManager, RateCalculator
from shared_models import (
    BFRangeConfig,
    GoalConfig,
    PhaseType,
    SimulationConfig,
    TemplateType,
    TrainingLevel,
    UserProfile,
)


class TestBFRangeConfigValidation(unittest.TestCase):
    """Test BFRangeConfig validation with weight constraints"""

    def test_valid_bf_range_config_without_weight(self):
        """Test that BFRangeConfig works without weight constraint"""
        config = BFRangeConfig(min_bf_pct=10.0, max_bf_pct=15.0)
        self.assertEqual(config.min_bf_pct, 10.0)
        self.assertEqual(config.max_bf_pct, 15.0)
        self.assertIsNone(config.max_weight_lbs)

    def test_valid_bf_range_config_with_weight(self):
        """Test that BFRangeConfig works with weight constraint"""
        config = BFRangeConfig(min_bf_pct=10.0, max_bf_pct=15.0, max_weight_lbs=180.0)
        self.assertEqual(config.min_bf_pct, 10.0)
        self.assertEqual(config.max_bf_pct, 15.0)
        self.assertEqual(config.max_weight_lbs, 180.0)

    def test_invalid_weight_constraint_negative(self):
        """Test that negative weight constraint raises error"""
        with self.assertRaises(ValueError) as cm:
            BFRangeConfig(min_bf_pct=10.0, max_bf_pct=15.0, max_weight_lbs=-10.0)
        self.assertIn("must be positive", str(cm.exception))

    def test_invalid_weight_constraint_zero(self):
        """Test that zero weight constraint raises error"""
        with self.assertRaises(ValueError) as cm:
            BFRangeConfig(min_bf_pct=10.0, max_bf_pct=15.0, max_weight_lbs=0.0)
        self.assertIn("must be positive", str(cm.exception))

    def test_invalid_weight_constraint_too_low(self):
        """Test that unreasonably low weight constraint raises error"""
        with self.assertRaises(ValueError) as cm:
            BFRangeConfig(min_bf_pct=10.0, max_bf_pct=15.0, max_weight_lbs=50.0)
        self.assertIn("must be reasonable (80-500 lbs)", str(cm.exception))

    def test_invalid_weight_constraint_too_high(self):
        """Test that unreasonably high weight constraint raises error"""
        with self.assertRaises(ValueError) as cm:
            BFRangeConfig(min_bf_pct=10.0, max_bf_pct=15.0, max_weight_lbs=600.0)
        self.assertIn("must be reasonable (80-500 lbs)", str(cm.exception))


class TestGoalFeasibilityValidation(unittest.TestCase):
    """Test goal feasibility validation with weight constraints"""

    def setUp(self):
        """Set up test user profile"""
        self.user_profile = UserProfile(
            birth_date="01/01/1990",
            height_in=70.0,
            gender="male",
            training_level=TrainingLevel.INTERMEDIATE,
            scan_history=[
                {
                    "date": "01/01/2023",
                    "total_weight_lbs": 170.0,
                    "total_lean_mass_lbs": 140.0,
                    "fat_mass_lbs": 30.0,
                    "body_fat_percentage": 17.6,
                    "arms_lean_lbs": 20.0,
                    "legs_lean_lbs": 50.0,
                }
            ],
        )

    def test_validation_passes_with_no_weight_constraint(self):
        """Test that validation passes when no weight constraint is provided"""
        goal_config = GoalConfig(metric_type="almi", target_percentile=0.75)
        bf_range_config = BFRangeConfig(min_bf_pct=10.0, max_bf_pct=15.0)

        is_feasible, error_msg, min_weight = (
            validate_goal_feasibility_with_weight_constraint(
                self.user_profile, goal_config, bf_range_config, 33.0
            )
        )

        self.assertTrue(is_feasible)
        self.assertIsNone(error_msg)
        self.assertIsNone(min_weight)

    def test_validation_passes_with_no_bf_range_config(self):
        """Test that validation passes when no bf_range_config is provided"""
        goal_config = GoalConfig(metric_type="almi", target_percentile=0.75)

        is_feasible, error_msg, min_weight = (
            validate_goal_feasibility_with_weight_constraint(
                self.user_profile, goal_config, None, 33.0
            )
        )

        self.assertTrue(is_feasible)
        self.assertIsNone(error_msg)
        self.assertIsNone(min_weight)

    def test_validation_skips_non_almi_goals(self):
        """Test that validation is skipped for non-ALMI goals"""
        goal_config = GoalConfig(metric_type="ffmi", target_percentile=0.75)
        bf_range_config = BFRangeConfig(
            min_bf_pct=10.0, max_bf_pct=15.0, max_weight_lbs=160.0
        )

        is_feasible, error_msg, min_weight = (
            validate_goal_feasibility_with_weight_constraint(
                self.user_profile, goal_config, bf_range_config, 33.0
            )
        )

        self.assertTrue(is_feasible)
        self.assertIsNone(error_msg)
        self.assertIsNone(min_weight)

    @patch("core.calculate_value_from_percentile_cached")
    def test_validation_fails_with_restrictive_weight_constraint(self, mock_calc):
        """Test that validation fails when weight constraint is too restrictive"""
        # Mock LMS calculation to return a high ALMI target
        mock_calc.return_value = (
            10.0  # High ALMI value that requires significant muscle mass
        )

        goal_config = GoalConfig(metric_type="almi", target_percentile=0.95)
        bf_range_config = BFRangeConfig(
            min_bf_pct=10.0, max_bf_pct=15.0, max_weight_lbs=160.0
        )

        is_feasible, error_msg, min_weight = (
            validate_goal_feasibility_with_weight_constraint(
                self.user_profile, goal_config, bf_range_config, 33.0
            )
        )

        self.assertFalse(is_feasible)
        self.assertIsNotNone(error_msg)
        self.assertIn("Goal not achievable within weight constraint", error_msg)
        self.assertIn("160.0 lbs", error_msg)
        self.assertIsNotNone(min_weight)

    @patch("core.calculate_value_from_percentile_cached")
    def test_validation_passes_with_reasonable_weight_constraint(self, mock_calc):
        """Test that validation passes when weight constraint is reasonable"""
        # Mock LMS calculation to return a reasonable ALMI target
        mock_calc.return_value = 8.5  # Reasonable ALMI value

        goal_config = GoalConfig(metric_type="almi", target_percentile=0.75)
        bf_range_config = BFRangeConfig(
            min_bf_pct=10.0, max_bf_pct=15.0, max_weight_lbs=200.0
        )

        is_feasible, error_msg, min_weight = (
            validate_goal_feasibility_with_weight_constraint(
                self.user_profile, goal_config, bf_range_config, 33.0
            )
        )

        self.assertTrue(is_feasible)
        self.assertIsNone(error_msg)
        self.assertIsNotNone(min_weight)
        self.assertLess(min_weight, 200.0)


class TestPhaseTransitionWithWeightConstraints(unittest.TestCase):
    """Test phase transitions with weight constraints"""

    def setUp(self):
        """Set up phase transition manager"""
        self.rate_calculator = RateCalculator()
        self.transition_manager = PhaseTransitionManager(self.rate_calculator)

    def test_transition_without_weight_constraint(self):
        """Test that transitions work normally without weight constraints"""
        from shared_models import PhaseConfig

        phase_config = PhaseConfig(
            phase_type=PhaseType.BULK,
            target_bf_pct=20.0,
            min_duration_weeks=8,
            max_duration_weeks=24,
            rate_pct_per_week=0.5,
            rationale="Test bulk phase",
        )

        # Should not transition before target reached
        should_transition = self.transition_manager.should_transition(
            current_bf_pct=18.0,
            current_phase=PhaseType.BULK,
            phase_config=phase_config,
            weeks_in_phase=10,
            gender="male",
        )
        self.assertFalse(should_transition)

        # Should transition when target reached
        should_transition = self.transition_manager.should_transition(
            current_bf_pct=20.0,
            current_phase=PhaseType.BULK,
            phase_config=phase_config,
            weeks_in_phase=10,
            gender="male",
        )
        self.assertTrue(should_transition)

    def test_transition_with_weight_constraint_not_triggered(self):
        """Test that weight constraints don't trigger when under limit"""
        from shared_models import PhaseConfig

        phase_config = PhaseConfig(
            phase_type=PhaseType.BULK,
            target_bf_pct=20.0,
            min_duration_weeks=8,
            max_duration_weeks=24,
            rate_pct_per_week=0.5,
            rationale="Test bulk phase",
        )

        bf_range_config = BFRangeConfig(
            min_bf_pct=10.0, max_bf_pct=15.0, max_weight_lbs=200.0
        )

        # Should not transition when under weight limit
        should_transition = self.transition_manager.should_transition(
            current_bf_pct=18.0,
            current_phase=PhaseType.BULK,
            phase_config=phase_config,
            weeks_in_phase=10,
            gender="male",
            current_weight_lbs=180.0,
            bf_range_config=bf_range_config,
        )
        self.assertFalse(should_transition)

    def test_transition_with_weight_constraint_triggered(self):
        """Test that weight constraints trigger transitions when limit reached"""
        from shared_models import PhaseConfig

        phase_config = PhaseConfig(
            phase_type=PhaseType.BULK,
            target_bf_pct=20.0,
            min_duration_weeks=8,
            max_duration_weeks=24,
            rate_pct_per_week=0.5,
            rationale="Test bulk phase",
        )

        bf_range_config = BFRangeConfig(
            min_bf_pct=10.0, max_bf_pct=15.0, max_weight_lbs=200.0
        )

        # Should transition when weight limit reached during bulk
        should_transition = self.transition_manager.should_transition(
            current_bf_pct=18.0,
            current_phase=PhaseType.BULK,
            phase_config=phase_config,
            weeks_in_phase=10,
            gender="male",
            current_weight_lbs=200.0,
            bf_range_config=bf_range_config,
        )
        self.assertTrue(should_transition)

    def test_weight_constraint_only_affects_bulk_phases(self):
        """Test that weight constraints only trigger during bulk phases"""
        from shared_models import PhaseConfig

        phase_config = PhaseConfig(
            phase_type=PhaseType.CUT,
            target_bf_pct=12.0,
            min_duration_weeks=8,
            max_duration_weeks=24,
            rate_pct_per_week=0.75,
            rationale="Test cut phase",
        )

        bf_range_config = BFRangeConfig(
            min_bf_pct=10.0, max_bf_pct=15.0, max_weight_lbs=200.0
        )

        # Should not transition on weight during cut (weight would be decreasing)
        should_transition = self.transition_manager.should_transition(
            current_bf_pct=15.0,
            current_phase=PhaseType.CUT,
            phase_config=phase_config,
            weeks_in_phase=10,
            gender="male",
            current_weight_lbs=200.0,
            bf_range_config=bf_range_config,
        )
        self.assertFalse(should_transition)


class TestTemplateGenerationWithWeightConstraints(unittest.TestCase):
    """Test template generation with weight constraints"""

    def setUp(self):
        """Set up template engine"""
        self.rate_calculator = RateCalculator()
        self.template_engine = PhaseTemplateEngine(self.rate_calculator)

        self.user_profile = UserProfile(
            birth_date="01/01/1990",
            height_in=70.0,
            gender="male",
            training_level=TrainingLevel.INTERMEDIATE,
            scan_history=[
                {
                    "date": "01/01/2023",
                    "total_weight_lbs": 170.0,
                    "total_lean_mass_lbs": 140.0,
                    "fat_mass_lbs": 30.0,
                    "body_fat_percentage": 17.6,
                    "arms_lean_lbs": 20.0,
                    "legs_lean_lbs": 50.0,
                }
            ],
        )

    def test_template_generation_without_weight_constraint(self):
        """Test that template generation works normally without weight constraints"""
        sequence = self.template_engine.generate_sequence(
            TemplateType.CUT_FIRST, self.user_profile
        )

        self.assertIsNotNone(sequence)
        self.assertTrue(len(sequence.phases) > 0)

        # Find bulk phases and check they have reasonable targets
        bulk_phases = [p for p in sequence.phases if p.phase_type == PhaseType.BULK]
        self.assertTrue(len(bulk_phases) > 0)

        for bulk_phase in bulk_phases:
            self.assertGreater(
                bulk_phase.target_bf_pct, 15.0
            )  # Should allow reasonable bulk targets

    def test_template_generation_with_restrictive_weight_constraint(self):
        """Test that template generation adjusts targets with restrictive weight constraints"""
        # Very restrictive weight constraint (should force lower BF% than 15%)
        bf_range_config = BFRangeConfig(
            min_bf_pct=10.0, max_bf_pct=15.0, max_weight_lbs=160.0
        )

        sequence = self.template_engine.generate_sequence(
            TemplateType.CUT_FIRST, self.user_profile, bf_range_config=bf_range_config
        )

        self.assertIsNotNone(sequence)

        # Find bulk phases and check they have weight-constrained targets
        bulk_phases = [p for p in sequence.phases if p.phase_type == PhaseType.BULK]

        for bulk_phase in bulk_phases:
            # Should have lower targets due to weight constraint
            self.assertIn("160.0 lbs", bulk_phase.rationale)

    def test_effective_max_bf_calculation(self):
        """Test the effective max BF% calculation from weight constraints"""
        # Test with reasonable weight constraint
        max_bf = self.template_engine._estimate_effective_max_bf_from_weight(
            self.user_profile, 200.0
        )

        # Should return a reasonable BF% (current lean is 140 lbs, so 200-140=60 lbs fat)
        # 60/200 = 30% BF, which should be capped at male maximum of 25%
        self.assertLessEqual(max_bf, 25.0)
        self.assertGreater(max_bf, 10.0)

    def test_effective_max_bf_with_very_restrictive_weight(self):
        """Test effective max BF% with weight below current lean mass"""
        # Weight constraint below current lean mass (140 lbs)
        max_bf = self.template_engine._estimate_effective_max_bf_from_weight(
            self.user_profile, 130.0
        )

        # Should return very low BF% indicating cutting scenario
        self.assertEqual(max_bf, 5.0)


class TestMonteCarloSimulationWithWeightConstraints(unittest.TestCase):
    """Test end-to-end Monte Carlo simulation with weight constraints"""

    def setUp(self):
        """Set up simulation components"""
        self.user_profile = UserProfile(
            birth_date="01/01/1990",
            height_in=70.0,
            gender="male",
            training_level=TrainingLevel.INTERMEDIATE,
            scan_history=[
                {
                    "date": "01/01/2023",
                    "total_weight_lbs": 170.0,
                    "total_lean_mass_lbs": 140.0,
                    "fat_mass_lbs": 30.0,
                    "body_fat_percentage": 17.6,
                    "arms_lean_lbs": 20.0,
                    "legs_lean_lbs": 50.0,
                }
            ],
        )

        self.goal_config = GoalConfig(metric_type="almi", target_percentile=0.75)

    def test_simulation_runs_without_weight_constraint(self):
        """Test that simulation runs normally without weight constraints"""
        engine = create_simulation_engine(
            user_profile=self.user_profile,
            goal_config=self.goal_config,
            run_count=10,
            random_seed=42,
        )

        # Should run without errors
        results = engine.run_simulation()
        self.assertIsNotNone(results)
        self.assertEqual(len(results.trajectories), 10)

    def test_simulation_runs_with_reasonable_weight_constraint(self):
        """Test that simulation runs with reasonable weight constraints"""
        bf_range_config = BFRangeConfig(
            min_bf_pct=10.0, max_bf_pct=15.0, max_weight_lbs=200.0
        )

        config = SimulationConfig(
            user_profile=self.user_profile,
            goal_config=self.goal_config,
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.CUT_FIRST,
            variance_factor=0.25,
            bf_range_config=bf_range_config,
            random_seed=42,
            run_count=10,
        )

        engine = MonteCarloEngine(config)

        # Should run without errors
        results = engine.run_simulation()
        self.assertIsNotNone(results)
        self.assertEqual(len(results.trajectories), 10)

    def test_simulation_fails_with_impossible_weight_constraint(self):
        """Test that simulation fails upfront with impossible weight constraints"""
        # Very restrictive weight constraint that creates unsafe phase targets
        bf_range_config = BFRangeConfig(
            min_bf_pct=10.0, max_bf_pct=15.0, max_weight_lbs=150.0
        )

        config = SimulationConfig(
            user_profile=self.user_profile,
            goal_config=self.goal_config,
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.CUT_FIRST,
            variance_factor=0.25,
            bf_range_config=bf_range_config,
            random_seed=42,
            run_count=10,
        )

        # Should raise PhaseConfigError during initialization due to unsafe targets
        from phase_planning import PhaseConfigError

        with self.assertRaises(PhaseConfigError) as cm:
            MonteCarloEngine(config)

        self.assertIn("below safe minimum", str(cm.exception))


class TestInverseLMSCalculation(unittest.TestCase):
    """Test inverse LMS calculation for weight constraint validation"""

    @patch("core.load_lms_data")
    @patch("core.get_value_from_zscore")
    def test_calculate_value_from_percentile_cached(
        self, mock_get_value, mock_load_lms
    ):
        """Test the inverse LMS calculation function"""

        # Mock LMS data loading
        def mock_l_func(age):
            return 1.0

        def mock_m_func(age):
            return 8.0

        def mock_s_func(age):
            return 0.15

        mock_load_lms.return_value = (mock_l_func, mock_m_func, mock_s_func)

        # Mock inverse calculation
        mock_get_value.return_value = 8.5

        result = calculate_value_from_percentile_cached(
            target_percentile=0.75, age=30.0, metric="appendicular_LMI", gender_code=0
        )

        self.assertEqual(result, 8.5)
        mock_load_lms.assert_called_once()
        mock_get_value.assert_called_once()

    def test_calculate_value_from_percentile_invalid_input(self):
        """Test that invalid percentiles return NaN"""
        result = calculate_value_from_percentile_cached(
            target_percentile=1.5,  # Invalid percentile > 1.0
            age=30.0,
            metric="appendicular_LMI",
            gender_code=0,
        )

        self.assertTrue(np.isnan(result))

    def test_calculate_value_from_percentile_edge_cases(self):
        """Test edge cases for percentile calculation"""
        # Test 0th percentile
        calculate_value_from_percentile_cached(
            target_percentile=0.0, age=30.0, metric="appendicular_LMI", gender_code=0
        )

        # Test 100th percentile
        calculate_value_from_percentile_cached(
            target_percentile=1.0, age=30.0, metric="appendicular_LMI", gender_code=0
        )

        # Should handle edge cases without crashing
        # Note: Actual values depend on LMS data availability in test environment


if __name__ == "__main__":
    unittest.main(verbosity=2)
