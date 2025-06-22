"""
Comprehensive test suite for Monte Carlo simulation engine

This module contains unit tests, integration tests, and canned profile tests
for the Monte Carlo forecasting engine. Tests cover all major functionality
including P-ratio calculations, phase transitions, goal detection, and
statistical properties of simulation results.
"""

import tempfile
import unittest
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np

from mc_forecast import (
    BF_THRESHOLDS,
    P_RATIO_DEFAULTS,
    TRAINING_VARIANCE,
    GoalConfig,
    MonteCarloEngine,
    PhaseType,
    SimulationConfig,
    TemplateType,
    TrainingLevel,
    UserProfile,
    create_simulation_engine,
)


class TestPRatioCalculations(unittest.TestCase):
    """Test P-ratio calculation logic"""

    def setUp(self):
        """Set up test user profile"""
        self.user_profile = UserProfile(
            birth_date="04/26/1982",
            height_in=66.0,
            gender="male",
            training_level=TrainingLevel.INTERMEDIATE,
            scan_history=[
                {
                    "date": "04/07/2022",
                    "total_weight_lbs": 150.0,
                    "total_lean_mass_lbs": 110.0,
                    "fat_mass_lbs": 40.0,
                    "body_fat_percentage": 26.7,
                    "arms_lean_lbs": 15.0,
                    "legs_lean_lbs": 35.0,
                }
            ],
        )

        self.goal_config = GoalConfig(metric_type="almi", target_percentile=0.75)

        self.config = SimulationConfig(
            user_profile=self.user_profile,
            goal_config=self.goal_config,
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.CUT_FIRST,
            variance_factor=0.25,
            random_seed=42,
        )

        self.engine = MonteCarloEngine(self.config)

    def test_bulk_p_ratio_independent_of_bf(self):
        """Test that bulking P-ratio is independent of body fat percentage"""

        # Test various body fat levels
        bf_levels = [10, 15, 20, 25, 30]

        bulk_ratios = []
        for bf in bf_levels:
            ratio = self.engine._get_p_ratio(PhaseType.BULK, bf)
            bulk_ratios.append(ratio)

        # All bulk ratios should be similar (within expected range)
        expected_min, expected_max = P_RATIO_DEFAULTS["bulk_any_bf"]

        for ratio in bulk_ratios:
            self.assertGreaterEqual(ratio, expected_min - 0.05)
            self.assertLessEqual(ratio, expected_max + 0.05)

        # Variance between ratios should be minimal
        ratio_variance = np.var(bulk_ratios)
        self.assertLess(
            ratio_variance,
            0.01,
            "Bulk P-ratios should not vary significantly with body fat",
        )

    def test_cut_p_ratio_depends_on_bf(self):
        """Test that cutting P-ratio depends on body fat level"""

        # High body fat cutting (>25% male)
        high_bf_ratio = self.engine._get_p_ratio(PhaseType.CUT, 30)

        # Moderate body fat cutting (15-25% male)
        mod_bf_ratio = self.engine._get_p_ratio(PhaseType.CUT, 18)

        # High BF cutting should have lower lean loss ratio
        self.assertLess(
            high_bf_ratio,
            mod_bf_ratio,
            "High body fat cutting should preserve more lean mass",
        )

        # Verify ratios are in expected ranges
        expected_high_min, expected_high_max = P_RATIO_DEFAULTS["cut_high_bf"]
        expected_mod_min, expected_mod_max = P_RATIO_DEFAULTS["cut_moderate_bf"]

        self.assertGreaterEqual(high_bf_ratio, expected_high_min - 0.05)
        self.assertLessEqual(high_bf_ratio, expected_high_max + 0.05)

        self.assertGreaterEqual(mod_bf_ratio, expected_mod_min - 0.05)
        self.assertLessEqual(mod_bf_ratio, expected_mod_max + 0.05)

    def test_female_bf_thresholds(self):
        """Test that female body fat thresholds are used correctly"""

        # Create female profile
        female_profile = self.user_profile
        female_profile.gender = "female"

        female_config = SimulationConfig(
            user_profile=female_profile,
            goal_config=self.goal_config,
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.CUT_FIRST,
            variance_factor=0.25,
            random_seed=42,
        )

        female_engine = MonteCarloEngine(female_config)

        # Test female thresholds (30% vs 25% for males)
        high_bf_ratio = female_engine._get_p_ratio(PhaseType.CUT, 35)  # High for female
        mod_bf_ratio = female_engine._get_p_ratio(
            PhaseType.CUT, 25
        )  # Moderate for female

        self.assertLess(high_bf_ratio, mod_bf_ratio)


class TestPhaseTransitions(unittest.TestCase):
    """Test phase transition logic"""

    def setUp(self):
        """Set up test configuration"""
        self.user_profile = UserProfile(
            birth_date="04/26/1982",
            height_in=66.0,
            gender="male",
            training_level=TrainingLevel.INTERMEDIATE,
            scan_history=[
                {
                    "date": "04/07/2022",
                    "total_weight_lbs": 180.0,
                    "total_lean_mass_lbs": 135.0,
                    "fat_mass_lbs": 45.0,
                    "body_fat_percentage": 25.0,  # Right at threshold
                    "arms_lean_lbs": 18.0,
                    "legs_lean_lbs": 45.0,
                }
            ],
        )

        self.goal_config = GoalConfig(metric_type="almi", target_percentile=0.85)

    def test_cut_first_template_logic(self):
        """Test Cut-First template initial phase selection"""

        # High BF should start with cut
        from dataclasses import replace

        high_bf_profile = replace(
            self.user_profile,
            scan_history=[
                replace(self.user_profile.scan_history[0], body_fat_percentage=28.0)
            ],
        )

        config = SimulationConfig(
            user_profile=high_bf_profile,
            goal_config=self.goal_config,
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.CUT_FIRST,
            variance_factor=0.25,
        )

        engine = MonteCarloEngine(config)
        initial_state = engine._create_initial_state(high_bf_profile.scan_history[0])
        initial_phase = engine._determine_initial_phase(initial_state)

        self.assertEqual(
            initial_phase,
            PhaseType.CUT,
            "High body fat should start with cutting phase",
        )

        # Low BF should start with bulk
        low_bf_profile = replace(
            self.user_profile,
            scan_history=[
                replace(self.user_profile.scan_history[0], body_fat_percentage=15.0)
            ],
        )

        config.user_profile = low_bf_profile
        engine = MonteCarloEngine(config)
        initial_state = engine._create_initial_state(low_bf_profile.scan_history[0])
        initial_phase = engine._determine_initial_phase(initial_state)

        self.assertEqual(
            initial_phase,
            PhaseType.BULK,
            "Low body fat should start with bulking phase",
        )

    def test_bulk_first_template_logic(self):
        """Test Bulk-First template initial phase selection"""

        config = SimulationConfig(
            user_profile=self.user_profile,
            goal_config=self.goal_config,
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.BULK_FIRST,
            variance_factor=0.25,
        )

        engine = MonteCarloEngine(config)
        initial_state = engine._create_initial_state(self.user_profile.scan_history[0])
        initial_phase = engine._determine_initial_phase(initial_state)

        # Should start with bulk unless extremely high BF
        self.assertEqual(
            initial_phase, PhaseType.BULK, "Bulk-First template should prefer bulking"
        )

        # Test extremely high BF override
        from dataclasses import replace

        extreme_bf_profile = replace(
            self.user_profile,
            scan_history=[
                replace(self.user_profile.scan_history[0], body_fat_percentage=35.0)
            ],
        )

        config.user_profile = extreme_bf_profile
        engine = MonteCarloEngine(config)
        initial_state = engine._create_initial_state(extreme_bf_profile.scan_history[0])
        initial_phase = engine._determine_initial_phase(initial_state)

        self.assertEqual(
            initial_phase,
            PhaseType.CUT,
            "Extremely high BF should override Bulk-First preference",
        )

    def test_phase_transition_thresholds(self):
        """Test body fat thresholds trigger phase transitions"""

        config = SimulationConfig(
            user_profile=self.user_profile,
            goal_config=self.goal_config,
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.CUT_FIRST,
            variance_factor=0.25,
        )

        engine = MonteCarloEngine(config)

        # Test cut -> bulk transition (reaching acceptable BF)
        cut_state = engine._create_initial_state(self.user_profile.scan_history[0])
        cut_state.body_fat_pct = 19.0  # Just below 20% threshold

        should_transition = engine._should_transition_phase(
            cut_state, PhaseType.CUT, 10
        )
        self.assertTrue(
            should_transition, "Should transition from cut to bulk at 19% BF"
        )

        # Test bulk -> cut transition (reaching upper limit)
        bulk_state = engine._create_initial_state(self.user_profile.scan_history[0])
        bulk_state.body_fat_pct = 24.0  # Above 20% + 3% = 23%

        should_transition = engine._should_transition_phase(
            bulk_state, PhaseType.BULK, 15
        )
        self.assertTrue(
            should_transition, "Should transition from bulk to cut at 24% BF"
        )

    def test_minimum_phase_durations(self):
        """Test minimum phase duration constraints"""

        config = SimulationConfig(
            user_profile=self.user_profile,
            goal_config=self.goal_config,
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.CUT_FIRST,
            variance_factor=0.25,
        )

        engine = MonteCarloEngine(config)
        state = engine._create_initial_state(self.user_profile.scan_history[0])
        state.body_fat_pct = 15.0  # Below threshold

        # Should not transition from cut before 8 weeks
        should_transition = engine._should_transition_phase(state, PhaseType.CUT, 6)
        self.assertFalse(
            should_transition, "Should not transition from cut before 8 weeks"
        )

        # Should transition after 8 weeks
        should_transition = engine._should_transition_phase(state, PhaseType.CUT, 10)
        self.assertTrue(should_transition, "Should transition from cut after 8 weeks")

        # Test bulk minimum duration
        state.body_fat_pct = 25.0  # Above threshold

        should_transition = engine._should_transition_phase(state, PhaseType.BULK, 10)
        self.assertFalse(
            should_transition, "Should not transition from bulk before 12 weeks"
        )

        should_transition = engine._should_transition_phase(state, PhaseType.BULK, 15)
        self.assertTrue(should_transition, "Should transition from bulk after 12 weeks")


class TestGoalDetection(unittest.TestCase):
    """Test goal achievement detection"""

    def setUp(self):
        """Set up test configuration"""
        self.user_profile = UserProfile(
            birth_date="04/26/1982",
            height_in=66.0,
            gender="male",
            training_level=TrainingLevel.INTERMEDIATE,
            scan_history=[
                {
                    "date": "04/07/2022",
                    "total_weight_lbs": 160.0,
                    "total_lean_mass_lbs": 120.0,
                    "fat_mass_lbs": 40.0,
                    "body_fat_percentage": 25.0,
                    "arms_lean_lbs": 16.0,
                    "legs_lean_lbs": 40.0,
                }
            ],
        )

    def test_almi_goal_detection(self):
        """Test ALMI goal achievement detection"""

        goal_config = GoalConfig(metric_type="almi", target_percentile=0.75)

        config = SimulationConfig(
            user_profile=self.user_profile,
            goal_config=goal_config,
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.CUT_FIRST,
            variance_factor=0.25,
        )

        engine = MonteCarloEngine(config)

        # Create state with high ALMI (should achieve goal)
        high_almi_state = engine._create_initial_state(
            self.user_profile.scan_history[0]
        )
        high_almi_state.almi = 9.3  # Should be >75th percentile for 43-year-old male

        goal_achieved = engine._goal_achieved(high_almi_state)
        self.assertTrue(goal_achieved, "High ALMI should achieve 75th percentile goal")

        # Create state with low ALMI (should not achieve goal)
        low_almi_state = engine._create_initial_state(self.user_profile.scan_history[0])
        low_almi_state.almi = 8.9  # Should be <75th percentile for 43-year-old male

        goal_achieved = engine._goal_achieved(low_almi_state)
        self.assertFalse(
            goal_achieved, "Low ALMI should not achieve 75th percentile goal"
        )

    def test_ffmi_goal_detection(self):
        """Test FFMI goal achievement detection"""

        goal_config = GoalConfig(metric_type="ffmi", target_percentile=0.90)

        config = SimulationConfig(
            user_profile=self.user_profile,
            goal_config=goal_config,
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.CUT_FIRST,
            variance_factor=0.25,
        )

        engine = MonteCarloEngine(config)

        # Create state with high FFMI (should achieve goal)
        high_ffmi_state = engine._create_initial_state(
            self.user_profile.scan_history[0]
        )
        high_ffmi_state.ffmi = 21.2  # Should be >90th percentile for 43-year-old male

        goal_achieved = engine._goal_achieved(high_ffmi_state)
        self.assertTrue(goal_achieved, "High FFMI should achieve 90th percentile goal")

        # Create state with low FFMI (should not achieve goal)
        low_ffmi_state = engine._create_initial_state(self.user_profile.scan_history[0])
        low_ffmi_state.ffmi = 19.9  # Should be <90th percentile for 43-year-old male

        goal_achieved = engine._goal_achieved(low_ffmi_state)
        self.assertFalse(
            goal_achieved, "Low FFMI should not achieve 90th percentile goal"
        )


class TestTrainingVariance(unittest.TestCase):
    """Test training-level variance calculations"""

    def test_variance_factors(self):
        """Test that variance factors match expected values"""

        # Test all training levels have expected variance
        for level in TrainingLevel:
            expected_variance = TRAINING_VARIANCE[level]

            # Create engine with this training level
            user_profile = UserProfile(
                birth_date="04/26/1982",
                height_in=66.0,
                gender="male",
                training_level=level,
                scan_history=[
                    {
                        "date": "04/07/2022",
                        "total_weight_lbs": 160.0,
                        "total_lean_mass_lbs": 120.0,
                        "fat_mass_lbs": 40.0,
                        "body_fat_percentage": 25.0,
                        "arms_lean_lbs": 16.0,
                        "legs_lean_lbs": 40.0,
                    }
                ],
            )

            goal_config = GoalConfig(metric_type="almi", target_percentile=0.75)

            config = SimulationConfig(
                user_profile=user_profile,
                goal_config=goal_config,
                training_level=level,
                template=TemplateType.CUT_FIRST,
                variance_factor=expected_variance,
            )

            engine = MonteCarloEngine(config)

            self.assertEqual(
                engine.variance_factor,
                expected_variance,
                f"Variance factor for {level.value} should be {expected_variance}",
            )

    def test_variance_hierarchy(self):
        """Test that variance decreases with training experience"""

        novice_variance = TRAINING_VARIANCE[TrainingLevel.NOVICE]
        intermediate_variance = TRAINING_VARIANCE[TrainingLevel.INTERMEDIATE]
        advanced_variance = TRAINING_VARIANCE[TrainingLevel.ADVANCED]

        self.assertGreater(
            novice_variance,
            intermediate_variance,
            "Novice variance should be higher than intermediate",
        )
        self.assertGreater(
            intermediate_variance,
            advanced_variance,
            "Intermediate variance should be higher than advanced",
        )


class TestCannedProfiles(unittest.TestCase):
    """Test simulation with realistic user profiles"""

    def test_novice_overweight_male(self):
        """Test novice overweight male scenario"""

        user_profile = UserProfile(
            birth_date="04/26/1990",  # 33 years old
            height_in=70.0,  # 5'10"
            gender="male",
            training_level=TrainingLevel.NOVICE,
            scan_history=[
                {
                    "date": "04/07/2022",
                    "total_weight_lbs": 200.0,
                    "total_lean_mass_lbs": 140.0,
                    "fat_mass_lbs": 60.0,
                    "body_fat_percentage": 30.0,  # High BF
                    "arms_lean_lbs": 18.0,  # Adjusted for realistic 60th percentile ALMI
                    "legs_lean_lbs": 42.0,  # Adjusted for realistic 60th percentile ALMI
                }
            ],
        )

        goal_config = GoalConfig(metric_type="almi", target_percentile=0.75)

        # Run simulation
        engine = create_simulation_engine(
            user_profile=user_profile,
            goal_config=goal_config,
            template=TemplateType.CUT_FIRST,
            run_count=100,  # Smaller for testing
            random_seed=42,
        )

        results = engine.run_simulation()

        # Validate results
        self.assertIsNotNone(results)
        self.assertGreater(results.goal_achievement_week, 0)
        self.assertLess(results.goal_achievement_week, 260)  # <5 years
        self.assertGreater(results.convergence_quality, 0.5)
        self.assertGreaterEqual(
            len(results.trajectories), 50
        )  # Most runs should complete

        # Should start with cutting phase (high BF)
        initial_phase = engine._determine_initial_phase(
            engine._create_initial_state(user_profile.scan_history[0])
        )
        self.assertEqual(initial_phase, PhaseType.CUT)

        print(
            f"Novice overweight male: Goal achieved at week {results.goal_achievement_week}, "
            f"age {results.goal_achievement_age:.1f}"
        )

    def test_novice_lean_female(self):
        """Test novice lean female scenario"""

        user_profile = UserProfile(
            birth_date="04/26/1995",  # 28 years old
            height_in=64.0,  # 5'4"
            gender="female",
            training_level=TrainingLevel.NOVICE,
            scan_history=[
                {
                    "date": "04/07/2022",
                    "total_weight_lbs": 130.0,
                    "total_lean_mass_lbs": 95.0,
                    "fat_mass_lbs": 35.0,
                    "body_fat_percentage": 27.0,  # Moderate BF for female
                    "arms_lean_lbs": 10.0,
                    "legs_lean_lbs": 30.0,
                }
            ],
        )

        goal_config = GoalConfig(metric_type="ffmi", target_percentile=0.90)

        # Run simulation
        engine = create_simulation_engine(
            user_profile=user_profile,
            goal_config=goal_config,
            template=TemplateType.BULK_FIRST,
            run_count=100,
            random_seed=42,
        )

        results = engine.run_simulation()

        # Validate results
        self.assertIsNotNone(results)
        self.assertGreater(results.goal_achievement_week, 0)
        self.assertLess(results.goal_achievement_week, 260)
        self.assertGreater(results.convergence_quality, 0.5)

        print(
            f"Novice lean female: Goal achieved at week {results.goal_achievement_week}, "
            f"age {results.goal_achievement_age:.1f}"
        )

    def test_intermediate_male(self):
        """Test intermediate male scenario"""

        user_profile = UserProfile(
            birth_date="04/26/1985",  # 38 years old
            height_in=72.0,  # 6'0"
            gender="male",
            training_level=TrainingLevel.INTERMEDIATE,
            scan_history=[
                {
                    "date": "04/07/2022",
                    "total_weight_lbs": 180.0,
                    "total_lean_mass_lbs": 145.0,
                    "fat_mass_lbs": 35.0,
                    "body_fat_percentage": 19.4,  # Good BF
                    "arms_lean_lbs": 20.0,  # Adjusted for realistic 70th percentile ALMI
                    "legs_lean_lbs": 46.0,  # Adjusted for realistic 70th percentile ALMI
                }
            ],
        )

        goal_config = GoalConfig(metric_type="almi", target_percentile=0.85)

        # Run simulation
        engine = create_simulation_engine(
            user_profile=user_profile,
            goal_config=goal_config,
            template=TemplateType.CUT_FIRST,
            run_count=100,
            random_seed=42,
        )

        results = engine.run_simulation()

        # Validate results
        self.assertIsNotNone(results)
        self.assertGreater(results.goal_achievement_week, 0)
        self.assertLess(results.goal_achievement_week, 260)

        # Intermediate should have lower variance than novice
        self.assertEqual(
            engine.variance_factor, TRAINING_VARIANCE[TrainingLevel.INTERMEDIATE]
        )

        print(
            f"Intermediate male: Goal achieved at week {results.goal_achievement_week}, "
            f"age {results.goal_achievement_age:.1f}"
        )

    def test_advanced_female(self):
        """Test advanced female scenario"""

        user_profile = UserProfile(
            birth_date="04/26/1980",  # 43 years old
            height_in=66.0,  # 5'6"
            gender="female",
            training_level=TrainingLevel.ADVANCED,
            scan_history=[
                {
                    "date": "04/07/2022",
                    "total_weight_lbs": 140.0,
                    "total_lean_mass_lbs": 105.0,
                    "fat_mass_lbs": 35.0,
                    "body_fat_percentage": 25.0,  # Athletic for female
                    "arms_lean_lbs": 12.0,
                    "legs_lean_lbs": 35.0,
                }
            ],
        )

        goal_config = GoalConfig(metric_type="ffmi", target_percentile=0.95)

        # Run simulation
        engine = create_simulation_engine(
            user_profile=user_profile,
            goal_config=goal_config,
            template=TemplateType.BULK_FIRST,
            run_count=100,
            random_seed=42,
        )

        results = engine.run_simulation()

        # Validate results
        self.assertIsNotNone(results)
        self.assertGreater(results.goal_achievement_week, 0)

        # Advanced goals may take longer
        self.assertLess(
            results.goal_achievement_week, 300
        )  # Up to ~6 years for 95th percentile

        # Advanced should have lowest variance
        self.assertEqual(
            engine.variance_factor, TRAINING_VARIANCE[TrainingLevel.ADVANCED]
        )

        print(
            f"Advanced female: Goal achieved at week {results.goal_achievement_week}, "
            f"age {results.goal_achievement_age:.1f}"
        )


class TestStatisticalProperties(unittest.TestCase):
    """Test statistical properties of simulation results"""

    def test_simulation_convergence(self):
        """Test that simulations produce stable statistics"""

        user_profile = UserProfile(
            birth_date="04/26/1985",
            height_in=68.0,
            gender="male",
            training_level=TrainingLevel.INTERMEDIATE,
            scan_history=[
                {
                    "date": "04/07/2022",
                    "total_weight_lbs": 170.0,
                    "total_lean_mass_lbs": 130.0,
                    "fat_mass_lbs": 40.0,
                    "body_fat_percentage": 23.5,
                    "arms_lean_lbs": 16.0,  # Reduced from 18.0 to give starting ALMI ~8.5 (65th percentile)
                    "legs_lean_lbs": 40.0,  # Reduced from 45.0 to give starting ALMI ~8.5 (65th percentile)
                }
            ],
        )

        goal_config = GoalConfig(metric_type="almi", target_percentile=0.80)

        # Run multiple simulations with same parameters
        results_list = []

        for seed in [42, 123, 456]:
            engine = create_simulation_engine(
                user_profile=user_profile,
                goal_config=goal_config,
                run_count=500,
                random_seed=seed,
            )

            results = engine.run_simulation()
            results_list.append(results.goal_achievement_week)

        # Check that results are reasonably stable
        print(f"Debug: results_list = {results_list}")
        mean_weeks = np.mean(results_list)
        std_weeks = np.std(results_list)
        print(f"Debug: mean_weeks = {mean_weeks}, std_weeks = {std_weeks}")

        if mean_weeks == 0:
            self.fail(
                f"All simulations failed to achieve goal or returned invalid results: {results_list}"
            )

        cv = std_weeks / mean_weeks  # Coefficient of variation

        self.assertLess(
            cv, 0.3, "Simulation results should be reasonably stable across runs"
        )
        print(f"Convergence test: Mean={mean_weeks:.1f} weeks, CV={cv:.3f}")

    def test_percentile_band_ordering(self):
        """Test that percentile bands are properly ordered"""

        user_profile = UserProfile(
            birth_date="04/26/1985",
            height_in=68.0,
            gender="male",
            training_level=TrainingLevel.INTERMEDIATE,
            scan_history=[
                {
                    "date": "04/07/2022",
                    "total_weight_lbs": 170.0,
                    "total_lean_mass_lbs": 130.0,
                    "fat_mass_lbs": 40.0,
                    "body_fat_percentage": 23.5,
                    "arms_lean_lbs": 18.0,
                    "legs_lean_lbs": 45.0,
                }
            ],
        )

        goal_config = GoalConfig(metric_type="almi", target_percentile=0.75)

        engine = create_simulation_engine(
            user_profile=user_profile,
            goal_config=goal_config,
            run_count=200,
            random_seed=42,
        )

        results = engine.run_simulation()

        # Check percentile band ordering at a sample time point
        if len(results.percentile_bands["p50"]) > 10:
            week_10_data = {
                "p10": results.percentile_bands["p10"][10].weight_lbs,
                "p25": results.percentile_bands["p25"][10].weight_lbs,
                "p50": results.percentile_bands["p50"][10].weight_lbs,
                "p75": results.percentile_bands["p75"][10].weight_lbs,
                "p90": results.percentile_bands["p90"][10].weight_lbs,
            }

            # Check ordering
            self.assertLessEqual(week_10_data["p10"], week_10_data["p25"])
            self.assertLessEqual(week_10_data["p25"], week_10_data["p50"])
            self.assertLessEqual(week_10_data["p50"], week_10_data["p75"])
            self.assertLessEqual(week_10_data["p75"], week_10_data["p90"])


class TestAgeBasedMaxDuration(unittest.TestCase):
    """Test age-based maximum duration calculation"""

    def test_age_based_duration_brackets(self):
        """Test that different age brackets get correct maximum durations"""

        # Test cases: (age, expected_weeks, expected_years)
        # Avoid exact boundary ages due to floating point precision
        test_cases = [
            (25, 520, 10.0),  # Young: 10 years
            (39, 520, 10.0),  # Young: still 10 years (< 41)
            (43, 416, 8.0),  # Middle-aged: 8 years
            (54, 416, 8.0),  # Middle-aged: still 8 years (< 56)
            (58, 260, 5.0),  # Older: 5 years
            (69, 260, 5.0),  # Older: still 5 years (< 71)
            (73, 156, 3.0),  # Elderly: 3 years
            (85, 156, 3.0),  # Very elderly: still 3 years
        ]

        for age, expected_weeks, expected_years in test_cases:
            with self.subTest(age=age):
                # Create birth date for desired age
                birth_date = datetime.now() - timedelta(days=age * 365.25)
                birth_date_str = birth_date.strftime("%m/%d/%Y")

                user_profile = UserProfile(
                    birth_date=birth_date_str,
                    height_in=68.0,
                    gender="male",
                    training_level=TrainingLevel.INTERMEDIATE,
                    scan_history=[
                        {
                            "date": "04/07/2022",
                            "total_weight_lbs": 170.0,
                            "total_lean_mass_lbs": 130.0,
                            "fat_mass_lbs": 40.0,
                            "body_fat_percentage": 23.5,
                            "arms_lean_lbs": 18.0,
                            "legs_lean_lbs": 45.0,
                        }
                    ],
                )

                goal_config = GoalConfig(metric_type="almi", target_percentile=0.75)

                config = SimulationConfig(
                    user_profile=user_profile,
                    goal_config=goal_config,
                    training_level=TrainingLevel.INTERMEDIATE,
                    template=TemplateType.CUT_FIRST,
                    variance_factor=0.25,
                )

                engine = MonteCarloEngine(config)

                self.assertEqual(
                    engine.max_duration_weeks,
                    expected_weeks,
                    f"Age {age} should have {expected_weeks} weeks max duration",
                )
                self.assertAlmostEqual(
                    engine.max_duration_weeks / 52,
                    expected_years,
                    places=1,
                    msg=f"Age {age} should have ~{expected_years} years max duration",
                )

    def test_max_duration_override(self):
        """Test that max_duration_weeks override works correctly"""

        # Create 60-year-old (normally gets 5 years = 260 weeks)
        birth_date = datetime.now() - timedelta(days=60 * 365.25)
        birth_date_str = birth_date.strftime("%m/%d/%Y")

        user_profile = UserProfile(
            birth_date=birth_date_str,
            height_in=68.0,
            gender="male",
            training_level=TrainingLevel.INTERMEDIATE,
            scan_history=[
                {
                    "date": "04/07/2022",
                    "total_weight_lbs": 170.0,
                    "total_lean_mass_lbs": 130.0,
                    "fat_mass_lbs": 40.0,
                    "body_fat_percentage": 23.5,
                    "arms_lean_lbs": 18.0,
                    "legs_lean_lbs": 45.0,
                }
            ],
        )

        goal_config = GoalConfig(metric_type="almi", target_percentile=0.75)

        # Test without override (should use age-based default)
        config_default = SimulationConfig(
            user_profile=user_profile,
            goal_config=goal_config,
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.CUT_FIRST,
            variance_factor=0.25,
        )

        engine_default = MonteCarloEngine(config_default)
        self.assertEqual(
            engine_default.max_duration_weeks,
            260,
            "60-year-old should get 260 weeks (5 years) by default",
        )

        # Test with override
        override_weeks = 780  # 15 years
        config_override = SimulationConfig(
            user_profile=user_profile,
            goal_config=goal_config,
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.CUT_FIRST,
            variance_factor=0.25,
            max_duration_weeks=override_weeks,
        )

        engine_override = MonteCarloEngine(config_override)
        self.assertEqual(
            engine_override.max_duration_weeks,
            override_weeks,
            "Override should take precedence over age-based default",
        )


class TestDataStructures(unittest.TestCase):
    """Test data structure integrity and validation"""

    def test_simulation_state_creation(self):
        """Test SimulationState creation and ALMI/FFMI calculations"""

        user_profile = UserProfile(
            birth_date="04/26/1985",
            height_in=68.0,
            gender="male",
            training_level=TrainingLevel.INTERMEDIATE,
            scan_history=[
                {
                    "date": "04/07/2022",
                    "total_weight_lbs": 170.0,
                    "total_lean_mass_lbs": 130.0,
                    "fat_mass_lbs": 40.0,
                    "body_fat_percentage": 23.5,
                    "arms_lean_lbs": 18.0,
                    "legs_lean_lbs": 45.0,
                }
            ],
        )

        config = SimulationConfig(
            user_profile=user_profile,
            goal_config=GoalConfig(metric_type="almi", target_percentile=0.75),
            training_level=TrainingLevel.INTERMEDIATE,
            template=TemplateType.CUT_FIRST,
            variance_factor=0.25,
        )

        engine = MonteCarloEngine(config)
        state = engine._create_initial_state(user_profile.scan_history[0])

        # Validate state properties
        self.assertEqual(state.week, 0)
        self.assertAlmostEqual(state.weight_lbs, 170.0, places=1)
        self.assertAlmostEqual(state.body_fat_pct, 23.5, places=1)

        # Validate ALMI calculation
        expected_alm_kg = (18.0 + 45.0) * 0.453592  # ALM in kg
        expected_almi = expected_alm_kg / (1.7272**2)  # Height in meters squared
        self.assertAlmostEqual(state.almi, expected_almi, places=2)

        # Validate FFMI calculation
        expected_tlm_kg = 130.0 * 0.453592
        expected_ffmi = expected_tlm_kg / (1.7272**2)
        self.assertAlmostEqual(state.ffmi, expected_ffmi, places=2)

    def test_simulation_results_completeness(self):
        """Test that SimulationResults contains all required data"""

        user_profile = UserProfile(
            birth_date="04/26/1985",
            height_in=68.0,
            gender="male",
            training_level=TrainingLevel.INTERMEDIATE,
            scan_history=[
                {
                    "date": "04/07/2022",
                    "total_weight_lbs": 170.0,
                    "total_lean_mass_lbs": 130.0,
                    "fat_mass_lbs": 40.0,
                    "body_fat_percentage": 23.5,
                    "arms_lean_lbs": 17.5,  # Adjusted for realistic 65th percentile ALMI
                    "legs_lean_lbs": 40.5,  # Adjusted for realistic 65th percentile ALMI
                }
            ],
        )

        goal_config = GoalConfig(metric_type="almi", target_percentile=0.75)

        engine = create_simulation_engine(
            user_profile=user_profile,
            goal_config=goal_config,
            run_count=50,
            random_seed=42,
        )

        results = engine.run_simulation()

        # Check all required fields are present
        self.assertIsInstance(results.trajectories, list)
        self.assertGreater(len(results.trajectories), 0)

        self.assertIsInstance(results.median_checkpoints, list)

        self.assertIsInstance(results.representative_path, list)
        self.assertGreater(len(results.representative_path), 0)

        self.assertIsInstance(results.percentile_bands, dict)
        self.assertIn("p50", results.percentile_bands)

        self.assertIsInstance(results.goal_achievement_week, int)
        self.assertGreater(results.goal_achievement_week, 0)

        self.assertIsInstance(results.goal_achievement_age, float)
        self.assertGreater(results.goal_achievement_age, 0)

        self.assertIsInstance(results.convergence_quality, float)
        self.assertGreaterEqual(results.convergence_quality, 0)
        self.assertLessEqual(results.convergence_quality, 1)


def run_performance_test():
    """Manual performance test for simulation speed"""
    print("\n" + "=" * 50)
    print("PERFORMANCE TEST")
    print("=" * 50)

    user_profile = UserProfile(
        birth_date="04/26/1985",
        height_in=68.0,
        gender="male",
        training_level=TrainingLevel.INTERMEDIATE,
        scan_history=[
            {
                "date": "04/07/2022",
                "total_weight_lbs": 170.0,
                "total_lean_mass_lbs": 130.0,
                "fat_mass_lbs": 40.0,
                "body_fat_percentage": 23.5,
                "arms_lean_lbs": 18.0,
                "legs_lean_lbs": 45.0,
            }
        ],
    )

    goal_config = GoalConfig(metric_type="almi", target_percentile=0.75)

    import time

    # Test with 2000 runs (production size)
    start_time = time.time()

    engine = create_simulation_engine(
        user_profile=user_profile,
        goal_config=goal_config,
        run_count=2000,
        random_seed=42,
    )

    results = engine.run_simulation()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"2000-run simulation completed in {elapsed_time:.2f} seconds")
    print(
        f"Goal achieved at week {results.goal_achievement_week}, age {results.goal_achievement_age:.1f}"
    )
    print(f"Convergence quality: {results.convergence_quality:.3f}")
    print(f"Total phases: {results.total_phases}")

    # Performance targets from plan
    if elapsed_time < 10:
        print("✅ PASS: Simulation completed in <10 seconds")
    else:
        print("❌ FAIL: Simulation took >10 seconds")


if __name__ == "__main__":
    # Run unit tests
    print("Running Monte Carlo Engine Test Suite...")
    unittest.main(verbosity=2, exit=False)

    # Run performance test
    run_performance_test()
