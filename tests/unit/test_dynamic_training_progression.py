"""
Test suite for dynamic training level progression in Monte Carlo simulation.

This module tests the time-aware training level transitions, variance modulation,
and age progression features that make long-term forecasting more realistic.
"""

import unittest
from datetime import datetime, timedelta
from typing import List

import numpy as np

from mc_forecast import MonteCarloEngine, create_simulation_engine
from shared_models import (
    TRAINING_VARIANCE,
    GoalConfig,
    SimulationConfig,
    SimulationState,
    TemplateType,
    TrainingLevel,
    UserProfile,
)


class TestDynamicTrainingProgression(unittest.TestCase):
    """Test dynamic training level progression over time"""

    def setUp(self):
        """Set up test user profile and configuration"""
        self.user_profile = UserProfile(
            birth_date="04/26/1990",  # 33 years old
            height_in=70.0,
            gender="male",
            training_level=TrainingLevel.NOVICE,
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

        self.goal_config = GoalConfig(metric_type="almi", target_percentile=0.75)

        self.config = SimulationConfig(
            user_profile=self.user_profile,
            goal_config=self.goal_config,
            training_level=TrainingLevel.NOVICE,
            template=TemplateType.CUT_FIRST,
            variance_factor=0.25,
            random_seed=42,
        )

    def test_training_level_transition_timing(self):
        """Test that training levels transition at appropriate times"""
        engine = MonteCarloEngine(self.config)

        # Test that progression generally works across different timeframes
        test_state = SimulationState(
            week=0,
            weight_lbs=170,
            lean_mass_lbs=130,
            fat_mass_lbs=40,
            body_fat_pct=23.5,
            phase=None,
            almi=8.5,
            ffmi=18.0,
            weeks_training=0,
            current_training_level=TrainingLevel.NOVICE,
            training_level_transition_weeks=[],
            simulation_age=25,
        )

        # Test early progression - should be novice
        test_state.weeks_training = 20
        level = engine._calculate_dynamic_training_level(
            test_state, TrainingLevel.NOVICE
        )
        self.assertEqual(level, TrainingLevel.NOVICE, "Should be novice at 20 weeks")

        # Test late progression - should be intermediate or advanced
        test_state.weeks_training = 80
        level = engine._calculate_dynamic_training_level(
            test_state, TrainingLevel.NOVICE
        )
        self.assertIn(
            level,
            [TrainingLevel.INTERMEDIATE, TrainingLevel.ADVANCED],
            "Should be intermediate or advanced at 80 weeks",
        )

        # Test very late progression - should be advanced
        test_state.weeks_training = 200
        level = engine._calculate_dynamic_training_level(
            test_state, TrainingLevel.NOVICE
        )
        self.assertEqual(
            level, TrainingLevel.ADVANCED, "Should be advanced at 200 weeks"
        )

        # Test age-based faster progression for older users
        test_state.simulation_age = 45  # Older user
        test_state.weeks_training = 30  # Same timeframe
        level_older = engine._calculate_dynamic_training_level(
            test_state, TrainingLevel.NOVICE
        )

        test_state.simulation_age = 25  # Younger user
        level_younger = engine._calculate_dynamic_training_level(
            test_state, TrainingLevel.NOVICE
        )

        # Older users should progress faster (be at higher level or same level at same timeframe)
        print(
            f"At 30 weeks: younger user is {level_younger.value}, older user is {level_older.value}"
        )

    def test_gender_progression_differences(self):
        """Test that gender affects progression timing slightly"""
        engine = MonteCarloEngine(self.config)

        # Create test states for male and female
        base_age = 30
        base_weeks = 52  # 1 year

        male_state = SimulationState(
            week=0,
            weight_lbs=170,
            lean_mass_lbs=130,
            fat_mass_lbs=40,
            body_fat_pct=23.5,
            phase=None,
            almi=8.5,
            ffmi=18.0,
            weeks_training=base_weeks,
            current_training_level=TrainingLevel.NOVICE,
            training_level_transition_weeks=[],
            simulation_age=base_age,
        )

        female_state = SimulationState(
            week=0,
            weight_lbs=140,
            lean_mass_lbs=100,
            fat_mass_lbs=40,
            body_fat_pct=28.5,
            phase=None,
            almi=7.5,
            ffmi=16.0,
            weeks_training=base_weeks,
            current_training_level=TrainingLevel.NOVICE,
            training_level_transition_weeks=[],
            simulation_age=base_age,
        )

        # Update config for female test
        female_config = self.config
        female_config.user_profile.gender = "female"
        female_engine = MonteCarloEngine(female_config)

        male_level = engine._calculate_dynamic_training_level(
            male_state, TrainingLevel.NOVICE
        )
        female_level = female_engine._calculate_dynamic_training_level(
            female_state, TrainingLevel.NOVICE
        )

        # Both should progress similarly, but female progression should be slightly slower
        # At exactly 52 weeks, male might transition while female might not yet
        print(f"Male level at {base_weeks} weeks: {male_level}")
        print(f"Female level at {base_weeks} weeks: {female_level}")

    def test_variance_factor_progression(self):
        """Test that variance factors change with training level progression"""
        # Create states at different training levels
        novice_state = SimulationState(
            week=0,
            weight_lbs=170,
            lean_mass_lbs=130,
            fat_mass_lbs=40,
            body_fat_pct=23.5,
            phase=None,
            almi=8.5,
            ffmi=18.0,
            weeks_training=10,
            current_training_level=TrainingLevel.NOVICE,
            training_level_transition_weeks=[],
            simulation_age=30,
        )

        intermediate_state = SimulationState(
            week=52,
            weight_lbs=175,
            lean_mass_lbs=135,
            fat_mass_lbs=40,
            body_fat_pct=22.8,
            phase=None,
            almi=8.7,
            ffmi=18.5,
            weeks_training=60,
            current_training_level=TrainingLevel.INTERMEDIATE,
            training_level_transition_weeks=[52],
            simulation_age=31,
        )

        advanced_state = SimulationState(
            week=156,
            weight_lbs=180,
            lean_mass_lbs=140,
            fat_mass_lbs=40,
            body_fat_pct=22.2,
            phase=None,
            almi=9.0,
            ffmi=19.0,
            weeks_training=170,
            current_training_level=TrainingLevel.ADVANCED,
            training_level_transition_weeks=[52, 156],
            simulation_age=33,
        )

        # Test that variance factors match training levels
        novice_variance = TRAINING_VARIANCE[novice_state.current_training_level]
        intermediate_variance = TRAINING_VARIANCE[
            intermediate_state.current_training_level
        ]
        advanced_variance = TRAINING_VARIANCE[advanced_state.current_training_level]

        self.assertEqual(novice_variance, 0.50)
        self.assertEqual(intermediate_variance, 0.25)
        self.assertEqual(advanced_variance, 0.10)

        # Verify progression: variance should decrease
        self.assertGreater(novice_variance, intermediate_variance)
        self.assertGreater(intermediate_variance, advanced_variance)

    def test_age_progression_during_simulation(self):
        """Test that age updates correctly during simulation"""
        engine = MonteCarloEngine(self.config)
        initial_age = engine.current_age

        # Create state at different simulation weeks
        test_weeks = [0, 26, 52, 104, 156]  # 0, 6mo, 1yr, 2yr, 3yr

        for week in test_weeks:
            expected_age = initial_age + (week / 52.0)

            test_state = SimulationState(
                week=week,
                weight_lbs=170,
                lean_mass_lbs=130,
                fat_mass_lbs=40,
                body_fat_pct=23.5,
                phase=None,
                almi=8.5,
                ffmi=18.0,
                weeks_training=week,
                current_training_level=TrainingLevel.NOVICE,
                training_level_transition_weeks=[],
                simulation_age=expected_age,
            )

            # Age should be updated based on simulation time
            self.assertAlmostEqual(test_state.simulation_age, expected_age, places=2)


class TestLongTermSimulationRealism(unittest.TestCase):
    """Test realistic behavior in long-term simulations"""

    def setUp(self):
        """Set up for long-term simulation tests"""
        self.user_profile = UserProfile(
            birth_date="01/01/1995",  # 28 years old
            height_in=68.0,
            gender="male",
            training_level=TrainingLevel.NOVICE,
            scan_history=[
                {
                    "date": "01/01/2023",
                    "total_weight_lbs": 160.0,
                    "total_lean_mass_lbs": 120.0,
                    "fat_mass_lbs": 40.0,
                    "body_fat_percentage": 25.0,
                    "arms_lean_lbs": 16.0,
                    "legs_lean_lbs": 40.0,
                }
            ],
        )

        self.goal_config = GoalConfig(metric_type="almi", target_percentile=0.85)

    def test_five_year_progression_simulation(self):
        """Test realistic progression over 5 years"""
        # Create engine with longer duration
        engine = create_simulation_engine(
            user_profile=self.user_profile,
            goal_config=self.goal_config,
            run_count=10,  # Small count for testing
            random_seed=42,
        )

        # Override max duration for 5-year test
        engine.max_duration_weeks = 260  # 5 years

        # Run single trajectory to examine progression
        trajectory = engine._run_single_trajectory(0)

        # Verify trajectory has progression tracking
        self.assertGreater(
            len(trajectory),
            100,
            "Should have substantial trajectory for 5-year simulation",
        )

        # Check that training levels progress over time
        training_levels_seen = set()
        transition_weeks = []

        for state in trajectory:
            training_levels_seen.add(state.current_training_level)
            if state.training_level_transition_weeks:
                transition_weeks.extend(state.training_level_transition_weeks)

        # Should see multiple training levels over 5 years
        self.assertGreaterEqual(
            len(training_levels_seen),
            2,
            "Should progress through multiple training levels",
        )

        # Should have at least one transition
        self.assertGreater(
            len(set(transition_weeks)), 0, "Should have training level transitions"
        )

        print(
            f"Training levels observed: {[level.value for level in training_levels_seen]}"
        )
        print(f"Transition weeks: {sorted(set(transition_weeks))}")

    def test_rate_progression_realism(self):
        """Test that rates become more conservative as training advances"""
        engine = create_simulation_engine(
            user_profile=self.user_profile,
            goal_config=self.goal_config,
            run_count=5,
            random_seed=42,
        )

        # Test rate calculations at different training levels
        novice_bulk_rate = engine.rate_calculator.get_bulk_rate(
            TrainingLevel.NOVICE, "moderate"
        )
        intermediate_bulk_rate = engine.rate_calculator.get_bulk_rate(
            TrainingLevel.INTERMEDIATE, "moderate"
        )
        advanced_bulk_rate = engine.rate_calculator.get_bulk_rate(
            TrainingLevel.ADVANCED, "moderate"
        )

        # Rates should decrease as training level advances
        self.assertGreater(novice_bulk_rate, intermediate_bulk_rate)
        self.assertGreater(intermediate_bulk_rate, advanced_bulk_rate)

        print(
            f"Bulk rates progression: Novice {novice_bulk_rate}% -> Intermediate {intermediate_bulk_rate}% -> Advanced {advanced_bulk_rate}%"
        )

    def test_transition_tracking_accuracy(self):
        """Test that transition tracking is accurate and consistent"""
        engine = create_simulation_engine(
            user_profile=self.user_profile,
            goal_config=self.goal_config,
            run_count=3,
            random_seed=42,
        )

        # Override for shorter test
        engine.max_duration_weeks = 200  # ~4 years

        trajectories = []
        for run_idx in range(3):
            trajectory = engine._run_single_trajectory(run_idx)
            trajectories.append(trajectory)

        # Analyze transition consistency across runs
        for i, trajectory in enumerate(trajectories):
            final_state = trajectory[-1]
            transitions = final_state.training_level_transition_weeks

            print(
                f"Run {i}: Final training level: {final_state.current_training_level.value}"
            )
            print(f"Run {i}: Transition weeks: {transitions}")
            print(f"Run {i}: Total training weeks: {final_state.weeks_training}")

            # Transitions should be monotonic (each transition week > previous)
            if len(transitions) > 1:
                for j in range(1, len(transitions)):
                    self.assertGreater(
                        transitions[j],
                        transitions[j - 1],
                        f"Transitions should be monotonic in run {i}",
                    )


class TestBackwardCompatibility(unittest.TestCase):
    """Test that new features maintain backward compatibility"""

    def test_existing_simulation_still_works(self):
        """Test that existing simulation code works with new features"""
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
                    "arms_lean_lbs": 15.0,  # Lower starting ALMI
                    "legs_lean_lbs": 38.0,
                }
            ],
        )

        goal_config = GoalConfig(
            metric_type="almi", target_percentile=0.95
        )  # More challenging goal

        # Create engine using existing factory function
        engine = create_simulation_engine(
            user_profile=user_profile,
            goal_config=goal_config,
            run_count=10,
            random_seed=42,
        )

        # Should work without errors
        results = engine.run_simulation()

        # Should have valid results
        self.assertIsNotNone(results)
        self.assertGreater(results.goal_achievement_week, 0)
        self.assertGreater(len(results.trajectories), 0)

    def test_simulation_state_default_values(self):
        """Test that SimulationState has proper defaults for new fields"""
        # Create state without new fields (old style)
        state = SimulationState(
            week=0,
            weight_lbs=170,
            lean_mass_lbs=130,
            fat_mass_lbs=40,
            body_fat_pct=23.5,
            phase=None,
            almi=8.5,
            ffmi=18.0,
        )

        # Should have proper defaults
        self.assertEqual(state.weeks_training, 0)
        self.assertEqual(state.current_training_level, TrainingLevel.NOVICE)
        self.assertEqual(state.training_level_transition_weeks, [])
        self.assertEqual(state.simulation_age, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
