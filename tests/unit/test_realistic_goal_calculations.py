#!/usr/bin/env python3
"""
Unit tests for realistic goal calculations in RecompTracker.

These tests ensure goal calculations produce realistic, science-based projections
for lean mass gain and timeframes, rather than overly optimistic estimates.
"""

import os
import sys
import unittest
from datetime import datetime

# Add the parent directory to Python path to import core
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core import (
    LEAN_MASS_GAIN_RATES,
    calculate_progressive_gain_over_time,
    extract_data_from_config,
    get_conservative_gain_rate,
    parse_gender,
    run_analysis_from_data,
)


class TestRealisticGoalCalculations(unittest.TestCase):
    """Test realistic goal calculations based on muscle building science."""

    def setUp(self):
        """Set up test data for realistic goal testing."""
        # Example user: 42-year-old male, 66" tall
        self.user_info = {
            "birth_date": "04/26/1982",
            "height_in": 66.0,
            "gender": "male",
            "gender_code": 0,
        }

        # Single scan scenario (worst case for TLM estimation)
        self.single_scan = [
            {
                "date": "04/07/2022",
                "total_weight_lbs": 143.2,
                "total_lean_mass_lbs": 106.3,
                "fat_mass_lbs": 32.6,
                "body_fat_percentage": 22.8,
                "arms_lean_lbs": 12.4,
                "legs_lean_lbs": 37.3,
            }
        ]

        # Progressive scan history (better for realistic rate detection)
        self.multi_scan_progressive = [
            {
                "date": "01/01/2022",
                "total_weight_lbs": 140.0,
                "total_lean_mass_lbs": 105.0,
                "fat_mass_lbs": 35.0,
                "body_fat_percentage": 25.0,
                "arms_lean_lbs": 12.0,
                "legs_lean_lbs": 36.0,
            },
            {
                "date": "07/01/2022",
                "total_weight_lbs": 145.0,
                "total_lean_mass_lbs": 108.0,  # 3 lbs gain in 6 months
                "fat_mass_lbs": 37.0,
                "body_fat_percentage": 25.5,
                "arms_lean_lbs": 12.5,
                "legs_lean_lbs": 37.0,
            },
            {
                "date": "01/01/2023",
                "total_weight_lbs": 148.0,
                "total_lean_mass_lbs": 110.0,  # 2 lbs gain in 6 months
                "fat_mass_lbs": 38.0,
                "body_fat_percentage": 25.7,
                "arms_lean_lbs": 13.0,
                "legs_lean_lbs": 37.5,
            },
        ]

    def test_conservative_gain_rates_are_realistic(self):
        """Test that conservative gain rates match research-backed expectations."""
        # Test base rates without age adjustment for 25-year-old
        young_user = {**self.user_info, "age": 25}

        # Novice rates should be realistic (not >1 kg/month)
        novice_rate, _ = get_conservative_gain_rate(young_user, "novice", 25)
        self.assertLessEqual(
            novice_rate, 1.0, "Novice rate should not exceed 1 kg/month"
        )
        self.assertGreaterEqual(
            novice_rate, 0.3, "Novice rate should be at least 0.3 kg/month"
        )

        # Intermediate rates should be moderate
        intermediate_rate, _ = get_conservative_gain_rate(
            young_user, "intermediate", 25
        )
        self.assertLessEqual(
            intermediate_rate, 0.6, "Intermediate rate should not exceed 0.6 kg/month"
        )
        self.assertGreaterEqual(
            intermediate_rate,
            0.15,
            "Intermediate rate should be at least 0.15 kg/month",
        )

        # Advanced rates should be slow
        advanced_rate, _ = get_conservative_gain_rate(young_user, "advanced", 25)
        self.assertLessEqual(
            advanced_rate, 0.3, "Advanced rate should not exceed 0.3 kg/month"
        )
        self.assertGreaterEqual(
            advanced_rate, 0.05, "Advanced rate should be at least 0.05 kg/month"
        )

    def test_age_adjustments_reduce_rates_appropriately(self):
        """Test that age adjustments properly reduce gain rates for older individuals."""
        # Test 42-year-old (should have rate reduction)
        older_user = {**self.user_info, "age": 42}

        young_rate, _ = get_conservative_gain_rate(older_user, "intermediate", 25)
        older_rate, _ = get_conservative_gain_rate(older_user, "intermediate", 42)

        self.assertLess(
            older_rate,
            young_rate,
            "Older users should have reduced lean mass gain rates",
        )

        # Rate reduction should be reasonable (not more than 50% reduction)
        reduction_factor = older_rate / young_rate
        self.assertGreaterEqual(
            reduction_factor,
            0.5,
            "Age adjustment should not reduce rates by more than 50%",
        )

    def test_single_scan_90th_percentile_goal_realistic_timeframe(self):
        """Test that 90th percentile goal from single scan requires realistic timeframe."""
        # ALMI goal: 90th percentile
        almi_goal = {"target_percentile": 0.90, "target_age": None}

        try:
            df_results, goal_calculations, _, _ = run_analysis_from_data(
                self.user_info, self.single_scan, almi_goal, None
            )

            if "almi" in goal_calculations:
                gc = goal_calculations["almi"]
                current_age = 42  # Approximate age from birth date

                # Target age should be realistic (3+ years minimum for 90th percentile)
                target_age = gc.get("target_age", current_age)
                years_to_goal = target_age - current_age

                self.assertGreaterEqual(
                    years_to_goal,
                    2.0,
                    f"90th percentile goal should take at least 2 years, got {years_to_goal:.1f}",
                )

                # Total lean mass gain should be realistic for 90th percentile goal
                tlm_change = gc.get("tlm_change_needed_lbs", 0)
                if years_to_goal > 0:
                    annual_tlm_gain = tlm_change / years_to_goal
                    # 90th percentile is ambitious - allow up to 12 lbs/year for intermediate/novice
                    self.assertLessEqual(
                        annual_tlm_gain,
                        12.0,
                        f"Annual TLM gain of {annual_tlm_gain:.1f} lbs/year exceeds even optimistic expectations",
                    )

                # ALM gain should be reasonable portion of TLM gain
                alm_change = gc.get("alm_change_needed_lbs", 0)
                if tlm_change > 0:
                    alm_ratio = alm_change / tlm_change
                    self.assertGreaterEqual(
                        alm_ratio,
                        0.3,
                        "ALM should be at least 30% of total lean mass gain",
                    )
                    self.assertLessEqual(
                        alm_ratio,
                        0.7,
                        "ALM should not exceed 70% of total lean mass gain",
                    )

        except Exception as e:
            self.fail(f"Goal calculation failed: {e}")

    def test_multi_scan_advanced_trainee_realistic_projections(self):
        """Test that advanced trainee with multi-scan history gets realistic projections."""
        # ALMI goal: 85th percentile (more moderate goal)
        almi_goal = {"target_percentile": 0.85, "target_age": None}

        try:
            df_results, goal_calculations, _, _ = run_analysis_from_data(
                self.user_info, self.multi_scan_progressive, almi_goal, None
            )

            if "almi" in goal_calculations:
                gc = goal_calculations["almi"]
                current_age = 42

                target_age = gc.get("target_age", current_age)
                years_to_goal = target_age - current_age

                # 85th percentile should be achievable in reasonable time
                self.assertGreaterEqual(
                    years_to_goal,
                    1.0,
                    "85th percentile goal should take at least 1 year",
                )
                self.assertLessEqual(
                    years_to_goal,
                    7.0,
                    "85th percentile goal should be achievable within 7 years",
                )

                # Monthly lean mass gain should match expected advanced rates
                tlm_change = gc.get("tlm_change_needed_lbs", 0)
                if years_to_goal > 0:
                    monthly_tlm_gain_lbs = tlm_change / (years_to_goal * 12)
                    # Advanced rate ~0.15 kg/month = ~0.33 lbs/month
                    self.assertLessEqual(
                        monthly_tlm_gain_lbs,
                        0.6,
                        f"Monthly TLM gain {monthly_tlm_gain_lbs:.2f} lbs/month too high for advanced",
                    )

        except Exception as e:
            self.fail(f"Advanced trainee goal calculation failed: {e}")

    def test_extremely_high_percentile_capped_timeframe(self):
        """Test that extremely high percentile goals (95th+) are capped appropriately."""
        # ALMI goal: 95th percentile (very ambitious)
        almi_goal = {"target_percentile": 0.95, "target_age": None}

        try:
            df_results, goal_calculations, _, _ = run_analysis_from_data(
                self.user_info, self.single_scan, almi_goal, None
            )

            if "almi" in goal_calculations:
                gc = goal_calculations["almi"]
                current_age = 42

                target_age = gc.get("target_age", current_age)
                years_to_goal = target_age - current_age

                # Should be capped at reasonable maximum (10 years)
                self.assertLessEqual(
                    years_to_goal,
                    10.0,
                    f"95th percentile goal timeframe should be capped at 10 years, got {years_to_goal:.1f}",
                )

                # If timeframe is at maximum, should indicate this is very ambitious
                if years_to_goal >= 8.0:
                    # This is expected for very high percentile goals
                    pass

        except Exception as e:
            self.fail(f"High percentile goal calculation failed: {e}")

    def test_goal_body_composition_changes_realistic(self):
        """Test that goal body composition changes are physiologically realistic."""
        almi_goal = {"target_percentile": 0.80, "target_age": None}

        try:
            df_results, goal_calculations, _, _ = run_analysis_from_data(
                self.user_info, self.single_scan, almi_goal, None
            )

            if "almi" in goal_calculations:
                gc = goal_calculations["almi"]

                # Weight change should be reasonable
                weight_change = gc.get("weight_change", 0)
                self.assertLessEqual(
                    abs(weight_change),
                    20.0,
                    f"Weight change {weight_change:.1f} lbs is excessive",
                )

                # Lean change should be positive and reasonable
                lean_change = gc.get("lean_change", 0)
                self.assertGreater(
                    lean_change, 0, "Lean mass change should be positive"
                )
                self.assertLessEqual(
                    lean_change, 25.0, f"Lean change {lean_change:.1f} lbs is excessive"
                )

                # Fat change should be reasonable (can be positive or negative)
                fat_change = gc.get("fat_change", 0)
                self.assertLessEqual(
                    abs(fat_change),
                    15.0,
                    f"Fat change {abs(fat_change):.1f} lbs is excessive",
                )

        except Exception as e:
            self.fail(f"Goal body composition calculation failed: {e}")

    def test_example_config_specific_case(self):
        """Test the specific example config case mentioned by user."""
        # Load the actual example config data
        example_single_scan = [
            {
                "date": "04/07/2022",
                "total_weight_lbs": 143.2,
                "total_lean_mass_lbs": 106.3,
                "fat_mass_lbs": 32.6,
                "body_fat_percentage": 22.8,
                "arms_lean_lbs": 12.4,
                "legs_lean_lbs": 37.3,
            }
        ]

        almi_goal = {"target_percentile": 0.90, "target_age": None}

        try:
            df_results, goal_calculations, _, _ = run_analysis_from_data(
                self.user_info, example_single_scan, almi_goal, None
            )

            if "almi" in goal_calculations:
                gc = goal_calculations["almi"]
                current_age = 42

                target_age = gc.get("target_age", current_age)
                years_to_goal = target_age - current_age
                tlm_change = gc.get("tlm_change_needed_lbs", 0)

                # This should NOT be the unrealistic 23.7 lbs in 1 year
                if years_to_goal <= 1.5:
                    self.assertLessEqual(
                        tlm_change,
                        8.0,
                        f"TLM gain of {tlm_change:.1f} lbs in {years_to_goal:.1f} years is unrealistic",
                    )

                # Print actual values for debugging
                print("\n--- Example Config Test Results ---")
                print(f"Years to 90th percentile: {years_to_goal:.1f}")
                print(f"TLM change needed: {tlm_change:.1f} lbs")
                print(f"Annual TLM rate: {tlm_change / years_to_goal:.1f} lbs/year")
                if "alm_change_needed_lbs" in gc:
                    print(f"ALM change needed: {gc['alm_change_needed_lbs']:.1f} lbs")

        except Exception as e:
            self.fail(f"Example config goal calculation failed: {e}")


class TestProgressiveGainModel(unittest.TestCase):
    """Test progressive gain rate modeling for long-term goals."""

    def setUp(self):
        """Set up test data for progressive gain testing."""
        self.user_info_male = {
            "birth_date": "04/26/1982",
            "height_in": 66.0,
            "gender": "male",
            "gender_code": 0,
        }

        self.user_info_female = {
            "birth_date": "04/26/1982",
            "height_in": 64.0,
            "gender": "female",
            "gender_code": 1,
        }

    def test_single_year_novice_uses_novice_rates(self):
        """Test that 1-year goals for novice trainees use only novice rates."""
        total_gain, explanation = calculate_progressive_gain_over_time(
            self.user_info_male, "novice", 25, 1.0
        )

        # Should be 1 year * 12 months * 0.45 kg/month = 5.4 kg
        expected_gain = 0.45 * 12 * 1.0
        self.assertAlmostEqual(total_gain, expected_gain, places=1)
        self.assertIn("novice rate", explanation)
        self.assertNotIn("intermediate", explanation)
        self.assertNotIn("advanced", explanation)

    def test_three_year_novice_progression(self):
        """Test that 3-year goals properly blend novice → intermediate rates."""
        total_gain, explanation = calculate_progressive_gain_over_time(
            self.user_info_male, "novice", 25, 3.0
        )

        # Year 1: 0.45 * 12 = 5.4 kg (novice)
        # Years 2-3: 0.25 * 12 * 2 = 6.0 kg (intermediate)
        # Total: 11.4 kg
        expected_gain = (0.45 * 12 * 1) + (0.25 * 12 * 2)
        self.assertAlmostEqual(total_gain, expected_gain, places=1)

        self.assertIn("Year 1", explanation)
        self.assertIn("novice rate", explanation)
        self.assertIn("Years 2-3", explanation)
        self.assertIn("intermediate rate", explanation)

    def test_five_year_full_progression(self):
        """Test that 5-year goals include all three phases: novice → intermediate → advanced."""
        total_gain, explanation = calculate_progressive_gain_over_time(
            self.user_info_male, "novice", 25, 5.0
        )

        # Year 1: 0.45 * 12 = 5.4 kg (novice)
        # Years 2-3: 0.25 * 12 * 2 = 6.0 kg (intermediate)
        # Years 4-5: 0.12 * 12 * 2 = 2.88 kg (advanced)
        # Total: 14.28 kg
        expected_gain = (0.45 * 12 * 1) + (0.25 * 12 * 2) + (0.12 * 12 * 2)
        self.assertAlmostEqual(total_gain, expected_gain, places=1)

        self.assertIn("Year 1", explanation)
        self.assertIn("Years 2-3", explanation)
        self.assertIn("Years 4+", explanation)
        self.assertIn("advanced rate", explanation)

    def test_intermediate_start_skips_novice_phase(self):
        """Test that intermediate trainees skip novice rates."""
        total_gain, explanation = calculate_progressive_gain_over_time(
            self.user_info_male, "intermediate", 25, 2.0
        )

        # All 2 years at intermediate rate: 0.25 * 12 * 2 = 6.0 kg
        expected_gain = 0.25 * 12 * 2.0
        self.assertAlmostEqual(total_gain, expected_gain, places=1)

        self.assertIn("intermediate rate", explanation)
        self.assertNotIn("novice", explanation)

    def test_advanced_start_uses_only_advanced_rates(self):
        """Test that advanced trainees use only advanced rates."""
        total_gain, explanation = calculate_progressive_gain_over_time(
            self.user_info_male, "advanced", 25, 3.0
        )

        # All 3 years at advanced rate: 0.12 * 12 * 3 = 4.32 kg
        expected_gain = 0.12 * 12 * 3.0
        self.assertAlmostEqual(total_gain, expected_gain, places=1)

        self.assertIn("advanced rate", explanation)
        self.assertNotIn("novice", explanation)
        self.assertNotIn("intermediate", explanation)

    def test_age_adjustments_apply_to_all_phases(self):
        """Test that age adjustments are applied correctly across all training phases."""
        # 42-year-old should have ~20% reduction (1.2 decades over 30 * 10% = 12% reduction)
        total_gain_young, _ = calculate_progressive_gain_over_time(
            self.user_info_male, "novice", 25, 3.0
        )

        total_gain_older, _ = calculate_progressive_gain_over_time(
            self.user_info_male, "novice", 42, 3.0
        )

        # Older person should have reduced gains
        self.assertLess(total_gain_older, total_gain_young)

        # Age factor: 1 - (1.2 * 0.1) = 0.88
        expected_ratio = 0.88
        actual_ratio = total_gain_older / total_gain_young
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=1)

    def test_female_vs_male_rates(self):
        """Test that female rates are appropriately lower than male rates."""
        male_gain, _ = calculate_progressive_gain_over_time(
            self.user_info_male, "novice", 25, 3.0
        )

        female_gain, _ = calculate_progressive_gain_over_time(
            self.user_info_female, "novice", 25, 3.0
        )

        # Female gains should be lower than male gains
        self.assertLess(female_gain, male_gain)

        # Should be roughly 55-60% of male gains based on rate ratios
        ratio = female_gain / male_gain
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 0.7)

    def test_progressive_vs_fixed_rate_comparison(self):
        """Test that progressive model gives different (more conservative) results than fixed rate for long terms."""
        # For a 5-year goal, progressive model should be more conservative than fixed intermediate rate
        progressive_gain, _ = calculate_progressive_gain_over_time(
            self.user_info_male, "intermediate", 25, 5.0
        )

        # Fixed intermediate rate for 5 years: 0.25 * 12 * 5 = 15.0 kg
        fixed_rate_gain = 0.25 * 12 * 5.0

        # Progressive should be less than fixed rate (includes advanced years)
        self.assertLess(progressive_gain, fixed_rate_gain)

        # Difference should be meaningful (at least 1 kg less)
        self.assertGreater(fixed_rate_gain - progressive_gain, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
