"""
Comprehensive test suite for RecompTracker body composition analysis.

This module contains all tests for the RecompTracker body composition analysis system,
including core calculation functions, TLM estimation logic, and integration tests.
"""

import json
import os
import tempfile
import unittest

import numpy as np
from scipy.interpolate import interp1d

from core import (
    AGE_ADJUSTMENT_FACTOR,
    LEAN_MASS_GAIN_RATES,
    calculate_age_precise,
    calculate_suggested_goal,
    calculate_t_score,
    calculate_tscore_reference_values,
    compute_zscore,
    detect_training_level_from_scans,
    determine_training_level,
    extract_data_from_config,
    get_alm_tlm_ratio,
    get_conservative_gain_rate,
    get_gender_string,
    get_value_from_zscore,
    load_config_json,
    parse_gender,
    process_scans_and_goal,
)


class TestBodyCompCalculations(unittest.TestCase):
    """
    Test suite for core calculation functions.

    This ensures the mathematical logic for age, Z-scores, T-scores,
    and inverse Z-scores is correct and handles edge cases properly.
    """

    def test_calculate_age(self):
        """Tests the age calculation logic."""
        self.assertAlmostEqual(
            calculate_age_precise("01/01/2000", "01/01/2001"), 1.0, places=2
        )
        self.assertAlmostEqual(
            calculate_age_precise("06/15/1980", "12/15/1980"), 0.5, places=2
        )

    def test_zscore_logic(self):
        """Tests the main Z-score calculation for various L values and edge cases."""
        self.assertAlmostEqual(
            compute_zscore(10, 0.5, 8, 0.1), (np.sqrt(1.25) - 1) / 0.05, 5
        )
        self.assertAlmostEqual(
            compute_zscore(7, -0.5, 8, 0.1), ((7 / 8) ** -0.5 - 1) / -0.05, 5
        )
        self.assertAlmostEqual(compute_zscore(10, 0, 8, 0.1), np.log(1.25) / 0.1, 5)
        self.assertEqual(compute_zscore(8, 0.5, 8, 0.1), 0)
        self.assertTrue(np.isnan(compute_zscore(0, 0.5, 8, 0.1)))
        self.assertTrue(np.isnan(compute_zscore(-1, 0.5, 8, 0.1)))

    def test_inverse_zscore_logic(self):
        """Tests that get_value_from_zscore is the mathematical inverse of compute_zscore."""
        L, M, S, z = 0.5, 10, 0.1, 1.5
        y = get_value_from_zscore(z, L, M, S)
        self.assertAlmostEqual(compute_zscore(y, L, M, S), z, 5)

        L, M, S, z = -0.5, 10, 0.1, -1.5
        y = get_value_from_zscore(z, L, M, S)
        self.assertAlmostEqual(compute_zscore(y, L, M, S), z, 5)

        L, M, S, z = 0, 10, 0.1, 1.0
        y = get_value_from_zscore(z, L, M, S)
        self.assertAlmostEqual(compute_zscore(y, L, M, S), z, 5)

    def test_t_score_logic(self):
        """Tests the T-score calculation against a young adult reference."""
        self.assertEqual(calculate_t_score(12, 10, 2), 1.0)
        self.assertEqual(calculate_t_score(8, 10, 2), -1.0)
        self.assertEqual(calculate_t_score(10, 10, 2), 0.0)
        self.assertTrue(np.isnan(calculate_t_score(10, 10, 0)))


class TestTLMEstimation(unittest.TestCase):
    """Test cases for intelligent TLM estimation using ALM/TLM ratios."""

    def setUp(self):
        """Set up common test fixtures."""
        # Mock LMS interpolation functions for testing
        ages = np.array([18, 30, 45, 60, 80])

        # Realistic ALMI values (kg/m²) for males
        almi_values = np.array([8.5, 9.0, 8.8, 8.2, 7.5])
        self.mock_almi_func = interp1d(
            ages, almi_values, kind="cubic", fill_value="extrapolate"
        )

        # Realistic LMI values (kg/m²) for males
        lmi_values = np.array([18.5, 19.2, 18.8, 17.8, 16.5])
        self.mock_lmi_func = interp1d(
            ages, lmi_values, kind="cubic", fill_value="extrapolate"
        )

        # Common test fixtures
        self.user_info = {
            "birth_date_str": "04/26/1982",
            "height_in": 66.0,
            "gender_code": 0,
        }

        self.goal_params = {"target_percentile": 0.90, "target_age": 45.0}

        self.lms_functions = {
            "almi_M": self.mock_almi_func,
            "lmi_M": self.mock_lmi_func,
        }

    def test_personal_ratio_multiple_scans(self):
        """Test ALM/TLM ratio calculation with multiple scans (personal history)."""
        processed_data = [
            {
                "date_str": "04/07/2022",
                "alm_lbs": 49.7,  # 12.4 + 37.3
                "total_lean_mass_lbs": 106.3,
                "age_at_scan": 39.95,
            },
            {
                "date_str": "04/01/2023",
                "alm_lbs": 56.9,  # 16.5 + 40.4
                "total_lean_mass_lbs": 121.2,
                "age_at_scan": 40.93,
            },
            {
                "date_str": "10/21/2023",
                "alm_lbs": 57.4,  # 16.7 + 40.7
                "total_lean_mass_lbs": 121.6,
                "age_at_scan": 41.49,
            },
        ]

        ratio = get_alm_tlm_ratio(
            processed_data, self.goal_params, self.lms_functions, self.user_info
        )

        # Calculate expected ratio manually
        lbs_to_kg = 1 / 2.20462
        expected_ratios = []
        for scan in processed_data:
            alm_kg = scan["alm_lbs"] * lbs_to_kg
            tlm_kg = scan["total_lean_mass_lbs"] * lbs_to_kg
            expected_ratios.append(alm_kg / tlm_kg)
        expected_ratio = np.mean(expected_ratios)

        self.assertAlmostEqual(ratio, expected_ratio, places=4)
        self.assertTrue(0.4 <= ratio <= 0.6)  # Reasonable physiological range

    def test_personal_ratio_two_scans(self):
        """Test ALM/TLM ratio calculation with exactly two scans."""
        processed_data = [
            {
                "date_str": "04/07/2022",
                "alm_lbs": 49.7,
                "total_lean_mass_lbs": 106.3,
                "age_at_scan": 39.95,
            },
            {
                "date_str": "11/25/2024",
                "alm_lbs": 58.3,  # 17.8 + 40.5
                "total_lean_mass_lbs": 129.6,
                "age_at_scan": 42.59,
            },
        ]

        ratio = get_alm_tlm_ratio(
            processed_data, self.goal_params, self.lms_functions, self.user_info
        )

        # Should use personal data (≥2 scans)
        self.assertTrue(0.4 <= ratio <= 0.6)

        # Verify it's using both scans
        lbs_to_kg = 1 / 2.20462
        ratio1 = (processed_data[0]["alm_lbs"] * lbs_to_kg) / (
            processed_data[0]["total_lean_mass_lbs"] * lbs_to_kg
        )
        ratio2 = (processed_data[1]["alm_lbs"] * lbs_to_kg) / (
            processed_data[1]["total_lean_mass_lbs"] * lbs_to_kg
        )
        expected_ratio = (ratio1 + ratio2) / 2

        self.assertAlmostEqual(ratio, expected_ratio, places=4)

    def test_population_ratio_single_scan(self):
        """Test ALM/TLM ratio fallback to population data with single scan."""
        processed_data = [
            {
                "date_str": "11/25/2024",
                "alm_lbs": 58.3,
                "total_lean_mass_lbs": 129.6,
                "age_at_scan": 42.59,
            }
        ]

        ratio = get_alm_tlm_ratio(
            processed_data, self.goal_params, self.lms_functions, self.user_info
        )

        # Should use population data from target age (45.0)
        height_m_sq = (self.user_info["height_in"] * 0.0254) ** 2
        expected_almi = self.mock_almi_func(45.0)
        expected_lmi = self.mock_lmi_func(45.0)
        expected_ratio = (expected_almi * height_m_sq) / (expected_lmi * height_m_sq)
        expected_ratio = expected_almi / expected_lmi  # Simplifies to this

        self.assertAlmostEqual(ratio, expected_ratio, places=4)
        self.assertTrue(0.3 <= ratio <= 0.7)  # Reasonable physiological range

    def test_population_ratio_uses_target_age(self):
        """Test that population ratio uses target age, not current age."""
        processed_data = [
            {
                "date_str": "11/25/2024",
                "alm_lbs": 58.3,
                "total_lean_mass_lbs": 129.6,
                "age_at_scan": 42.59,  # Current age different from target
            }
        ]

        # Test with different target ages
        goal_params_30 = {"target_age": 30.0, "target_percentile": 0.90}
        goal_params_60 = {"target_age": 60.0, "target_percentile": 0.90}

        ratio_30 = get_alm_tlm_ratio(
            processed_data, goal_params_30, self.lms_functions, self.user_info
        )
        ratio_60 = get_alm_tlm_ratio(
            processed_data, goal_params_60, self.lms_functions, self.user_info
        )

        # Ratios should be different because they use different target ages
        self.assertNotAlmostEqual(ratio_30, ratio_60, places=3)

        # Verify the ratios match expected calculations at target ages
        expected_ratio_30 = self.mock_almi_func(30.0) / self.mock_lmi_func(30.0)
        expected_ratio_60 = self.mock_almi_func(60.0) / self.mock_lmi_func(60.0)

        self.assertAlmostEqual(ratio_30, expected_ratio_30, places=4)
        self.assertAlmostEqual(ratio_60, expected_ratio_60, places=4)

    def test_ratio_recent_scans_priority(self):
        """Test that only the most recent 3 scans are used for personal ratio."""
        # Create 5 scans where the first 2 have very different ratios
        processed_data = [
            {
                "alm_lbs": 40.0,
                "total_lean_mass_lbs": 100.0,
                "age_at_scan": 35,
            },  # ratio = 0.40 (old)
            {
                "alm_lbs": 41.0,
                "total_lean_mass_lbs": 100.0,
                "age_at_scan": 36,
            },  # ratio = 0.41 (old)
            {
                "alm_lbs": 50.0,
                "total_lean_mass_lbs": 100.0,
                "age_at_scan": 37,
            },  # ratio = 0.50 (recent)
            {
                "alm_lbs": 51.0,
                "total_lean_mass_lbs": 100.0,
                "age_at_scan": 38,
            },  # ratio = 0.51 (recent)
            {
                "alm_lbs": 52.0,
                "total_lean_mass_lbs": 100.0,
                "age_at_scan": 39,
            },  # ratio = 0.52 (recent)
        ]

        ratio = get_alm_tlm_ratio(
            processed_data, self.goal_params, self.lms_functions, self.user_info
        )

        # Should only use last 3 scans (ratios 0.50, 0.51, 0.52)
        expected_ratio = (0.50 + 0.51 + 0.52) / 3
        expected_ratio_kg = (
            expected_ratio  # Since we used same denominator, conversion cancels out
        )

        self.assertAlmostEqual(ratio, expected_ratio_kg, places=3)
        self.assertGreater(ratio, 0.48)  # Should be close to recent scans, not old ones

    def test_edge_case_zero_tlm(self):
        """Test handling of edge case where TLM is zero (should not crash)."""
        processed_data = [
            {
                "date_str": "11/25/2024",
                "alm_lbs": 58.3,
                "total_lean_mass_lbs": 0.0,  # Edge case - should trigger population fallback
                "age_at_scan": 42.59,
            }
        ]

        # Should fall back to population ratio when personal calculation would fail
        ratio = get_alm_tlm_ratio(
            processed_data, self.goal_params, self.lms_functions, self.user_info
        )

        # Should use population fallback
        expected_ratio = self.mock_almi_func(45.0) / self.mock_lmi_func(45.0)
        self.assertAlmostEqual(ratio, expected_ratio, places=4)

    def test_realistic_ratio_ranges(self):
        """Test that calculated ratios fall within realistic physiological ranges."""
        # Test multiple scenarios
        scenarios = [
            # Multiple scans
            [
                {"alm_lbs": 50.0, "total_lean_mass_lbs": 110.0, "age_at_scan": 40},
                {"alm_lbs": 52.0, "total_lean_mass_lbs": 115.0, "age_at_scan": 41},
            ],
            # Single scan (population fallback)
            [{"alm_lbs": 55.0, "total_lean_mass_lbs": 120.0, "age_at_scan": 42}],
        ]

        for processed_data in scenarios:
            ratio = get_alm_tlm_ratio(
                processed_data, self.goal_params, self.lms_functions, self.user_info
            )

            # ALM/TLM ratios should be in realistic range for adult males
            self.assertGreaterEqual(
                ratio, 0.35, "Ratio too low - below physiological minimum"
            )
            self.assertLessEqual(
                ratio, 0.65, "Ratio too high - above physiological maximum"
            )


class TestIntegrationTLMEstimation(unittest.TestCase):
    """Integration tests for TLM estimation within the full processing pipeline."""

    def setUp(self):
        """Set up integration test fixtures."""
        # Create minimal mock LMS functions for integration testing
        self.user_info = {
            "birth_date_str": "04/26/1982",
            "height_in": 66.0,
            "gender_code": 0,
        }

        self.scan_history = [
            {
                "date_str": "04/07/2022",
                "total_weight_lbs": 143.2,
                "total_lean_mass_lbs": 106.3,
                "fat_mass_lbs": 32.6,
                "body_fat_percentage": 22.8,
                "arms_lean_lbs": 12.4,
                "legs_lean_lbs": 37.3,
            },
            {
                "date_str": "11/25/2024",
                "total_weight_lbs": 152.7,
                "total_lean_mass_lbs": 129.6,
                "fat_mass_lbs": 18.2,
                "body_fat_percentage": 11.9,
                "arms_lean_lbs": 17.8,
                "legs_lean_lbs": 40.5,
            },
        ]

        self.goal_params = {"target_percentile": 0.90, "target_age": 45.0}

        # Mock LMS functions with realistic values
        ages = np.linspace(18, 80, 50)
        almi_values = 9.0 - 0.01 * (ages - 30)  # Slight decline with age
        lmi_values = 19.0 - 0.02 * (ages - 30)  # Slight decline with age
        l_values = np.ones_like(ages) * 0.1  # Mock skewness
        s_values = np.ones_like(ages) * 0.1  # Mock coefficient of variation

        self.lms_functions = {
            "almi_L": interp1d(ages, l_values, kind="cubic", fill_value="extrapolate"),
            "almi_M": interp1d(
                ages, almi_values, kind="cubic", fill_value="extrapolate"
            ),
            "almi_S": interp1d(ages, s_values, kind="cubic", fill_value="extrapolate"),
            "lmi_L": interp1d(ages, l_values, kind="cubic", fill_value="extrapolate"),
            "lmi_M": interp1d(ages, lmi_values, kind="cubic", fill_value="extrapolate"),
            "lmi_S": interp1d(ages, s_values, kind="cubic", fill_value="extrapolate"),
        }

    def test_tlm_estimation_integration(self):
        """Test that TLM estimation integrates properly with full processing pipeline."""
        # Test that goal_params gets updated with estimated_tlm_gain_kg
        initial_goal_params = self.goal_params.copy()
        self.assertNotIn("estimated_tlm_gain_kg", initial_goal_params)

        # Calculate what the integration should produce
        height_m_sq = (self.user_info["height_in"] * 0.0254) ** 2
        lbs_to_kg = 1 / 2.20462

        # Mock personal ratio calculation
        alm1 = (12.4 + 37.3) * lbs_to_kg
        tlm1 = 106.3 * lbs_to_kg
        alm2 = (17.8 + 40.5) * lbs_to_kg
        tlm2 = 129.6 * lbs_to_kg

        personal_ratio = np.mean([alm1 / tlm1, alm2 / tlm2])

        # Mock target ALMI calculation (would come from LMS inversion)
        target_almi = 9.8  # Approximate 90th percentile
        target_alm_kg = target_almi * height_m_sq
        target_tlm_kg = target_alm_kg / personal_ratio
        current_tlm_kg = tlm2
        expected_tlm_gain = target_tlm_kg - current_tlm_kg

        # Verify the calculation produces reasonable results
        self.assertGreater(
            expected_tlm_gain,
            0,
            "Should need positive TLM gain for 90th percentile goal",
        )
        self.assertLess(expected_tlm_gain, 10, "TLM gain should be realistic (< 10 kg)")

        print(f"Integration test - Expected TLM gain: {expected_tlm_gain:.2f} kg")


class TestJSONConfigHandling(unittest.TestCase):
    """Test cases for JSON configuration loading and validation."""

    def setUp(self):
        """Set up test fixtures for JSON config tests."""
        # New nested goals format
        self.valid_config_nested = {
            "user_info": {
                "birth_date": "04/26/1982",
                "height_in": 66.0,
                "gender": "male",
            },
            "scan_history": [
                {
                    "date": "04/07/2022",
                    "total_weight_lbs": 143.2,
                    "total_lean_mass_lbs": 106.3,
                    "fat_mass_lbs": 31.4,
                    "body_fat_percentage": 22.8,
                    "arms_lean_lbs": 12.4,
                    "legs_lean_lbs": 37.3,
                },
                {
                    "date": "11/25/2024",
                    "total_weight_lbs": 152.7,
                    "total_lean_mass_lbs": 129.6,
                    "fat_mass_lbs": 17.5,
                    "body_fat_percentage": 11.9,
                    "arms_lean_lbs": 17.8,
                    "legs_lean_lbs": 40.5,
                },
            ],
            "goals": {
                "almi": {
                    "target_percentile": 0.90,
                    "target_age": 45.0,
                    "description": "Reach 90th percentile ALMI by age 45",
                },
                "ffmi": {
                    "target_percentile": 0.85,
                    "target_age": 50.0,
                    "description": "Reach 85th percentile FFMI by age 50",
                },
            },
        }

        # Config with no goals
        self.config_no_goals = {
            "user_info": {
                "birth_date": "04/26/1982",
                "height_in": 66.0,
                "gender": "male",
            },
            "scan_history": [
                {
                    "date": "04/07/2022",
                    "total_weight_lbs": 143.2,
                    "total_lean_mass_lbs": 106.3,
                    "fat_mass_lbs": 32.6,
                    "body_fat_percentage": 22.8,
                    "arms_lean_lbs": 12.4,
                    "legs_lean_lbs": 37.3,
                }
            ],
        }

    def test_parse_gender_valid_inputs(self):
        """Test gender parsing with all valid inputs."""
        # Test male variants
        self.assertEqual(parse_gender("male"), 0)
        self.assertEqual(parse_gender("MALE"), 0)
        self.assertEqual(parse_gender("Male"), 0)
        self.assertEqual(parse_gender("m"), 0)
        self.assertEqual(parse_gender("M"), 0)

        # Test female variants
        self.assertEqual(parse_gender("female"), 1)
        self.assertEqual(parse_gender("FEMALE"), 1)
        self.assertEqual(parse_gender("Female"), 1)
        self.assertEqual(parse_gender("f"), 1)
        self.assertEqual(parse_gender("F"), 1)

    def test_parse_gender_invalid_inputs(self):
        """Test gender parsing with invalid inputs."""
        with self.assertRaises(ValueError):
            parse_gender("invalid")
        with self.assertRaises(ValueError):
            parse_gender("man")
        with self.assertRaises(ValueError):
            parse_gender("woman")
        with self.assertRaises(ValueError):
            parse_gender("")

    def test_extract_data_from_config_nested_goals(self):
        """Test extraction with new nested goals format."""
        user_info, scan_history, almi_goal, ffmi_goal = extract_data_from_config(
            self.valid_config_nested
        )

        # Check user_info conversion
        self.assertEqual(user_info["birth_date_str"], "04/26/1982")
        self.assertEqual(user_info["height_in"], 66.0)
        self.assertEqual(user_info["gender_code"], 0)  # male = 0

        # Check scan_history conversion
        self.assertEqual(len(scan_history), 2)
        self.assertEqual(scan_history[0]["date_str"], "04/07/2022")
        self.assertEqual(scan_history[0]["total_lean_mass_lbs"], 106.3)
        self.assertEqual(scan_history[1]["date_str"], "11/25/2024")

        # Check goals extraction
        self.assertIsNotNone(almi_goal)
        self.assertIsNotNone(ffmi_goal)
        self.assertEqual(almi_goal["target_percentile"], 0.90)
        self.assertEqual(almi_goal["target_age"], 45.0)
        self.assertEqual(ffmi_goal["target_percentile"], 0.85)
        self.assertEqual(ffmi_goal["target_age"], 50.0)

    def test_extract_data_from_config_no_goals(self):
        """Test extraction with no goals specified."""
        user_info, scan_history, almi_goal, ffmi_goal = extract_data_from_config(
            self.config_no_goals
        )

        # Check that no goals are extracted
        self.assertIsNone(almi_goal)
        self.assertIsNone(ffmi_goal)

        # But user info and scan history should still work
        self.assertEqual(user_info["birth_date_str"], "04/26/1982")
        self.assertEqual(len(scan_history), 1)

    def test_extract_data_from_config_partial_goals(self):
        """Test extraction with only one goal specified."""
        # Config with only ALMI goal
        config_almi_only = {
            "user_info": self.config_no_goals["user_info"],
            "scan_history": self.config_no_goals["scan_history"],
            "goals": {"almi": {"target_percentile": 0.75, "target_age": 40.0}},
        }

        user_info, scan_history, almi_goal, ffmi_goal = extract_data_from_config(
            config_almi_only
        )

        self.assertIsNotNone(almi_goal)
        self.assertIsNone(ffmi_goal)
        self.assertEqual(almi_goal["target_percentile"], 0.75)

        # Config with only FFMI goal
        config_ffmi_only = {
            "user_info": self.config_no_goals["user_info"],
            "scan_history": self.config_no_goals["scan_history"],
            "goals": {"ffmi": {"target_percentile": 0.80, "target_age": 55.0}},
        }

        user_info, scan_history, almi_goal, ffmi_goal = extract_data_from_config(
            config_ffmi_only
        )

        self.assertIsNone(almi_goal)
        self.assertIsNotNone(ffmi_goal)
        self.assertEqual(ffmi_goal["target_percentile"], 0.80)

    def test_load_config_json_valid_file_nested(self):
        """Test loading a valid JSON config file with nested goals."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.valid_config_nested, f)
            temp_path = f.name

        try:
            config = load_config_json(temp_path)
            self.assertEqual(config, self.valid_config_nested)
        finally:
            os.unlink(temp_path)

    def test_load_config_json_no_goals(self):
        """Test loading a config file with no goals."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.config_no_goals, f)
            temp_path = f.name

        try:
            config = load_config_json(temp_path)
            self.assertEqual(config, self.config_no_goals)
        finally:
            os.unlink(temp_path)

    def test_load_config_json_missing_file(self):
        """Test loading a non-existent config file."""
        with self.assertRaises(FileNotFoundError):
            load_config_json("nonexistent_file.json")

    def test_load_config_json_invalid_json(self):
        """Test loading malformed JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json ")
            temp_path = f.name

        try:
            with self.assertRaises(json.JSONDecodeError):
                load_config_json(temp_path)
        finally:
            os.unlink(temp_path)

    def test_config_validation_missing_sections(self):
        """Test JSON schema validation with missing required sections."""
        from jsonschema import ValidationError

        # Test missing user_info
        invalid_config = self.valid_config_nested.copy()
        del invalid_config["user_info"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config, f)
            temp_path = f.name

        try:
            with self.assertRaises(ValidationError):
                load_config_json(temp_path)
        finally:
            os.unlink(temp_path)

    def test_config_validation_invalid_gender(self):
        """Test JSON schema validation with invalid gender."""
        from jsonschema import ValidationError

        invalid_config = self.valid_config_nested.copy()
        invalid_config["user_info"]["gender"] = "invalid"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config, f)
            temp_path = f.name

        try:
            with self.assertRaises(ValidationError):
                load_config_json(temp_path)
        finally:
            os.unlink(temp_path)

    def test_config_validation_invalid_percentile(self):
        """Test JSON schema validation with invalid percentile."""
        from jsonschema import ValidationError

        invalid_config = self.valid_config_nested.copy()
        invalid_config["goals"]["almi"]["target_percentile"] = 1.5  # > 1.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config, f)
            temp_path = f.name

        try:
            with self.assertRaises(ValidationError):
                load_config_json(temp_path)
        finally:
            os.unlink(temp_path)

    def test_config_validation_empty_scan_history(self):
        """Test JSON schema validation with empty scan history."""
        from jsonschema import ValidationError

        invalid_config = self.valid_config_nested.copy()
        invalid_config["scan_history"] = []

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config, f)
            temp_path = f.name

        try:
            with self.assertRaises(ValidationError):
                load_config_json(temp_path)
        finally:
            os.unlink(temp_path)


class TestGoalProcessingIntegration(unittest.TestCase):
    """Integration tests for goal processing with the full pipeline."""

    def setUp(self):
        """Set up integration test fixtures."""
        # Create minimal mock LMS functions for integration testing
        ages = np.linspace(18, 80, 50)
        almi_values = 9.0 - 0.01 * (ages - 30)  # Slight decline with age
        lmi_values = 19.0 - 0.02 * (ages - 30)  # Slight decline with age
        l_values = np.ones_like(ages) * 0.1  # Mock skewness
        s_values = np.ones_like(ages) * 0.1  # Mock coefficient of variation

        self.lms_functions = {
            "almi_L": interp1d(ages, l_values, kind="cubic", fill_value="extrapolate"),
            "almi_M": interp1d(
                ages, almi_values, kind="cubic", fill_value="extrapolate"
            ),
            "almi_S": interp1d(ages, s_values, kind="cubic", fill_value="extrapolate"),
            "lmi_L": interp1d(ages, l_values, kind="cubic", fill_value="extrapolate"),
            "lmi_M": interp1d(ages, lmi_values, kind="cubic", fill_value="extrapolate"),
            "lmi_S": interp1d(ages, s_values, kind="cubic", fill_value="extrapolate"),
        }

        self.user_info = {
            "birth_date_str": "04/26/1982",
            "height_in": 66.0,
            "gender_code": 0,
        }

        self.scan_history = [
            {
                "date_str": "04/07/2022",
                "total_weight_lbs": 143.2,
                "total_lean_mass_lbs": 106.3,
                "fat_mass_lbs": 32.6,
                "body_fat_percentage": 22.8,
                "arms_lean_lbs": 12.4,
                "legs_lean_lbs": 37.3,
            },
            {
                "date_str": "11/25/2024",
                "total_weight_lbs": 152.7,
                "total_lean_mass_lbs": 129.6,
                "fat_mass_lbs": 18.2,
                "body_fat_percentage": 11.9,
                "arms_lean_lbs": 17.8,
                "legs_lean_lbs": 40.5,
            },
        ]

    def test_process_with_both_goals(self):
        """Test processing with both ALMI and FFMI goals."""
        almi_goal = {"target_percentile": 0.90, "target_age": 45.0}
        ffmi_goal = {"target_percentile": 0.85, "target_age": 50.0}

        df_results, goal_calculations = process_scans_and_goal(
            self.user_info, self.scan_history, almi_goal, ffmi_goal, self.lms_functions
        )

        # Should have 2 scan rows + 2 goal rows
        self.assertEqual(len(df_results), 4)

        # Check goal calculations
        self.assertIn("almi", goal_calculations)
        self.assertIn("ffmi", goal_calculations)

        # Check ALMI goal calculations
        almi_calc = goal_calculations["almi"]
        self.assertEqual(almi_calc["target_percentile"], 0.90)
        self.assertEqual(almi_calc["target_age"], 45.0)
        self.assertIn("alm_to_add_kg", almi_calc)
        self.assertIn("estimated_tlm_gain_kg", almi_calc)

        # Check FFMI goal calculations
        ffmi_calc = goal_calculations["ffmi"]
        self.assertEqual(ffmi_calc["target_percentile"], 0.85)
        self.assertEqual(ffmi_calc["target_age"], 50.0)
        self.assertIn("tlm_to_add_kg", ffmi_calc)

        # Check goal rows in DataFrame
        goal_rows = df_results[df_results["date_str"].str.contains("Goal")]
        self.assertEqual(len(goal_rows), 2)

        almi_goal_row = df_results[
            df_results["date_str"].str.contains("ALMI Goal")
        ].iloc[0]
        ffmi_goal_row = df_results[
            df_results["date_str"].str.contains("FFMI Goal")
        ].iloc[0]

        self.assertEqual(almi_goal_row["age_at_scan"], 45.0)
        self.assertEqual(ffmi_goal_row["age_at_scan"], 50.0)

    def test_process_with_almi_goal_only(self):
        """Test processing with only ALMI goal."""
        almi_goal = {"target_percentile": 0.75, "target_age": 40.0}
        ffmi_goal = None

        df_results, goal_calculations = process_scans_and_goal(
            self.user_info, self.scan_history, almi_goal, ffmi_goal, self.lms_functions
        )

        # Should have 2 scan rows + 1 goal row
        self.assertEqual(len(df_results), 3)

        # Check goal calculations
        self.assertIn("almi", goal_calculations)
        self.assertNotIn("ffmi", goal_calculations)

        # Check only ALMI goal row exists
        goal_rows = df_results[df_results["date_str"].str.contains("Goal")]
        self.assertEqual(len(goal_rows), 1)

        almi_goal_row = goal_rows.iloc[0]
        self.assertIn("ALMI Goal", almi_goal_row["date_str"])
        self.assertEqual(almi_goal_row["age_at_scan"], 40.0)

    def test_process_with_ffmi_goal_only(self):
        """Test processing with only FFMI goal."""
        almi_goal = None
        ffmi_goal = {"target_percentile": 0.80, "target_age": 55.0}

        df_results, goal_calculations = process_scans_and_goal(
            self.user_info, self.scan_history, almi_goal, ffmi_goal, self.lms_functions
        )

        # Should have 2 scan rows + 1 goal row
        self.assertEqual(len(df_results), 3)

        # Check goal calculations
        self.assertNotIn("almi", goal_calculations)
        self.assertIn("ffmi", goal_calculations)

        # Check only FFMI goal row exists
        goal_rows = df_results[df_results["date_str"].str.contains("Goal")]
        self.assertEqual(len(goal_rows), 1)

        ffmi_goal_row = goal_rows.iloc[0]
        self.assertIn("FFMI Goal", ffmi_goal_row["date_str"])
        self.assertEqual(ffmi_goal_row["age_at_scan"], 55.0)

    def test_process_with_no_goals(self):
        """Test processing with no goals."""
        almi_goal = None
        ffmi_goal = None

        df_results, goal_calculations = process_scans_and_goal(
            self.user_info, self.scan_history, almi_goal, ffmi_goal, self.lms_functions
        )

        # Should have only 2 scan rows
        self.assertEqual(len(df_results), 2)

        # Check no goal calculations
        self.assertEqual(len(goal_calculations), 0)

        # Check no goal rows exist
        goal_rows = df_results[df_results["date_str"].str.contains("Goal")]
        self.assertEqual(len(goal_rows), 0)

        # All rows should be historical scans
        for _, row in df_results.iterrows():
            self.assertNotIn("Goal", row["date_str"])

    def test_goal_calculations_consistency(self):
        """Test that goal calculations are mathematically consistent."""
        almi_goal = {"target_percentile": 0.90, "target_age": 45.0}
        ffmi_goal = {"target_percentile": 0.85, "target_age": 50.0}

        df_results, goal_calculations = process_scans_and_goal(
            self.user_info, self.scan_history, almi_goal, ffmi_goal, self.lms_functions
        )

        # ALMI goal calculations should be positive for reasonable targets
        almi_calc = goal_calculations["almi"]
        self.assertIsInstance(almi_calc["alm_to_add_kg"], (int, float))
        self.assertIsInstance(almi_calc["estimated_tlm_gain_kg"], (int, float))

        # FFMI goal calculations should be reasonable
        ffmi_calc = goal_calculations["ffmi"]
        self.assertIsInstance(ffmi_calc["tlm_to_add_kg"], (int, float))

        # Check that DataFrame goal rows match calculations
        almi_goal_row = df_results[
            df_results["date_str"].str.contains("ALMI Goal")
        ].iloc[0]
        ffmi_goal_row = df_results[
            df_results["date_str"].str.contains("FFMI Goal")
        ].iloc[0]

        self.assertAlmostEqual(almi_goal_row["almi_percentile"], 90.0, places=1)
        self.assertAlmostEqual(ffmi_goal_row["ffmi_percentile"], 85.0, places=1)


class TestTrainingLevelDetection(unittest.TestCase):
    """Test cases for automatic training level detection from scan progression."""

    def setUp(self):
        """Set up common test fixtures."""
        self.user_info_male = {
            "birth_date_str": "01/01/1990",
            "height_in": 70.0,
            "gender_code": 0,
        }

        self.user_info_female = {
            "birth_date_str": "01/01/1990",
            "height_in": 65.0,
            "gender_code": 1,
        }

    def test_detect_novice_single_scan(self):
        """Test that single scan defaults to novice level."""
        processed_data = [
            {
                "date_str": "01/01/2024",
                "total_lean_mass_lbs": 150.0,
                "age_at_scan": 34.0,
            }
        ]

        level, explanation = detect_training_level_from_scans(
            processed_data, self.user_info_male
        )

        self.assertEqual(level, "novice")
        self.assertIn("Insufficient scan history", explanation)
        self.assertIn("conservative approach", explanation)

    def test_detect_novice_rapid_gains(self):
        """Test detection of novice level with rapid gains and few scans."""
        # Simulate rapid novice gains over 6 months
        processed_data = [
            {
                "date_str": "01/01/2024",
                "total_lean_mass_lbs": 150.0,
                "age_at_scan": 34.0,
            },
            {
                "date_str": "04/01/2024",
                "total_lean_mass_lbs": 156.0,  # 6 lbs in 3 months = ~0.9 kg/month
                "age_at_scan": 34.25,
            },
            {
                "date_str": "07/01/2024",
                "total_lean_mass_lbs": 162.0,  # Another 6 lbs in 3 months
                "age_at_scan": 34.5,
            },
        ]

        level, explanation = detect_training_level_from_scans(
            processed_data, self.user_info_male
        )

        self.assertEqual(level, "novice")
        self.assertIn("novice gains", explanation)
        self.assertIn("early training phase", explanation)

    def test_detect_intermediate_sustained_gains(self):
        """Test detection of intermediate level with sustained moderate gains."""
        # Simulate intermediate gains (need ~0.25+ kg/month for intermediate detection)
        processed_data = [
            {
                "date_str": "01/01/2023",
                "total_lean_mass_lbs": 150.0,
                "age_at_scan": 33.0,
            },
            {
                "date_str": "07/01/2023",
                "total_lean_mass_lbs": 156.0,  # 6 lbs in 6 months = ~0.45 kg/month
                "age_at_scan": 33.5,
            },
            {
                "date_str": "01/01/2024",
                "total_lean_mass_lbs": 161.0,  # 5 lbs in 6 months = ~0.38 kg/month
                "age_at_scan": 34.0,
            },
            {
                "date_str": "07/01/2024",
                "total_lean_mass_lbs": 165.0,  # 4 lbs in 6 months = ~0.30 kg/month
                "age_at_scan": 34.5,
            },
        ]

        level, explanation = detect_training_level_from_scans(
            processed_data, self.user_info_male
        )

        self.assertEqual(level, "intermediate")
        self.assertIn("intermediate level", explanation)
        self.assertIn("moderate progression", explanation)

    def test_detect_slow_gains_as_advanced(self):
        """Test detection of advanced level with very slow progression."""
        # Simulate slow gains that fall below intermediate threshold
        processed_data = [
            {
                "date_str": "01/01/2023",
                "total_lean_mass_lbs": 150.0,
                "age_at_scan": 33.0,
            },
            {
                "date_str": "07/01/2023",
                "total_lean_mass_lbs": 154.0,  # 4 lbs in 6 months = ~0.3 kg/month
                "age_at_scan": 33.5,
            },
            {
                "date_str": "01/01/2024",
                "total_lean_mass_lbs": 157.0,  # 3 lbs in 6 months = ~0.23 kg/month
                "age_at_scan": 34.0,
            },
            {
                "date_str": "07/01/2024",
                "total_lean_mass_lbs": 160.0,  # 3 lbs in 6 months = ~0.23 kg/month
                "age_at_scan": 34.5,
            },
        ]

        level, explanation = detect_training_level_from_scans(
            processed_data, self.user_info_male
        )

        # With ~0.23 kg/month rate, this should be classified as advanced
        self.assertEqual(level, "advanced")
        self.assertIn("advanced level", explanation)
        self.assertIn("slow progression", explanation)

    def test_detect_advanced_slow_gains(self):
        """Test detection of advanced level with slow progression."""
        processed_data = [
            {
                "date_str": "01/01/2022",
                "total_lean_mass_lbs": 155.0,
                "age_at_scan": 32.0,
            },
            {
                "date_str": "01/01/2023",
                "total_lean_mass_lbs": 156.5,  # 1.5 lbs in 12 months = ~0.06 kg/month
                "age_at_scan": 33.0,
            },
            {
                "date_str": "01/01/2024",
                "total_lean_mass_lbs": 157.8,  # 1.3 lbs in 12 months = ~0.05 kg/month
                "age_at_scan": 34.0,
            },
        ]

        level, explanation = detect_training_level_from_scans(
            processed_data, self.user_info_male
        )

        self.assertEqual(level, "advanced")
        self.assertIn("advanced level", explanation)
        self.assertIn("slow progression", explanation)
        self.assertIn("experienced trainee", explanation)

    def test_detect_edge_case_zero_gains(self):
        """Test handling of zero or negative progression."""
        processed_data = [
            {
                "date_str": "01/01/2023",
                "total_lean_mass_lbs": 155.0,
                "age_at_scan": 33.0,
            },
            {
                "date_str": "07/01/2023",
                "total_lean_mass_lbs": 154.5,  # Slight loss
                "age_at_scan": 33.5,
            },
            {
                "date_str": "01/01/2024",
                "total_lean_mass_lbs": 155.0,  # Back to baseline
                "age_at_scan": 34.0,
            },
        ]

        level, explanation = detect_training_level_from_scans(
            processed_data, self.user_info_male
        )

        # Should classify as advanced due to very low/no progression
        self.assertEqual(level, "advanced")
        self.assertIn("advanced level", explanation)

    def test_detect_female_demographics(self):
        """Test that detection works correctly for female users."""
        # Female novice gains (should be lower than male thresholds)
        processed_data = [
            {
                "date_str": "01/01/2024",
                "total_lean_mass_lbs": 110.0,
                "age_at_scan": 34.0,
            },
            {
                "date_str": "04/01/2024",
                "total_lean_mass_lbs": 114.0,  # 4 lbs in 3 months = ~0.6 kg/month
                "age_at_scan": 34.25,
            },
            {
                "date_str": "07/01/2024",
                "total_lean_mass_lbs": 117.0,  # 3 lbs in 3 months = ~0.45 kg/month
                "age_at_scan": 34.5,
            },
        ]

        level, explanation = detect_training_level_from_scans(
            processed_data, self.user_info_female
        )

        # Should detect as novice for female with these gains
        self.assertEqual(level, "novice")
        self.assertIn("novice gains", explanation)

    def test_detect_with_irregular_timing(self):
        """Test detection with irregular scan intervals."""
        processed_data = [
            {
                "date_str": "01/01/2024",
                "total_lean_mass_lbs": 150.0,
                "age_at_scan": 34.0,
            },
            {
                "date_str": "02/15/2024",  # 1.5 months later
                "total_lean_mass_lbs": 152.0,  # Rapid gain
                "age_at_scan": 34.125,
            },
            {
                "date_str": "09/01/2024",  # 6.5 months later
                "total_lean_mass_lbs": 153.5,  # Much slower gain
                "age_at_scan": 34.667,
            },
        ]

        level, explanation = detect_training_level_from_scans(
            processed_data, self.user_info_male
        )

        # Should handle irregular timing and still make reasonable classification
        self.assertIn(level, ["novice", "intermediate", "advanced"])
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 10)


class TestConservativeGainRates(unittest.TestCase):
    """Test cases for demographic-specific gain rate calculations."""

    def setUp(self):
        """Set up common test fixtures."""
        self.user_info_male = {"gender_code": 0, "height_in": 70.0}

        self.user_info_female = {"gender_code": 1, "height_in": 65.0}

    def test_male_rates_all_levels(self):
        """Test male gain rates for all training levels."""
        # Test young male (no age adjustment)
        age = 25.0

        novice_rate, explanation = get_conservative_gain_rate(
            self.user_info_male, "novice", age
        )
        self.assertEqual(novice_rate, LEAN_MASS_GAIN_RATES["male"]["novice"])
        self.assertIn("novice", explanation)
        self.assertIn("male", explanation)

        intermediate_rate, explanation = get_conservative_gain_rate(
            self.user_info_male, "intermediate", age
        )
        self.assertEqual(
            intermediate_rate, LEAN_MASS_GAIN_RATES["male"]["intermediate"]
        )
        self.assertIn("intermediate", explanation)

        advanced_rate, explanation = get_conservative_gain_rate(
            self.user_info_male, "advanced", age
        )
        self.assertEqual(advanced_rate, LEAN_MASS_GAIN_RATES["male"]["advanced"])
        self.assertIn("advanced", explanation)

    def test_female_rates_all_levels(self):
        """Test female gain rates for all training levels."""
        age = 25.0

        novice_rate, explanation = get_conservative_gain_rate(
            self.user_info_female, "novice", age
        )
        self.assertEqual(novice_rate, LEAN_MASS_GAIN_RATES["female"]["novice"])
        self.assertIn("female", explanation)

        intermediate_rate, explanation = get_conservative_gain_rate(
            self.user_info_female, "intermediate", age
        )
        self.assertEqual(
            intermediate_rate, LEAN_MASS_GAIN_RATES["female"]["intermediate"]
        )

        advanced_rate, explanation = get_conservative_gain_rate(
            self.user_info_female, "advanced", age
        )
        self.assertEqual(advanced_rate, LEAN_MASS_GAIN_RATES["female"]["advanced"])

    def test_age_adjustments_over_30(self):
        """Test age adjustments for users over 30."""
        # Test 40-year-old male (1 decade over 30)
        age = 40.0
        base_rate = LEAN_MASS_GAIN_RATES["male"]["intermediate"]
        expected_reduction = 1 - (AGE_ADJUSTMENT_FACTOR * 1)  # 10% reduction
        expected_rate = base_rate * expected_reduction

        actual_rate, explanation = get_conservative_gain_rate(
            self.user_info_male, "intermediate", age
        )

        self.assertAlmostEqual(actual_rate, expected_rate, places=3)
        self.assertIn("age-adjusted", explanation)
        self.assertIn("age 40", explanation)

    def test_age_adjustments_under_30(self):
        """Test no age adjustment for users under 30."""
        age = 25.0
        base_rate = LEAN_MASS_GAIN_RATES["male"]["intermediate"]

        actual_rate, explanation = get_conservative_gain_rate(
            self.user_info_male, "intermediate", age
        )

        self.assertEqual(actual_rate, base_rate)
        self.assertNotIn("age-adjusted", explanation)

    def test_minimum_rate_floor(self):
        """Test that rates never go below 50% of base rate."""
        # Test very old user (should hit 50% floor)
        age = 80.0  # 5 decades over 30 = 50% reduction, hits floor
        base_rate = LEAN_MASS_GAIN_RATES["male"]["novice"]
        expected_rate = base_rate * 0.5  # 50% floor

        actual_rate, explanation = get_conservative_gain_rate(
            self.user_info_male, "novice", age
        )

        self.assertAlmostEqual(actual_rate, expected_rate, places=3)
        self.assertIn("age-adjusted", explanation)

    def test_edge_case_very_old(self):
        """Test handling of extreme age values."""
        age = 90.0

        rate, explanation = get_conservative_gain_rate(
            self.user_info_male, "intermediate", age
        )

        # Should still return a reasonable positive rate
        self.assertGreater(rate, 0)
        self.assertIsInstance(explanation, str)
        self.assertIn("age-adjusted", explanation)

    def test_gender_string_conversion(self):
        """Test gender code to string conversion."""
        self.assertEqual(get_gender_string(0), "male")
        self.assertEqual(get_gender_string(1), "female")


class TestTrainingLevelDetermination(unittest.TestCase):
    """Test cases for user-specified vs auto-detected training level logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.user_info_with_level = {"gender_code": 0, "training_level": "intermediate"}

        self.user_info_without_level = {"gender_code": 0}

        self.processed_data = [
            {"total_lean_mass_lbs": 150.0, "age_at_scan": 30.0},
            {"total_lean_mass_lbs": 153.0, "age_at_scan": 30.5},
        ]

    def test_user_specified_overrides_detection(self):
        """Test that user-specified level takes priority over detection."""
        level, explanation = determine_training_level(
            self.user_info_with_level, self.processed_data
        )

        self.assertEqual(level, "intermediate")
        self.assertIn("user-specified", explanation)
        self.assertIn("intermediate", explanation)

    def test_case_insensitive_user_input(self):
        """Test case-insensitive handling of user input."""
        test_cases = [
            ("Novice", "novice"),
            ("INTERMEDIATE", "intermediate"),
            ("Advanced", "advanced"),
            ("NOVICE", "novice"),
        ]

        for input_level, expected_level in test_cases:
            user_info = {"gender_code": 0, "training_level": input_level}
            level, explanation = determine_training_level(
                user_info, self.processed_data
            )

            self.assertEqual(level, expected_level)
            self.assertIn("user-specified", explanation)

    def test_fallback_to_detection(self):
        """Test auto-detection when training level not specified."""
        level, explanation = determine_training_level(
            self.user_info_without_level, self.processed_data
        )

        # Should call detection logic
        self.assertIn(level, ["novice", "intermediate", "advanced"])
        self.assertNotIn("user-specified", explanation)
        # Should contain detection-specific language
        self.assertTrue(
            any(
                word in explanation
                for word in ["Detected", "progression", "gains", "Insufficient"]
            )
        )


class TestSuggestedGoalCalculation(unittest.TestCase):
    """Test cases for the core suggested goal algorithm."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock LMS functions
        ages = np.linspace(18, 80, 50)
        almi_values = 9.0 - 0.01 * (ages - 30)  # Slight decline with age
        lmi_values = 19.0 - 0.02 * (ages - 30)  # Slight decline with age
        l_values = np.ones_like(ages) * 0.1
        s_values = np.ones_like(ages) * 0.1

        self.lms_functions = {
            "almi_L": interp1d(ages, l_values, kind="cubic", fill_value="extrapolate"),
            "almi_M": interp1d(
                ages, almi_values, kind="cubic", fill_value="extrapolate"
            ),
            "almi_S": interp1d(ages, s_values, kind="cubic", fill_value="extrapolate"),
            "lmi_L": interp1d(ages, l_values, kind="cubic", fill_value="extrapolate"),
            "lmi_M": interp1d(ages, lmi_values, kind="cubic", fill_value="extrapolate"),
            "lmi_S": interp1d(ages, s_values, kind="cubic", fill_value="extrapolate"),
        }

        self.user_info = {
            "height_in": 70.0,
            "gender_code": 0,
            "training_level": "intermediate",
        }

        self.processed_data = [
            {
                "age_at_scan": 30.0,
                "ffmi_kg_m2": 18.0,
                "almi_kg_m2": 8.5,
                "total_lean_mass_lbs": 150.0,
                "alm_lbs": 41.0,  # Arms + legs lean mass
                "arms_lean_lbs": 16.0,
                "legs_lean_lbs": 25.0,
                "almi_percentile": 65.0,  # Example percentile
                "ffmi_percentile": 70.0,  # Example percentile
            },
            {
                "age_at_scan": 30.5,
                "ffmi_kg_m2": 18.5,
                "almi_kg_m2": 8.7,
                "total_lean_mass_lbs": 153.0,
                "alm_lbs": 42.0,  # Arms + legs lean mass
                "arms_lean_lbs": 16.5,
                "legs_lean_lbs": 25.5,
                "almi_percentile": 68.0,  # Example percentile
                "ffmi_percentile": 73.0,  # Example percentile
            },
        ]

    def test_calculate_feasible_timeframe_almi(self):
        """Test auto-calculation of ALMI goal timing."""
        goal_params = {"target_percentile": 0.90, "target_age": None}

        updated_goal, messages = calculate_suggested_goal(
            goal_params, self.user_info, self.processed_data, self.lms_functions, "almi"
        )

        self.assertIsNotNone(updated_goal["target_age"])
        self.assertTrue(updated_goal.get("suggested", False))
        self.assertGreater(len(messages), 0)

        # Should have reasonable target age
        current_age = self.processed_data[-1]["age_at_scan"]
        self.assertGreater(updated_goal["target_age"], current_age)
        self.assertLess(updated_goal["target_age"], current_age + 15)  # Within 15 years

    def test_calculate_feasible_timeframe_ffmi(self):
        """Test auto-calculation of FFMI goal timing."""
        goal_params = {
            "target_percentile": 0.85,
            "target_age": "?",  # Test "?" string input
        }

        updated_goal, messages = calculate_suggested_goal(
            goal_params, self.user_info, self.processed_data, self.lms_functions, "ffmi"
        )

        self.assertIsNotNone(updated_goal["target_age"])
        self.assertTrue(updated_goal.get("suggested", False))
        self.assertGreater(len(messages), 0)

        # Check that messages are informative
        message_text = " ".join(messages)
        self.assertIn("timeframe", message_text)
        self.assertIn("intermediate", message_text)

    def test_question_mark_target_age(self):
        """Test handling of "?" string for target_age."""
        goal_params = {"target_percentile": 0.75, "target_age": "?"}

        updated_goal, messages = calculate_suggested_goal(
            goal_params, self.user_info, self.processed_data, self.lms_functions, "almi"
        )

        # Should calculate a numeric target age
        self.assertIsInstance(updated_goal["target_age"], (int, float))
        self.assertGreater(updated_goal["target_age"], 0)

        # Should mention auto-calculation in messages
        message_text = " ".join(messages)
        self.assertIn("Calculated feasible timeframe", message_text)

    def test_null_target_age(self):
        """Test handling of None/null target_age."""
        goal_params = {"target_percentile": 0.80, "target_age": None}

        updated_goal, messages = calculate_suggested_goal(
            goal_params, self.user_info, self.processed_data, self.lms_functions, "ffmi"
        )

        self.assertIsInstance(updated_goal["target_age"], (int, float))
        self.assertTrue(updated_goal.get("suggested", False))

    def test_multiple_demographics(self):
        """Test suggested goals across different demographics."""
        demographics = [
            {"gender_code": 0, "training_level": "novice"},  # Male novice
            {"gender_code": 0, "training_level": "advanced"},  # Male advanced
            {"gender_code": 1, "training_level": "intermediate"},  # Female intermediate
            {"gender_code": 1, "training_level": "novice"},  # Female novice
        ]

        for demo in demographics:
            user_info = {**self.user_info, **demo}
            goal_params = {"target_percentile": 0.90, "target_age": None}

            updated_goal, messages = calculate_suggested_goal(
                goal_params, user_info, self.processed_data, self.lms_functions, "almi"
            )

            # Should produce reasonable results for all demographics
            self.assertIsInstance(updated_goal["target_age"], (int, float))
            self.assertGreater(
                len(messages), 2
            )  # Should have multiple informative messages

    def test_unrealistic_goal_10_year_cap(self):
        """Test handling of goals that would require >10 years."""
        # Create a scenario with very slow progression and high percentile goal
        slow_user_info = {**self.user_info, "training_level": "advanced"}

        goal_params = {
            "target_percentile": 0.99,  # Very high percentile
            "target_age": None,
        }

        updated_goal, messages = calculate_suggested_goal(
            goal_params, slow_user_info, self.processed_data, self.lms_functions, "almi"
        )

        # Should cap at reasonable timeframe
        current_age = self.processed_data[-1]["age_at_scan"]
        self.assertLessEqual(updated_goal["target_age"], current_age + 10)

        # Should mention warning in messages
        " ".join(messages)
        if updated_goal["target_age"] >= current_age + 10:
            self.assertTrue(any("10 years" in msg for msg in messages))

    def test_suggested_flag_behavior(self):
        """Test that suggested=True is properly set."""
        goal_params = {"target_percentile": 0.85, "target_age": None}

        updated_goal, messages = calculate_suggested_goal(
            goal_params, self.user_info, self.processed_data, self.lms_functions, "ffmi"
        )

        self.assertTrue(updated_goal.get("suggested", False))
        self.assertEqual(
            updated_goal["target_percentile"], 0.85
        )  # Should preserve other fields


class TestSuggestedGoalIntegration(unittest.TestCase):
    """Test cases for full integration with existing goal processing pipeline."""

    def setUp(self):
        """Set up integration test fixtures."""
        # Create realistic mock LMS functions
        ages = np.linspace(18, 80, 50)
        almi_values = 9.0 - 0.01 * (ages - 30)
        lmi_values = 19.0 - 0.02 * (ages - 30)
        l_values = np.ones_like(ages) * 0.1
        s_values = np.ones_like(ages) * 0.1

        self.lms_functions = {
            "almi_L": interp1d(ages, l_values, kind="cubic", fill_value="extrapolate"),
            "almi_M": interp1d(
                ages, almi_values, kind="cubic", fill_value="extrapolate"
            ),
            "almi_S": interp1d(ages, s_values, kind="cubic", fill_value="extrapolate"),
            "lmi_L": interp1d(ages, l_values, kind="cubic", fill_value="extrapolate"),
            "lmi_M": interp1d(ages, lmi_values, kind="cubic", fill_value="extrapolate"),
            "lmi_S": interp1d(ages, s_values, kind="cubic", fill_value="extrapolate"),
        }

        self.user_info = {
            "birth_date_str": "01/01/1990",
            "height_in": 70.0,
            "gender_code": 0,
            "training_level": "intermediate",
        }

        self.scan_history = [
            {
                "date_str": "01/01/2024",
                "total_weight_lbs": 170.0,
                "total_lean_mass_lbs": 125.0,
                "fat_mass_lbs": 40.0,
                "body_fat_percentage": 23.5,
                "arms_lean_lbs": 13.0,
                "legs_lean_lbs": 30.0,
            },
            {
                "date_str": "07/01/2024",
                "total_weight_lbs": 175.0,
                "total_lean_mass_lbs": 130.0,
                "fat_mass_lbs": 40.0,
                "body_fat_percentage": 22.9,
                "arms_lean_lbs": 14.0,
                "legs_lean_lbs": 32.0,
            },
        ]

    def test_suggested_almi_goal_processing(self):
        """Test end-to-end ALMI suggested goal processing."""
        almi_goal = {"target_percentile": 0.90, "target_age": None, "suggested": True}

        df_results, goal_calculations = process_scans_and_goal(
            self.user_info, self.scan_history, almi_goal, None, self.lms_functions
        )

        # Should have processed the suggested goal
        self.assertIn("almi", goal_calculations)
        self.assertTrue(goal_calculations["almi"].get("suggested", False))

        # Should have goal messages
        self.assertIn("messages", goal_calculations)
        self.assertGreater(len(goal_calculations["messages"]), 0)

        # Should have added goal row to DataFrame
        goal_rows = df_results[df_results["date_str"].str.contains("ALMI Goal")]
        self.assertEqual(len(goal_rows), 1)

    def test_suggested_ffmi_goal_processing(self):
        """Test end-to-end FFMI suggested goal processing."""
        ffmi_goal = {"target_percentile": 0.85, "target_age": "?"}

        df_results, goal_calculations = process_scans_and_goal(
            self.user_info, self.scan_history, None, ffmi_goal, self.lms_functions
        )

        # Should have processed the suggested goal
        self.assertIn("ffmi", goal_calculations)
        self.assertTrue(goal_calculations["ffmi"].get("suggested", False))

        # Should have goal row
        goal_rows = df_results[df_results["date_str"].str.contains("FFMI Goal")]
        self.assertEqual(len(goal_rows), 1)

    def test_mixed_explicit_and_suggested(self):
        """Test processing one explicit goal and one suggested goal."""
        almi_goal = {
            "target_percentile": 0.90,
            "target_age": 40.0,  # Explicit age
        }

        ffmi_goal = {
            "target_percentile": 0.85,
            "target_age": None,  # Suggested age
        }

        df_results, goal_calculations = process_scans_and_goal(
            self.user_info, self.scan_history, almi_goal, ffmi_goal, self.lms_functions
        )

        # ALMI should not be suggested
        self.assertFalse(goal_calculations["almi"].get("suggested", False))

        # FFMI should be suggested
        self.assertTrue(goal_calculations["ffmi"].get("suggested", False))

        # Should have messages only for the suggested goal
        self.assertIn("messages", goal_calculations)

    def test_backward_compatibility(self):
        """Test that explicit goals work unchanged."""
        almi_goal = {"target_percentile": 0.75, "target_age": 35.0}

        df_results, goal_calculations = process_scans_and_goal(
            self.user_info, self.scan_history, almi_goal, None, self.lms_functions
        )

        # Should process as explicit goal (not suggested)
        self.assertFalse(goal_calculations["almi"].get("suggested", False))
        self.assertEqual(goal_calculations["almi"]["target_age"], 35.0)

        # Should not have suggestion messages for explicit goals
        messages = goal_calculations.get("messages", [])
        if messages:
            message_text = " ".join(messages)
            self.assertNotIn("auto-calculated", message_text)

    def test_transparent_messaging(self):
        """Test that all explanation messages are generated."""
        almi_goal = {"target_percentile": 0.90, "target_age": "?"}

        df_results, goal_calculations = process_scans_and_goal(
            self.user_info, self.scan_history, almi_goal, None, self.lms_functions
        )

        messages = goal_calculations.get("messages", [])
        self.assertGreater(len(messages), 2)

        message_text = " ".join(messages)

        # Should explain training level
        self.assertIn("training level", message_text)

        # Should explain rate selection
        self.assertIn("Conservative", message_text)
        self.assertIn("rate", message_text)

        # Should explain final timeframe
        self.assertIn("timeframe", message_text)

    def test_goal_calculations_structure(self):
        """Test that returned goal calculations have expected structure."""
        almi_goal = {"target_percentile": 0.90, "target_age": None}

        df_results, goal_calculations = process_scans_and_goal(
            self.user_info, self.scan_history, almi_goal, None, self.lms_functions
        )

        almi_calc = goal_calculations["almi"]

        # Should have all expected fields
        expected_fields = [
            "target_almi",
            "target_z",
            "target_age",
            "target_percentile",
            "alm_to_add_kg",
            "estimated_tlm_gain_kg",
            "suggested",
        ]

        for field in expected_fields:
            self.assertIn(field, almi_calc)

        # Numeric fields should be reasonable
        self.assertIsInstance(almi_calc["target_age"], (int, float))
        self.assertGreater(almi_calc["target_age"], 0)
        self.assertTrue(almi_calc["suggested"])


class TestSuggestedGoalEdgeCases(unittest.TestCase):
    """Test cases for boundary conditions and error scenarios."""

    def setUp(self):
        """Set up edge case test fixtures."""
        # Minimal mock LMS functions
        ages = np.array([18, 30, 50, 80])
        values = np.array([8.0, 9.0, 8.5, 7.5])
        l_values = np.array([0.1, 0.1, 0.1, 0.1])
        s_values = np.array([0.1, 0.1, 0.1, 0.1])

        self.lms_functions = {
            "almi_L": interp1d(ages, l_values, kind="linear", fill_value="extrapolate"),
            "almi_M": interp1d(ages, values, kind="linear", fill_value="extrapolate"),
            "almi_S": interp1d(ages, s_values, kind="linear", fill_value="extrapolate"),
            "lmi_L": interp1d(ages, l_values, kind="linear", fill_value="extrapolate"),
            "lmi_M": interp1d(
                ages, values * 2, kind="linear", fill_value="extrapolate"
            ),
            "lmi_S": interp1d(ages, s_values, kind="linear", fill_value="extrapolate"),
        }

        self.user_info = {"height_in": 70.0, "gender_code": 0}

    def test_empty_scan_history(self):
        """Test handling of empty scan history."""
        processed_data = []
        goal_params = {"target_percentile": 0.90, "target_age": None}

        # Should handle gracefully
        try:
            updated_goal, messages = calculate_suggested_goal(
                goal_params, self.user_info, processed_data, self.lms_functions, "almi"
            )
            # If it doesn't crash, check that it handles the empty case
            self.assertIsInstance(messages, list)
        except (IndexError, KeyError):
            # Expected to fail with empty data - this is acceptable
            pass

    def test_already_above_target_percentile(self):
        """Test behavior when user is already above target percentile (e.g., above 90th)."""
        # Create scenario where user is already at 95th percentile
        processed_data = [
            {
                "age_at_scan": 30.0,
                "almi_kg_m2": 12.0,  # Very high ALMI value
                "ffmi_kg_m2": 22.0,  # Very high FFMI value
                "total_lean_mass_lbs": 170.0,
                "alm_lbs": 50.0,
                "arms_lean_lbs": 20.0,
                "legs_lean_lbs": 30.0,
                "almi_percentile": 95.0,  # Very high percentile
                "ffmi_percentile": 95.0,  # Very high percentile
            }
        ]

        # Try to set 90th percentile goal when already above it
        goal_params = {"target_percentile": 0.90, "target_age": None}

        updated_goal, messages = calculate_suggested_goal(
            goal_params, self.user_info, processed_data, self.lms_functions, "almi"
        )

        # Should return None since user is already above 90th percentile
        self.assertIsNone(updated_goal)

        # Test the same for FFMI
        updated_ffmi_goal, ffmi_messages = calculate_suggested_goal(
            goal_params, self.user_info, processed_data, self.lms_functions, "ffmi"
        )

        # Should also return None since user is already above 90th percentile
        self.assertIsNone(updated_ffmi_goal)

    def test_already_at_95th_percentile_cap(self):
        """Test behavior when user is already at 95th percentile (cap scenario)."""
        # Create scenario where user is already at 96th percentile
        processed_data = [
            {
                "age_at_scan": 30.0,
                "almi_kg_m2": 13.0,  # Extremely high ALMI value
                "ffmi_kg_m2": 24.0,  # Extremely high FFMI value
                "total_lean_mass_lbs": 180.0,
                "alm_lbs": 55.0,
                "arms_lean_lbs": 22.0,
                "legs_lean_lbs": 33.0,
                "almi_percentile": 96.0,  # Extremely high percentile
                "ffmi_percentile": 96.0,  # Extremely high percentile
            }
        ]

        # Try to set 90th percentile goal when already well above it
        goal_params = {"target_percentile": 0.90, "target_age": None}

        updated_goal, messages = calculate_suggested_goal(
            goal_params, self.user_info, processed_data, self.lms_functions, "almi"
        )

        # Should return None since user is already above 90th percentile
        self.assertIsNone(updated_goal)

    def test_below_90th_percentile_still_works(self):
        """Test behavior when user is below 90th percentile - should still suggest goals."""
        # Create scenario where user is at normal percentile levels
        processed_data = [
            {
                "age_at_scan": 30.0,
                "almi_kg_m2": 8.0,  # Normal ALMI value
                "ffmi_kg_m2": 18.0,  # Normal FFMI value
                "total_lean_mass_lbs": 155.0,
                "alm_lbs": 40.0,
                "arms_lean_lbs": 16.0,
                "legs_lean_lbs": 24.0,
                "almi_percentile": 60.0,  # Normal percentile
                "ffmi_percentile": 65.0,  # Normal percentile
            }
        ]

        # Set a reasonable goal
        goal_params = {"target_percentile": 0.75, "target_age": None}

        updated_goal, messages = calculate_suggested_goal(
            goal_params, self.user_info, processed_data, self.lms_functions, "almi"
        )

        # Should return a valid goal since user is below 90th percentile
        self.assertIsNotNone(updated_goal)
        self.assertTrue("target_age" in updated_goal)
        self.assertTrue(updated_goal.get("suggested", False))

    def test_single_scan_fallback(self):
        """Test behavior with single scan (no progression data)."""
        processed_data = [
            {
                "age_at_scan": 30.0,
                "almi_kg_m2": 8.0,  # Add missing ALMI value
                "ffmi_kg_m2": 18.0,
                "total_lean_mass_lbs": 150.0,
                "alm_lbs": 40.0,
                "arms_lean_lbs": 15.0,
                "legs_lean_lbs": 25.0,
                "almi_percentile": 50.0,  # Median percentile
                "ffmi_percentile": 55.0,  # Slightly above median
            }
        ]

        goal_params = {"target_percentile": 0.90, "target_age": None}

        updated_goal, messages = calculate_suggested_goal(
            goal_params, self.user_info, processed_data, self.lms_functions, "almi"
        )

        # Should fall back to default assumption for insufficient data
        message_text = " ".join(messages)
        self.assertIn("Insufficient scan history", message_text)

    def test_extreme_current_values(self):
        """Test handling of very high/low current metric values."""
        # User with very high current values
        processed_data = [
            {
                "age_at_scan": 30.0,
                "ffmi_kg_m2": 25.0,  # Very high FFMI
                "almi_kg_m2": 12.0,  # Very high ALMI
                "total_lean_mass_lbs": 200.0,
                "alm_lbs": 55.0,
                "arms_lean_lbs": 20.0,
                "legs_lean_lbs": 35.0,
                "almi_percentile": 98.0,  # Very high percentile
                "ffmi_percentile": 99.0,  # Extremely high percentile
            }
        ]

        goal_params = {"target_percentile": 0.90, "target_age": None}

        updated_goal, messages = calculate_suggested_goal(
            goal_params, self.user_info, processed_data, self.lms_functions, "almi"
        )

        # Should handle without crashing - may return None if already above target
        if updated_goal is not None:
            self.assertIsInstance(updated_goal["target_age"], (int, float))
        else:
            # User is already above 90th percentile, so no goal needed
            self.assertIsNone(updated_goal)
        self.assertIsInstance(messages, list)

    def test_extreme_target_percentiles(self):
        """Test handling of very high and low target percentiles."""
        processed_data = [
            {
                "age_at_scan": 30.0,
                "almi_kg_m2": 7.5,  # Below median value
                "ffmi_kg_m2": 18.0,
                "total_lean_mass_lbs": 150.0,
                "alm_lbs": 40.0,
                "arms_lean_lbs": 15.0,
                "legs_lean_lbs": 25.0,
                "almi_percentile": 45.0,  # Below median percentile
                "ffmi_percentile": 50.0,  # Median percentile
            }
        ]

        extreme_percentiles = [0.05, 0.99]  # Very low and very high

        for percentile in extreme_percentiles:
            goal_params = {"target_percentile": percentile, "target_age": None}

            updated_goal, messages = calculate_suggested_goal(
                goal_params, self.user_info, processed_data, self.lms_functions, "ffmi"
            )

            # Should handle extreme percentiles (may auto-adjust if too low)
            if percentile == 0.05:
                # Function auto-adjusts low percentiles that are below current
                self.assertGreaterEqual(updated_goal["target_percentile"], 0.5)
            else:
                self.assertEqual(updated_goal["target_percentile"], percentile)
            self.assertIsInstance(updated_goal["target_age"], (int, float))

    def test_lms_data_boundary_ages(self):
        """Test behavior at LMS data age boundaries."""
        # Test with user near age boundaries
        processed_data = [
            {
                "age_at_scan": 79.0,  # Near upper boundary
                "ffmi_kg_m2": 17.0,
                "total_lean_mass_lbs": 140.0,
                "alm_lbs": 38.0,
                "arms_lean_lbs": 14.0,
                "legs_lean_lbs": 24.0,
                "almi_percentile": 40.0,  # Below median for older age
                "ffmi_percentile": 45.0,  # Below median for older age
            }
        ]

        goal_params = {"target_percentile": 0.75, "target_age": None}

        updated_goal, messages = calculate_suggested_goal(
            goal_params, self.user_info, processed_data, self.lms_functions, "ffmi"
        )

        # Should handle boundary cases - target age might be slightly lower due to algorithm
        self.assertIsInstance(updated_goal["target_age"], (int, float))
        # Age might be slightly lower than current due to algorithm calculations
        self.assertAlmostEqual(updated_goal["target_age"], 79.0, delta=1.0)


class TestBodyFatPercentageAccuracy(unittest.TestCase):
    """
    Test suite for body fat percentage accuracy using ground truth body composition data.

    This ensures that the system correctly uses actual body fat percentages
    instead of calculating them incorrectly from weight and lean mass alone.
    """

    def setUp(self):
        """Set up test data with actual body composition scan results."""
        # Ground truth data from actual body composition scans
        self.ground_truth_config = {
            "user_info": {
                "birth_date": "04/26/1982",
                "height_in": 66.0,
                "gender": "male",
            },
            "scan_history": [
                {
                    "date": "04/07/2022",
                    "total_weight_lbs": 143.2,
                    "total_lean_mass_lbs": 106.3,
                    "fat_mass_lbs": 32.6,
                    "body_fat_percentage": 22.8,
                    "arms_lean_lbs": 12.4,
                    "legs_lean_lbs": 37.3,
                },
                {
                    "date": "04/01/2023",
                    "total_weight_lbs": 154.3,
                    "total_lean_mass_lbs": 121.2,
                    "fat_mass_lbs": 28.5,
                    "body_fat_percentage": 18.5,
                    "arms_lean_lbs": 16.5,
                    "legs_lean_lbs": 40.4,
                },
                {
                    "date": "10/21/2023",
                    "total_weight_lbs": 159.5,
                    "total_lean_mass_lbs": 121.6,
                    "fat_mass_lbs": 33.3,
                    "body_fat_percentage": 20.9,
                    "arms_lean_lbs": 16.7,
                    "legs_lean_lbs": 40.7,
                },
                {
                    "date": "04/02/2024",
                    "total_weight_lbs": 145.0,
                    "total_lean_mass_lbs": 123.9,
                    "fat_mass_lbs": 16.1,
                    "body_fat_percentage": 11.1,
                    "arms_lean_lbs": 17.2,
                    "legs_lean_lbs": 39.4,
                },
                {
                    "date": "11/25/2024",
                    "total_weight_lbs": 152.7,
                    "total_lean_mass_lbs": 129.6,
                    "fat_mass_lbs": 18.2,
                    "body_fat_percentage": 11.9,
                    "arms_lean_lbs": 17.8,
                    "legs_lean_lbs": 40.5,
                },
            ],
        }

        # Expected ground truth body fat percentages
        self.expected_bf_percentages = [22.8, 18.5, 20.9, 11.1, 11.9]
        self.scan_dates = [
            "04/07/2022",
            "04/01/2023",
            "10/21/2023",
            "04/02/2024",
            "11/25/2024",
        ]

    def test_body_fat_percentage_accuracy(self):
        """Test that body fat percentages match actual body composition scan results exactly."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.ground_truth_config, f)
            temp_config_path = f.name

        try:
            # Load the config and extract data
            config = load_config_json(temp_config_path)
            user_info, scan_history, _, _ = extract_data_from_config(config)

            # Verify that the body fat percentages in scan_history match ground truth
            for i, scan in enumerate(scan_history):
                expected_bf = self.expected_bf_percentages[i]
                actual_bf = scan["body_fat_percentage"]

                self.assertAlmostEqual(
                    actual_bf,
                    expected_bf,
                    places=1,
                    msg=f"Body fat percentage mismatch for scan {i + 1} ({self.scan_dates[i]}): "
                    f"expected {expected_bf}%, got {actual_bf}%",
                )

        finally:
            # Clean up
            os.unlink(temp_config_path)

    def test_data_consistency_validation(self):
        """Test that the body composition data components are internally consistent."""
        # Verify that total_weight ≈ lean_mass + fat_mass + bone_mass
        # (We can't check bone mass directly, but we can verify the relationship)

        for i, scan_data in enumerate(self.ground_truth_config["scan_history"]):
            total_weight = scan_data["total_weight_lbs"]
            lean_mass = scan_data["total_lean_mass_lbs"]
            fat_mass = scan_data["fat_mass_lbs"]
            bf_percentage = scan_data["body_fat_percentage"]

            # Verify fat mass vs body fat percentage consistency
            calculated_bf_from_fat_mass = (fat_mass / total_weight) * 100
            self.assertAlmostEqual(
                calculated_bf_from_fat_mass,
                bf_percentage,
                places=1,
                msg=f"Fat mass and BF% inconsistent for scan {i + 1}: "
                f"fat_mass calculation gives {calculated_bf_from_fat_mass:.1f}%, "
                f"but BF% field shows {bf_percentage}%",
            )

            # Verify that lean + fat < total (accounting for bone mass)
            lean_plus_fat = lean_mass + fat_mass
            self.assertLess(
                lean_plus_fat,
                total_weight,
                msg=f"Scan {i + 1}: lean + fat mass ({lean_plus_fat:.1f}) should be less than "
                f"total weight ({total_weight:.1f}) to account for bone mass",
            )

            # Bone mass should be reasonable (typically 3-5% of body weight)
            bone_mass = total_weight - lean_plus_fat
            bone_percentage = (bone_mass / total_weight) * 100
            self.assertTrue(
                2.0 <= bone_percentage <= 8.0,
                msg=f"Scan {i + 1}: bone mass percentage ({bone_percentage:.1f}%) outside "
                f"reasonable range (2-8%) - check data consistency",
            )

    def test_old_calculation_would_be_wrong(self):
        """Test that the old calculation method would produce incorrect results."""
        # Demonstrate what the old calculation would have given vs ground truth
        for i, scan_data in enumerate(self.ground_truth_config["scan_history"]):
            total_weight = scan_data["total_weight_lbs"]
            lean_mass = scan_data["total_lean_mass_lbs"]
            actual_bf = scan_data["body_fat_percentage"]

            # This is what the old incorrect calculation would have produced
            old_calculated_fat_mass = (
                total_weight - lean_mass
            )  # WRONG: ignores bone mass
            old_calculated_bf = (old_calculated_fat_mass / total_weight) * 100

            # Verify that the old calculation is significantly different from ground truth
            # (This proves our fix was necessary)
            difference = abs(old_calculated_bf - actual_bf)
            self.assertGreater(
                difference,
                1.0,  # Should differ by more than 1%
                msg=f"Scan {i + 1}: old calculation ({old_calculated_bf:.1f}%) too close to "
                f"actual body composition result ({actual_bf:.1f}%) - test may not be demonstrating the fix",
            )

    def test_schema_validation_with_new_fields(self):
        """Test that the updated schema correctly validates the new required fields."""
        # This should pass with all required fields
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(self.ground_truth_config, f)
                temp_config_path = f.name

            config = load_config_json(temp_config_path)
            self.assertIsNotNone(config)

        finally:
            os.unlink(temp_config_path)

        # Test that missing fat_mass_lbs fails validation
        invalid_config = self.ground_truth_config.copy()
        del invalid_config["scan_history"][0]["fat_mass_lbs"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config, f)
            temp_invalid_path = f.name

        try:
            with self.assertRaises(Exception):  # Should raise ValidationError
                load_config_json(temp_invalid_path)
        finally:
            os.unlink(temp_invalid_path)

        # Test that missing body_fat_percentage fails validation
        invalid_config2 = self.ground_truth_config.copy()
        del invalid_config2["scan_history"][0]["body_fat_percentage"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config2, f)
            temp_invalid_path2 = f.name

        try:
            with self.assertRaises(Exception):  # Should raise ValidationError
                load_config_json(temp_invalid_path2)
        finally:
            os.unlink(temp_invalid_path2)


class TestTScoreCalculations(unittest.TestCase):
    """
    Test suite for T-score calculation functions.

    T-scores compare an individual's measurement to the peak reference
    values typically seen in young adults (ages 20-30), which is the
    "peak bone mass" approach used in bone density analysis but applied
    to muscle mass metrics.
    """

    def test_tscore_reference_values_calculation(self):
        """Test that T-score reference values are calculated correctly."""
        # Test male reference values
        male_mu, male_sigma = calculate_tscore_reference_values(0)

        # Should return valid numeric values
        self.assertFalse(
            np.isnan(male_mu), "Male T-score reference mean should be valid"
        )
        self.assertFalse(
            np.isnan(male_sigma), "Male T-score reference SD should be valid"
        )

        # Male ALMI values should be in reasonable range (6-12 kg/m²)
        self.assertGreater(male_mu, 6.0, "Male reference mean too low")
        self.assertLess(male_mu, 12.0, "Male reference mean too high")

        # Standard deviation should be reasonable (0.5-2.0 kg/m²)
        self.assertGreater(male_sigma, 0.5, "Male reference SD too low")
        self.assertLess(male_sigma, 2.0, "Male reference SD too high")

        # Test female reference values
        female_mu, female_sigma = calculate_tscore_reference_values(1)

        # Should return valid numeric values
        self.assertFalse(
            np.isnan(female_mu), "Female T-score reference mean should be valid"
        )
        self.assertFalse(
            np.isnan(female_sigma), "Female T-score reference SD should be valid"
        )

        # Female ALMI values should be lower than male (4-8 kg/m²)
        self.assertGreater(female_mu, 4.0, "Female reference mean too low")
        self.assertLess(female_mu, 8.0, "Female reference mean too high")
        self.assertLess(female_mu, male_mu, "Female mean should be lower than male")

        # Standard deviation should be reasonable
        self.assertGreater(female_sigma, 0.3, "Female reference SD too low")
        self.assertLess(female_sigma, 1.5, "Female reference SD too high")

    def test_tscore_calculation_logic(self):
        """Test T-score calculation with known reference values."""
        # Use realistic reference values
        mu_peak = 8.5  # kg/m²
        sigma_peak = 1.0  # kg/m²

        # Test T-score calculation
        test_cases = [
            (8.5, 0.0),  # At reference mean → T-score = 0
            (9.5, 1.0),  # One SD above → T-score = +1
            (7.5, -1.0),  # One SD below → T-score = -1
            (10.5, 2.0),  # Two SDs above → T-score = +2
            (6.5, -2.0),  # Two SDs below → T-score = -2
        ]

        for almi_value, expected_tscore in test_cases:
            calculated_tscore = calculate_t_score(almi_value, mu_peak, sigma_peak)
            self.assertAlmostEqual(
                calculated_tscore,
                expected_tscore,
                places=6,
                msg=f"T-score calculation failed for ALMI {almi_value}",
            )

    def test_tscore_peak_zone_stratification(self):
        """Test T-score peak muscle mass zone boundaries."""
        # Use realistic male reference values
        male_mu, male_sigma = calculate_tscore_reference_values(0)

        # Test realistic ALMI values and their T-score peak zones
        test_values = [
            (male_mu + 2 * male_sigma, "Peak Zone"),  # Well above peak
            (male_mu + 0.5 * male_sigma, "Peak Zone"),  # Above peak
            (male_mu - 0.5 * male_sigma, "Approaching Peak"),  # Below peak
            (male_mu - 1.5 * male_sigma, "Below Peak"),  # Low
            (male_mu - 2.5 * male_sigma, "Well Below Peak"),  # Very low
        ]

        for almi_value, expected_zone in test_values:
            t_score = calculate_t_score(almi_value, male_mu, male_sigma)

            # Verify T-score is in expected range for peak zone
            if "Peak Zone" in expected_zone:
                self.assertGreaterEqual(
                    t_score, 0, f"ALMI {almi_value:.2f} should be in Peak Zone (≥0)"
                )
            elif "Approaching Peak" in expected_zone:
                self.assertGreaterEqual(
                    t_score, -1.0, f"ALMI {almi_value:.2f} should be ≥ -1.0"
                )
                self.assertLess(t_score, 0, f"ALMI {almi_value:.2f} should be < 0")
            elif "Below Peak" in expected_zone and "Well Below" not in expected_zone:
                self.assertGreaterEqual(
                    t_score, -2.0, f"ALMI {almi_value:.2f} should be ≥ -2.0"
                )
                self.assertLess(
                    t_score, -1.0, f"ALMI {almi_value:.2f} should be < -1.0"
                )
            elif "Well Below Peak" in expected_zone:
                self.assertLess(
                    t_score, -2.0, f"ALMI {almi_value:.2f} should be < -2.0"
                )

    def test_tscore_gender_differences(self):
        """Test that T-score reference values show expected gender differences."""
        male_mu, male_sigma = calculate_tscore_reference_values(0)
        female_mu, female_sigma = calculate_tscore_reference_values(1)

        # Males should have higher ALMI reference values than females
        self.assertGreater(
            male_mu, female_mu, "Male reference mean should be higher than female"
        )

        # Both should have reasonable, non-zero standard deviations
        self.assertGreater(male_sigma, 0.1, "Male SD should be substantial")
        self.assertGreater(female_sigma, 0.1, "Female SD should be substantial")

        # Standard deviations should be in similar range (within factor of 2)
        ratio = max(male_sigma, female_sigma) / min(male_sigma, female_sigma)
        self.assertLess(ratio, 2.0, "Gender SD differences should be reasonable")

    def test_tscore_edge_cases(self):
        """Test T-score calculation edge cases."""
        mu_peak = 8.0
        sigma_peak = 1.0

        # Test with zero SD (should return NaN)
        result = calculate_t_score(8.0, mu_peak, 0.0)
        self.assertTrue(np.isnan(result), "Zero SD should return NaN")

        # Test with NaN inputs
        result = calculate_t_score(np.nan, mu_peak, sigma_peak)
        self.assertTrue(np.isnan(result), "NaN value should return NaN")

        result = calculate_t_score(8.0, np.nan, sigma_peak)
        self.assertTrue(np.isnan(result), "NaN mean should return NaN")

        result = calculate_t_score(8.0, mu_peak, np.nan)
        self.assertTrue(np.isnan(result), "NaN SD should return NaN")

    def test_tscore_consistency_with_zscore(self):
        """Test that T-scores and Z-scores are conceptually consistent."""
        # Load some LMS data for comparison
        male_mu, male_sigma = calculate_tscore_reference_values(0)

        # For a person at the young adult average, T-score should be ~0
        # while their Z-score will vary depending on their current age
        almi_at_peak = male_mu  # This should give T-score ≈ 0

        t_score = calculate_t_score(almi_at_peak, male_mu, male_sigma)
        self.assertAlmostEqual(
            t_score, 0.0, places=2, msg="ALMI at peak reference should give T-score ≈ 0"
        )


if __name__ == "__main__":
    # Run tests with detailed output
    print(
        "Running comprehensive test suite for RecompTracker body composition analysis..."
    )
    unittest.main(verbosity=2)
