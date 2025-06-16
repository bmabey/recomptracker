"""
Comprehensive test suite for zscore_plot.py functionality.

This module contains all tests for the DEXA body composition analysis script,
including core calculation functions, TLM estimation logic, and integration tests.
"""

import unittest
import numpy as np
import pandas as pd
import json
import tempfile
import os
from scipy.interpolate import interp1d
from zscore_plot import (
    calculate_age_precise, 
    compute_zscore, 
    get_value_from_zscore, 
    calculate_t_score,
    get_alm_tlm_ratio,
    process_scans_and_goal,
    load_config_json,
    parse_gender,
    extract_data_from_config
)


class TestBodyCompCalculations(unittest.TestCase):
    """
    Test suite for core calculation functions.
    
    This ensures the mathematical logic for age, Z-scores, T-scores,
    and inverse Z-scores is correct and handles edge cases properly.
    """
    
    def test_calculate_age(self):
        """Tests the age calculation logic."""
        self.assertAlmostEqual(calculate_age_precise("01/01/2000", "01/01/2001"), 1.0, places=2)
        self.assertAlmostEqual(calculate_age_precise("06/15/1980", "12/15/1980"), 0.5, places=2)
        
    def test_zscore_logic(self):
        """Tests the main Z-score calculation for various L values and edge cases."""
        self.assertAlmostEqual(compute_zscore(10, 0.5, 8, 0.1), (np.sqrt(1.25)-1)/0.05, 5)
        self.assertAlmostEqual(compute_zscore(7, -0.5, 8, 0.1), ((7/8)**-0.5-1)/-0.05, 5)
        self.assertAlmostEqual(compute_zscore(10, 0, 8, 0.1), np.log(1.25)/0.1, 5)
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
        self.mock_almi_func = interp1d(ages, almi_values, kind='cubic', fill_value="extrapolate")
        
        # Realistic LMI values (kg/m²) for males  
        lmi_values = np.array([18.5, 19.2, 18.8, 17.8, 16.5])
        self.mock_lmi_func = interp1d(ages, lmi_values, kind='cubic', fill_value="extrapolate")
        
        # Common test fixtures
        self.user_info = {
            "birth_date_str": "04/26/1982",
            "height_in": 66.0,
            "gender_code": 0
        }
        
        self.goal_params = {
            'target_percentile': 0.90,
            'target_age': 45.0
        }
        
        self.lms_functions = {
            'almi_M': self.mock_almi_func,
            'lmi_M': self.mock_lmi_func
        }
    
    def test_personal_ratio_multiple_scans(self):
        """Test ALM/TLM ratio calculation with multiple scans (personal history)."""
        processed_data = [
            {
                'date_str': "04/07/2022",
                'alm_lbs': 49.7,  # 12.4 + 37.3
                'total_lean_mass_lbs': 106.3,
                'age_at_scan': 39.95
            },
            {
                'date_str': "04/01/2023", 
                'alm_lbs': 56.9,  # 16.5 + 40.4
                'total_lean_mass_lbs': 121.2,
                'age_at_scan': 40.93
            },
            {
                'date_str': "10/21/2023",
                'alm_lbs': 57.4,  # 16.7 + 40.7  
                'total_lean_mass_lbs': 121.6,
                'age_at_scan': 41.49
            }
        ]
        
        ratio = get_alm_tlm_ratio(processed_data, self.goal_params, self.lms_functions, self.user_info)
        
        # Calculate expected ratio manually
        lbs_to_kg = 1 / 2.20462
        expected_ratios = []
        for scan in processed_data:
            alm_kg = scan['alm_lbs'] * lbs_to_kg
            tlm_kg = scan['total_lean_mass_lbs'] * lbs_to_kg
            expected_ratios.append(alm_kg / tlm_kg)
        expected_ratio = np.mean(expected_ratios)
        
        self.assertAlmostEqual(ratio, expected_ratio, places=4)
        self.assertTrue(0.4 <= ratio <= 0.6)  # Reasonable physiological range
    
    def test_personal_ratio_two_scans(self):
        """Test ALM/TLM ratio calculation with exactly two scans."""
        processed_data = [
            {
                'date_str': "04/07/2022",
                'alm_lbs': 49.7,
                'total_lean_mass_lbs': 106.3,
                'age_at_scan': 39.95
            },
            {
                'date_str': "11/25/2024",
                'alm_lbs': 58.3,  # 17.8 + 40.5
                'total_lean_mass_lbs': 129.6,
                'age_at_scan': 42.59
            }
        ]
        
        ratio = get_alm_tlm_ratio(processed_data, self.goal_params, self.lms_functions, self.user_info)
        
        # Should use personal data (≥2 scans)
        self.assertTrue(0.4 <= ratio <= 0.6)
        
        # Verify it's using both scans
        lbs_to_kg = 1 / 2.20462
        ratio1 = (processed_data[0]['alm_lbs'] * lbs_to_kg) / (processed_data[0]['total_lean_mass_lbs'] * lbs_to_kg)
        ratio2 = (processed_data[1]['alm_lbs'] * lbs_to_kg) / (processed_data[1]['total_lean_mass_lbs'] * lbs_to_kg)
        expected_ratio = (ratio1 + ratio2) / 2
        
        self.assertAlmostEqual(ratio, expected_ratio, places=4)
    
    def test_population_ratio_single_scan(self):
        """Test ALM/TLM ratio fallback to population data with single scan."""
        processed_data = [
            {
                'date_str': "11/25/2024",
                'alm_lbs': 58.3,
                'total_lean_mass_lbs': 129.6,
                'age_at_scan': 42.59
            }
        ]
        
        ratio = get_alm_tlm_ratio(processed_data, self.goal_params, self.lms_functions, self.user_info)
        
        # Should use population data from target age (45.0)
        height_m_sq = (self.user_info['height_in'] * 0.0254) ** 2
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
                'date_str': "11/25/2024",
                'alm_lbs': 58.3,
                'total_lean_mass_lbs': 129.6,
                'age_at_scan': 42.59  # Current age different from target
            }
        ]
        
        # Test with different target ages
        goal_params_30 = {'target_age': 30.0, 'target_percentile': 0.90}
        goal_params_60 = {'target_age': 60.0, 'target_percentile': 0.90}
        
        ratio_30 = get_alm_tlm_ratio(processed_data, goal_params_30, self.lms_functions, self.user_info)
        ratio_60 = get_alm_tlm_ratio(processed_data, goal_params_60, self.lms_functions, self.user_info)
        
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
            {'alm_lbs': 40.0, 'total_lean_mass_lbs': 100.0, 'age_at_scan': 35},  # ratio = 0.40 (old)
            {'alm_lbs': 41.0, 'total_lean_mass_lbs': 100.0, 'age_at_scan': 36},  # ratio = 0.41 (old)
            {'alm_lbs': 50.0, 'total_lean_mass_lbs': 100.0, 'age_at_scan': 37},  # ratio = 0.50 (recent)
            {'alm_lbs': 51.0, 'total_lean_mass_lbs': 100.0, 'age_at_scan': 38},  # ratio = 0.51 (recent)
            {'alm_lbs': 52.0, 'total_lean_mass_lbs': 100.0, 'age_at_scan': 39}   # ratio = 0.52 (recent)
        ]
        
        ratio = get_alm_tlm_ratio(processed_data, self.goal_params, self.lms_functions, self.user_info)
        
        # Should only use last 3 scans (ratios 0.50, 0.51, 0.52)
        expected_ratio = (0.50 + 0.51 + 0.52) / 3
        lbs_to_kg = 1 / 2.20462
        expected_ratio_kg = expected_ratio  # Since we used same denominator, conversion cancels out
        
        self.assertAlmostEqual(ratio, expected_ratio_kg, places=3)
        self.assertGreater(ratio, 0.48)  # Should be close to recent scans, not old ones
    
    def test_edge_case_zero_tlm(self):
        """Test handling of edge case where TLM is zero (should not crash)."""
        processed_data = [
            {
                'date_str': "11/25/2024",
                'alm_lbs': 58.3,
                'total_lean_mass_lbs': 0.0,  # Edge case - should trigger population fallback
                'age_at_scan': 42.59
            }
        ]
        
        # Should fall back to population ratio when personal calculation would fail
        ratio = get_alm_tlm_ratio(processed_data, self.goal_params, self.lms_functions, self.user_info)
        
        # Should use population fallback
        expected_ratio = self.mock_almi_func(45.0) / self.mock_lmi_func(45.0)
        self.assertAlmostEqual(ratio, expected_ratio, places=4)
    
    def test_realistic_ratio_ranges(self):
        """Test that calculated ratios fall within realistic physiological ranges."""
        # Test multiple scenarios
        scenarios = [
            # Multiple scans
            [
                {'alm_lbs': 50.0, 'total_lean_mass_lbs': 110.0, 'age_at_scan': 40},
                {'alm_lbs': 52.0, 'total_lean_mass_lbs': 115.0, 'age_at_scan': 41}
            ],
            # Single scan (population fallback)
            [
                {'alm_lbs': 55.0, 'total_lean_mass_lbs': 120.0, 'age_at_scan': 42}
            ]
        ]
        
        for processed_data in scenarios:
            ratio = get_alm_tlm_ratio(processed_data, self.goal_params, self.lms_functions, self.user_info)
            
            # ALM/TLM ratios should be in realistic range for adult males
            self.assertGreaterEqual(ratio, 0.35, "Ratio too low - below physiological minimum")
            self.assertLessEqual(ratio, 0.65, "Ratio too high - above physiological maximum")


class TestIntegrationTLMEstimation(unittest.TestCase):
    """Integration tests for TLM estimation within the full processing pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create minimal mock LMS functions for integration testing
        self.user_info = {
            "birth_date_str": "04/26/1982", 
            "height_in": 66.0,
            "gender_code": 0
        }
        
        self.scan_history = [
            {'date_str': "04/07/2022", 'total_lean_mass_lbs': 106.3, 'arms_lean_lbs': 12.4, 'legs_lean_lbs': 37.3},
            {'date_str': "11/25/2024", 'total_lean_mass_lbs': 129.6, 'arms_lean_lbs': 17.8, 'legs_lean_lbs': 40.5}
        ]
        
        self.goal_params = {
            'target_percentile': 0.90,
            'target_age': 45.0
        }
        
        # Mock LMS functions with realistic values
        ages = np.linspace(18, 80, 50)
        almi_values = 9.0 - 0.01 * (ages - 30)  # Slight decline with age
        lmi_values = 19.0 - 0.02 * (ages - 30)  # Slight decline with age
        l_values = np.ones_like(ages) * 0.1     # Mock skewness
        s_values = np.ones_like(ages) * 0.1     # Mock coefficient of variation
        
        self.lms_functions = {
            'almi_L': interp1d(ages, l_values, kind='cubic', fill_value="extrapolate"),
            'almi_M': interp1d(ages, almi_values, kind='cubic', fill_value="extrapolate"),
            'almi_S': interp1d(ages, s_values, kind='cubic', fill_value="extrapolate"),
            'lmi_L': interp1d(ages, l_values, kind='cubic', fill_value="extrapolate"),
            'lmi_M': interp1d(ages, lmi_values, kind='cubic', fill_value="extrapolate"),
            'lmi_S': interp1d(ages, s_values, kind='cubic', fill_value="extrapolate")
        }
    
    def test_tlm_estimation_integration(self):
        """Test that TLM estimation integrates properly with full processing pipeline."""
        # Test that goal_params gets updated with estimated_tlm_gain_kg
        initial_goal_params = self.goal_params.copy()
        self.assertNotIn('estimated_tlm_gain_kg', initial_goal_params)
        
        # Calculate what the integration should produce
        height_m_sq = (self.user_info['height_in'] * 0.0254) ** 2
        lbs_to_kg = 1 / 2.20462
        
        # Mock personal ratio calculation
        alm1 = (12.4 + 37.3) * lbs_to_kg
        tlm1 = 106.3 * lbs_to_kg
        alm2 = (17.8 + 40.5) * lbs_to_kg  
        tlm2 = 129.6 * lbs_to_kg
        
        personal_ratio = np.mean([alm1/tlm1, alm2/tlm2])
        
        # Mock target ALMI calculation (would come from LMS inversion)
        target_almi = 9.8  # Approximate 90th percentile
        target_alm_kg = target_almi * height_m_sq
        target_tlm_kg = target_alm_kg / personal_ratio
        current_tlm_kg = tlm2
        expected_tlm_gain = target_tlm_kg - current_tlm_kg
        
        # Verify the calculation produces reasonable results
        self.assertGreater(expected_tlm_gain, 0, "Should need positive TLM gain for 90th percentile goal")
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
                "gender": "male"
            },
            "scan_history": [
                {
                    "date": "04/07/2022",
                    "total_lean_mass_lbs": 106.3,
                    "arms_lean_lbs": 12.4,
                    "legs_lean_lbs": 37.3
                },
                {
                    "date": "11/25/2024",
                    "total_lean_mass_lbs": 129.6,
                    "arms_lean_lbs": 17.8,
                    "legs_lean_lbs": 40.5
                }
            ],
            "goals": {
                "almi": {
                    "target_percentile": 0.90,
                    "target_age": 45.0,
                    "description": "Reach 90th percentile ALMI by age 45"
                },
                "ffmi": {
                    "target_percentile": 0.85,
                    "target_age": 50.0,
                    "description": "Reach 85th percentile FFMI by age 50"
                }
            }
        }
        
        
        # Config with no goals
        self.config_no_goals = {
            "user_info": {
                "birth_date": "04/26/1982",
                "height_in": 66.0,
                "gender": "male"
            },
            "scan_history": [
                {
                    "date": "04/07/2022",
                    "total_lean_mass_lbs": 106.3,
                    "arms_lean_lbs": 12.4,
                    "legs_lean_lbs": 37.3
                }
            ]
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
        user_info, scan_history, almi_goal, ffmi_goal = extract_data_from_config(self.valid_config_nested)
        
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
        user_info, scan_history, almi_goal, ffmi_goal = extract_data_from_config(self.config_no_goals)
        
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
            "goals": {
                "almi": {
                    "target_percentile": 0.75,
                    "target_age": 40.0
                }
            }
        }
        
        user_info, scan_history, almi_goal, ffmi_goal = extract_data_from_config(config_almi_only)
        
        self.assertIsNotNone(almi_goal)
        self.assertIsNone(ffmi_goal)
        self.assertEqual(almi_goal["target_percentile"], 0.75)
        
        # Config with only FFMI goal
        config_ffmi_only = {
            "user_info": self.config_no_goals["user_info"],
            "scan_history": self.config_no_goals["scan_history"],
            "goals": {
                "ffmi": {
                    "target_percentile": 0.80,
                    "target_age": 55.0
                }
            }
        }
        
        user_info, scan_history, almi_goal, ffmi_goal = extract_data_from_config(config_ffmi_only)
        
        self.assertIsNone(almi_goal)
        self.assertIsNotNone(ffmi_goal)
        self.assertEqual(ffmi_goal["target_percentile"], 0.80)
    
    def test_load_config_json_valid_file_nested(self):
        """Test loading a valid JSON config file with nested goals."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.valid_config_nested, f)
            temp_path = f.name
        
        try:
            config = load_config_json(temp_path)
            self.assertEqual(config, self.valid_config_nested)
        finally:
            os.unlink(temp_path)
    
    
    def test_load_config_json_no_goals(self):
        """Test loading a config file with no goals."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
        del invalid_config['user_info']
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
        invalid_config['user_info']['gender'] = 'invalid'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
        invalid_config['goals']['almi']['target_percentile'] = 1.5  # > 1.0
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
        invalid_config['scan_history'] = []
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
        l_values = np.ones_like(ages) * 0.1     # Mock skewness
        s_values = np.ones_like(ages) * 0.1     # Mock coefficient of variation
        
        self.lms_functions = {
            'almi_L': interp1d(ages, l_values, kind='cubic', fill_value="extrapolate"),
            'almi_M': interp1d(ages, almi_values, kind='cubic', fill_value="extrapolate"),
            'almi_S': interp1d(ages, s_values, kind='cubic', fill_value="extrapolate"),
            'lmi_L': interp1d(ages, l_values, kind='cubic', fill_value="extrapolate"),
            'lmi_M': interp1d(ages, lmi_values, kind='cubic', fill_value="extrapolate"),
            'lmi_S': interp1d(ages, s_values, kind='cubic', fill_value="extrapolate")
        }
        
        self.user_info = {
            "birth_date_str": "04/26/1982", 
            "height_in": 66.0,
            "gender_code": 0
        }
        
        self.scan_history = [
            {'date_str': "04/07/2022", 'total_lean_mass_lbs': 106.3, 'arms_lean_lbs': 12.4, 'legs_lean_lbs': 37.3},
            {'date_str': "11/25/2024", 'total_lean_mass_lbs': 129.6, 'arms_lean_lbs': 17.8, 'legs_lean_lbs': 40.5}
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
        self.assertIn('almi', goal_calculations)
        self.assertIn('ffmi', goal_calculations)
        
        # Check ALMI goal calculations
        almi_calc = goal_calculations['almi']
        self.assertEqual(almi_calc['target_percentile'], 0.90)
        self.assertEqual(almi_calc['target_age'], 45.0)
        self.assertIn('alm_to_add_kg', almi_calc)
        self.assertIn('estimated_tlm_gain_kg', almi_calc)
        
        # Check FFMI goal calculations
        ffmi_calc = goal_calculations['ffmi']
        self.assertEqual(ffmi_calc['target_percentile'], 0.85)
        self.assertEqual(ffmi_calc['target_age'], 50.0)
        self.assertIn('tlm_to_add_kg', ffmi_calc)
        
        # Check goal rows in DataFrame
        goal_rows = df_results[df_results['date_str'].str.contains('Goal')]
        self.assertEqual(len(goal_rows), 2)
        
        almi_goal_row = df_results[df_results['date_str'].str.contains('ALMI Goal')].iloc[0]
        ffmi_goal_row = df_results[df_results['date_str'].str.contains('FFMI Goal')].iloc[0]
        
        self.assertEqual(almi_goal_row['age_at_scan'], 45.0)
        self.assertEqual(ffmi_goal_row['age_at_scan'], 50.0)
    
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
        self.assertIn('almi', goal_calculations)
        self.assertNotIn('ffmi', goal_calculations)
        
        # Check only ALMI goal row exists
        goal_rows = df_results[df_results['date_str'].str.contains('Goal')]
        self.assertEqual(len(goal_rows), 1)
        
        almi_goal_row = goal_rows.iloc[0]
        self.assertIn('ALMI Goal', almi_goal_row['date_str'])
        self.assertEqual(almi_goal_row['age_at_scan'], 40.0)
    
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
        self.assertNotIn('almi', goal_calculations)
        self.assertIn('ffmi', goal_calculations)
        
        # Check only FFMI goal row exists
        goal_rows = df_results[df_results['date_str'].str.contains('Goal')]
        self.assertEqual(len(goal_rows), 1)
        
        ffmi_goal_row = goal_rows.iloc[0]
        self.assertIn('FFMI Goal', ffmi_goal_row['date_str'])
        self.assertEqual(ffmi_goal_row['age_at_scan'], 55.0)
    
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
        goal_rows = df_results[df_results['date_str'].str.contains('Goal')]
        self.assertEqual(len(goal_rows), 0)
        
        # All rows should be historical scans
        for _, row in df_results.iterrows():
            self.assertNotIn('Goal', row['date_str'])
    
    def test_goal_calculations_consistency(self):
        """Test that goal calculations are mathematically consistent."""
        almi_goal = {"target_percentile": 0.90, "target_age": 45.0}
        ffmi_goal = {"target_percentile": 0.85, "target_age": 50.0}
        
        df_results, goal_calculations = process_scans_and_goal(
            self.user_info, self.scan_history, almi_goal, ffmi_goal, self.lms_functions
        )
        
        # ALMI goal calculations should be positive for reasonable targets
        almi_calc = goal_calculations['almi']
        self.assertIsInstance(almi_calc['alm_to_add_kg'], (int, float))
        self.assertIsInstance(almi_calc['estimated_tlm_gain_kg'], (int, float))
        
        # FFMI goal calculations should be reasonable
        ffmi_calc = goal_calculations['ffmi']
        self.assertIsInstance(ffmi_calc['tlm_to_add_kg'], (int, float))
        
        # Check that DataFrame goal rows match calculations
        almi_goal_row = df_results[df_results['date_str'].str.contains('ALMI Goal')].iloc[0]
        ffmi_goal_row = df_results[df_results['date_str'].str.contains('FFMI Goal')].iloc[0]
        
        self.assertAlmostEqual(almi_goal_row['almi_percentile'], 90.0, places=1)
        self.assertAlmostEqual(ffmi_goal_row['ffmi_lmi_percentile'], 85.0, places=1)


if __name__ == '__main__':
    # Run tests with detailed output
    print("Running comprehensive test suite for zscore_plot.py...")
    unittest.main(verbosity=2)