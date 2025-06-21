#!/usr/bin/env python3
"""
V1 URL State Backward Compatibility Integration Tests

These tests verify that the V1 URL state format remains supported and functional
as the application evolves. This is critical for maintaining backward compatibility
with URLs shared by users after the public release.

Key difference from other URL tests: These tests manually construct V1 format data
according to the specification, rather than deriving it from the application state.

Run with: python -m pytest test_v1_url_state_compatibility.py -v
"""

import base64
import json

# Import schema validation utilities
import sys
import urllib.parse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import functions from webapp
from webapp import (
    decode_state_from_url,
    expand_compact_config,
    initialize_session_state,
)

sys.path.append(str(Path(__file__).parent.parent.parent / "docs"))
from v1_schema_validator import load_v1_schema, validate_v1_config, validate_v1_url


class TestV1URLStateSpecification:
    """Test manually constructed V1 URL state data according to specification."""

    def create_v1_minimal_config(self):
        """Create minimal valid V1 configuration according to spec."""
        return {
            "u": {"bd": "04/26/1982", "h": 66.0, "g": "m"},
            "s": [["04/07/2022", 143.2, 106.3, 32.6, 22.8, 12.4, 37.3]],
        }

    def create_v1_complete_config(self):
        """Create complete V1 configuration with all optional fields."""
        return {
            "u": {
                "bd": "04/26/1982",
                "h": 66.0,
                "g": "m",
                "tl": "intermediate",
                "hd": "5'6\"",
            },
            "s": [
                ["04/07/2022", 143.2, 106.3, 32.6, 22.8, 12.4, 37.3],
                ["04/01/2023", 154.3, 121.2, 28.5, 18.5, 16.5, 40.4],
                ["10/15/2023", 158.7, 125.1, 27.8, 17.5, 17.2, 42.1],
            ],
            "ag": {"tp": 0.90, "ta": 30},
            "fg": {"tp": 0.75, "ta": 35},
        }

    def create_v1_edge_case_config(self):
        """Create V1 configuration with edge cases."""
        return {
            "u": {"bd": "12/31/1990", "h": 72.5, "g": "f", "tl": "advanced"},
            "s": [["01/01/2024", 135.8, 95.2, 35.6, 26.2, 10.8, 32.4]],
            "ag": {
                "tp": 0.99  # No target age
            },
            "fg": {
                "tp": 0.01,  # Minimum percentile
                "ta": 25,
            },
        }

    def create_v1_maximum_scans_config(self):
        """Create V1 configuration with maximum 20 scans."""
        config = {
            "u": {"bd": "06/15/1985", "h": 68.0, "g": "m", "tl": "novice"},
            "s": [],
            "ag": {"tp": 0.85},
            "fg": {"tp": 0.80},
        }

        # Add exactly 20 scans
        for i in range(20):
            month = (i % 12) + 1
            day = (i % 28) + 1
            year = 2020 + (i // 12)
            config["s"].append(
                [
                    f"{month:02d}/{day:02d}/{year}",
                    150.0 + i * 0.5,
                    120.0 + i * 0.3,
                    25.0 + i * 0.1,
                    16.5 + i * 0.05,
                    15.0 + i * 0.1,
                    38.0 + i * 0.2,
                ]
            )

        return config

    def encode_v1_config_to_url(self, v1_config):
        """Encode V1 config to URL format according to specification."""
        # Step 1: Minify JSON
        json_str = json.dumps(v1_config, separators=(",", ":"))

        # Step 2: Base64 encode
        encoded_data = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")

        # Step 3: Create URL (URL encoding handled by browser/framework)
        base_url = "http://localhost:8501"
        share_url = f"{base_url}?data={urllib.parse.quote(encoded_data)}"

        return share_url, encoded_data

    def test_v1_minimal_config_round_trip(self):
        """Test minimal V1 config encodes and decodes correctly."""
        v1_config = self.create_v1_minimal_config()

        # Validate against schema first
        is_valid, error = validate_v1_config(v1_config, raise_on_error=False)
        assert is_valid, f"V1 config should be valid according to schema: {error}"

        # Encode to URL
        share_url, encoded_data = self.encode_v1_config_to_url(v1_config)

        # Verify encoding
        assert "data=" in share_url
        assert len(encoded_data) > 0

        # Validate complete URL against schema
        url_valid, url_error, decoded_config = validate_v1_url(
            share_url, raise_on_error=False
        )
        assert url_valid, f"V1 URL should be valid according to schema: {url_error}"

        # Decode back manually
        decoded_bytes = base64.b64decode(encoded_data.encode("utf-8"))
        decoded_json = decoded_bytes.decode("utf-8")
        recovered_config = json.loads(decoded_json)

        # Verify perfect round-trip
        assert recovered_config == v1_config
        assert decoded_config == v1_config  # From schema validator

        # Expand to application format
        user_info, scan_history, almi_goal, ffmi_goal, height_display = (
            expand_compact_config(v1_config)
        )

        # Verify expansion
        assert user_info["birth_date"] == "04/26/1982"
        assert user_info["height_in"] == 66.0
        assert user_info["gender"] == "male"
        assert user_info["training_level"] == ""  # Not specified

        assert len(scan_history) == 1
        assert scan_history[0]["date"] == "04/07/2022"
        assert scan_history[0]["total_weight_lbs"] == 143.2
        assert scan_history[0]["body_fat_percentage"] == 22.8

        # Default goals (no goals specified in minimal config)
        assert almi_goal["target_percentile"] == 0.75
        assert almi_goal["target_age"] == "?"

    def test_v1_complete_config_round_trip(self):
        """Test complete V1 config with all optional fields."""
        v1_config = self.create_v1_complete_config()

        # Validate against schema
        is_valid, error = validate_v1_config(v1_config, raise_on_error=False)
        assert is_valid, f"Complete V1 config should be valid: {error}"

        # Encode and decode
        share_url, encoded_data = self.encode_v1_config_to_url(v1_config)

        # Validate URL too
        url_valid, url_error, decoded_from_url = validate_v1_url(
            share_url, raise_on_error=False
        )
        assert url_valid, f"Complete V1 URL should be valid: {url_error}"

        recovered_config = json.loads(base64.b64decode(encoded_data).decode("utf-8"))

        assert recovered_config == v1_config

        # Expand and verify all fields
        user_info, scan_history, almi_goal, ffmi_goal, height_display = (
            expand_compact_config(v1_config)
        )

        # User info with all optional fields
        assert user_info["training_level"] == "intermediate"
        assert height_display == "5'6\""

        # Multiple scans
        assert len(scan_history) == 3
        assert scan_history[2]["date"] == "10/15/2023"
        assert scan_history[2]["total_weight_lbs"] == 158.7

        # Goals with target ages
        assert almi_goal["target_percentile"] == 0.90
        assert almi_goal["target_age"] == 30
        assert ffmi_goal["target_percentile"] == 0.75
        assert ffmi_goal["target_age"] == 35

    def test_v1_edge_case_config(self):
        """Test V1 config with edge cases and boundary values."""
        v1_config = self.create_v1_edge_case_config()

        # Encode and decode
        share_url, encoded_data = self.encode_v1_config_to_url(v1_config)
        recovered_config = json.loads(base64.b64decode(encoded_data).decode("utf-8"))

        assert recovered_config == v1_config

        # Expand and verify edge cases
        user_info, scan_history, almi_goal, ffmi_goal, height_display = (
            expand_compact_config(v1_config)
        )

        # Female, advanced training level
        assert user_info["gender"] == "female"
        assert user_info["training_level"] == "advanced"
        assert user_info["height_in"] == 72.5  # Decimal height

        # High body fat percentage
        assert scan_history[0]["body_fat_percentage"] == 26.2

        # Extreme percentiles
        assert almi_goal["target_percentile"] == 0.99
        assert almi_goal["target_age"] == "?"  # No target age specified
        assert ffmi_goal["target_percentile"] == 0.01

    def test_v1_maximum_scans_config(self):
        """Test V1 config with maximum 20 scans."""
        v1_config = self.create_v1_maximum_scans_config()

        # Verify we have exactly 20 scans
        assert len(v1_config["s"]) == 20

        # Encode and decode
        share_url, encoded_data = self.encode_v1_config_to_url(v1_config)
        recovered_config = json.loads(base64.b64decode(encoded_data).decode("utf-8"))

        assert recovered_config == v1_config

        # Expand and verify
        user_info, scan_history, almi_goal, ffmi_goal, height_display = (
            expand_compact_config(v1_config)
        )

        assert len(scan_history) == 20
        assert scan_history[0]["date"] == "01/01/2020"
        assert scan_history[19]["date"] == "08/20/2021"
        assert scan_history[19]["total_weight_lbs"] == 159.5

    def test_v1_url_length_constraints(self):
        """Test that V1 URLs stay within reasonable length limits."""
        # Test with maximum scans
        v1_config = self.create_v1_maximum_scans_config()
        share_url, encoded_data = self.encode_v1_config_to_url(v1_config)

        # Verify URL length is reasonable (most browsers support 2000+ chars)
        assert len(share_url) < 3000, f"URL too long: {len(share_url)} characters"

        # Test compression ratio
        original_json = json.dumps(
            {
                "user_info": {
                    "birth_date": "06/15/1985",
                    "height_in": 68.0,
                    "gender": "male",
                    "training_level": "novice",
                },
                "scan_history": [
                    {
                        "date": f"{(i % 12) + 1:02d}/{(i % 28) + 1:02d}/{2020 + (i // 12)}",
                        "total_weight_lbs": 150.0 + i * 0.5,
                        "total_lean_mass_lbs": 120.0 + i * 0.3,
                        "fat_mass_lbs": 25.0 + i * 0.1,
                        "body_fat_percentage": 16.5 + i * 0.05,
                        "arms_lean_lbs": 15.0 + i * 0.1,
                        "legs_lean_lbs": 38.0 + i * 0.2,
                    }
                    for i in range(20)
                ],
                "goals": {
                    "almi": {"target_percentile": 0.85},
                    "ffmi": {"target_percentile": 0.80},
                },
            },
            separators=(",", ":"),
        )

        compact_json = json.dumps(v1_config, separators=(",", ":"))
        compression_ratio = len(compact_json) / len(original_json)

        # Should achieve meaningful compression
        assert compression_ratio < 0.7, (
            f"Compression ratio too low: {compression_ratio:.2f}"
        )


class TestV1URLWebAppIntegration:
    """Test V1 URL integration with the web application."""

    class MockSessionState:
        """Mock Streamlit session state for testing."""

        def __init__(self):
            self._data = {}

        def __setattr__(self, name, value):
            if name.startswith("_"):
                super().__setattr__(name, value)
            else:
                self._data[name] = value

        def __getattr__(self, name):
            if name in self._data:
                return self._data[name]
            raise AttributeError(f'st.session_state has no attribute "{name}"')

        def __contains__(self, name):
            return name in self._data

        def get(self, name, default=None):
            return self._data.get(name, default)

    def create_v1_test_url_data(self):
        """Create test V1 URL data for webapp integration."""
        v1_config = {
            "u": {"bd": "08/15/1990", "h": 70.0, "g": "f", "tl": "intermediate"},
            "s": [
                ["05/10/2023", 140.5, 110.2, 25.3, 18.0, 13.8, 35.6],
                ["11/20/2023", 145.2, 115.8, 24.4, 16.8, 14.5, 37.2],
            ],
            "ag": {"tp": 0.85, "ta": 32},
            "fg": {"tp": 0.70},
        }

        # Encode to base64 (same as URL encoding process)
        json_str = json.dumps(v1_config, separators=(",", ":"))
        encoded_data = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")

        return encoded_data, v1_config

    @patch("streamlit.query_params")
    def test_v1_url_loading_in_webapp(self, mock_query_params):
        """Test that V1 URLs load correctly in the webapp."""
        encoded_data, original_v1_config = self.create_v1_test_url_data()

        # Mock URL parameters
        mock_query_params.__contains__ = MagicMock(return_value=True)
        mock_query_params.__getitem__ = MagicMock(return_value=encoded_data)

        mock_session_state = self.MockSessionState()

        with (
            patch("streamlit.session_state", mock_session_state),
            patch("streamlit.success") as mock_success,
        ):
            # Test URL decoding
            result = decode_state_from_url()

            # Should successfully load
            assert result

            # Verify success message was shown
            mock_success.assert_called_once_with("Configuration loaded from URL!")

            # Verify session state was populated correctly
            assert mock_session_state.user_info["birth_date"] == "08/15/1990"
            assert mock_session_state.user_info["height_in"] == 70.0
            assert mock_session_state.user_info["gender"] == "female"
            assert mock_session_state.user_info["training_level"] == "intermediate"

            assert len(mock_session_state.scan_history) == 2
            assert mock_session_state.scan_history[0]["date"] == "05/10/2023"
            assert mock_session_state.scan_history[1]["body_fat_percentage"] == 16.8

            assert mock_session_state.almi_goal["target_percentile"] == 0.85
            assert mock_session_state.almi_goal["target_age"] == 32
            assert mock_session_state.ffmi_goal["target_percentile"] == 0.70
            assert mock_session_state.ffmi_goal["target_age"] == "?"  # Not specified

    @patch("streamlit.query_params")
    def test_v1_url_session_state_initialization(self, mock_query_params):
        """Test complete session state initialization with V1 URL."""
        encoded_data, original_v1_config = self.create_v1_test_url_data()

        mock_query_params.__contains__ = MagicMock(return_value=True)
        mock_query_params.__getitem__ = MagicMock(return_value=encoded_data)

        mock_session_state = self.MockSessionState()

        with (
            patch("streamlit.session_state", mock_session_state),
            patch("streamlit.success"),
        ):
            # Initialize session state (this should load from URL and set all attributes)
            initialize_session_state()

            # Verify all required session state attributes exist
            required_attributes = [
                "user_info",
                "scan_history",
                "almi_goal",
                "ffmi_goal",
                "analysis_results",
                "url_loaded",
            ]

            for attr in required_attributes:
                assert hasattr(mock_session_state, attr), (
                    f"Missing session state attribute: {attr}"
                )

            # Verify loaded data matches V1 config
            assert mock_session_state.user_info["birth_date"] == "08/15/1990"
            assert (
                mock_session_state.analysis_results is None
            )  # Should be initialized to None
            assert mock_session_state.url_loaded

    def test_v1_config_validation_rules(self):
        """Test V1 config against specification validation rules."""
        # Test invalid birth date format
        invalid_config = {
            "u": {"bd": "1990-08-15", "h": 70.0, "g": "f"},  # Wrong date format
            "s": [["05/10/2023", 140.5, 110.2, 25.3, 18.0, 13.8, 35.6]],
        }

        # Should still parse (validation happens at application level)
        user_info, scan_history, almi_goal, ffmi_goal, height_display = (
            expand_compact_config(invalid_config)
        )
        assert user_info["birth_date"] == "1990-08-15"  # Preserved as-is

        # Test boundary values
        boundary_config = {
            "u": {"bd": "01/01/2000", "h": 12.0, "g": "m"},  # Minimum height
            "s": [["12/31/2023", 100.0, 80.0, 15.0, 15.0, 8.0, 25.0]],
            "ag": {"tp": 0.01},  # Minimum percentile
            "fg": {"tp": 0.99},  # Maximum percentile
        }

        user_info, scan_history, almi_goal, ffmi_goal, height_display = (
            expand_compact_config(boundary_config)
        )
        assert user_info["height_in"] == 12.0
        assert almi_goal["target_percentile"] == 0.01
        assert ffmi_goal["target_percentile"] == 0.99


class TestV1JSONSchemaValidation:
    """Test V1 configs against the JSON Schema specification."""

    def test_schema_loads_correctly(self):
        """Test that the V1 JSON Schema loads and is valid."""
        schema = load_v1_schema()

        # Verify basic schema structure
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["title"] == "V1 URL State Schema"
        assert schema["version"] == "1.0.0"
        assert "properties" in schema

        # Verify required top-level properties
        assert "u" in schema["properties"]  # User info
        assert schema["required"] == ["u"]  # Only user info is required

        # Optional top-level properties
        optional_props = ["s", "ag", "fg"]  # scans, ALMI goal, FFMI goal
        for prop in optional_props:
            assert prop in schema["properties"]

    def test_all_example_configs_validate(self):
        """Test that all our example V1 configs validate against the schema."""
        test_instance = TestV1URLStateSpecification()

        # Test all config variations
        configs = [
            ("minimal", test_instance.create_v1_minimal_config()),
            ("complete", test_instance.create_v1_complete_config()),
            ("edge_case", test_instance.create_v1_edge_case_config()),
            ("maximum_scans", test_instance.create_v1_maximum_scans_config()),
        ]

        for config_name, config in configs:
            is_valid, error = validate_v1_config(config, raise_on_error=False)
            assert is_valid, (
                f"{config_name} config should validate against schema: {error}"
            )

    def test_schema_catches_invalid_configs(self):
        """Test that the schema properly rejects invalid V1 configs."""
        # Missing required user info
        invalid_config = {"s": [["04/07/2022", 143.2, 106.3, 32.6, 22.8, 12.4, 37.3]]}
        is_valid, error = validate_v1_config(invalid_config, raise_on_error=False)
        assert not is_valid, "Config missing user info should be invalid"
        assert "'u' is a required property" in error

        # Invalid gender code
        invalid_gender = {
            "u": {"bd": "04/26/1982", "h": 66.0, "g": "x"},  # Invalid gender
            "s": [["04/07/2022", 143.2, 106.3, 32.6, 22.8, 12.4, 37.3]],
        }
        is_valid, error = validate_v1_config(invalid_gender, raise_on_error=False)
        assert not is_valid, "Invalid gender code should be rejected"

        # Invalid date format
        invalid_date = {
            "u": {"bd": "1982-04-26", "h": 66.0, "g": "m"},  # Wrong date format
            "s": [["04/07/2022", 143.2, 106.3, 32.6, 22.8, 12.4, 37.3]],
        }
        is_valid, error = validate_v1_config(invalid_date, raise_on_error=False)
        assert not is_valid, "Invalid date format should be rejected"

        # Scan with wrong number of elements
        invalid_scan = {
            "u": {"bd": "04/26/1982", "h": 66.0, "g": "m"},
            "s": [["04/07/2022", 143.2, 106.3, 32.6, 22.8]],  # Missing elements
        }
        is_valid, error = validate_v1_config(invalid_scan, raise_on_error=False)
        assert not is_valid, "Scan with wrong number of elements should be rejected"

        # Percentile out of range
        invalid_percentile = {
            "u": {"bd": "04/26/1982", "h": 66.0, "g": "m"},
            "s": [["04/07/2022", 143.2, 106.3, 32.6, 22.8, 12.4, 37.3]],
            "ag": {"tp": 1.5},  # > 0.99
        }
        is_valid, error = validate_v1_config(invalid_percentile, raise_on_error=False)
        assert not is_valid, "Percentile out of range should be rejected"

    def test_schema_boundary_values(self):
        """Test schema validation with boundary values."""
        # Minimum valid percentile
        min_percentile = {
            "u": {"bd": "04/26/1982", "h": 66.0, "g": "m"},
            "s": [["04/07/2022", 143.2, 106.3, 32.6, 22.8, 12.4, 37.3]],
            "ag": {"tp": 0.01},  # Minimum allowed
        }
        is_valid, error = validate_v1_config(min_percentile, raise_on_error=False)
        assert is_valid, f"Minimum percentile should be valid: {error}"

        # Maximum valid percentile
        max_percentile = {
            "u": {"bd": "04/26/1982", "h": 66.0, "g": "m"},
            "s": [["04/07/2022", 143.2, 106.3, 32.6, 22.8, 12.4, 37.3]],
            "ag": {"tp": 0.99},  # Maximum allowed
        }
        is_valid, error = validate_v1_config(max_percentile, raise_on_error=False)
        assert is_valid, f"Maximum percentile should be valid: {error}"

        # Exactly 20 scans (maximum)
        twenty_scans = {
            "u": {"bd": "04/26/1982", "h": 66.0, "g": "m"},
            "s": [["04/07/2022", 143.2, 106.3, 32.6, 22.8, 12.4, 37.3]] * 20,
        }
        is_valid, error = validate_v1_config(twenty_scans, raise_on_error=False)
        assert is_valid, f"20 scans should be valid: {error}"

        # 21 scans (over limit)
        twenty_one_scans = {
            "u": {"bd": "04/26/1982", "h": 66.0, "g": "m"},
            "s": [["04/07/2022", 143.2, 106.3, 32.6, 22.8, 12.4, 37.3]] * 21,
        }
        is_valid, error = validate_v1_config(twenty_one_scans, raise_on_error=False)
        assert not is_valid, "21 scans should exceed the limit"


class TestV1BackwardCompatibilityGuarantee:
    """Test that V1 format maintains backward compatibility guarantees."""

    def test_v1_format_stability(self):
        """Test that V1 format structure is stable and well-defined."""
        # Create reference V1 config
        reference_config = {
            "u": {"bd": "04/26/1982", "h": 66.0, "g": "m", "tl": "intermediate"},
            "s": [["04/07/2022", 143.2, 106.3, 32.6, 22.8, 12.4, 37.3]],
            "ag": {"tp": 0.90, "ta": 30},
            "fg": {"tp": 0.75},
        }

        # Test that all required V1 fields are recognized
        user_info, scan_history, almi_goal, ffmi_goal, height_display = (
            expand_compact_config(reference_config)
        )

        # User info mapping
        assert user_info["birth_date"] == reference_config["u"]["bd"]
        assert user_info["height_in"] == reference_config["u"]["h"]
        assert user_info["gender"] == (
            "male" if reference_config["u"]["g"] == "m" else "female"
        )
        assert user_info["training_level"] == reference_config["u"]["tl"]

        # Scan history mapping
        scan = reference_config["s"][0]
        assert scan_history[0]["date"] == scan[0]
        assert scan_history[0]["total_weight_lbs"] == scan[1]
        assert scan_history[0]["total_lean_mass_lbs"] == scan[2]
        assert scan_history[0]["fat_mass_lbs"] == scan[3]
        assert scan_history[0]["body_fat_percentage"] == scan[4]
        assert scan_history[0]["arms_lean_lbs"] == scan[5]
        assert scan_history[0]["legs_lean_lbs"] == scan[6]

        # Goals mapping
        assert almi_goal["target_percentile"] == reference_config["ag"]["tp"]
        assert almi_goal["target_age"] == reference_config["ag"]["ta"]
        assert ffmi_goal["target_percentile"] == reference_config["fg"]["tp"]
        assert ffmi_goal["target_age"] == "?"  # Not specified, should default

    def test_v1_missing_optional_fields_handling(self):
        """Test graceful handling of missing optional V1 fields."""
        # Config with minimal required fields only
        minimal_config = {
            "u": {"bd": "04/26/1982", "h": 66.0, "g": "m"},
            "s": [["04/07/2022", 143.2, 106.3, 32.6, 22.8, 12.4, 37.3]],
            # No goals, no training level, no height display
        }

        user_info, scan_history, almi_goal, ffmi_goal, height_display = (
            expand_compact_config(minimal_config)
        )

        # Should provide sensible defaults
        assert user_info["training_level"] == ""
        assert height_display == ""
        assert almi_goal["target_percentile"] == 0.75  # Default
        assert almi_goal["target_age"] == "?"
        assert ffmi_goal["target_percentile"] == 0.75  # Default
        assert ffmi_goal["target_age"] == "?"

    def test_v1_future_compatibility_resilience(self):
        """Test that V1 configs will work even with future application changes."""
        # Simulate a V1 config that might be loaded in a future version
        v1_config = {
            "u": {"bd": "04/26/1982", "h": 66.0, "g": "m"},
            "s": [["04/07/2022", 143.2, 106.3, 32.6, 22.8, 12.4, 37.3]],
            "ag": {"tp": 0.85},
        }

        # Should expand without errors
        user_info, scan_history, almi_goal, ffmi_goal, height_display = (
            expand_compact_config(v1_config)
        )

        # All required fields should be present with appropriate defaults
        required_user_fields = ["birth_date", "height_in", "gender", "training_level"]
        for field in required_user_fields:
            assert field in user_info

        required_scan_fields = [
            "date",
            "total_weight_lbs",
            "total_lean_mass_lbs",
            "fat_mass_lbs",
            "body_fat_percentage",
            "arms_lean_lbs",
            "legs_lean_lbs",
        ]
        for field in required_scan_fields:
            assert field in scan_history[0]

        required_goal_fields = ["target_percentile", "target_age"]
        for field in required_goal_fields:
            assert field in almi_goal
            assert field in ffmi_goal


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
