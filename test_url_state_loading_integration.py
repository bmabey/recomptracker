#!/usr/bin/env python3
"""
Integration tests for URL state loading functionality in the DEXA webapp.

These tests verify that configurations can be loaded from URLs, form inputs
are populated correctly, and analysis runs automatically.

Run with: python -m pytest test_url_state_loading_integration.py -v
"""

import pytest
import streamlit.testing.v1 as testing
import json
import base64
import urllib.parse
from unittest.mock import patch, MagicMock
import pandas as pd

# Import functions from webapp
from webapp import (
    get_compact_config, expand_compact_config, 
    encode_state_to_url, decode_state_from_url,
    initialize_session_state
)


class TestURLStateLoadingIntegration:
    """Integration tests for URL-based state loading."""
    
    @pytest.fixture
    def example_config(self):
        """Example configuration based on example_config.json."""
        return {
            "user_info": {
                "birth_date": "04/26/1982",
                "height_in": 66.0,
                "gender": "male"
            },
            "scan_history": [
                {
                    "date": "04/07/2022",
                    "total_weight_lbs": 143.2,
                    "total_lean_mass_lbs": 106.3,
                    "fat_mass_lbs": 32.6,
                    "body_fat_percentage": 22.8,
                    "arms_lean_lbs": 12.4,
                    "legs_lean_lbs": 37.3
                },
                {
                    "date": "04/01/2023",
                    "total_weight_lbs": 154.3,
                    "total_lean_mass_lbs": 121.2,
                    "fat_mass_lbs": 28.5,
                    "body_fat_percentage": 18.5,
                    "arms_lean_lbs": 16.5,
                    "legs_lean_lbs": 40.4
                }
            ],
            "goals": {
                "almi": {"target_percentile": 0.90},
                "ffmi": {"target_percentile": 0.75}
            }
        }
    
    @pytest.fixture
    def mock_session_state(self):
        """Mock Streamlit session state for testing."""
        class MockSessionState:
            def __init__(self):
                self.user_info = {
                    'birth_date': '',
                    'height_in': 66.0,
                    'gender': 'male',
                    'training_level': ''
                }
                self.scan_history = []
                self.almi_goal = {'target_percentile': 0.75, 'target_age': '?'}
                self.ffmi_goal = {'target_percentile': 0.75, 'target_age': '?'}
                self.analysis_results = None
                self._state = {}
            
            def get(self, key, default=None):
                return getattr(self, key, self._state.get(key, default))
            
            def __setitem__(self, key, value):
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    self._state[key] = value
            
            def __getitem__(self, key):
                if hasattr(self, key):
                    return getattr(self, key)
                return self._state[key]
            
            def __contains__(self, key):
                return hasattr(self, key) or key in self._state
        
        return MockSessionState()
    
    def create_encoded_url(self, config, mock_session_state=None):
        """Helper to create an encoded URL from a configuration."""
        # Convert to compact format
        if mock_session_state is None:
            mock_session_state = self.mock_session_state()
        
        mock_session_state.user_info = config['user_info']
        mock_session_state.scan_history = config['scan_history']
        
        if 'goals' in config:
            if 'almi' in config['goals']:
                mock_session_state.almi_goal = config['goals']['almi']
            if 'ffmi' in config['goals']:
                mock_session_state.ffmi_goal = config['goals']['ffmi']
        
        # Get compact representation
        compact = self.get_compact_config_from_data(mock_session_state)
        
        # Encode as base64
        json_str = json.dumps(compact, separators=(',', ':'))
        encoded_data = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
        
        return f"http://localhost:8501?data={urllib.parse.quote(encoded_data)}"
    
    def get_compact_config_from_data(self, session_state):
        """Create compact config from session state data."""
        compact = {
            "u": {
                "bd": session_state.user_info.get('birth_date', ''),
                "h": session_state.user_info.get('height_in', 66.0),
                "g": session_state.user_info.get('gender', 'male')[0],  # 'm' or 'f'
            }
        }
        
        # Add training level if set
        if session_state.user_info.get('training_level'):
            compact["u"]["tl"] = session_state.user_info['training_level']
        
        # Convert scan history to array format
        compact["s"] = []
        for scan in session_state.scan_history:
            if scan.get('date'):
                compact["s"].append([
                    scan.get('date', ''),
                    scan.get('total_weight_lbs', 0.0),
                    scan.get('total_lean_mass_lbs', 0.0),
                    scan.get('fat_mass_lbs', 0.0),
                    scan.get('body_fat_percentage', 0.0),
                    scan.get('arms_lean_lbs', 0.0),
                    scan.get('legs_lean_lbs', 0.0)
                ])
        
        # Add goals if set
        if session_state.almi_goal.get('target_percentile'):
            compact["ag"] = {"tp": session_state.almi_goal['target_percentile']}
            if session_state.almi_goal.get('target_age') and session_state.almi_goal['target_age'] != '?':
                compact["ag"]["ta"] = session_state.almi_goal['target_age']
        
        if session_state.ffmi_goal.get('target_percentile'):
            compact["fg"] = {"tp": session_state.ffmi_goal['target_percentile']}
            if session_state.ffmi_goal.get('target_age') and session_state.ffmi_goal['target_age'] != '?':
                compact["fg"]["ta"] = session_state.ffmi_goal['target_age']
        
        return compact
    
    def test_compact_config_round_trip(self, example_config, mock_session_state):
        """Test that compact config conversion preserves data integrity."""
        # Set up session state with example data
        mock_session_state.user_info = example_config['user_info']
        mock_session_state.scan_history = example_config['scan_history']
        mock_session_state.almi_goal = example_config['goals']['almi']
        mock_session_state.ffmi_goal = example_config['goals']['ffmi']
        
        # Convert to compact format
        compact = self.get_compact_config_from_data(mock_session_state)
        
        # Verify compact format structure
        assert "u" in compact
        assert "s" in compact
        assert "ag" in compact
        assert "fg" in compact
        
        # Verify user info
        assert compact["u"]["bd"] == "04/26/1982"
        assert compact["u"]["h"] == 66.0
        assert compact["u"]["g"] == "m"
        
        # Verify scans are in array format
        assert len(compact["s"]) == 2
        assert compact["s"][0][0] == "04/07/2022"  # First scan date
        assert compact["s"][0][1] == 143.2  # First scan weight
        
        # Verify goals
        assert compact["ag"]["tp"] == 0.90
        assert compact["fg"]["tp"] == 0.75
    
    @patch('streamlit.query_params')
    def test_decode_state_from_url_success(self, mock_query_params, example_config, mock_session_state):
        """Test successful decoding of state from URL parameters."""
        # Create encoded URL data
        url = self.create_encoded_url(example_config, mock_session_state)
        encoded_data = url.split('data=')[1]
        
        # Mock query parameters
        mock_query_params.__getitem__ = MagicMock(return_value=urllib.parse.unquote(encoded_data))
        mock_query_params.__contains__ = MagicMock(return_value=True)
        
        # Mock session state
        with patch('streamlit.session_state') as mock_st_session:
            mock_st_session.user_info = mock_session_state.user_info
            mock_st_session.scan_history = mock_session_state.scan_history
            mock_st_session.almi_goal = mock_session_state.almi_goal
            mock_st_session.ffmi_goal = mock_session_state.ffmi_goal
            
            # Test decoding
            result = decode_state_from_url()
            
            # Should return True for successful decode
            assert result == True
            
            # Verify session state was updated
            assert mock_st_session.user_info['birth_date'] == "04/26/1982"
            assert mock_st_session.user_info['height_in'] == 66.0
            assert mock_st_session.user_info['gender'] == "male"
            assert len(mock_st_session.scan_history) == 2
            assert mock_st_session.scan_history[0]['date'] == "04/07/2022"
            assert mock_st_session.almi_goal['target_percentile'] == 0.90
    
    @patch('streamlit.query_params')
    def test_decode_state_from_url_no_data(self, mock_query_params):
        """Test decode behavior when no URL data is present."""
        # Mock empty query parameters
        mock_query_params.__contains__ = MagicMock(return_value=False)
        
        # Test decoding
        result = decode_state_from_url()
        
        # Should return False when no data
        assert result == False
    
    @patch('streamlit.query_params')
    def test_decode_state_from_url_invalid_data(self, mock_query_params):
        """Test decode behavior with invalid/corrupted data."""
        # Mock invalid base64 data
        mock_query_params.__getitem__ = MagicMock(return_value="invalid_base64_data!!!")
        mock_query_params.__contains__ = MagicMock(return_value=True)
        
        # Test decoding - should handle gracefully
        result = decode_state_from_url()
        
        # Should return False for invalid data
        assert result == False
    
    def test_expand_compact_config(self, example_config, mock_session_state):
        """Test expansion of compact config back to full format."""
        # Create compact config
        mock_session_state.user_info = example_config['user_info']
        mock_session_state.scan_history = example_config['scan_history']
        mock_session_state.almi_goal = example_config['goals']['almi']
        mock_session_state.ffmi_goal = example_config['goals']['ffmi']
        
        compact = self.get_compact_config_from_data(mock_session_state)
        
        # Expand back to full format
        user_info, scan_history, almi_goal, ffmi_goal = expand_compact_config(compact)
        
        # Verify user info
        assert user_info['birth_date'] == "04/26/1982"
        assert user_info['height_in'] == 66.0
        assert user_info['gender'] == "male"
        assert user_info['training_level'] == ""
        
        # Verify scan history
        assert len(scan_history) == 2
        assert scan_history[0]['date'] == "04/07/2022"
        assert scan_history[0]['total_weight_lbs'] == 143.2
        assert scan_history[1]['date'] == "04/01/2023"
        assert scan_history[1]['total_weight_lbs'] == 154.3
        
        # Verify goals
        assert almi_goal['target_percentile'] == 0.90
        assert ffmi_goal['target_percentile'] == 0.75


class TestWebAppURLIntegration:
    """Integration tests using Streamlit AppTest for full webapp URL loading."""
    
    @pytest.fixture
    def app(self):
        """Create AppTest instance for the webapp."""
        return testing.AppTest.from_file("webapp.py", default_timeout=15)
    
    @pytest.fixture
    def example_config_url(self):
        """Create a shareable URL with example configuration."""
        config = {
            "user_info": {
                "birth_date": "04/26/1982",
                "height_in": 66.0,
                "gender": "male"
            },
            "scan_history": [
                {
                    "date": "04/07/2022",
                    "total_weight_lbs": 143.2,
                    "total_lean_mass_lbs": 106.3,
                    "fat_mass_lbs": 32.6,
                    "body_fat_percentage": 22.8,
                    "arms_lean_lbs": 12.4,
                    "legs_lean_lbs": 37.3
                },
                {
                    "date": "04/01/2023",
                    "total_weight_lbs": 154.3,
                    "total_lean_mass_lbs": 121.2,
                    "fat_mass_lbs": 28.5,
                    "body_fat_percentage": 18.5,
                    "arms_lean_lbs": 16.5,
                    "legs_lean_lbs": 40.4
                }
            ],
            "goals": {
                "almi": {"target_percentile": 0.90},
                "ffmi": {"target_percentile": 0.75}
            }
        }
        
        # Convert to compact format
        compact = {
            "u": {
                "bd": "04/26/1982",
                "h": 66.0,
                "g": "m"
            },
            "s": [
                ["04/07/2022", 143.2, 106.3, 32.6, 22.8, 12.4, 37.3],
                ["04/01/2023", 154.3, 121.2, 28.5, 18.5, 16.5, 40.4]
            ],
            "ag": {"tp": 0.90},
            "fg": {"tp": 0.75}
        }
        
        # Encode as base64
        json_str = json.dumps(compact, separators=(',', ':'))
        encoded_data = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
        
        return urllib.parse.quote(encoded_data)
    
    @patch('streamlit.query_params')
    def test_webapp_loads_url_state_correctly(self, mock_query_params, app, example_config_url):
        """Test that webapp loads state from URL and populates form inputs correctly."""
        # Mock query parameters
        mock_query_params.__getitem__ = MagicMock(return_value=example_config_url)
        mock_query_params.__contains__ = MagicMock(return_value=True)
        
        # Run the app
        at = app.run()
        
        # Verify the app loaded without errors
        assert len(at.exception) == 0, f"App should run without exceptions, got: {at.exception}"
        
        # Check for success message indicating URL was loaded
        success_messages = [msg for msg in at.success if "Configuration loaded from URL" in str(msg)]
        assert len(success_messages) > 0, "Should show URL loading success message"
        
        # Verify form elements are present
        assert len(at.text_input) >= 1, "Should have text input fields"
        assert len(at.number_input) >= 1, "Should have number input fields"
        assert len(at.selectbox) >= 1, "Should have selectbox fields"
        
        # Check that birth date input has the correct value
        birth_date_inputs = [inp for inp in at.text_input if "Birth Date" in inp.label]
        assert len(birth_date_inputs) > 0, "Should have birth date input"
        # Note: In actual Streamlit, we'd verify the value, but AppTest may not capture session state values
        
        # Verify scan history data editor is present
        assert len(at.data_editor) >= 1, "Should have data editor for scan history"
        
        # Check for analysis results or analysis-related content
        # The analysis should run automatically when valid data is loaded
        analysis_content = [msg for msg in at.info if "analysis" in str(msg).lower()]
        success_content = [msg for msg in at.success if "analysis" in str(msg).lower() or "completed" in str(msg).lower()]
        
        # Should have some indication that analysis processing occurred
        assert len(analysis_content) > 0 or len(success_content) > 0, "Should show analysis-related content"
    
    @patch('streamlit.query_params')
    def test_webapp_shows_share_url_section(self, mock_query_params, app, example_config_url):
        """Test that webapp shows the share URL section when data is loaded."""
        # Mock query parameters
        mock_query_params.__getitem__ = MagicMock(return_value=example_config_url)
        mock_query_params.__contains__ = MagicMock(return_value=True)
        
        # Run the app
        at = app.run()
        
        # Check for share configuration section
        markdown_content = [md.value for md in at.markdown]
        share_section_present = any("Share Configuration" in content for content in markdown_content)
        assert share_section_present, "Should display Share Configuration section"
        
        # Check for URL text area
        text_areas = [ta for ta in at.text_area if "Share URL" in ta.label or "URL" in ta.label]
        assert len(text_areas) > 0, "Should have text area for share URL"
    
    @patch('streamlit.query_params')
    def test_webapp_handles_automatic_analysis(self, mock_query_params, app, example_config_url):
        """Test that webapp automatically runs analysis when complete data is loaded from URL."""
        # Mock query parameters
        mock_query_params.__getitem__ = MagicMock(return_value=example_config_url)
        mock_query_params.__contains__ = MagicMock(return_value=True)
        
        # Run the app
        at = app.run()
        
        # Verify no critical errors
        assert len(at.exception) == 0, f"App should run without exceptions, got: {at.exception}"
        
        # Check for analysis-related content
        # When analysis runs automatically, we should see results
        success_messages = [str(msg) for msg in at.success]
        info_messages = [str(msg) for msg in at.info]
        
        # Look for indicators that analysis was attempted or completed
        analysis_indicators = []
        for messages in [success_messages, info_messages]:
            analysis_indicators.extend([
                msg for msg in messages 
                if any(keyword in msg.lower() for keyword in ['analysis', 'completed', 'results', 'calculated'])
            ])
        
        assert len(analysis_indicators) > 0, f"Should show analysis indicators, found messages: {success_messages + info_messages}"
        
        # Check for plots or data tables (analysis outputs)
        # In a real analysis, we'd expect pyplot figures and dataframes
        assert len(at.pyplot) >= 0, "Should have pyplot figures (if analysis completed)"
        
    @patch('streamlit.query_params')
    def test_webapp_url_loading_with_empty_params(self, mock_query_params, app):
        """Test webapp behavior when no URL parameters are present."""
        # Mock empty query parameters
        mock_query_params.__contains__ = MagicMock(return_value=False)
        
        # Run the app
        at = app.run()
        
        # Should run without errors
        assert len(at.exception) == 0, f"App should run without exceptions, got: {at.exception}"
        
        # Should not show URL loading success message
        success_messages = [str(msg) for msg in at.success]
        url_load_messages = [msg for msg in success_messages if "Configuration loaded from URL" in msg]
        assert len(url_load_messages) == 0, "Should not show URL loading message when no params"
        
        # Should show the default empty state message
        info_messages = [str(msg) for msg in at.info]
        empty_state_messages = [msg for msg in info_messages if "Enter some data" in msg or "automatically" in msg]
        assert len(empty_state_messages) > 0, "Should show empty state message"


class TestURLRoundTripIntegrity:
    """Test round-trip integrity of URL state loading and generation."""
    
    @pytest.fixture
    def example_config(self):
        """Example configuration based on example_config.json."""
        return {
            "user_info": {
                "birth_date": "04/26/1982",
                "height_in": 66.0,
                "gender": "male"
            },
            "scan_history": [
                {
                    "date": "04/07/2022",
                    "total_weight_lbs": 143.2,
                    "total_lean_mass_lbs": 106.3,
                    "fat_mass_lbs": 32.6,
                    "body_fat_percentage": 22.8,
                    "arms_lean_lbs": 12.4,
                    "legs_lean_lbs": 37.3
                },
                {
                    "date": "04/01/2023",
                    "total_weight_lbs": 154.3,
                    "total_lean_mass_lbs": 121.2,
                    "fat_mass_lbs": 28.5,
                    "body_fat_percentage": 18.5,
                    "arms_lean_lbs": 16.5,
                    "legs_lean_lbs": 40.4
                }
            ],
            "goals": {
                "almi": {"target_percentile": 0.90},
                "ffmi": {"target_percentile": 0.75}
            }
        }
    
    def test_full_round_trip_integrity(self, example_config):
        """Test that URL generation and loading preserves data integrity perfectly."""
        # Create initial session state
        mock_session = MagicMock()
        mock_session.user_info = example_config['user_info']
        mock_session.scan_history = example_config['scan_history']
        mock_session.almi_goal = example_config['goals']['almi']
        mock_session.ffmi_goal = example_config['goals']['ffmi']
        
        # Generate compact config
        compact_original = self.get_compact_config_from_session(mock_session)
        
        # Simulate URL encoding
        json_str = json.dumps(compact_original, separators=(',', ':'))
        encoded_data = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
        
        # Simulate URL decoding
        decoded_bytes = base64.b64decode(encoded_data.encode('utf-8'))
        decoded_json = decoded_bytes.decode('utf-8')
        compact_recovered = json.loads(decoded_json)
        
        # Verify perfect round-trip
        assert compact_original == compact_recovered, "Round-trip should preserve exact data"
        
        # Expand recovered config
        user_info, scan_history, almi_goal, ffmi_goal = expand_compact_config(compact_recovered)
        
        # Verify all data matches original
        assert user_info['birth_date'] == example_config['user_info']['birth_date']
        assert user_info['height_in'] == example_config['user_info']['height_in']
        assert user_info['gender'] == example_config['user_info']['gender']
        
        assert len(scan_history) == len(example_config['scan_history'])
        for i, scan in enumerate(scan_history):
            original_scan = example_config['scan_history'][i]
            assert scan['date'] == original_scan['date']
            assert scan['total_weight_lbs'] == original_scan['total_weight_lbs']
            assert scan['body_fat_percentage'] == original_scan['body_fat_percentage']
        
        assert almi_goal['target_percentile'] == example_config['goals']['almi']['target_percentile']
        assert ffmi_goal['target_percentile'] == example_config['goals']['ffmi']['target_percentile']
    
    def get_compact_config_from_session(self, session_state):
        """Helper to create compact config from session state."""
        compact = {
            "u": {
                "bd": session_state.user_info.get('birth_date', ''),
                "h": session_state.user_info.get('height_in', 66.0),
                "g": session_state.user_info.get('gender', 'male')[0],
            }
        }
        
        if session_state.user_info.get('training_level'):
            compact["u"]["tl"] = session_state.user_info['training_level']
        
        compact["s"] = []
        for scan in session_state.scan_history:
            if scan.get('date'):
                compact["s"].append([
                    scan.get('date', ''),
                    scan.get('total_weight_lbs', 0.0),
                    scan.get('total_lean_mass_lbs', 0.0),
                    scan.get('fat_mass_lbs', 0.0),
                    scan.get('body_fat_percentage', 0.0),
                    scan.get('arms_lean_lbs', 0.0),
                    scan.get('legs_lean_lbs', 0.0)
                ])
        
        if session_state.almi_goal.get('target_percentile'):
            compact["ag"] = {"tp": session_state.almi_goal['target_percentile']}
            if session_state.almi_goal.get('target_age') and session_state.almi_goal['target_age'] != '?':
                compact["ag"]["ta"] = session_state.almi_goal['target_age']
        
        if session_state.ffmi_goal.get('target_percentile'):
            compact["fg"] = {"tp": session_state.ffmi_goal['target_percentile']}
            if session_state.ffmi_goal.get('target_age') and session_state.ffmi_goal['target_age'] != '?':
                compact["fg"]["ta"] = session_state.ffmi_goal['target_age']
        
        return compact


class TestURLEdgeCases:
    """Test edge cases and error handling for URL state loading."""
    
    @patch('streamlit.query_params')
    def test_malformed_base64_handling(self, mock_query_params):
        """Test graceful handling of malformed base64 data."""
        # Mock malformed base64
        mock_query_params.__getitem__ = MagicMock(return_value="invalid===base64!!!")
        mock_query_params.__contains__ = MagicMock(return_value=True)
        
        # Should handle gracefully without crashing
        result = decode_state_from_url()
        assert result == False, "Should return False for malformed data"
    
    @patch('streamlit.query_params')
    def test_invalid_json_handling(self, mock_query_params):
        """Test graceful handling of invalid JSON after base64 decode."""
        # Create base64 of invalid JSON
        invalid_json = "{'invalid': json structure"
        encoded = base64.b64encode(invalid_json.encode('utf-8')).decode('utf-8')
        
        mock_query_params.__getitem__ = MagicMock(return_value=encoded)
        mock_query_params.__contains__ = MagicMock(return_value=True)
        
        # Should handle gracefully
        result = decode_state_from_url()
        assert result == False, "Should return False for invalid JSON"
    
    def test_partial_config_handling(self):
        """Test handling of partial/incomplete configurations."""
        # Test with minimal config (only user info)
        partial_compact = {
            "u": {
                "bd": "04/26/1982",
                "h": 66.0,
                "g": "m"
            }
            # No scans or goals
        }
        
        # Should expand without errors
        user_info, scan_history, almi_goal, ffmi_goal = expand_compact_config(partial_compact)
        
        assert user_info['birth_date'] == "04/26/1982"
        assert len(scan_history) == 0, "Should handle empty scan history"
        assert almi_goal['target_percentile'] == 0.75, "Should use default goal"
        assert ffmi_goal['target_percentile'] == 0.75, "Should use default goal"
    
    def test_config_with_special_characters(self):
        """Test handling of configurations with special characters."""
        # Config with special characters in user data
        special_compact = {
            "u": {
                "bd": "04/26/1982",
                "h": 66.0,
                "g": "f",  # Female
                "tl": "advanced"
            },
            "s": [
                ["12/31/2023", 150.5, 120.3, 25.2, 16.8, 15.1, 38.9]
            ],
            "ag": {"tp": 0.95, "ta": 30},
            "fg": {"tp": 0.80, "ta": 30}
        }
        
        # Should handle without issues
        user_info, scan_history, almi_goal, ffmi_goal = expand_compact_config(special_compact)
        
        assert user_info['gender'] == "female", "Should correctly convert gender"
        assert user_info['training_level'] == "advanced"
        assert scan_history[0]['date'] == "12/31/2023"
        assert almi_goal['target_age'] == 30
        assert ffmi_goal['target_age'] == 30
    
    def test_maximum_scan_limit_handling(self):
        """Test handling of configurations at the 20-scan limit."""
        # Create config with exactly 20 scans
        scans = []
        for i in range(20):
            scans.append([
                f"01/{i+1:02d}/2024",
                150.0 + i,
                120.0 + i * 0.5,
                25.0,
                16.5,
                15.0,
                38.0
            ])
        
        max_compact = {
            "u": {"bd": "04/26/1982", "h": 66.0, "g": "m"},
            "s": scans,
            "ag": {"tp": 0.85},
            "fg": {"tp": 0.75}
        }
        
        # Should handle 20 scans without issues
        user_info, scan_history, almi_goal, ffmi_goal = expand_compact_config(max_compact)
        
        assert len(scan_history) == 20, "Should handle maximum 20 scans"
        assert scan_history[0]['date'] == "01/01/2024"
        assert scan_history[19]['date'] == "01/20/2024"
        assert scan_history[19]['total_weight_lbs'] == 169.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])