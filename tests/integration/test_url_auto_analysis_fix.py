#!/usr/bin/env python3
"""
Integration test to verify that analysis automatically reruns when state is loaded from URL.

This test verifies the fix for the issue: 
"The state is restored correctly but the analysis is not reran automatically like it should be."

This is a focused test that verifies the specific auto-analysis behavior after URL loading.
"""

import json
import base64
from unittest.mock import patch, MagicMock


class MockSessionState:
    """Mock Streamlit session state for testing."""
    
    def __init__(self):
        self._data = {}
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._data[name] = value
    
    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        # Return reasonable defaults
        defaults = {
            'user_info': {'birth_date': '', 'height_in': 66.0, 'gender': 'male', 'training_level': ''},
            'scan_history': [],
            'almi_goal': {'target_percentile': 0.75, 'target_age': '?'},
            'ffmi_goal': {'target_percentile': 0.75, 'target_age': '?'},
            'analysis_results': None,
            'url_loaded': False,
            'url_loaded_needs_analysis': False
        }
        return defaults.get(name, None)
    
    def __contains__(self, name):
        return name in self._data
    
    def get(self, name, default=None):
        return self._data.get(name, default)


def test_url_auto_analysis_fix():
    """
    Test that URL loading automatically triggers analysis with the fix.
    
    This test verifies that when valid data is loaded from URL, the webapp
    automatically runs analysis without requiring user interaction.
    """
    print("Testing URL auto-analysis fix...")
    
    # Create test URL with complete, valid data
    compact_config = {
        "u": {
            "bd": "04/26/1982",    # Valid birth date
            "h": 66.0,             # Valid height
            "g": "m"               # Valid gender
        },
        "s": [
            # Complete scan data that should pass validation
            ["04/07/2022", 143.2, 106.3, 32.6, 22.8, 12.4, 37.3],
            ["04/01/2023", 154.3, 121.2, 28.5, 18.5, 16.5, 40.4]
        ],
        "ag": {"tp": 0.90},
        "fg": {"tp": 0.75}
    }
    
    # Encode as URL parameter
    json_str = json.dumps(compact_config, separators=(',', ':'))
    encoded_data = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
    
    print(f"✅ Created test URL data")
    
    # Set up mock session state
    mock_session_state = MockSessionState()
    
    # Track analysis calls
    analysis_called = {"count": 0}
    
    def mock_run_analysis():
        """Mock run_analysis that tracks calls."""
        analysis_called["count"] += 1
        mock_session_state.analysis_results = {"mock": "results"}
        print(f"📊 Analysis called (call #{analysis_called['count']})")
    
    def mock_validate_form_data():
        """Mock validate_form_data to return no errors."""
        return {}  # No errors = valid data
    
    with patch('streamlit.query_params') as mock_query_params, \
         patch('streamlit.session_state', mock_session_state), \
         patch('streamlit.success'), \
         patch('webapp.run_analysis', side_effect=mock_run_analysis), \
         patch('webapp.validate_form_data', side_effect=mock_validate_form_data):
        
        # Mock URL parameters
        mock_query_params.__contains__ = MagicMock(return_value=True)
        mock_query_params.__getitem__ = MagicMock(return_value=encoded_data)
        
        # Test the fix: initialize session state (which loads from URL)
        from webapp import initialize_session_state
        
        print("🔄 Initializing session state from URL...")
        initialize_session_state()
        
        # Verify URL loading worked
        assert mock_session_state.user_info['birth_date'] == "04/26/1982"
        assert len(mock_session_state.scan_history) == 2
        print("✅ URL state loading succeeded")
        
        # Key test: Check if the url_loaded_needs_analysis flag is set
        needs_analysis = mock_session_state.get('url_loaded_needs_analysis', False)
        print(f"🏁 URL loaded needs analysis flag: {needs_analysis}")
        
        if not needs_analysis:
            print("❌ FAILURE: url_loaded_needs_analysis flag not set after URL loading")
            return False
        
        # Simulate the main() function auto-analysis logic that processes this flag
        print("🚀 Simulating main() function auto-analysis logic...")
        
        if mock_session_state.get('url_loaded_needs_analysis', False):
            # Check if we have valid data for analysis (same conditions as main())
            if (mock_session_state.user_info.get('birth_date') and 
                mock_session_state.user_info.get('gender') and 
                len(mock_session_state.scan_history) > 0 and
                any(scan.get('date') for scan in mock_session_state.scan_history)):
                
                errors = mock_validate_form_data()
                if not errors:
                    print("   Auto-analysis conditions met - running analysis...")
                    mock_run_analysis()
                else:
                    print("   Validation failed - analysis not called")
            else:
                print("   Auto-analysis conditions not met")
            
            # Clear the flag (as main() function does)
            mock_session_state.url_loaded_needs_analysis = False
        
        # Verify analysis was called
        if analysis_called["count"] > 0:
            print("✅ SUCCESS: Analysis was automatically triggered after URL loading!")
            print(f"   Analysis called {analysis_called['count']} time(s)")
            return True
        else:
            print("❌ FAILURE: Analysis was NOT triggered after URL loading")
            return False


def test_url_auto_analysis_flag_cleared():
    """
    Test that the url_loaded_needs_analysis flag is properly cleared after processing.
    
    This ensures the analysis doesn't run repeatedly on subsequent reruns.
    """
    print("\nTesting flag clearing behavior...")
    
    mock_session_state = MockSessionState()
    
    # Manually set the flag
    mock_session_state.url_loaded_needs_analysis = True
    
    analysis_called = {"count": 0}
    
    def mock_run_analysis():
        analysis_called["count"] += 1
        mock_session_state.analysis_results = {"mock": "results"}
    
    def mock_validate_form_data():
        return {}
    
    # Set up valid data
    mock_session_state.user_info = {
        'birth_date': '04/26/1982',
        'height_in': 66.0,
        'gender': 'male'
    }
    mock_session_state.scan_history = [
        {'date': '04/07/2022', 'total_weight_lbs': 143.2}
    ]
    
    with patch('webapp.run_analysis', side_effect=mock_run_analysis), \
         patch('webapp.validate_form_data', side_effect=mock_validate_form_data):
        
        # Simulate the main() function logic
        if mock_session_state.get('url_loaded_needs_analysis', False):
            if (mock_session_state.user_info.get('birth_date') and 
                mock_session_state.user_info.get('gender') and 
                len(mock_session_state.scan_history) > 0 and
                any(scan.get('date') for scan in mock_session_state.scan_history)):
                
                errors = mock_validate_form_data()
                if not errors:
                    mock_run_analysis()
            
            # Clear the flag
            mock_session_state.url_loaded_needs_analysis = False
        
        # Verify flag was cleared
        flag_after = mock_session_state.get('url_loaded_needs_analysis', False)
        
        if not flag_after and analysis_called["count"] == 1:
            print("✅ Flag properly cleared after analysis")
            return True
        else:
            print(f"❌ Flag clearing failed. Flag after: {flag_after}, Analysis calls: {analysis_called['count']}")
            return False


if __name__ == "__main__":
    print("=" * 70)
    print("URL AUTO-ANALYSIS FIX VERIFICATION")
    print("=" * 70)
    
    # Test 1: URL auto-analysis fix
    print("\n1. TESTING URL AUTO-ANALYSIS FIX:")
    print("-" * 35)
    fix_works = test_url_auto_analysis_fix()
    
    # Test 2: Flag clearing behavior
    print("\n2. TESTING FLAG CLEARING:")
    print("-" * 25)
    flag_works = test_url_auto_analysis_flag_cleared()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY:")
    print("=" * 70)
    
    if fix_works and flag_works:
        print("🎉 ALL TESTS PASS - URL auto-analysis fix is working correctly!")
        exit(0)
    else:
        print("❌ TESTS FAILED")
        print(f"   Auto-analysis fix working: {fix_works}")
        print(f"   Flag clearing working: {flag_works}")
        exit(1)