#!/usr/bin/env python3
"""
Integration test to reproduce and verify fix for URL loading session state bug.

This test simulates the complete user workflow:
1. User enters data and generates a share URL
2. New session loads state from that URL
3. App tries to access all session state attributes

Bug: When URL loading succeeds, initialize_session_state() returns early,
skipping initialization of analysis_results and other attributes.

This should be kept as a permanent regression test.
"""

import base64
import json
from unittest.mock import MagicMock, patch


class MockSessionState:
    """Mock Streamlit session state that tracks attribute access."""

    def __init__(self):
        self._data = {}
        self._accessed_attributes = set()

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __getattr__(self, name):
        self._accessed_attributes.add(name)
        if name in self._data:
            return self._data[name]
        raise AttributeError(
            f'st.session_state has no attribute "{name}". Did you forget to initialize it?'
        )

    def __contains__(self, name):
        return name in self._data

    def get(self, name, default=None):
        self._accessed_attributes.add(name)
        return self._data.get(name, default)

    def has_attribute(self, name):
        return name in self._data

    def get_accessed_attributes(self):
        return self._accessed_attributes.copy()


def create_test_url():
    """Create a realistic test URL with user data."""
    # Simulate user-entered data
    compact_config = {
        "u": {"bd": "04/26/1982", "h": 66.0, "g": "m"},
        "s": [
            ["04/07/2022", 143.2, 106.3, 32.6, 22.8, 12.4, 37.3],
            ["04/01/2023", 154.3, 121.2, 28.5, 18.5, 16.5, 40.4],
        ],
        "ag": {"tp": 0.90},
        "fg": {"tp": 0.75},
    }

    # Encode as URL parameter (same as webapp does)
    json_str = json.dumps(compact_config, separators=(",", ":"))
    encoded_data = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")
    # Note: Streamlit query_params provides unquoted data, so we return the raw base64

    return encoded_data, compact_config


def test_url_loading_session_state_bug():
    """
    Reproduce the session state initialization bug when loading from URL.

    This test should initially FAIL, demonstrating the bug exists.
    After the fix is implemented, it should PASS.
    """
    print("Testing URL loading session state bug reproduction...")

    # Step 1: Create test URL (simulating user sharing URL)
    url_param, original_config = create_test_url()
    print(f"✅ Created test URL parameter (length: {len(url_param)})")

    # Step 2: Simulate new session loading from URL
    mock_session_state = MockSessionState()

    with (
        patch("streamlit.query_params") as mock_query_params,
        patch("streamlit.session_state", mock_session_state),
        patch("streamlit.success"),
    ):
        # Mock URL parameters
        mock_query_params.__contains__ = MagicMock(return_value=True)
        mock_query_params.__getitem__ = MagicMock(return_value=url_param)

        # Step 3: Initialize session state (this is where the bug occurs)
        from webapp import initialize_session_state

        print("📡 Loading state from URL...")
        initialize_session_state()

        # Step 4: Verify URL loading worked
        assert mock_session_state.has_attribute("user_info"), (
            "user_info should be loaded from URL"
        )
        assert mock_session_state.has_attribute("scan_history"), (
            "scan_history should be loaded from URL"
        )
        assert mock_session_state.user_info["birth_date"] == "04/26/1982"
        assert len(mock_session_state.scan_history) == 2
        print("✅ URL state loading succeeded")

        # Step 5: This is where the bug manifests - trying to access analysis_results
        print(
            "🔍 Attempting to access analysis_results (this should fail with the bug)..."
        )

        try:
            # This simulates what display_results() does
            analysis_results = mock_session_state.analysis_results

            # If we get here, the bug is NOT present (or already fixed)
            print(f"✅ analysis_results accessible: {analysis_results}")
            return True, "Bug not reproduced - analysis_results was initialized"

        except AttributeError as e:
            # This is the expected bug behavior
            expected_msg = 'st.session_state has no attribute "analysis_results"'
            if expected_msg in str(e):
                print(f"🐛 BUG REPRODUCED: {e}")
                return False, str(e)
            else:
                raise AssertionError(f"Unexpected AttributeError: {e}")


def test_all_required_attributes_after_url_loading():
    """
    Test that all required session state attributes are accessible after URL loading.

    This test verifies the fix by ensuring ALL session state attributes are
    properly initialized, even when loading from URL.
    """
    print("\nTesting complete session state initialization after URL loading...")

    url_param, _ = create_test_url()
    mock_session_state = MockSessionState()

    with (
        patch("streamlit.query_params") as mock_query_params,
        patch("streamlit.session_state", mock_session_state),
        patch("streamlit.success"),
    ):
        mock_query_params.__contains__ = MagicMock(return_value=True)
        mock_query_params.__getitem__ = MagicMock(return_value=url_param)

        # Initialize session state
        from webapp import initialize_session_state

        initialize_session_state()

        # Test access to all required attributes
        required_attributes = [
            "user_info",
            "scan_history",
            "almi_goal",
            "ffmi_goal",
            "analysis_results",  # This is the one that fails with the bug
            "url_loaded",
        ]

        print("🔍 Checking all required session state attributes...")

        missing_attributes = []
        for attr in required_attributes:
            try:
                value = getattr(mock_session_state, attr)
                print(f"✅ {attr}: {type(value).__name__}")
            except AttributeError as e:
                print(f"❌ {attr}: MISSING - {e}")
                missing_attributes.append(attr)

        if missing_attributes:
            return False, f"Missing attributes: {missing_attributes}"

        # Verify the loaded data is correct
        assert mock_session_state.user_info["birth_date"] == "04/26/1982"
        assert mock_session_state.user_info["height_in"] == 66.0
        assert len(mock_session_state.scan_history) == 2
        assert mock_session_state.almi_goal["target_percentile"] == 0.90
        assert mock_session_state.analysis_results is None  # Should be None initially

        print("✅ All session state attributes properly initialized and accessible")
        return True, "All attributes accessible"


def test_normal_initialization_still_works():
    """
    Test that normal initialization (without URL) still works correctly.

    This ensures our fix doesn't break the normal startup flow.
    """
    print("\nTesting normal initialization (no URL)...")

    mock_session_state = MockSessionState()

    with (
        patch("streamlit.query_params") as mock_query_params,
        patch("streamlit.session_state", mock_session_state),
    ):
        # Mock empty query parameters (no URL data)
        mock_query_params.__contains__ = MagicMock(return_value=False)

        # Initialize session state
        from webapp import initialize_session_state

        initialize_session_state()

        # Verify default initialization
        assert mock_session_state.user_info["birth_date"] == ""
        assert mock_session_state.user_info["height_in"] == 66.0
        assert mock_session_state.user_info["gender"] == "male"
        assert len(mock_session_state.scan_history) == 0
        assert mock_session_state.analysis_results is None

        print("✅ Normal initialization works correctly")
        return True


if __name__ == "__main__":
    print("=" * 70)
    print("URL LOADING SESSION STATE BUG - REPRODUCTION TEST")
    print("=" * 70)

    # Test 1: Reproduce the bug
    print("\n1. REPRODUCING THE BUG:")
    print("-" * 30)
    bug_reproduced, bug_message = test_url_loading_session_state_bug()

    if bug_reproduced:
        print("✅ Bug is already fixed!")
    else:
        print(f"🐛 Bug confirmed: {bug_message}")

    # Test 2: Test complete attribute access (will fail until fixed)
    print("\n2. TESTING COMPLETE ATTRIBUTE ACCESS:")
    print("-" * 40)
    try:
        all_attrs_work, attr_message = test_all_required_attributes_after_url_loading()
        if all_attrs_work:
            print("✅ All attributes accessible - fix is working!")
        else:
            print(f"❌ Fix needed: {attr_message}")
    except Exception as e:
        print(f"❌ Error accessing attributes: {e}")
        all_attrs_work = False

    # Test 3: Verify normal init still works
    print("\n3. TESTING NORMAL INITIALIZATION:")
    print("-" * 35)
    normal_init_works = test_normal_initialization_still_works()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY:")
    print("=" * 70)

    if bug_reproduced and all_attrs_work and normal_init_works:
        print("🎉 ALL TESTS PASS - Bug has been fixed!")
        exit(0)
    elif not bug_reproduced and not all_attrs_work:
        print("🐛 Bug confirmed and needs to be fixed in webapp.py")
        print("   Issue: initialize_session_state() returns early after URL loading,")
        print("   skipping initialization of analysis_results and other attributes.")
        exit(1)
    else:
        print("⚠️  Mixed results - investigate further")
        exit(1)
