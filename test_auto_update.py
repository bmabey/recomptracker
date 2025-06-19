#!/usr/bin/env python3
"""
Test automatic URL updating functionality
"""

import json

# Mock session state for testing auto-update logic
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
        self._state = {}
    
    def get(self, key, default=None):
        return self._state.get(key, default)
    
    def __setitem__(self, key, value):
        self._state[key] = value
    
    def __getitem__(self, key):
        return self._state[key]
    
    def __contains__(self, key):
        return key in self._state

def get_compact_config(session_state):
    """Convert session state to compact JSON format."""
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
        if scan.get('date'):  # Only include scans with dates
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

def test_auto_update_logic():
    """Test the auto-update detection logic."""
    print("Testing auto-update URL logic...")
    
    # Create mock session state
    session_state = MockSessionState()
    
    # Test 1: No meaningful data - should not trigger update
    compact = get_compact_config(session_state)
    has_data = (session_state.user_info.get('birth_date') or 
                session_state.user_info.get('height_in', 0) != 66.0 or
                len(session_state.scan_history) > 0)
    
    print(f"Test 1 - Empty state: has_data = {has_data} (should be False)")
    assert not has_data, "Empty state should not trigger updates"
    
    # Test 2: Add birth date - should trigger update
    session_state.user_info['birth_date'] = '04/26/1982'
    compact1 = get_compact_config(session_state)
    hash1 = hash(json.dumps(compact1, sort_keys=True))
    
    has_data = (session_state.user_info.get('birth_date') or 
                session_state.user_info.get('height_in', 0) != 66.0 or
                len(session_state.scan_history) > 0)
    
    print(f"Test 2 - With birth date: has_data = {has_data} (should be True)")
    assert has_data, "State with birth date should trigger updates"
    
    # Test 3: Add scan - should change hash
    session_state.scan_history = [{
        'date': '04/07/2022',
        'total_weight_lbs': 143.2,
        'total_lean_mass_lbs': 106.3,
        'fat_mass_lbs': 32.6,
        'body_fat_percentage': 22.8,
        'arms_lean_lbs': 12.4,
        'legs_lean_lbs': 37.3
    }]
    
    compact2 = get_compact_config(session_state)
    hash2 = hash(json.dumps(compact2, sort_keys=True))
    
    print(f"Test 3 - Hash changed: {hash1 != hash2} (should be True)")
    assert hash1 != hash2, "Adding scan should change state hash"
    
    # Test 4: No change - hash should be same
    compact3 = get_compact_config(session_state)
    hash3 = hash(json.dumps(compact3, sort_keys=True))
    
    print(f"Test 4 - Hash unchanged: {hash2 == hash3} (should be True)")
    assert hash2 == hash3, "Unchanged state should have same hash"
    
    # Test 5: Change goal - should change hash
    session_state.almi_goal['target_percentile'] = 0.90
    compact4 = get_compact_config(session_state)
    hash4 = hash(json.dumps(compact4, sort_keys=True))
    
    print(f"Test 5 - Goal change: {hash3 != hash4} (should be True)")
    assert hash3 != hash4, "Changing goal should change state hash"
    
    print("✅ All auto-update logic tests passed!")
    
    # Show compact format example
    print(f"\nFinal compact config:")
    print(json.dumps(compact4, indent=2))

if __name__ == "__main__":
    test_auto_update_logic()