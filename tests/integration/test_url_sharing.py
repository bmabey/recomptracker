#!/usr/bin/env python3
"""
Test script for URL sharing functionality
"""

import json
import base64
import urllib.parse

# Mock session state for testing
class MockSessionState:
    def __init__(self):
        self.user_info = {
            'birth_date': '04/26/1982',
            'height_in': 66.0,
            'gender': 'male',
            'training_level': 'intermediate'
        }
        self.scan_history = [
            {
                'date': '04/07/2022',
                'total_weight_lbs': 143.2,
                'total_lean_mass_lbs': 106.3,
                'fat_mass_lbs': 32.6,
                'body_fat_percentage': 22.8,
                'arms_lean_lbs': 12.4,
                'legs_lean_lbs': 37.3
            },
            {
                'date': '04/01/2023',
                'total_weight_lbs': 154.3,
                'total_lean_mass_lbs': 121.2,
                'fat_mass_lbs': 28.5,
                'body_fat_percentage': 18.5,
                'arms_lean_lbs': 16.5,
                'legs_lean_lbs': 40.4
            }
        ]
        self.almi_goal = {'target_percentile': 0.90, 'target_age': '?'}
        self.ffmi_goal = {'target_percentile': 0.75, 'target_age': 42}

# Import functions from webapp (we'll need to modify them slightly)
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

def expand_compact_config(compact_config):
    """Convert compact JSON format back to full session state format."""
    # User info
    user_info = {
        'birth_date': compact_config.get("u", {}).get("bd", ''),
        'height_in': compact_config.get("u", {}).get("h", 66.0),
        'gender': 'male' if compact_config.get("u", {}).get("g", 'm') == 'm' else 'female',
        'training_level': compact_config.get("u", {}).get("tl", '')
    }
    
    # Scan history
    scan_history = []
    for scan_array in compact_config.get("s", []):
        if len(scan_array) >= 7:
            scan_history.append({
                'date': scan_array[0],
                'total_weight_lbs': scan_array[1],
                'total_lean_mass_lbs': scan_array[2],
                'fat_mass_lbs': scan_array[3],
                'body_fat_percentage': scan_array[4],
                'arms_lean_lbs': scan_array[5],
                'legs_lean_lbs': scan_array[6]
            })
    
    # Goals
    almi_goal = {'target_percentile': 0.75, 'target_age': '?'}
    if "ag" in compact_config:
        almi_goal['target_percentile'] = compact_config["ag"].get("tp", 0.75)
        almi_goal['target_age'] = compact_config["ag"].get("ta", '?')
    
    ffmi_goal = {'target_percentile': 0.75, 'target_age': '?'}
    if "fg" in compact_config:
        ffmi_goal['target_percentile'] = compact_config["fg"].get("tp", 0.75)
        ffmi_goal['target_age'] = compact_config["fg"].get("ta", '?')
    
    return user_info, scan_history, almi_goal, ffmi_goal

def test_url_sharing():
    """Test the URL sharing functionality."""
    print("Testing URL sharing functionality...")
    
    # Create mock session state
    session_state = MockSessionState()
    
    # Get compact config
    compact_config = get_compact_config(session_state)
    print(f"Compact config: {json.dumps(compact_config, indent=2)}")
    
    # Convert to JSON string
    json_str = json.dumps(compact_config, separators=(',', ':'))
    print(f"JSON string length: {len(json_str)} characters")
    
    # Encode as base64
    encoded_data = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
    print(f"Base64 encoded length: {len(encoded_data)} characters")
    
    # Create URL
    base_url = "http://localhost:8501"
    share_url = f"{base_url}?data={urllib.parse.quote(encoded_data)}"
    print(f"Share URL length: {len(share_url)} characters")
    print(f"Share URL: {share_url}")
    
    # Test round-trip: decode back
    decoded_bytes = base64.b64decode(encoded_data.encode('utf-8'))
    decoded_json = decoded_bytes.decode('utf-8')
    recovered_config = json.loads(decoded_json)
    
    # Expand back to full format
    user_info, scan_history, almi_goal, ffmi_goal = expand_compact_config(recovered_config)
    
    # Verify data integrity
    assert user_info['birth_date'] == session_state.user_info['birth_date']
    assert user_info['height_in'] == session_state.user_info['height_in']
    assert user_info['gender'] == session_state.user_info['gender']
    assert user_info['training_level'] == session_state.user_info['training_level']
    
    assert len(scan_history) == len(session_state.scan_history)
    assert scan_history[0]['date'] == session_state.scan_history[0]['date']
    assert scan_history[0]['total_weight_lbs'] == session_state.scan_history[0]['total_weight_lbs']
    
    assert almi_goal['target_percentile'] == session_state.almi_goal['target_percentile']
    assert ffmi_goal['target_percentile'] == session_state.ffmi_goal['target_percentile']
    assert ffmi_goal['target_age'] == session_state.ffmi_goal['target_age']
    
    print("✅ All tests passed! URL sharing works correctly.")
    
    # Calculate compression ratio
    original_json = json.dumps({
        "user_info": session_state.user_info,
        "scan_history": session_state.scan_history,
        "goals": {
            "almi": session_state.almi_goal,
            "ffmi": session_state.ffmi_goal
        }
    }, separators=(',', ':'))
    
    compression_ratio = len(json_str) / len(original_json)
    print(f"Compression ratio: {compression_ratio:.2f} ({len(original_json)} → {len(json_str)} chars)")

if __name__ == "__main__":
    test_url_sharing()