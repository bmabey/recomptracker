#!/usr/bin/env python3
"""
Test URL sharing with 20 scans to verify size limits
"""

import base64
import json
import urllib.parse
from datetime import datetime, timedelta


# Mock session state with 20 scans
class MockSessionState20:
    def __init__(self):
        self.user_info = {
            "birth_date": "04/26/1982",
            "height_in": 66.0,
            "gender": "male",
            "training_level": "advanced",
        }

        # Generate 20 scans
        self.scan_history = []
        base_date = datetime(2020, 1, 1)
        for i in range(20):
            scan_date = base_date + timedelta(days=i * 30)  # Monthly scans
            self.scan_history.append(
                {
                    "date": scan_date.strftime("%m/%d/%Y"),
                    "total_weight_lbs": 140.0 + i * 0.5,  # Gradual weight gain
                    "total_lean_mass_lbs": 100.0 + i * 0.8,  # Gradual lean mass gain
                    "fat_mass_lbs": 35.0 - i * 0.3,  # Gradual fat loss
                    "body_fat_percentage": 25.0 - i * 0.5,  # BF% reduction
                    "arms_lean_lbs": 12.0 + i * 0.1,
                    "legs_lean_lbs": 35.0 + i * 0.2,
                }
            )

        self.almi_goal = {"target_percentile": 0.90, "target_age": "?"}
        self.ffmi_goal = {"target_percentile": 0.85, "target_age": 45}


def get_compact_config(session_state):
    """Convert session state to compact JSON format."""
    compact = {
        "u": {
            "bd": session_state.user_info.get("birth_date", ""),
            "h": session_state.user_info.get("height_in", 66.0),
            "g": session_state.user_info.get("gender", "male")[0],  # 'm' or 'f'
        }
    }

    # Add training level if set
    if session_state.user_info.get("training_level"):
        compact["u"]["tl"] = session_state.user_info["training_level"]

    # Convert scan history to array format
    compact["s"] = []
    for scan in session_state.scan_history:
        if scan.get("date"):  # Only include scans with dates
            compact["s"].append(
                [
                    scan.get("date", ""),
                    scan.get("total_weight_lbs", 0.0),
                    scan.get("total_lean_mass_lbs", 0.0),
                    scan.get("fat_mass_lbs", 0.0),
                    scan.get("body_fat_percentage", 0.0),
                    scan.get("arms_lean_lbs", 0.0),
                    scan.get("legs_lean_lbs", 0.0),
                ]
            )

    # Add goals if set
    if session_state.almi_goal.get("target_percentile"):
        compact["ag"] = {"tp": session_state.almi_goal["target_percentile"]}
        if (
            session_state.almi_goal.get("target_age")
            and session_state.almi_goal["target_age"] != "?"
        ):
            compact["ag"]["ta"] = session_state.almi_goal["target_age"]

    if session_state.ffmi_goal.get("target_percentile"):
        compact["fg"] = {"tp": session_state.ffmi_goal["target_percentile"]}
        if (
            session_state.ffmi_goal.get("target_age")
            and session_state.ffmi_goal["target_age"] != "?"
        ):
            compact["fg"]["ta"] = session_state.ffmi_goal["target_age"]

    return compact


def test_20_scans():
    """Test URL sharing with 20 scans."""
    print("Testing URL sharing with 20 scans...")

    # Create mock session state with 20 scans
    session_state = MockSessionState20()

    # Get compact config
    compact_config = get_compact_config(session_state)

    # Convert to JSON string
    json_str = json.dumps(compact_config, separators=(",", ":"))
    print(f"Compact JSON length with 20 scans: {len(json_str)} characters")

    # Encode as base64
    encoded_data = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")
    print(f"Base64 encoded length: {len(encoded_data)} characters")

    # Create URL
    base_url = "http://localhost:8501"
    share_url = f"{base_url}?data={urllib.parse.quote(encoded_data)}"
    print(f"Share URL length: {len(share_url)} characters")

    # Check if URL is within reasonable limits
    if len(share_url) < 2000:
        print("✅ URL length is within browser limits (< 2000 chars)")
    elif len(share_url) < 8192:
        print("⚠️  URL length is acceptable but long (< 8192 chars)")
    else:
        print("❌ URL length exceeds practical limits (> 8192 chars)")

    # Show sample of the compact format
    print(f"Sample compact config (first scan): {compact_config['s'][0]}")
    print(f"Number of scans: {len(compact_config['s'])}")

    # Calculate what the original JSON would be
    original_json = json.dumps(
        {
            "user_info": session_state.user_info,
            "scan_history": session_state.scan_history,
            "goals": {"almi": session_state.almi_goal, "ffmi": session_state.ffmi_goal},
        },
        separators=(",", ":"),
    )

    compression_ratio = len(json_str) / len(original_json)
    print(
        f"Compression ratio: {compression_ratio:.2f} ({len(original_json)} → {len(json_str)} chars)"
    )
    print(f"Space saved: {len(original_json) - len(json_str)} characters")


if __name__ == "__main__":
    test_20_scans()
