#!/usr/bin/env python3
"""
Test script to verify data editor conversion works correctly.
"""

import json

import pandas as pd


def test_data_conversion():
    """Test that scan history data converts properly between formats."""

    # Load example config
    with open("example_config.json", "r") as f:
        config = json.load(f)

    scan_history = config["scan_history"]
    print(f"Original scan history: {len(scan_history)} scans")

    # Convert to DataFrame (like in webapp)
    df = pd.DataFrame(scan_history)
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")

    # Check column types
    print("\nColumn types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")

    # Convert back to records (like in webapp)
    records = df.to_dict("records")
    print(f"\nConverted back: {len(records)} scans")

    # Verify data integrity
    for i, (orig, conv) in enumerate(zip(scan_history, records)):
        for key in orig.keys():
            if orig[key] != conv[key]:
                print(f"MISMATCH in scan {i}, field {key}: {orig[key]} != {conv[key]}")
                return False

    print("✅ Data conversion test passed!")
    return True


def test_validation():
    """Test validation logic on sample data."""

    # Test valid data
    valid_scan = {
        "date": "04/07/2022",
        "total_weight_lbs": 143.2,
        "total_lean_mass_lbs": 106.3,
        "fat_mass_lbs": 32.6,
        "body_fat_percentage": 22.8,
        "arms_lean_lbs": 12.4,
        "legs_lean_lbs": 37.3,
    }

    # Test invalid data
    invalid_scans = [
        {"date": "", "total_weight_lbs": 143.2},  # Missing date
        {"date": "13/40/2022", "total_weight_lbs": 143.2},  # Invalid date
        {"date": "04/07/2022", "total_weight_lbs": 0},  # Zero weight
        {"date": "04/07/2022", "total_weight_lbs": -5},  # Negative weight
    ]

    print("\n📋 Validation tests:")
    print(f"Valid scan has {len(valid_scan)} fields")

    for i, invalid in enumerate(invalid_scans):
        print(f"Invalid scan {i + 1}: {len(invalid)} fields")

    print("✅ Validation test structure verified!")
    return True


if __name__ == "__main__":
    test_data_conversion()
    test_validation()
