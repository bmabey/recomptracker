#!/usr/bin/env python3
"""
V1 URL State Schema Validation Utilities

This module provides utilities for validating V1 URL state data against the
JSON Schema specification. This ensures programmatic validation of V1 format
compliance and can be used for testing, runtime validation, and debugging.
"""

import json
import base64
import urllib.parse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import jsonschema
from jsonschema import validate, ValidationError, Draft202012Validator


def load_v1_schema() -> Dict[str, Any]:
    """Load the V1 URL state JSON Schema."""
    schema_path = Path(__file__).parent / "v1-url-state-schema.json"
    
    if not schema_path.exists():
        raise FileNotFoundError(f"V1 schema not found at {schema_path}")
    
    with open(schema_path, 'r') as f:
        return json.load(f)


def validate_v1_config(config: Dict[str, Any], raise_on_error: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Validate a V1 config against the JSON Schema.
    
    Args:
        config: The V1 compact config to validate
        raise_on_error: Whether to raise ValidationError on validation failure
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Raises:
        ValidationError: If validation fails and raise_on_error is True
    """
    try:
        schema = load_v1_schema()
        validate(instance=config, schema=schema, cls=Draft202012Validator)
        return True, None
    except ValidationError as e:
        error_msg = f"V1 schema validation failed: {e.message}"
        if raise_on_error:
            raise ValidationError(error_msg) from e
        return False, error_msg
    except Exception as e:
        error_msg = f"Schema validation error: {str(e)}"
        if raise_on_error:
            raise
        return False, error_msg


def validate_v1_url(url: str, raise_on_error: bool = True) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate a complete V1 URL by decoding and validating against schema.
    
    Args:
        url: Complete URL with V1 state data
        raise_on_error: Whether to raise errors on validation failure
        
    Returns:
        Tuple of (is_valid, error_message, decoded_config)
        
    Raises:
        Various exceptions if decoding/validation fails and raise_on_error is True
    """
    try:
        # Extract data parameter from URL
        if '?data=' not in url:
            error_msg = "URL missing 'data' parameter"
            if raise_on_error:
                raise ValueError(error_msg)
            return False, error_msg, None
        
        encoded_data = url.split('?data=')[1].split('&')[0]  # Handle multiple params
        encoded_data = urllib.parse.unquote(encoded_data)
        
        # Decode base64
        try:
            decoded_bytes = base64.b64decode(encoded_data)
            json_str = decoded_bytes.decode('utf-8')
        except Exception as e:
            error_msg = f"Failed to decode base64 data: {str(e)}"
            if raise_on_error:
                raise ValueError(error_msg) from e
            return False, error_msg, None
        
        # Parse JSON
        try:
            config = json.loads(json_str)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON: {str(e)}"
            if raise_on_error:
                raise ValueError(error_msg) from e
            return False, error_msg, None
        
        # Validate against schema
        is_valid, validation_error = validate_v1_config(config, raise_on_error=raise_on_error)
        
        return is_valid, validation_error, config
        
    except Exception as e:
        if raise_on_error:
            raise
        return False, str(e), None


def get_schema_errors(config: Dict[str, Any]) -> List[str]:
    """
    Get all validation errors for a config without raising exceptions.
    
    Args:
        config: The V1 compact config to validate
        
    Returns:
        List of error messages (empty if valid)
    """
    try:
        schema = load_v1_schema()
        validator = Draft202012Validator(schema)
        errors = []
        
        for error in validator.iter_errors(config):
            path = " -> ".join(str(p) for p in error.absolute_path) if error.absolute_path else "root"
            errors.append(f"Path '{path}': {error.message}")
        
        return errors
    except Exception as e:
        return [f"Schema loading error: {str(e)}"]


def create_v1_config_example() -> Dict[str, Any]:
    """Create a valid V1 config example for testing."""
    return {
        "u": {
            "bd": "04/26/1982",
            "h": 66.0,
            "g": "m",
            "tl": "intermediate"
        },
        "s": [
            ["04/07/2022", 143.2, 106.3, 32.6, 22.8, 12.4, 37.3],
            ["04/01/2023", 154.3, 121.2, 28.5, 18.5, 16.5, 40.4]
        ],
        "ag": {
            "tp": 0.90,
            "ta": 30
        },
        "fg": {
            "tp": 0.75
        }
    }


def validate_and_report(config: Dict[str, Any], config_name: str = "config") -> bool:
    """
    Validate a config and print a detailed report.
    
    Args:
        config: The V1 compact config to validate
        config_name: Name for the config (for reporting)
        
    Returns:
        True if valid, False otherwise
    """
    print(f"\n=== Validating {config_name} ===")
    
    errors = get_schema_errors(config)
    
    if not errors:
        print("✅ Valid V1 config")
        return True
    else:
        print(f"❌ Invalid V1 config ({len(errors)} errors):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        return False


def main():
    """Command-line interface for schema validation."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python v1_schema_validator.py <url_or_config_file>")
        print("\nValidate V1 URL state format against JSON Schema")
        print("Examples:")
        print("  python v1_schema_validator.py 'http://localhost:8501?data=eyJ1Ijp7...'")
        print("  python v1_schema_validator.py config.json")
        print("  python v1_schema_validator.py --example")
        sys.exit(1)
    
    arg = sys.argv[1]
    
    if arg == "--example":
        # Test with example config
        example_config = create_v1_config_example()
        validate_and_report(example_config, "example config")
        
        # Encode to URL and validate that too
        json_str = json.dumps(example_config, separators=(',', ':'))
        encoded_data = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
        test_url = f"http://localhost:8501?data={urllib.parse.quote(encoded_data)}"
        
        print(f"\n=== Validating generated URL ===")
        is_valid, error, decoded_config = validate_v1_url(test_url, raise_on_error=False)
        if is_valid:
            print("✅ Valid V1 URL")
            print(f"URL length: {len(test_url)} characters")
        else:
            print(f"❌ Invalid V1 URL: {error}")
        
    elif arg.startswith('http'):
        # Validate URL
        is_valid, error, config = validate_v1_url(arg, raise_on_error=False)
        if is_valid:
            print("✅ Valid V1 URL")
            print(f"Decoded config has {len(config.get('s', []))} scans")
        else:
            print(f"❌ Invalid V1 URL: {error}")
    
    else:
        # Validate config file
        try:
            with open(arg, 'r') as f:
                config = json.load(f)
            validate_and_report(config, f"file '{arg}'")
        except Exception as e:
            print(f"❌ Error loading file '{arg}': {e}")


if __name__ == "__main__":
    main()