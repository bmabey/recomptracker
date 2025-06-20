# V1 URL State Specification

**Version**: 1.0  
**Status**: Baseline for backward compatibility  
**Created**: 2025-06-20  

## Overview

This document defines the V1 URL state format used by the RecompTracker Body Composition Analysis web application for sharing user configurations via URLs. This format was established as the baseline for backward compatibility when the application was released publicly.

## Purpose

The URL state format allows users to:
- Share their complete analysis configuration via a single URL
- Save and restore their data across browser sessions
- Collaborate by sharing scan data and goals with others

## Format Structure

The URL state uses a compact JSON format that is base64-encoded and included as a URL parameter. The compact format minimizes URL length while preserving all necessary data.

### URL Parameter Format

```
https://your-domain.com?data=<base64_encoded_compact_json>
```

Where `<base64_encoded_compact_json>` is the base64 encoding of the compact JSON structure described below.

## Compact JSON Schema

### Root Structure

```json
{
  "u": {USER_INFO},
  "s": [SCAN_ARRAY, ...],
  "ag": {ALMI_GOAL},
  "fg": {FFMI_GOAL}
}
```

### User Info Object (`"u"`)

**Required Fields:**
- `"bd"` (string): Birth date in MM/DD/YYYY format
- `"h"` (number): Height in inches (decimal allowed)
- `"g"` (string): Gender code - `"m"` for male, `"f"` for female

**Optional Fields:**
- `"tl"` (string): Training level - `"novice"`, `"intermediate"`, or `"advanced"`
- `"hd"` (string): Height display format (e.g., `"5'10\""`) - preserved for UI display

**Example:**
```json
{
  "bd": "04/26/1982",
  "h": 66.0,
  "g": "m",
  "tl": "intermediate"
}
```

### Scan History Array (`"s"`)

An array of scan arrays, where each scan is represented as a 7-element array in this exact order:

**Scan Array Format:**
`[date, total_weight_lbs, total_lean_mass_lbs, fat_mass_lbs, body_fat_percentage, arms_lean_lbs, legs_lean_lbs]`

**Field Definitions:**
1. `date` (string): Scan date in MM/DD/YYYY format
2. `total_weight_lbs` (number): Total body weight in pounds
3. `total_lean_mass_lbs` (number): Total lean mass in pounds
4. `fat_mass_lbs` (number): Total fat mass in pounds
5. `body_fat_percentage` (number): Body fat percentage (0-100)
6. `arms_lean_lbs` (number): Arms lean mass in pounds
7. `legs_lean_lbs` (number): Legs lean mass in pounds

**Example:**
```json
[
  ["04/07/2022", 143.2, 106.3, 32.6, 22.8, 12.4, 37.3],
  ["04/01/2023", 154.3, 121.2, 28.5, 18.5, 16.5, 40.4]
]
```

### ALMI Goal Object (`"ag"`)

**Required Fields:**
- `"tp"` (number): Target percentile (0.01 to 0.99)

**Optional Fields:**
- `"ta"` (number): Target age in years

**Example:**
```json
{
  "tp": 0.90,
  "ta": 30
}
```

### FFMI Goal Object (`"fg"`)

**Required Fields:**
- `"tp"` (number): Target percentile (0.01 to 0.99)

**Optional Fields:**
- `"ta"` (number): Target age in years

**Example:**
```json
{
  "tp": 0.75,
  "ta": 35
}
```

## Complete Example

### Full Compact JSON
```json
{
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
    "tp": 0.90
  },
  "fg": {
    "tp": 0.75,
    "ta": 35
  }
}
```

### Minified JSON (for encoding)
```json
{"u":{"bd":"04/26/1982","h":66.0,"g":"m","tl":"intermediate"},"s":[["04/07/2022",143.2,106.3,32.6,22.8,12.4,37.3],["04/01/2023",154.3,121.2,28.5,18.5,16.5,40.4]],"ag":{"tp":0.90},"fg":{"tp":0.75,"ta":35}}
```

### Base64 Encoded
```
eyJ1Ijp7ImJkIjoiMDQvMjYvMTk4MiIsImgiOjY2LjAsImciOiJtIiwidGwiOiJpbnRlcm1lZGlhdGUifSwicyI6W1siMDQvMDcvMjAyMiIsMTQzLjIsMTA2LjMsMzIuNiwyMi44LDEyLjQsMzcuM10sWyIwNC8wMS8yMDIzIiwxNTQuMywxMjEuMiwyOC41LDE4LjUsMTYuNSw0MC40XV0sImFnIjp7InRwIjowLjkwfSwiZmciOnsidHAiOjAuNzUsInRhIjozNX19
```

### Complete URL
```
https://localhost:8501?data=eyJ1Ijp7ImJkIjoiMDQvMjYvMTk4MiIsImgiOjY2LjAsImciOiJtIiwidGwiOiJpbnRlcm1lZGlhdGUifSwicyI6W1siMDQvMDcvMjAyMiIsMTQzLjIsMTA2LjMsMzIuNiwyMi44LDEyLjQsMzcuM10sWyIwNC8wMS8yMDIzIiwxNTQuMywxMjEuMiwyOC41LDE4LjUsMTYuNSw0MC40XV0sImFnIjp7InRwIjowLjkwfSwiZmciOnsidHAiOjAuNzUsInRhIjozNX19
```

## Encoding Process

1. **Create Compact JSON**: Convert application state to compact format
2. **Minify JSON**: Remove whitespace using separators `(',', ':')`
3. **Base64 Encode**: Encode UTF-8 bytes to base64 string
4. **URL Encode**: URL-encode the base64 string for safe transmission
5. **Create URL**: Append as `data` parameter to base URL

## Decoding Process

1. **Extract Parameter**: Get `data` parameter from URL
2. **URL Decode**: Decode URL-encoded string
3. **Base64 Decode**: Decode base64 to UTF-8 bytes
4. **Parse JSON**: Parse JSON string to object
5. **Expand Format**: Convert compact format to full application state

## Constraints and Limits

- **Maximum Scans**: 20 scans supported for URL sharing
- **URL Length**: Typical URLs are 500-2000 characters
- **Date Format**: Must be MM/DD/YYYY
- **Numeric Precision**: Preserved as provided (typically 1-2 decimal places)
- **Required Data**: At minimum, user birth date, gender, and one complete scan

## Validation Rules

### User Info
- Birth date must be valid MM/DD/YYYY format
- Height must be positive number between 12 and 120 inches
- Gender must be 'm' or 'f'
- Training level, if provided, must be 'novice', 'intermediate', or 'advanced'

### Scan Data
- Date must be valid MM/DD/YYYY format
- All numeric values must be positive
- Body fat percentage must be between 0 and 100
- All 7 scan fields are required

### Goals
- Target percentile must be between 0.01 and 0.99
- Target age, if provided, must be positive number

## Backward Compatibility Commitment

This V1 specification serves as the baseline for backward compatibility. Future versions of the application must be able to:

1. **Decode V1 URLs**: Always support loading V1 format URLs
2. **Restore State**: Completely restore application state from V1 data
3. **Handle Missing Fields**: Gracefully handle optional fields that may not be present
4. **Provide Defaults**: Supply appropriate defaults for new fields not in V1
5. **Maintain Functionality**: Ensure V1 URLs provide the same core functionality

## Migration Path

If future versions introduce new URL formats (V2, V3, etc.):

1. **Detection**: Version detection should be automatic (e.g., presence of version field)
2. **Fallback**: Default to V1 format if no version specified
3. **Conversion**: Provide utilities to convert V1 to newer formats if needed
4. **Documentation**: Clearly document changes and migration paths

## Implementation Notes

- The compact format reduces URL length by ~40% compared to full JSON
- Field abbreviations are consistent and meaningful
- Array format for scans is space-efficient for multiple scans
- Base64 encoding handles special characters and ensures URL safety
- JSON minification removes unnecessary whitespace

## Testing Requirements

V1 backward compatibility must be verified by:

1. **Round-trip Testing**: Encode → URL → Decode → Verify data integrity
2. **Edge Case Testing**: Test minimal data, maximum scans, special characters
3. **Integration Testing**: Test complete application workflow with V1 URLs
4. **Regression Testing**: Ensure V1 URLs continue working after application updates

## Security Considerations

- URL data is client-side encoded, not encrypted
- No sensitive information should be included beyond body composition data
- URLs may be logged by web servers and browsers
- Consider URL length limits in various browsers and systems