# Plan 1: Fix WebApp Load Example Button Test

Think hard.

For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially.

If you create any temporary new files, scripts, or helper files for iteration, clean up these files by removing them at the end of the task.

Please write a high quality, general purpose solution. Implement a solution that works correctly for all valid inputs, not just the test cases. Do not hard-code values or create solutions that only work for specific test inputs. Instead, implement the actual logic that solves the problem generally.

Focus on understanding the problem requirements and implementing the correct algorithm. Tests are there to verify correctness, not to define the solution. Provide a principled implementation that follows best practices and software design principles.

If the task is unreasonable or infeasible, or if any of the tests are incorrect, please tell me. The solution should be robust, maintainable, and extendable.

## Problem Analysis

**Test**: `tests/integration/test_webapp_integration.py::TestWebAppIntegration::test_load_example_config_button`

**Failure**: The test expects the height input field to be `at.number_input[0]` with value `66.0`, but instead it's getting a NumberInput with label 'Target ALMI Percentile' and value `0.9`.

**Root Cause**: The test is using hardcoded indices to access UI elements (`at.number_input[0]`), but the webapp UI layout has changed, causing the height input to no longer be at index 0. The ALMI percentile input is now at index 0 instead.

## Solution Plan

### Step 1: Analyze UI Layout Structure
- Examine the webapp.py file to understand the current UI element ordering
- Identify where the height input field is positioned relative to other number inputs
- Understand how the "Load Example" button populates the session state

### Step 2: Fix Test Element Selection Strategy
- Replace hardcoded index-based selection with label-based or more robust selection
- Update the test to find UI elements by their labels or other identifying characteristics
- Ensure the test selects the correct height input field regardless of its position in the layout

### Step 3: Validate Load Example Functionality
- Verify that the "Load Example" button correctly populates all user info fields
- Ensure the example configuration data matches what the test expects
- Test that the height value is properly set in the UI after button click

### Step 4: Make Test More Robust
- Update all hardcoded element selections in the test (birth_date_input, gender_selectbox, height_input)
- Use element labels or other stable identifiers instead of array indices
- Add defensive checks to ensure elements exist before accessing their values

### Implementation Strategy

1. **Examine webapp.py** to understand the current UI structure and identify the height input field
2. **Update test selectors** to use label-based selection instead of index-based selection
3. **Verify Load Example button** functionality works correctly with the updated selectors
4. **Test edge cases** to ensure the solution is robust and handles UI changes gracefully

### Expected Outcome

The test should pass by correctly identifying and verifying the height input field value after the "Load Example" button is clicked, regardless of its position in the UI layout.