# Plan 2: Fix Goal Calculations Structure Test

Think hard.

For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially.

If you create any temporary new files, scripts, or helper files for iteration, clean up these files by removing them at the end of the task.

Please write a high quality, general purpose solution. Implement a solution that works correctly for all valid inputs, not just the test cases. Do not hard-code values or create solutions that only work for specific test inputs. Instead, implement the actual logic that solves the problem generally.

Focus on understanding the problem requirements and implementing the correct algorithm. Tests are there to verify correctness, not to define the solution. Provide a principled implementation that follows best practices and software design principles.

If the task is unreasonable or infeasible, or if any of the tests are incorrect, please tell me. The solution should be robust, maintainable, and extendable.

## Problem Analysis

**Test**: `tests/unit/test_zscore_calculations.py::TestSuggestedGoalIntegration::test_goal_calculations_structure`

**Failure**: The test expects specific field names in the goal calculations dictionary that don't match the current implementation:
- Expected: `target_almi` → Actual: `target_metric_value`
- Expected: `target_z` → Actual: `target_z_score`
- Expected: `alm_to_add_kg`, `estimated_tlm_gain_kg` → These fields exist in actual output

**Root Cause**: The goal calculation structure has evolved, and field names have changed. The test is checking for old field names that are no longer used in the current implementation.

## Solution Plan

### Step 1: Analyze Current Goal Calculation Structure
- Examine the `process_scans_and_goal` function in core.py to understand the current output structure
- Identify all fields that are actually generated in goal calculations
- Compare current structure with test expectations to identify mismatches

### Step 2: Determine Correct Solution Approach
- **Option A**: Update the goal calculation code to include/rename fields to match test expectations
- **Option B**: Update the test to match the current (presumably better) field naming convention
- **Decision Criteria**: Choose based on which naming convention is more logical and consistent

### Step 3: Implement Field Name Standardization
- If updating code: Add field aliases or rename fields to match test expectations
- If updating test: Update expected field names to match current implementation
- Ensure consistency across all goal calculation functions (ALMI and FFMI)

### Step 4: Verify Data Accuracy
- Ensure that the data content is correct regardless of field names
- Validate that all necessary information is present in the goal calculations
- Test that the structure works for both ALMI and FFMI goals

### Implementation Strategy

1. **Analyze actual vs expected structure** by examining both the test and the current implementation
2. **Determine the better naming convention** based on code consistency and clarity
3. **Implement the chosen approach** (either update code or test, whichever is more appropriate)
4. **Validate across all related tests** to ensure consistency

### Expected Outcome

The test should pass with a consistent and well-named goal calculations structure that contains all necessary fields for both ALMI and FFMI goal processing. The solution should maintain backward compatibility where possible and use clear, descriptive field names.