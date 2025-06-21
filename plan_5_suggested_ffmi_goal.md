# Plan 5: Fix Suggested FFMI Goal Processing Test

Think hard.

For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially.

If you create any temporary new files, scripts, or helper files for iteration, clean up these files by removing them at the end of the task.

Please write a high quality, general purpose solution. Implement a solution that works correctly for all valid inputs, not just the test cases. Do not hard-code values or create solutions that only work for specific test inputs. Instead, implement the actual logic that solves the problem generally.

Focus on understanding the problem requirements and implementing the correct algorithm. Tests are there to verify correctness, not to define the solution. Provide a principled implementation that follows best practices and software design principles.

If the task is unreasonable or infeasible, or if any of the tests are incorrect, please tell me. The solution should be robust, maintainable, and extendable.

## Problem Analysis

**Test**: `tests/unit/test_zscore_calculations.py::TestSuggestedGoalIntegration::test_suggested_ffmi_goal_processing`

**Failure**: `KeyError: 'ffmi'` - The FFMI goal calculation is completely missing from the results.

**Root Cause**: Similar to Plan 3, the output shows "Already at 94.9th percentile for FFMI, which is above target 85th percentile" and "no goal suggestion needed!" The system is skipping FFMI goal processing entirely when the user is already above their target, rather than including the goal with appropriate status.

**Expected Behavior**: Even for suggested goals where the user is already above target, the goal calculations should be included in the results with appropriate status indicators and the `suggested: True` flag preserved.

## Solution Plan

### Step 1: Identify FFMI Goal Processing Skip Logic
- Find where FFMI goal processing determines "already above target"
- Locate the code that outputs "no goal suggestion needed!" message
- Understand why this causes complete omission from results rather than inclusion with status

### Step 2: Analyze Suggested vs Explicit Goal Handling
- Determine if suggested goals should have different behavior than explicit goals when already above target
- Understand if the `target_age: "?"` parameter affects processing logic
- Review how suggested goals are intended to work in already-achieved scenarios

### Step 3: Implement Consistent Goal Inclusion for Suggested Goals
- Modify FFMI goal processing to always include goal data, even when already achieved
- Preserve the `suggested` flag (either from input or computed based on `target_age: "?"`)
- Add status fields to indicate goal is already achieved while maintaining structure

### Step 4: Handle "?" Target Age for Suggested Goals
- Ensure that `target_age: "?"` is properly recognized as a suggested goal indicator
- Implement logic to calculate suggested target age even when already above target
- Maintain consistency with how ALMI suggested goals are handled

### Implementation Strategy

1. **Locate FFMI goal processing functions** that check current vs target percentile
2. **Find skip conditions** that cause goals to be excluded from results
3. **Modify early exit logic** to include goal data with achievement status
4. **Handle suggested goal indicators** like `target_age: "?"` properly
5. **Ensure flag preservation** for suggested goals throughout the pipeline

### Expected Outcome

The test should pass by ensuring that FFMI goal calculations are always included in results for suggested goals, even when the user is already above their target. The results should include the `suggested: True` flag (or equivalent) and appropriate status information indicating the goal is already achieved, while maintaining the complete goal calculation structure expected by the test.