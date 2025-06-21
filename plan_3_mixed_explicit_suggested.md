# Plan 3: Fix Mixed Explicit and Suggested Goals Test

Think hard.

For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially.

If you create any temporary new files, scripts, or helper files for iteration, clean up these files by removing them at the end of the task.

Please write a high quality, general purpose solution. Implement a solution that works correctly for all valid inputs, not just the test cases. Do not hard-code values or create solutions that only work for specific test inputs. Instead, implement the actual logic that solves the problem generally.

Focus on understanding the problem requirements and implementing the correct algorithm. Tests are there to verify correctness, not to define the solution. Provide a principled implementation that follows best practices and software design principles.

If the task is unreasonable or infeasible, or if any of the tests are incorrect, please tell me. The solution should be robust, maintainable, and extendable.

## Problem Analysis

**Test**: `tests/unit/test_zscore_calculations.py::TestSuggestedGoalIntegration::test_mixed_explicit_and_suggested`

**Failure**: `KeyError: 'ffmi'` - The FFMI goal calculation is missing from the results entirely.

**Root Cause**: The output shows "Already at 94.9th percentile for FFMI, which is above target 85th percentile" and "no goal suggestion needed!" This indicates that when a user is already above their target percentile, the goal processing is being completely skipped instead of including the goal with appropriate status information.

**Expected Behavior**: Even when a user is already above their target, the goal calculation should still be included in the results with appropriate messaging about already meeting/exceeding the goal.

## Solution Plan

### Step 1: Identify Goal Processing Logic
- Examine the `process_scans_and_goal` function to understand how FFMI goals are processed
- Find where the "already above target" logic is implemented
- Identify why goals are being omitted instead of included with status

### Step 2: Understand Current vs Expected Behavior
- Determine if tests expect goals to always be included regardless of current status
- Analyze whether "already above target" should return goal data with status flags
- Review how this affects both explicit and suggested goal types

### Step 3: Implement Consistent Goal Inclusion
- Modify goal processing to always return goal calculation data
- Add status fields to indicate when targets are already met/exceeded
- Ensure both ALMI and FFMI goals follow consistent behavior patterns
- Include appropriate messaging while maintaining goal structure

### Step 4: Handle Edge Cases
- Test scenarios where user is at exactly the target percentile
- Verify behavior when user is slightly below target
- Ensure suggested goals work correctly in all scenarios

### Implementation Strategy

1. **Locate goal processing functions** that handle FFMI calculations
2. **Identify skip conditions** that cause goals to be omitted from results
3. **Modify logic** to always include goal data with appropriate status indicators
4. **Add status fields** like `already_achieved`, `target_met`, or similar
5. **Update messaging** to be informational rather than exclusionary

### Expected Outcome

The test should pass by ensuring that FFMI goal calculations are always included in results, even when the user is already above their target percentile. The goal data should include status information indicating the user has already achieved or exceeded their target, while maintaining all the structural data the test expects.