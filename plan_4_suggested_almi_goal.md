# Plan 4: Fix Suggested ALMI Goal Processing Test

Think hard.

For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially.

If you create any temporary new files, scripts, or helper files for iteration, clean up these files by removing them at the end of the task.

Please write a high quality, general purpose solution. Implement a solution that works correctly for all valid inputs, not just the test cases. Do not hard-code values or create solutions that only work for specific test inputs. Instead, implement the actual logic that solves the problem generally.

Focus on understanding the problem requirements and implementing the correct algorithm. Tests are there to verify correctness, not to define the solution. Provide a principled implementation that follows best practices and software design principles.

If the task is unreasonable or infeasible, or if any of the tests are incorrect, please tell me. The solution should be robust, maintainable, and extendable.

## Problem Analysis

**Test**: `tests/unit/test_zscore_calculations.py::TestSuggestedGoalIntegration::test_suggested_almi_goal_processing`

**Failure**: `AssertionError: False is not true` - The test expects `goal_calculations['almi'].get('suggested', False)` to be `True`, but it's returning `False`.

**Root Cause**: The ALMI goal is passed with `'suggested': True` in the input, but this flag is not being preserved or correctly set in the output goal calculations. The processing logic likely removes or doesn't propagate the `suggested` field.

**Expected Behavior**: When a goal has `suggested: True` in the input, the resulting goal calculations should maintain this flag to indicate the goal was system-suggested rather than user-specified.

## Solution Plan

### Step 1: Trace Suggested Flag Propagation
- Examine how the `suggested` flag flows through the goal processing pipeline
- Identify where in `process_scans_and_goal` or related functions the flag might be lost
- Find all locations where goal dictionaries are created or modified

### Step 2: Analyze Goal Processing Functions
- Look at ALMI-specific goal processing functions
- Understand how goal input parameters are transformed into output calculations
- Identify if the `suggested` flag should be preserved or computed differently

### Step 3: Implement Flag Preservation Logic
- Ensure the `suggested` flag from input goals is preserved in output calculations
- Add logic to explicitly set `suggested: True` when goals are system-generated
- Verify that explicit (user-specified) goals have `suggested: False` or no flag

### Step 4: Test Both Suggested and Explicit Goals
- Validate that suggested goals correctly show `suggested: True`
- Ensure explicit goals show `suggested: False`
- Test mixed scenarios where one goal is suggested and another is explicit

### Implementation Strategy

1. **Find goal processing entry points** where ALMI goals are handled
2. **Trace data flow** from input goal dict to output calculations dict
3. **Identify transformation points** where the `suggested` flag could be lost
4. **Add flag preservation logic** at appropriate points in the processing pipeline
5. **Validate consistency** across all goal types (ALMI, FFMI) and scenarios

### Expected Outcome

The test should pass by ensuring that when an ALMI goal is marked as `suggested: True` in the input, this flag is correctly preserved and appears as `suggested: True` in the output goal calculations. This allows the system to distinguish between user-specified goals and system-suggested goals for proper UI display and messaging.