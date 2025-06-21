#!/usr/bin/env python3
"""
Test runner for optional BF% integration tests.

This script runs the new optional BF% integration tests to verify that
the intelligent health-based BF% targeting works correctly in the webapp.

Run with: python run_optional_bf_tests.py
"""

import subprocess
import sys
from pathlib import Path


def run_optional_bf_tests():
    """Run just the optional BF% integration tests."""
    print("Running Optional BF% Integration Tests")
    print("=" * 50)

    try:
        # Run only the new TestOptionalBFGoalIntegration class
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "test_webapp_integration.py::TestOptionalBFGoalIntegration",
                "-v",
                "--tb=short",
                "--no-header",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print("\n🎉 All Optional BF% Integration Tests PASSED!")
            print("\nFeatures verified:")
            print("  ✅ Optional BF% goal functionality works in webapp")
            print("  ✅ Intelligent health-based targeting is functional")
            print("  ✅ Goal messages display correctly")
            print("  ✅ Different BF% health categories handled properly")
            print("  ✅ Training level affects feasibility calculations")
            print("  ✅ Backward compatibility with explicit BF% goals")
            print("  ✅ Error handling for empty/invalid inputs")
        else:
            print(f"\n❌ Tests failed with return code: {result.returncode}")
            return False

        return result.returncode == 0

    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def run_all_webapp_tests():
    """Run all webapp integration tests."""
    print("\nRunning All Webapp Integration Tests")
    print("=" * 50)

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "test_webapp_integration.py",
                "-v",
                "--tb=short",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )

        print("Test Summary:")
        # Extract summary line from pytest output
        lines = result.stdout.split("\n")
        for line in lines:
            if "passed" in line and (
                "failed" in line or "error" in line or line.strip().endswith("passed")
            ):
                print(f"  {line.strip()}")
                break

        return result.returncode == 0

    except Exception as e:
        print(f"Error running all tests: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Optional BF% Goal Integration")
    print("This verifies that the intelligent health-based BF% targeting")
    print("works correctly in the webapp interface.\n")

    # Run optional BF% specific tests
    bf_tests_passed = run_optional_bf_tests()

    # Optionally run all webapp tests for broader verification
    print("\n" + "=" * 60)

    # Check if running in interactive mode
    try:
        run_all = (
            input("Run all webapp tests for broader verification? (y/n): ")
            .lower()
            .strip()
        )
    except EOFError:
        # Non-interactive mode (like CI/automation)
        run_all = "n"
        print("Running in non-interactive mode - skipping full test suite")

    if run_all == "y":
        all_tests_passed = run_all_webapp_tests()

        if bf_tests_passed and all_tests_passed:
            print("\n🌟 ALL TESTS PASSED! Optional BF% feature is fully integrated.")
        else:
            print("\n⚠️  Some tests failed - review output above for details.")
    else:
        if bf_tests_passed:
            print("\n🌟 Optional BF% Integration Tests PASSED!")
        else:
            print("\n❌ Optional BF% Integration Tests FAILED!")

    exit(0 if bf_tests_passed else 1)
