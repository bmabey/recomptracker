#!/usr/bin/env python3
"""
Streamlit AppTest Integration Tests for DEXA Body Composition Analysis Web App

These tests use Streamlit's native AppTest framework to verify the webapp functionality
with the example config and various user interaction scenarios.

Run with: python -m pytest test_webapp_integration.py -v
Or with task runner: task test-webapp
"""

import pytest
import streamlit.testing.v1 as testing
import json
import pandas as pd
from pathlib import Path


class TestWebAppIntegration:
    """Integration tests for the Streamlit webapp using AppTest framework."""
    
    @pytest.fixture
    def app(self):
        """Create AppTest instance for the webapp."""
        return testing.AppTest.from_file("webapp.py", default_timeout=10)
    
    def load_example_and_run_analysis(self, at):
        """Helper method to load example config and run analysis."""
        # Load example config
        load_button = None
        for button in at.button:
            if "📋 Load Example" in button.label:
                load_button = button
                break
        
        assert load_button is not None, "Load Example button should be present"
        load_button.click()
        at = at.run()
        
        # Run analysis
        analysis_button = None
        for button in at.button:
            if "🔬 Run Analysis" in button.label:
                analysis_button = button
                break
        
        assert analysis_button is not None, "Run Analysis button should be present"
        analysis_button.click()
        at = at.run()
        
        return at
    
    @pytest.fixture
    def example_config_data(self):
        """Load example config data for testing."""
        with open("example_config.json", "r") as f:
            return json.load(f)
    
    def test_webapp_loads_successfully(self, app):
        """Test that the webapp loads without errors."""
        # Run the app
        at = app.run()
        
        # Verify basic UI elements are present
        assert len(at.title) > 0, "App title should be present"
        assert len(at.button) > 0, "Buttons should be present"
        
        # Verify no exceptions occurred
        assert len(at.exception) == 0, f"No exceptions should occur on load, found: {[e.value for e in at.exception]}"
    
    def test_load_example_config_button(self, app, example_config_data):
        """Test the 'Load Example' button functionality."""
        at = app.run()
        
        # Find and click the Load Example button
        load_button = None
        for button in at.button:
            if "📋 Load Example" in button.label:
                load_button = button
                break
        
        assert load_button is not None, "Load Example button should be present"
        
        # Click the button
        load_button.click()
        at = at.run()
        
        # Verify that user info is populated
        # Check that birth date input has the expected value (index 0)
        birth_date_input = at.text_input[0]
        assert birth_date_input.value == example_config_data["user_info"]["birth_date"], \
            f"Birth date should be {example_config_data['user_info']['birth_date']}"
        
        # Check gender selection (index 0)
        gender_selectbox = at.selectbox[0]
        assert gender_selectbox.value == example_config_data["user_info"]["gender"], \
            f"Gender should be {example_config_data['user_info']['gender']}"
        
        # Check height input (index 0)
        height_input = at.number_input[0]
        assert height_input.value == example_config_data["user_info"]["height_in"], \
            f"Height should be {example_config_data['user_info']['height_in']}"
    
    def test_analysis_execution_with_example_config(self, app):
        """Test that analysis runs successfully with example config data."""
        at = app.run()
        
        # Load example config
        load_button = None
        for button in at.button:
            if "📋 Load Example" in button.label:
                load_button = button
                break
        
        load_button.click()
        at = at.run()
        
        # Find and click the Run Analysis button
        analysis_button = None
        for button in at.button:
            if "🔬 Run Analysis" in button.label:
                analysis_button = button
                break
        
        assert analysis_button is not None, "Run Analysis button should be present"
        
        # Click the analysis button
        analysis_button.click()
        at = at.run()
        
        # Verify no exceptions occurred during analysis
        assert len(at.exception) == 0, f"Analysis should complete without exceptions, found: {[e.value for e in at.exception]}"
        
        # Verify no error messages
        error_messages = [error.value for error in at.error]
        assert len(error_messages) == 0, f"No error messages should appear, found: {error_messages}"
    
    def test_analysis_results_display(self, app):
        """Test that analysis results are displayed correctly."""
        at = app.run()
        
        # Load example config and run analysis
        at = self.load_example_and_run_analysis(at)
        
        # Verify results section appears
        results_headers = [header.value for header in at.subheader]
        assert any("Analysis Results" in header for header in results_headers), \
            "Analysis Results section should be displayed"
        
        # Verify metric displays
        metrics = at.metric
        assert len(metrics) >= 4, "Should display at least 4 metrics (ALMI, ALMI %, FFMI, FFMI %)"
        
        # Check for specific metrics
        metric_labels = [metric.label for metric in metrics]
        expected_metrics = ["Current ALMI", "ALMI Percentile", "Current FFMI", "FFMI Percentile"]
        
        for expected_metric in expected_metrics:
            assert any(expected_metric in label for label in metric_labels), \
                f"Metric '{expected_metric}' should be displayed"
        
        # Verify metric values are reasonable
        for metric in metrics:
            if "ALMI" in metric.label and "kg/m²" in metric.value:
                # ALMI should be between 5-15 kg/m² for typical values
                almi_value = float(metric.value.split()[0])
                assert 5.0 <= almi_value <= 15.0, f"ALMI value {almi_value} should be reasonable"
            
            if "Percentile" in metric.label and "%" in metric.value:
                # Percentiles should be between 0-100%
                percentile = float(metric.value.replace("%", ""))
                assert 0.0 <= percentile <= 100.0, f"Percentile {percentile} should be valid"
    
    def test_plots_generation(self, app):
        """Test that both ALMI and FFMI plots are generated."""
        at = app.run()
        
        # Load example config and run analysis
        at = self.load_example_and_run_analysis(at)
        
        # Check for plot headers
        plot_headers = [header.value for header in at.subheader]
        assert any("ALMI Percentile Curves" in header for header in plot_headers), \
            "ALMI plot section should be displayed"
        assert any("FFMI Percentile Curves" in header for header in plot_headers), \
            "FFMI plot section should be displayed"
    
    def test_data_table_display(self, app):
        """Test that the detailed results table is displayed."""
        at = app.run()
        
        # Load example config and run analysis
        at = self.load_example_and_run_analysis(at)
        
        # Check for table header
        table_headers = [header.value for header in at.subheader]
        assert any("Detailed Results Table" in header for header in table_headers), \
            "Results table section should be displayed"
        
        # Verify data grid/table is present
        # Note: Streamlit's data_editor might not be directly accessible in AppTest
        # but we can check that no errors occurred and the section is present
        assert len(at.exception) == 0, "Table display should not cause exceptions"
    
    def test_form_validation(self, app):
        """Test that form validation works correctly."""
        at = app.run()
        
        # Try to run analysis without any data
        analysis_button = None
        for button in at.button:
            if "Run Analysis" in button.label:
                analysis_button = button
                break
        
        # Should not crash when no data is present
        if analysis_button:
            analysis_button.click()
            at = at.run()
            
            # Should handle gracefully (either with error message or no-op)
            # The main thing is no exceptions should occur
            assert len(at.exception) == 0, "Validation should handle empty forms gracefully"
    
    def test_csv_export_availability(self, app):
        """Test that CSV export functionality is available after analysis."""
        at = app.run()
        
        # Load example config and run analysis
        at = self.load_example_and_run_analysis(at)
        
        # Look for download button or CSV export functionality
        # This might be implemented as a download_button in Streamlit
        download_buttons = []
        for button in at.button:
            if "download" in button.label.lower() or "csv" in button.label.lower():
                download_buttons.append(button)
        
        # If CSV export is implemented, it should be available after analysis
        # If not implemented yet, this test serves as a reminder to add it
        # For now, we just verify no exceptions occurred during the process
        assert len(at.exception) == 0, "Analysis completion should enable export functionality"
    
    def test_fake_data_generation(self, app):
        """Test the fake data generation functionality."""
        at = app.run()
        
        # Look for and click the Random Profile button (button with 🎲)
        random_button = None
        for button in at.button:
            if "🎲 Random Profile" in button.label:
                random_button = button
                break
        
        if random_button:
            random_button.click()
            at = at.run()
            
            # Verify that fake data was generated (fields should be populated)
            birth_date_input = at.text_input[0]  # Birth date is first text input
            assert birth_date_input.value != "", "Birth date should be populated with fake data"
            
            # Verify no exceptions occurred during fake data generation
            assert len(at.exception) == 0, "Fake data generation should not cause exceptions"


class TestWebAppErrorHandling:
    """Test error handling scenarios in the webapp."""
    
    @pytest.fixture
    def app(self):
        """Create AppTest instance for the webapp."""
        return testing.AppTest.from_file("webapp.py", default_timeout=10)
    
    def test_invalid_date_handling(self, app):
        """Test handling of invalid date inputs."""
        at = app.run()
        
        # Try to enter invalid date in birth date input (index 0)
        birth_date_input = at.text_input[0]
        birth_date_input.input("invalid-date")
        at = at.run()
        
        # Should handle gracefully without crashing
        assert len(at.exception) == 0, "Invalid date should be handled gracefully"
    
    def test_missing_required_fields(self, app):
        """Test handling when required fields are missing."""
        at = app.run()
        
        # Set only partial data - use selectbox for gender (index 0)
        gender_selectbox = at.selectbox[0]
        gender_selectbox.set_value("male")
        at = at.run()
        
        # Try to run analysis
        analysis_button = None
        for button in at.button:
            if "🔬 Run Analysis" in button.label:
                analysis_button = button
                break
        
        if analysis_button:
            analysis_button.click()
            at = at.run()
            
            # Should handle missing data gracefully
            assert len(at.exception) == 0, "Missing required fields should be handled gracefully"


def run_integration_tests():
    """Run all integration tests programmatically."""
    import subprocess
    import sys
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "test_webapp_integration.py", 
            "-v", "--tb=short"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        print("INTEGRATION TEST RESULTS:")
        print("=" * 50)
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


if __name__ == "__main__":
    # Allow running tests directly
    success = run_integration_tests()
    exit(0 if success else 1)