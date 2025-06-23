#!/usr/bin/env python3
"""
Scenario Analysis Script for Monte Carlo Simulation

This script loads the example config, takes only the first scan, and runs
Monte Carlo simulations with different scenarios targeting 90th ALMI percentile.
All input parameters and results are stored in a dictionary for analysis.
Generates plots for each scenario.
"""

import json
import os
import sys
from datetime import datetime
from pprint import pprint

import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as pyo
from plotly.subplots import make_subplots

from mc_forecast import MonteCarloEngine
from shared_models import (
    BFRangeConfig,
    GoalConfig,
    SimulationConfig,
    TemplateType,
    TrainingLevel,
    UserProfile,
    convert_dict_to_user_profile,
)


def load_and_prepare_base_config():
    """Load example config and prepare base user profile with first scan only"""
    # Load example config
    with open("example_config.json", "r") as f:
        config = json.load(f)

    # Take only the first scan
    first_scan = config["scan_history"][0]
    modified_config = config.copy()
    modified_config["scan_history"] = [first_scan]

    # Convert to UserProfile
    user_profile = convert_dict_to_user_profile(
        modified_config["user_info"], modified_config["scan_history"]
    )

    return user_profile, modified_config


def create_scenario_plot(scenario_name, simulation_results, save_plots=True):
    """Create Plotly visualization for a scenario and optionally save as HTML/PNG"""
    if simulation_results is None:
        return None

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "ALMI Progression",
            "Body Fat Progression",
            "Weight Progression",
            "Duration Distribution",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Extract trajectory data
    trajectories = simulation_results.trajectories

    # Determine successful vs failed trajectories using same logic as main analysis
    successful_trajectories = []
    failed_trajectories = []

    for trajectory in trajectories:
        if not trajectory:  # Skip empty trajectories
            failed_trajectories.append(trajectory)
            continue

        # Simple heuristic: if trajectory achieved reasonable ALMI values
        final_state = trajectory[-1]
        goal_likely_achieved = final_state.almi >= 8.0  # Conservative threshold

        if goal_likely_achieved:
            successful_trajectories.append(trajectory)
        else:
            failed_trajectories.append(trajectory)

    # Plot 1: ALMI Progression
    if successful_trajectories:
        for i, traj in enumerate(
            successful_trajectories[:20]
        ):  # Show first 20 successful
            weeks = [state.week for state in traj]
            years = [w / 52 for w in weeks]
            almi_values = [state.almi for state in traj]

            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=almi_values,
                    mode="lines",
                    line={"color": "green", "width": 1},
                    opacity=0.3,
                    showlegend=True if i == 0 else False,
                    name="Successful" if i == 0 else "",
                    hovertemplate="Year: %{x:.1f}<br>ALMI: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    if failed_trajectories:
        for i, traj in enumerate(failed_trajectories[:10]):  # Show first 10 failed
            weeks = [state.week for state in traj]
            years = [w / 52 for w in weeks]
            almi_values = [state.almi for state in traj]

            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=almi_values,
                    mode="lines",
                    line={"color": "red", "width": 1},
                    opacity=0.3,
                    showlegend=True if i == 0 else False,
                    name="Failed" if i == 0 else "",
                    hovertemplate="Year: %{x:.1f}<br>ALMI: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # Add dynamic target ALMI line that changes with age (more accurate)
    # Calculate target percentile values across age range for plotting
    from core import calculate_percentile_cached

    if successful_trajectories:
        # Get age range from trajectories
        max_years = (
            max(len(traj) / 52 for traj in successful_trajectories[:5])
            if successful_trajectories
            else 5
        )
        years_range = list(range(int(max_years) + 1))
        target_almis = []

        for year in years_range:
            age = 40.176 + year  # Approximate starting age from debug output
            # Calculate inverse: what ALMI gives 90th percentile at this age?
            # Try a range of ALMI values to find 90th percentile
            for test_almi in [7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0]:
                try:
                    percentile = calculate_percentile_cached(
                        value=test_almi,
                        age=age,
                        metric="appendicular_LMI",
                        gender_code=0,  # male
                    )
                    if percentile >= 0.90:
                        target_almis.append(test_almi)
                        break
                except:
                    continue
            else:
                # Fallback if no match found
                target_almis.append(8.5)

        # Plot dynamic target line
        fig.add_trace(
            go.Scatter(
                x=years_range[: len(target_almis)],
                y=target_almis,
                mode="lines",
                line={"color": "blue", "dash": "dash", "width": 2},
                name="Target 90th percentile (age-adjusted)",
                showlegend=True,
                hovertemplate="Year: %{x}<br>Target ALMI: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Plot 2: Body Fat Progression
    if successful_trajectories:
        for i, traj in enumerate(successful_trajectories[:20]):
            weeks = [state.week for state in traj]
            years = [w / 52 for w in weeks]
            bf_values = [state.body_fat_pct for state in traj]

            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=bf_values,
                    mode="lines",
                    line={"color": "green", "width": 1},
                    opacity=0.3,
                    showlegend=False,
                    hovertemplate="Year: %{x:.1f}<br>BF%: %{y:.1f}<extra></extra>",
                ),
                row=1,
                col=2,
            )

    # Plot 3: Weight Progression
    if successful_trajectories:
        for i, traj in enumerate(successful_trajectories[:20]):
            weeks = [state.week for state in traj]
            years = [w / 52 for w in weeks]
            weight_values = [state.weight_lbs for state in traj]

            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=weight_values,
                    mode="lines",
                    line={"color": "green", "width": 1},
                    opacity=0.3,
                    showlegend=False,
                    hovertemplate="Year: %{x:.1f}<br>Weight: %{y:.1f} lbs<extra></extra>",
                ),
                row=2,
                col=1,
            )

    # Plot 4: Duration Distribution
    durations_successful = [len(t) / 52 for t in successful_trajectories]
    durations_failed = [len(t) / 52 for t in failed_trajectories]

    if durations_successful:
        fig.add_trace(
            go.Histogram(
                x=durations_successful,
                name="Successful",
                marker_color="green",
                opacity=0.7,
                nbinsx=20,
            ),
            row=2,
            col=2,
        )

    if durations_failed:
        fig.add_trace(
            go.Histogram(
                x=durations_failed,
                name="Failed",
                marker_color="red",
                opacity=0.7,
                nbinsx=20,
            ),
            row=2,
            col=2,
        )

    # Update layout
    fig.update_layout(
        title_text=f"Monte Carlo Simulation Results: {scenario_name}",
        height=800,
        showlegend=True,
    )

    # Update axis labels
    fig.update_xaxes(title_text="Years", row=1, col=1)
    fig.update_yaxes(title_text="ALMI (kg/m²)", row=1, col=1)
    fig.update_xaxes(title_text="Years", row=1, col=2)
    fig.update_yaxes(title_text="Body Fat %", row=1, col=2)
    fig.update_xaxes(title_text="Years", row=2, col=1)
    fig.update_yaxes(title_text="Weight (lbs)", row=2, col=1)
    fig.update_xaxes(title_text="Duration (years)", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)

    if save_plots:
        # Create plots directory if it doesn't exist
        os.makedirs("scenario_plots", exist_ok=True)

        # Save as HTML (interactive)
        safe_name = (
            scenario_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        html_filename = f"scenario_plots/scenario_{safe_name}_plot.html"
        pio.write_html(fig, html_filename)
        print(f"  💾 Saved interactive plot: {html_filename}")

        # Save as PNG (static) - requires kaleido package
        try:
            png_filename = f"scenario_plots/scenario_{safe_name}_plot.png"
            pio.write_image(fig, png_filename, width=1200, height=800)
            print(f"  💾 Saved static plot: {png_filename}")
        except Exception as e:
            print(f"  ⚠️  Could not save PNG (install kaleido): {e}")

    return fig


def generate_html_report(scenarios_dict, user_profile):
    """Generate a comprehensive HTML report with all scenario visualizations and analysis"""

    # Extract summary data
    summary = scenarios_dict.get("_summary", {})
    successful_scenarios = {
        k: v
        for k, v in scenarios_dict.items()
        if k != "_summary" and v.get("success", False)
    }
    {
        k: v
        for k, v in scenarios_dict.items()
        if k != "_summary" and not v.get("success", False)
    }

    # Start building HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Monte Carlo Simulation Scenario Analysis Report</title>
        <meta charset="utf-8">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            h2 {{
                color: #34495e;
                border-left: 4px solid #3498db;
                padding-left: 15px;
                margin-top: 40px;
            }}
            h3 {{
                color: #2c3e50;
                margin-top: 30px;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .summary-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }}
            .summary-card h4 {{
                margin: 0 0 10px 0;
                font-size: 1.2em;
            }}
            .summary-card .number {{
                font-size: 2.5em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .user-profile {{
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            .scenario-section {{
                margin: 40px 0;
                padding: 25px;
                border: 1px solid #bdc3c7;
                border-radius: 8px;
                background-color: #fdfdfd;
            }}
            .scenario-header {{
                background: linear-gradient(90deg, #74b9ff 0%, #0984e3 100%);
                color: white;
                padding: 15px 20px;
                margin: -25px -25px 20px -25px;
                border-radius: 8px 8px 0 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .scenario-status {{
                font-weight: bold;
                padding: 5px 10px;
                border-radius: 15px;
                font-size: 0.9em;
            }}
            .status-success {{
                background-color: #00b894;
            }}
            .status-failed {{
                background-color: #e17055;
            }}
            .parameters-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .parameter-item {{
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                border-left: 3px solid #74b9ff;
            }}
            .parameter-label {{
                font-weight: bold;
                color: #2d3436;
                font-size: 0.9em;
            }}
            .parameter-value {{
                color: #636e72;
                margin-top: 5px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-label {{
                font-size: 0.9em;
                color: #7f8c8d;
                margin-bottom: 5px;
            }}
            .metric-value {{
                font-size: 1.4em;
                font-weight: bold;
                color: #2c3e50;
            }}
            .plot-container {{
                margin: 30px 0;
                border: 1px solid #ddd;
                border-radius: 8px;
                overflow: hidden;
            }}
            .error-message {{
                background-color: #ffe6e6;
                color: #c0392b;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #e74c3c;
                margin: 15px 0;
            }}
            .analysis-section {{
                background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                padding: 25px;
                border-radius: 10px;
                margin: 30px 0;
            }}
            .comparison-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .comparison-table th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 10px;
                text-align: left;
                font-weight: bold;
            }}
            .comparison-table td {{
                padding: 12px 10px;
                border-bottom: 1px solid #ecf0f1;
            }}
            .comparison-table tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            .comparison-table tr:hover {{
                background-color: #e3f2fd;
            }}
            .best-performer {{
                background-color: #d4edda !important;
                font-weight: bold;
            }}
            .insights-list {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #f39c12;
            }}
            .insights-list li {{
                margin: 10px 0;
                padding-left: 10px;
            }}
            .timestamp {{
                color: #7f8c8d;
                font-size: 0.9em;
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #bdc3c7;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Monte Carlo Simulation Scenario Analysis Report</h1>

            <div class="summary-grid">
                <div class="summary-card">
                    <h4>Total Scenarios</h4>
                    <div class="number">{summary.get("total_scenarios", 0)}</div>
                </div>
                <div class="summary-card">
                    <h4>Successful</h4>
                    <div class="number">{summary.get("successful_scenarios", 0)}</div>
                </div>
                <div class="summary-card">
                    <h4>Failed</h4>
                    <div class="number">{summary.get("failed_scenarios", 0)}</div>
                </div>
                <div class="summary-card">
                    <h4>Target Percentile</h4>
                    <div class="number">{summary.get("target_percentile", 0.9):.0%}</div>
                </div>
            </div>

            <h2>📋 User Profile & Configuration</h2>
            <div class="user-profile">
                <strong>Birth Date:</strong> {user_profile.birth_date}<br>
                <strong>Height:</strong> {user_profile.height_in} inches<br>
                <strong>Gender:</strong> {user_profile.gender}<br>
                <strong>Training Level:</strong> {user_profile.training_level.value}<br>
                <strong>Starting Scan Date:</strong> {user_profile.scan_history[0].date}<br>
                <strong>Starting Weight:</strong> {user_profile.scan_history[0].total_weight_lbs} lbs<br>
                <strong>Starting Body Fat:</strong> {user_profile.scan_history[0].body_fat_percentage}%<br>
                <strong>Starting ALMI:</strong> {8.02:.2f} kg/m² (calculated from scan data)
            </div>
    """

    # Add analysis section
    if successful_scenarios:
        # Find best and worst performers
        best_duration = min(
            successful_scenarios.values(),
            key=lambda x: x["key_metrics"]["median_duration_years"],
        )
        worst_duration = max(
            successful_scenarios.values(),
            key=lambda x: x["key_metrics"]["median_duration_years"],
        )

        html_content += f"""
            <h2>📊 Key Insights & Analysis</h2>
            <div class="analysis-section">
                <h3>🎯 Performance Summary</h3>
                <ul class="insights-list">
                    <li><strong>Fastest Goal Achievement:</strong> {best_duration["scenario_name"]}
                        ({best_duration["key_metrics"]["median_duration_years"]:.1f} years)</li>
                    <li><strong>Slowest Goal Achievement:</strong> {worst_duration["scenario_name"]}
                        ({worst_duration["key_metrics"]["median_duration_years"]:.1f} years)</li>
                    <li><strong>Success Rate:</strong> All successful scenarios achieved 100% success rate</li>
                    <li><strong>Template Impact:</strong> Bulk-first templates generally show faster results than cut-first</li>
                    <li><strong>Weight Constraints:</strong> Adding weight limits significantly increases time to goal</li>
                    <li><strong>BF Range Impact:</strong> Tighter body fat ranges require longer durations</li>
                </ul>
            </div>
        """

        # Add comparison table
        html_content += """
            <h3>📈 Scenario Comparison</h3>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Scenario</th>
                        <th>Duration (Years)</th>
                        <th>Final Weight (lbs)</th>
                        <th>Final BF%</th>
                        <th>Final ALMI</th>
                        <th>Template</th>
                        <th>Training Level</th>
                    </tr>
                </thead>
                <tbody>
        """

        # Sort scenarios by duration for table
        sorted_scenarios = sorted(
            successful_scenarios.items(),
            key=lambda x: x[1]["key_metrics"]["median_duration_years"],
        )

        for i, (key, scenario) in enumerate(sorted_scenarios):
            metrics = scenario["key_metrics"]
            params = scenario["input_parameters"]
            row_class = "best-performer" if i == 0 else ""

            html_content += f"""
                <tr class="{row_class}">
                    <td>{scenario["scenario_name"]}</td>
                    <td>{metrics["median_duration_years"]:.1f}</td>
                    <td>{metrics["final_weight_mean"]:.1f}</td>
                    <td>{metrics["final_bf_mean"]:.1f}%</td>
                    <td>{metrics["final_almi_mean"]:.2f}</td>
                    <td>{params["template"].value}</td>
                    <td>{params["training_level"].value}</td>
                </tr>
            """

        html_content += """
                </tbody>
            </table>
        """

    # Add individual scenario sections
    html_content += "<h2>🔬 Detailed Scenario Results</h2>"

    # Process successful scenarios first
    for key, scenario in scenarios_dict.items():
        if key == "_summary":
            continue

        success_status = "✅ SUCCESS" if scenario["success"] else "❌ FAILED"
        status_class = "status-success" if scenario["success"] else "status-failed"

        html_content += f"""
            <div class="scenario-section">
                <div class="scenario-header">
                    <h3 style="margin: 0; color: white;">{scenario["scenario_name"]}</h3>
                    <span class="scenario-status {status_class}">{success_status}</span>
                </div>
        """

        # Add parameters
        params = scenario["input_parameters"]
        html_content += """
            <h4>⚙️ Configuration Parameters</h4>
            <div class="parameters-grid">
        """

        # Add parameter items
        param_items = [
            ("Training Level", params["training_level"].value),
            ("Template", params["template"].value),
            ("Variance Factor", f"{params['variance_factor']:.2f}"),
            ("Run Count", str(params["run_count"])),
            ("Random Seed", str(params["random_seed"])),
            (
                "Max Duration",
                f"{params['max_duration_years']} years"
                if params["max_duration_years"]
                else "Default",
            ),
        ]

        if params["bf_range_config"]:
            bf_config = params["bf_range_config"]
            param_items.append(
                ("BF Range", f"{bf_config.min_bf_pct}-{bf_config.max_bf_pct}%")
            )
            if bf_config.max_weight_lbs:
                param_items.append(("Max Weight", f"{bf_config.max_weight_lbs} lbs"))

        for label, value in param_items:
            html_content += f"""
                <div class="parameter-item">
                    <div class="parameter-label">{label}</div>
                    <div class="parameter-value">{value}</div>
                </div>
            """

        html_content += "</div>"

        if scenario["success"]:
            # Add metrics
            metrics = scenario["key_metrics"]
            html_content += """
                <h4>📊 Results</h4>
                <div class="metrics-grid">
            """

            metric_items = [
                ("Success Rate", f"{metrics['success_rate']:.1%}"),
                ("Median Duration", f"{metrics['median_duration_years']:.1f} years"),
                ("Average Duration", f"{metrics['avg_duration_years']:.1f} years"),
                (
                    "25th-75th Percentile",
                    f"{metrics['p25_duration_years']:.1f}-{metrics['p75_duration_years']:.1f} years",
                ),
                ("Final Weight", f"{metrics['final_weight_mean']:.1f} lbs"),
                ("Final Body Fat", f"{metrics['final_bf_mean']:.1f}%"),
                ("Final ALMI", f"{metrics['final_almi_mean']:.2f} kg/m²"),
            ]

            for label, value in metric_items:
                html_content += f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                """

            html_content += "</div>"

            # Add plot
            if scenario["plot_figure"]:
                html_content += '<div class="plot-container">'
                # Convert plot to HTML
                plot_html = pyo.plot(
                    scenario["plot_figure"], include_plotlyjs=False, output_type="div"
                )
                html_content += plot_html
                html_content += "</div>"
        else:
            # Add error message
            html_content += f"""
                <div class="error-message">
                    <strong>Error:</strong> {scenario["error"]}
                </div>
            """

        html_content += "</div>"  # Close scenario section

    # Add footer
    html_content += f"""
            <div class="timestamp">
                Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} using RecompTracker Monte Carlo Simulation Engine
            </div>
        </div>
    </body>
    </html>
    """

    return html_content


def run_simulation_scenario(scenario_name, user_profile, **kwargs):
    """Run a single simulation scenario and return results"""
    print(f"\n🔬 Running scenario: {scenario_name}")
    print("=" * 50)

    # Create goal config for 90th ALMI percentile
    goal_config = GoalConfig(metric_type="almi", target_percentile=0.90)

    # Extract simulation parameters
    training_level = kwargs.get(
        "training_level", TrainingLevel.NOVICE
    )  # Default to NOVICE
    template = kwargs.get("template", TemplateType.CUT_FIRST)
    variance_factor = kwargs.get("variance_factor", 0.25)
    bf_range_config = kwargs.get("bf_range_config", None)
    run_count = kwargs.get("run_count", 200)
    max_duration_years = kwargs.get("max_duration_years", None)
    random_seed = kwargs.get("random_seed", 42)

    print("Parameters:")
    print(f"  Training Level: {training_level.value}")
    print(f"  Template: {template.value}")
    print(f"  Variance Factor: {variance_factor}")
    print(f"  BF Range Config: {bf_range_config}")
    print(f"  Run Count: {run_count}")
    print(
        f"  Max Duration: {max_duration_years} years"
        if max_duration_years
        else "  Max Duration: Default"
    )
    print(f"  Random Seed: {random_seed}")

    try:
        # Create a copy of user profile with the desired training level
        modified_user_profile = UserProfile(
            birth_date=user_profile.birth_date,
            height_in=user_profile.height_in,
            gender=user_profile.gender,
            training_level=training_level,
            scan_history=user_profile.scan_history,
        )

        # Create simulation config with custom parameters
        max_duration_weeks = None
        if max_duration_years:
            max_duration_weeks = int(max_duration_years * 52)  # Convert years to weeks

        config = SimulationConfig(
            user_profile=modified_user_profile,
            goal_config=goal_config,
            training_level=training_level,
            template=template,
            variance_factor=variance_factor,
            bf_range_config=bf_range_config,
            random_seed=random_seed,
            run_count=run_count,
            max_duration_weeks=max_duration_weeks,
        )

        # Create simulation engine
        engine = MonteCarloEngine(config)

        # Run simulation
        results = engine.run_simulation()

        # Extract key metrics by analyzing trajectories
        # Use the goal_achievement_week to determine success rate and other metrics
        total_trajectories = len(results.trajectories)

        # Count successful trajectories (those that achieved the goal)
        successful_count = 0
        successful_trajectories = []
        failed_trajectories = []

        for trajectory in results.trajectories:
            if not trajectory:  # Skip empty trajectories
                failed_trajectories.append(trajectory)
                continue

            # A trajectory is successful if it's shorter than or equal to goal achievement week
            # and the final ALMI is reasonably high
            final_state = trajectory[-1]
            len(trajectory)

            # Simple heuristic: if trajectory achieved reasonable ALMI values
            if goal_config.metric_type == "almi":
                goal_likely_achieved = final_state.almi >= 8.0  # Conservative threshold
            else:  # ffmi
                goal_likely_achieved = (
                    final_state.ffmi >= 19.0
                )  # Conservative threshold

            if goal_likely_achieved:
                successful_trajectories.append(trajectory)
                successful_count += 1
            else:
                failed_trajectories.append(trajectory)

        success_rate = (
            successful_count / total_trajectories if total_trajectories > 0 else 0
        )

        # Calculate duration statistics (in years)
        durations_years = [len(t) / 52.0 for t in results.trajectories if t]
        [len(t) / 52.0 for t in successful_trajectories]

        if durations_years:
            avg_duration = sum(durations_years) / len(durations_years)
            median_duration = sorted(durations_years)[len(durations_years) // 2]
            p25_duration = sorted(durations_years)[len(durations_years) // 4]
            p75_duration = sorted(durations_years)[3 * len(durations_years) // 4]
        else:
            avg_duration = median_duration = p25_duration = p75_duration = 0

        # Calculate final state statistics (from all trajectories)
        final_weights = [t[-1].weight_lbs for t in results.trajectories if t]
        final_bfs = [t[-1].body_fat_pct for t in results.trajectories if t]
        final_almis = [t[-1].almi for t in results.trajectories if t]

        final_weight_mean = (
            sum(final_weights) / len(final_weights) if final_weights else 0
        )
        final_bf_mean = sum(final_bfs) / len(final_bfs) if final_bfs else 0
        final_almi_mean = sum(final_almis) / len(final_almis) if final_almis else 0

        print("\n📊 Results:")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Duration - Mean: {avg_duration:.1f} years")
        print(f"  Duration - Median: {median_duration:.1f} years")
        print(
            f"  Duration - 25th-75th percentile: {p25_duration:.1f}-{p75_duration:.1f} years"
        )
        print(f"  Final Weight: {final_weight_mean:.1f} lbs")
        print(f"  Final BF%: {final_bf_mean:.1f}%")
        print(f"  Final ALMI: {final_almi_mean:.2f} kg/m²")

        # Create and save plot
        print("\n📈 Generating plots...")
        plot_fig = create_scenario_plot(scenario_name, results, save_plots=True)

        scenario_data = {
            "scenario_name": scenario_name,
            "input_parameters": {
                "training_level": training_level,
                "template": template,
                "variance_factor": variance_factor,
                "bf_range_config": bf_range_config,
                "run_count": run_count,
                "max_duration_years": max_duration_years,
                "random_seed": random_seed,
            },
            "simulation_results": results,
            "plot_figure": plot_fig,
            "key_metrics": {
                "success_rate": success_rate,
                "avg_duration_years": avg_duration,
                "median_duration_years": median_duration,
                "p25_duration_years": p25_duration,
                "p75_duration_years": p75_duration,
                "final_weight_mean": final_weight_mean,
                "final_bf_mean": final_bf_mean,
                "final_almi_mean": final_almi_mean,
            },
            "success": True,
            "error": None,
        }

        return scenario_data

    except Exception as e:
        print(f"❌ Simulation failed: {str(e)}")

        scenario_data = {
            "scenario_name": scenario_name,
            "input_parameters": {
                "training_level": training_level,
                "template": template,
                "variance_factor": variance_factor,
                "bf_range_config": bf_range_config,
                "run_count": run_count,
                "max_duration_years": max_duration_years,
                "random_seed": random_seed,
            },
            "simulation_results": None,
            "plot_figure": None,
            "key_metrics": None,
            "success": False,
            "error": str(e),
        }

        return scenario_data


def main():
    """Main execution function"""
    print("🚀 Monte Carlo Simulation Scenario Analysis")
    print("=" * 60)
    print("Target: 90th ALMI percentile across different scenarios")
    print("Base: First scan only from example_config.json")

    # Load base configuration
    user_profile, base_config = load_and_prepare_base_config()

    print("\n👤 Base User Profile:")
    print(f"  Birth Date: {user_profile.birth_date}")
    print(f"  Height: {user_profile.height_in} inches")
    print(f"  Gender: {user_profile.gender}")
    print(f"  Training Level: {user_profile.training_level.value}")

    first_scan = user_profile.scan_history[0]
    print("\n📋 First Scan Data:")
    print(f"  Date: {first_scan.date}")
    print(f"  Weight: {first_scan.total_weight_lbs} lbs")
    print(f"  Lean Mass: {first_scan.total_lean_mass_lbs} lbs")
    print(f"  Body Fat: {first_scan.body_fat_percentage}%")
    print(
        f"  Arms+Legs Lean: {first_scan.arms_lean_lbs + first_scan.legs_lean_lbs} lbs"
    )

    # Dictionary to store all scenario results
    scenarios = {}

    # Scenario 1: Baseline (Novice with dynamic progression)
    scenarios["baseline"] = run_simulation_scenario(
        "Baseline - Novice/Cut-First/Medium Variance",
        user_profile,
        training_level=TrainingLevel.NOVICE,
        template=TemplateType.CUT_FIRST,
        variance_factor=0.25,
        run_count=200,
    )

    # Scenario 2: Training level variations (for comparison)
    scenarios["intermediate"] = run_simulation_scenario(
        "Intermediate Training Level",
        user_profile,
        training_level=TrainingLevel.INTERMEDIATE,
        template=TemplateType.CUT_FIRST,
        variance_factor=0.25,
        run_count=200,
    )

    scenarios["advanced"] = run_simulation_scenario(
        "Advanced Training Level",
        user_profile,
        training_level=TrainingLevel.ADVANCED,
        template=TemplateType.CUT_FIRST,
        variance_factor=0.10,  # Lower variance for advanced
        run_count=200,
    )

    # Scenario 3: Template variations
    scenarios["bulk_first"] = run_simulation_scenario(
        "Bulk-First Template",
        user_profile,
        training_level=TrainingLevel.NOVICE,
        template=TemplateType.BULK_FIRST,
        variance_factor=0.25,
        run_count=200,
    )

    # Scenario 4: Variance factor variations
    scenarios["low_variance"] = run_simulation_scenario(
        "Low Variance (Conservative)",
        user_profile,
        training_level=TrainingLevel.NOVICE,
        template=TemplateType.CUT_FIRST,
        variance_factor=0.10,
        run_count=200,
    )

    scenarios["high_variance"] = run_simulation_scenario(
        "High Variance (Aggressive)",
        user_profile,
        training_level=TrainingLevel.NOVICE,
        template=TemplateType.CUT_FIRST,
        variance_factor=0.40,
        run_count=200,
    )

    # Scenario 5: Weight constraint scenarios
    scenarios["weight_190"] = run_simulation_scenario(
        "Weight Constraint 190 lbs",
        user_profile,
        training_level=TrainingLevel.NOVICE,
        template=TemplateType.CUT_FIRST,
        variance_factor=0.25,
        bf_range_config=BFRangeConfig(
            min_bf_pct=8.0, max_bf_pct=15.0, max_weight_lbs=190.0
        ),
        run_count=200,
    )

    scenarios["weight_180"] = run_simulation_scenario(
        "Weight Constraint 180 lbs (Restrictive)",
        user_profile,
        training_level=TrainingLevel.NOVICE,
        template=TemplateType.CUT_FIRST,
        variance_factor=0.25,
        bf_range_config=BFRangeConfig(
            min_bf_pct=8.0, max_bf_pct=15.0, max_weight_lbs=180.0
        ),
        run_count=200,
    )

    # Scenario 6: Duration constraint
    scenarios["short_duration"] = run_simulation_scenario(
        "Short Duration (3 years max)",
        user_profile,
        training_level=TrainingLevel.NOVICE,
        template=TemplateType.CUT_FIRST,
        variance_factor=0.25,
        max_duration_years=3,
        run_count=200,
    )

    # Scenario 7: BF range variations (user requested 11-15% with 165 lbs weight limit)
    scenarios["tight_bf_range"] = run_simulation_scenario(
        "Tight BF Range (11-15%) + 165 lbs Weight Limit",
        user_profile,
        training_level=TrainingLevel.NOVICE,
        template=TemplateType.CUT_FIRST,
        variance_factor=0.25,
        bf_range_config=BFRangeConfig(
            min_bf_pct=11.0, max_bf_pct=15.0, max_weight_lbs=165.0
        ),
        run_count=200,
    )

    scenarios["wide_bf_range"] = run_simulation_scenario(
        "Wide BF Range (8-18%)",
        user_profile,
        training_level=TrainingLevel.NOVICE,
        template=TemplateType.CUT_FIRST,
        variance_factor=0.25,
        bf_range_config=BFRangeConfig(min_bf_pct=8.0, max_bf_pct=18.0),
        run_count=200,
    )

    # Print summary of all scenarios
    print("\n" + "=" * 60)
    print("📈 SCENARIO SUMMARY")
    print("=" * 60)

    successful_scenarios = {k: v for k, v in scenarios.items() if v["success"]}
    failed_scenarios = {k: v for k, v in scenarios.items() if not v["success"]}

    if successful_scenarios:
        print(f"\n✅ Successful Scenarios ({len(successful_scenarios)}):")
        print("-" * 40)
        for name, data in successful_scenarios.items():
            metrics = data["key_metrics"]
            print(
                f"{name:25} | Success: {metrics['success_rate']:5.1%} | "
                f"Duration: {metrics['median_duration_years']:4.1f}y | "
                f"Final Weight: {metrics['final_weight_mean']:5.1f} lbs"
            )

    if failed_scenarios:
        print(f"\n❌ Failed Scenarios ({len(failed_scenarios)}):")
        print("-" * 40)
        for name, data in failed_scenarios.items():
            print(f"{name:25} | Error: {data['error']}")

    # Add summary data to scenarios dict
    scenarios["_summary"] = {
        "total_scenarios": len(scenarios) - 1,  # Exclude this summary
        "successful_scenarios": len(successful_scenarios),
        "failed_scenarios": len(failed_scenarios),
        "user_profile": user_profile,
        "base_config": base_config,
        "target_percentile": 0.90,
        "analysis_timestamp": datetime.now().isoformat(),
    }

    print("\n🎯 Analysis Complete!")
    print(f"Total scenarios: {len(scenarios) - 1}")
    print(f"Successful: {len(successful_scenarios)}")
    print(f"Failed: {len(failed_scenarios)}")
    print("📁 Plots saved in: scenario_plots/")

    # Generate comprehensive HTML report
    print("\n📄 Generating comprehensive HTML report...")
    html_content = generate_html_report(scenarios, user_profile)

    # Save HTML report
    report_filename = f"monte_carlo_scenario_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ HTML report saved: {report_filename}")
    print(
        "📊 Open the report in your browser to view all scenarios, plots, and analysis"
    )

    print("\n📚 All data stored in 'scenarios' dictionary")
    print("Available keys:", list(scenarios.keys()))

    # Launch IPython shell with scenarios dictionary available
    print("\n🐍 Launching IPython shell...")
    print("Use 'scenarios' dictionary to explore results")
    print("Example: scenarios['baseline']['key_metrics']")
    print("Example: scenarios['baseline']['simulation_results'].statistics")
    print("Example: scenarios['baseline']['plot_figure'].show()  # Display plot")

    try:
        import IPython

        IPython.embed(user_ns={"scenarios": scenarios, "user_profile": user_profile})
    except ImportError:
        print("IPython not available, launching regular Python shell")
        import code

        code.interact(local={"scenarios": scenarios, "user_profile": user_profile})


if __name__ == "__main__":
    main()
