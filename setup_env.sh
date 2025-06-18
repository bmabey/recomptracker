#!/bin/bash
# Virtual Environment Setup Script for Body Metrics Project

set -e

echo "🏗️  Setting up virtual environment for Body Metrics..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Option 1: Use pyenv with venv (preferred)
if command_exists pyenv; then
    echo "🐍 Using pyenv with venv..."
    # Get latest Python 3 version from pyenv
    latest_python=$(pyenv versions --bare | grep -E '^3\.[0-9]+\.[0-9]+$' | sort -V | tail -1)
    if [ -z "$latest_python" ]; then
        echo "Installing Python 3.11.0 via pyenv..."
        pyenv install 3.11.0
        latest_python="3.11.0"
    fi
    echo "Using Python $latest_python"
    pyenv local $latest_python
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Pyenv + venv environment created!"
    echo "To activate: source venv/bin/activate"
    echo "To deactivate: deactivate"

# Option 2: Use pipenv if available
elif command_exists pipenv; then
    echo "📦 Using pipenv..."
    pipenv install -r requirements.txt
    echo "✅ Pipenv environment created!"
    echo "To activate: pipenv shell"
    echo "To run analysis: pipenv run python run_analysis.py"

# Option 3: Standard venv fallback
else
    echo "🐍 Using standard venv..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    echo "✅ Virtual environment created!"
    echo "To activate: source venv/bin/activate"
    echo "To deactivate: deactivate"
fi

echo ""
echo "🎯 Quick start:"
echo "  python run_analysis.py --help-config"
echo "  python run_analysis.py example_config.json"