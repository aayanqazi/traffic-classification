#!/bin/bash

# Vehicle Counting Application Setup Script
# This script creates a virtual environment and installs dependencies

echo "🚗 Vehicle Counting Application Setup"
echo "======================================"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv vehicle_counter_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source vehicle_counter_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Test the installation
echo ""
echo "🧪 Testing installation..."
python3 -c "
try:
    from detector import VehicleDetector
    from tracker import VehicleTracker
    from counter import VehicleCounter
    from utils import setup_logging
    print('✅ All modules imported successfully')
    print('✅ Installation verified!')
except Exception as e:
    print(f'❌ Installation test failed: {e}')
    exit 1
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To run the application:"
echo "   source vehicle_counter_env/bin/activate"
echo "   python3 -m streamlit run app.py"
echo ""
echo "💡 Tip: You can also run './run.sh' to start the application"
