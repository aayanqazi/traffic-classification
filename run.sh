#!/bin/bash

# Vehicle Counting Application Run Script
# This script activates the virtual environment and starts the application

# Check if virtual environment exists
if [ ! -d "vehicle_counter_env" ]; then
    echo "❌ Virtual environment not found. Please run './setup.sh' first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source vehicle_counter_env/bin/activate

# Verify components can be imported
echo "🧪 Verifying components..."
if ! python3 -c "
try:
    from detector import VehicleDetector
    from tracker import VehicleTracker
    from counter import VehicleCounter
    print('✅ Components verified')
except Exception as e:
    print(f'❌ Component verification failed: {e}')
    exit 1
" 2>/dev/null; then
    echo "❌ Component verification failed. Please run './setup.sh' again."
    exit 1
fi

# Check if streamlit is available
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "❌ Streamlit not found. Please run './setup.sh' to install dependencies."
    exit 1
fi

# Start the application
echo "🚗 Starting Vehicle Counting Application..."
echo "📱 Open http://localhost:8501 in your browser"
echo "🛑 Press Ctrl+C to stop"
echo ""

python3 -m streamlit run app.py
