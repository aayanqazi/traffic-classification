# Vehicle Counting Application - Phase 1

An offline desktop application for real-time vehicle detection and counting using computer vision.

## 🚀 Features

- **Offline Operation**: No internet connection required
- **Real-time Detection**: Live vehicle detection using YOLOv8
- **Multi-Object Tracking**: Consistent vehicle tracking with ByteTrack
- **Vehicle Classification**: Counts cars, motorcycles, buses, and trucks
- **Line-Crossing Logic**: Prevents double counting with intelligent line-crossing detection
- **Live Statistics**: Real-time count updates and processing status
- **Video Controls**: Play, pause, and stop functionality
- **Export Results**: Saves results to JSON and CSV formats

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** - Web UI framework
- **OpenCV** - Computer vision operations
- **YOLOv8 (Ultralytics)** - Object detection
- **ByteTrack** - Multi-object tracking
- **PyTorch** - Deep learning framework
- **Supervision** - Computer vision utilities

## 📋 Requirements

- Python 3.10 or higher
- Sufficient RAM for video processing (4GB+ recommended)
- Video files in MP4, AVI, MOV, or MKV format

## 🚀 Installation

1. **Clone or download** this repository

2. **Quick Setup (Recommended)**:
   ```bash
   # Make scripts executable (first time only)
   chmod +x setup.sh run.sh

   # Run setup script
   ./setup.sh
   ```

3. **Manual Setup** (if you prefer step-by-step):
   ```bash
   # Create virtual environment
   python3 -m venv vehicle_counter_env
   source vehicle_counter_env/bin/activate  # On macOS/Linux
   # On Windows: vehicle_counter_env\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

   **Important**: If you get permission errors or "streamlit command not found", always use a virtual environment.

4. **Download YOLOv8 model** (automatic on first run):
   - The application will automatically download `yolov8n.pt` on startup
   - For offline use, ensure the model is cached before going offline

## 🎯 Usage

1. **Start the application**:
   ```bash
   ./run.sh
   ```

   **Alternative methods** (if the above doesn't work):
   ```bash
   # Method 2: Add to PATH (temporary for current session)
   export PATH="$HOME/Library/Python/3.9/bin:$PATH"
   streamlit run app.py

   # Method 3: Use full path
   /Users/arsalansabir/Library/Python/3.9/bin/streamlit run app.py

   # Method 4: Create and use virtual environment (recommended)
   python3 -m venv vehicle_env
   source vehicle_env/bin/activate  # On Windows: vehicle_env\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```

2. **Initialize components**:
   - Click "🔧 Initialize Components" in the sidebar
   - Wait for the success message

3. **Choose input method**:
   - **Upload File**: For videos < 2GB (MP4, AVI, MOV, MKV, DAV)
   - **Local File Path**: For any size video by entering the file path

4. **Upload or specify video**:
   - **Upload**: Click "Upload File" and select your video
   - **Local Path**: Enter full path (e.g., `/Users/username/videos/video.mp4`)
   - DAV files supported (may require conversion for best results)

5. **Start processing**:
   - Click "▶️ Start" to begin video analysis
   - Watch real-time detection and counting
   - Use "⏸️ Pause/Resume" to control playback
   - Use "⏹️ Stop" to end processing

6. **View results**:
   - Live counts displayed in the statistics panel
   - Final results saved to `output/` directory as JSON and CSV

## 📁 Project Structure

```
vehicle-counter-phase1/
├── app.py                 # Main Streamlit application
├── detector.py           # YOLOv8 vehicle detection module
├── tracker.py            # ByteTrack multi-object tracking
├── counter.py            # Vehicle counting with line-crossing logic
├── utils.py              # Utility functions and helpers
├── requirements.txt      # Python dependencies
├── setup.sh              # Automated setup script
├── run.sh                # Automated run script
├── README.md             # This file
├── .streamlit/           # Streamlit configuration
│   └── config.toml       # Upload size configuration
└── output/               # Generated results (created automatically)
    ├── video_results.json
    └── video_results.csv
```

## 🔧 Configuration

### Counting Line
- Default: Horizontal line at y=300 pixels
- Can be modified in future phases for interactive configuration

### Detection Settings
- Vehicle classes: car, motorcycle, bus, truck
- Confidence threshold: 0.5 (configurable in detector.py)
- Tracking parameters: Configurable in tracker.py

## 📊 Output Format

### JSON Output
```json
{
  "vehicle_counts": {
    "car": 15,
    "motorcycle": 3,
    "bus": 2,
    "truck": 5,
    "total": 25
  },
  "timestamp": "2024-01-07T12:00:00"
}
```

### CSV Output
```csv
car,motorcycle,bus,truck,total
15,3,2,5,25
```

## 🐛 Troubleshooting

### Common Issues

1. **"File must be 200.0MB or smaller"**
   - **Solution**: Use the "Local File Path" option for large files, or upload smaller files
   - The app now supports files up to 2GB via configuration

2. **"streamlit command not found"**
   - **Solution**: Use `python3 -m streamlit run app.py` instead of `streamlit run app.py`
   - **Alternative**: Add Python bin to PATH: `export PATH="$HOME/Library/Python/3.9/bin:$PATH"`
   - **Recommended**: Use a virtual environment as shown in the installation steps

2. **Permission errors when importing modules**
   - **Cause**: Global pip install conflicts with macOS permissions
   - **Solution**: Always use a virtual environment:
     ```bash
     python3 -m venv vehicle_counter_env
     source vehicle_counter_env/bin/activate
     pip install -r requirements.txt
     python3 -m streamlit run app.py
     ```

3. **ByteTrack initialization errors (track_thresh parameter)**
   - **Cause**: Supervision library API changed parameter names
   - **Solution**: The code has been updated to use correct parameters. If you get this error, make sure you're using the latest version of the code from this repository.

4. **"use_column_width parameter has been deprecated" warning**
   - **Cause**: Older Streamlit parameter deprecated
   - **Solution**: The code has been updated to use `use_container_width` instead. Update to the latest version of the code.

2. **Model Download Failed**
   - Ensure internet connection during first run
   - Check available disk space

2. **Video Not Loading**
   - Verify video format is supported
   - Check video file integrity
   - Ensure video codec is compatible with OpenCV

3. **Slow Processing**
   - Reduce video resolution in utils.py
   - Use a more powerful CPU/GPU
   - Consider frame skipping for very high FPS videos

4. **Memory Issues**
   - Process shorter video segments
   - Close other memory-intensive applications
   - Reduce tracking buffer size in tracker.py

## 🔄 Future Enhancements (Phase 2+)

- Interactive counting line drawing
- Multiple counting zones
- Vehicle speed calculation
- Direction-based counting
- Heat maps and analytics
- Batch processing capabilities
- Custom model training interface

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📞 Support

For issues or questions, please check the troubleshooting section or create an issue in the repository.
