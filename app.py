"""
Vehicle Counting Application - Phase 1
Offline Streamlit application for real-time vehicle detection and counting.
"""

import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
from pathlib import Path
import threading
from queue import Queue
import pandas as pd

# Import our modules
from detector import VehicleDetector
from tracker import VehicleTracker
from counter import VehicleCounter
from utils import (
    setup_logging, save_results_to_json, save_results_to_csv,
    get_video_info, validate_video_file, resize_frame,
    draw_tracking_info, draw_stats_panel
)
from loguru import logger

# Setup logging
setup_logging()


class VideoProcessor:
    """Handles video processing in a separate thread."""

    def __init__(self):
        self.is_processing = False
        self.is_paused = False
        self.should_stop = False
        self.frame_queue = Queue(maxsize=10)
        self.current_frame = None
        self.processing_thread = None

        # Initialize components
        self.detector = None
        self.tracker = None
        self.counter = None

        # Progress tracking (for thread-safe communication)
        self.progress = 0.0
        self.status_message = ""
        self.processing_complete = False

        # Thread-safe line position settings (accessible from background thread)
        self.line_y_position = 50  # Default: 50% from top
        self.line_start_x = 10  # Default: 10% from left
        self.line_end_x = 90  # Default: 90% from left

    def initialize_components(self, confidence_threshold=0.3, model_name='yolo26s.pt'):
        """Initialize detection, tracking, and counting components."""
        try:
            logger.info(f"Initializing with model: {model_name}, confidence: {confidence_threshold}")
            self.detector = VehicleDetector(model_path=model_name, confidence_threshold=confidence_threshold)
            self.tracker = VehicleTracker()
            self.counter = VehicleCounter()
            self.confidence_threshold = confidence_threshold
            
            # Speed recommendation based on model
            if 'yolo26' in model_name.lower():
                speed_note = "✨ YOLO26 - Latest model! 43% faster CPU inference + better motorcycle detection"
            elif 'yolo12' in model_name.lower():
                speed_note = "⚡ YOLOv12 - Attention-based architecture, good accuracy"
            elif 'n.pt' in model_name:
                speed_note = "⚡ Very fast nano model"
            elif 's.pt' in model_name:
                speed_note = "⚡ Good balance of speed and accuracy"
            elif 'm.pt' in model_name:
                speed_note = "⚠️ Slower model - increase detection interval for speed"
            else:
                speed_note = "⚠️ Very slow model - expect reduced FPS"
            
            st.success(f"✅ Components initialized with {model_name}")
            st.info(speed_note)
            return True
        except Exception as e:
            st.error(f"❌ Failed to initialize components: {e}")
            logger.error(f"Initialization failed: {e}")
            return False

    def process_video(self, video_path: str):
        """Process video in a separate thread."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Performance optimization: Process every Nth frame for detection
            # Get settings from session state or use defaults
            detection_interval = getattr(st.session_state, 'detection_interval', 2)
            max_resolution = getattr(st.session_state, 'max_resolution', (640, 480))
            
            frame_count = 0
            processed_count = 0
            start_time = time.time()
            last_detections = []  # Cache detections for skipped frames

            self.status_message = f"Processing frame {frame_count}/{total_frames}"

            while not self.should_stop:
                if self.is_paused:
                    time.sleep(0.1)
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                processed_count += 1

                # Resize frame for processing (use user-selected resolution)
                frame = resize_frame(frame, max_width=max_resolution[0], max_height=max_resolution[1])

                # Detect vehicles (only every Nth frame for performance)
                if processed_count % detection_interval == 0 or processed_count == 1:
                    detections = self.detector.detect_vehicles(frame)
                    last_detections = detections
                else:
                    # Use cached detections for intermediate frames
                    detections = last_detections

                # Track vehicles (every frame for smooth tracking)
                tracked_objects = self.tracker.update(detections, frame.shape[:2])

                # Update counting line position based on user settings
                # Read from thread-safe instance variables (updated by UI in main thread)
                line_y_percent = self.line_y_position
                line_start_x_percent = self.line_start_x
                line_end_x_percent = self.line_end_x

                # Debug: Log line position values every 30 frames to avoid spam
                if processed_count % 30 == 0:
                    logger.debug(f"Using line position: Y={line_y_percent}%, X={line_start_x_percent}%-{line_end_x_percent}%")

                # Update counting line position (must be before update_counts)
                frame = self.counter.draw_counting_line(
                    frame,
                    line_y_percent=line_y_percent,
                    line_start_x_percent=line_start_x_percent,
                    line_end_x_percent=line_end_x_percent
                )

                # Update counts (uses the counting line position set above)
                self.counter.update_counts(tracked_objects)

                # Draw results on frame
                frame = draw_tracking_info(frame, tracked_objects)
                
                # Calculate actual processing FPS
                elapsed = time.time() - start_time
                processing_fps = processed_count / elapsed if elapsed > 0 else 0
                frame = draw_stats_panel(frame, self.counter.get_counts(), processing_fps=processing_fps)

                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Update current frame and progress (thread-safe)
                self.current_frame = frame_rgb
                self.progress = min(frame_count / total_frames, 1.0)
                self.status_message = f"Processing frame {frame_count}/{total_frames} ({processing_fps:.1f} FPS)"

                # No artificial delay - process as fast as possible

            cap.release()

            # Save results
            self.save_results(video_path)

            self.status_message = "✅ Processing completed!"
            self.processing_complete = True

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self.status_message = f"❌ Processing failed: {str(e)}"
            self.processing_complete = True

        finally:
            self.is_processing = False

    def save_results(self, video_path: str):
        """Save counting results to files."""
        try:
            # Create output directory
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            # Get video filename without extension
            video_name = Path(video_path).stem

            # Combine regular counts and directional counts
            results = self.counter.get_counts()
            direction_counts = self.counter.get_direction_counts()
            results['direction_up'] = direction_counts.get('up', 0)
            results['direction_down'] = direction_counts.get('down', 0)

            # Save JSON
            json_path = output_dir / f"{video_name}_results.json"
            save_results_to_json(results, str(json_path))

            # Save CSV
            csv_path = output_dir / f"{video_name}_results.csv"
            save_results_to_csv(results, str(csv_path))

            logger.info(f"Results saved to {json_path} and {csv_path}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def start_processing(self, video_path: str):
        """Start video processing in a separate thread."""
        if self.processing_thread and self.processing_thread.is_alive():
            return

        self.is_processing = True
        self.should_stop = False
        self.is_paused = False
        self.processing_complete = False
        self.progress = 0.0
        self.status_message = "Starting processing..."

        self.processing_thread = threading.Thread(
            target=self.process_video,
            args=(video_path,)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def pause_processing(self):
        """Pause video processing."""
        self.is_paused = not self.is_paused

    def stop_processing(self):
        """Stop video processing."""
        self.should_stop = True
        self.is_processing = False
        self.processing_complete = True
        self.status_message = "⏹️ Processing stopped by user"


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Vehicle Counter - Phase 1",
        page_icon="🚗",
        layout="wide"
    )

    st.title("🚗 Vehicle Counting Application - Phase 1")
    st.markdown("Real-time vehicle detection and counting")

    # Initialize processor
    if 'processor' not in st.session_state:
        st.session_state.processor = VideoProcessor()

    processor = st.session_state.processor

    # Sidebar controls
    st.sidebar.header("🎛️ Controls")

    # Model selection
    st.sidebar.header("🤖 Model Selection")
    model_options = {
        # YOLO26 Models (Latest - January 2026) - RECOMMENDED
        "YOLO26n (Nano - Fastest) ⚡⚡⚡": "yolo26n.pt",
        "YOLO26s (Small - Recommended) ⭐": "yolo26s.pt",
        "YOLO26m (Medium - Balanced)": "yolo26m.pt",
        "YOLO26l (Large - Accurate)": "yolo26l.pt",
        
        # YOLOv12 Models (Alternative - Feb 2025)
        "YOLOv12n (Nano - Fast)": "yolo12n.pt",
        "YOLOv12s (Small - Good)": "yolo12s.pt",
        "YOLOv12m (Medium - Better)": "yolo12m.pt",
        
        # Legacy YOLOv8 Models (Outdated)
        "YOLOv8n (Legacy)": "yolov8n.pt",
        "YOLOv8s (Legacy)": "yolov8s.pt",
        "YOLOv8m (Legacy - Slow)": "yolov8m.pt",
    }
    
    selected_model = st.sidebar.selectbox(
        "Select YOLO Model",
        options=list(model_options.keys()),
        index=1,  # Default to YOLO26s
        help="YOLO26 is 43% faster with better accuracy! Released Jan 2026."
    )
    model_file = model_options[selected_model]
    
    st.sidebar.caption("✨ **NEW**: YOLO26 - 43% faster CPU inference + better small object detection")
    st.sidebar.caption("⚠️ First use will download the selected model (~10-50 MB)")

    # Detection threshold
    st.sidebar.header("🎯 Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Lower threshold = detect more objects (including small motorcycles). Higher = only high-confidence detections."
    )
    st.sidebar.caption("💡 **Tip**: Use 0.3-0.4 for better motorcycle detection")

    # Initialize button
    if st.sidebar.button("🔧 Initialize Components", type="primary"):
        with st.spinner(f"Initializing {selected_model}..."):
            if processor.initialize_components(confidence_threshold=confidence_threshold, model_name=model_file):
                st.session_state.initialized = True
                st.session_state.confidence_threshold = confidence_threshold
                st.session_state.model_name = model_file
            else:
                st.session_state.initialized = False

    # Video input options
    st.sidebar.header("📹 Video Input")

    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload File (<2GB)", "Local File Path"],
        help="Upload for small files, use path for large files"
    )

    uploaded_file = None
    video_path_input = ""

    if input_method == "Upload File (<2GB)":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Video File",
            type=['mp4', 'avi', 'mov', 'mkv', 'dav'],
            help="Upload a video file (max 2GB)"
        )
    else:
        video_path_input = st.sidebar.text_input(
            "Video File Path",
            help="Enter full path to video file (e.g., /Users/username/videos/video.mp4)",
            placeholder="/path/to/your/video.mp4"
        )

        if video_path_input and not os.path.exists(video_path_input):
            st.sidebar.error("❌ File not found. Please check the path.")
        elif video_path_input and os.path.exists(video_path_input):
            # Validate it's a video file
            if validate_video_file(video_path_input):
                st.sidebar.success("✅ Valid video file found")
                # Get file info
                try:
                    video_info = get_video_info(video_path_input)
                    if video_info:
                        st.sidebar.info(f"📊 {os.path.basename(video_path_input)}")
                        st.sidebar.text(f"Duration: {video_info.get('duration', 0):.1f}s")
                        st.sidebar.text(f"Resolution: {video_info.get('width', 0)}x{video_info.get('height', 0)}")
                        st.sidebar.text(f"FPS: {video_info.get('fps', 0):.1f}")
                except:
                    pass
            else:
                st.sidebar.error("❌ Invalid video file format")

    # Progress display (persistent)
    progress_bar = st.sidebar.empty()
    status_text = st.sidebar.empty()

    # Determine if we have a valid video input
    has_video_input = False
    final_video_path = None

    if input_method == "Upload File (<2GB)" and uploaded_file:
        has_video_input = True
    elif input_method == "Local File Path" and video_path_input and os.path.exists(video_path_input) and validate_video_file(video_path_input):
        has_video_input = True
        final_video_path = video_path_input

    # Processing controls
    if st.session_state.get('initialized', False) and has_video_input:
        col1, col2, col3 = st.sidebar.columns(3)

        with col1:
            if st.button("▶️ Start", disabled=processor.is_processing):
                if input_method == "Upload File (<2GB)":
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        st.session_state.video_path = tmp_file.name
                else:
                    # Use the provided path directly
                    st.session_state.video_path = final_video_path

                # Start processing
                processor.start_processing(st.session_state.video_path)

        with col2:
            if st.button("⏸️ Pause/Resume", disabled=not processor.is_processing):
                processor.pause_processing()

        with col3:
            if st.button("⏹️ Stop", disabled=not processor.is_processing):
                processor.stop_processing()

        # Update progress and status from thread-safe variables
        if processor.is_processing or processor.processing_complete:
            progress_bar.progress(processor.progress)
            status_text.text(processor.status_message)

    # Performance settings
    st.sidebar.header("⚡ Performance")
    if st.session_state.get('initialized', False):
        # Show current model info
        current_model = st.session_state.get('model_name', 'yolov8s.pt')
        st.sidebar.info(f"🤖 Model: {current_model}")
        st.sidebar.caption(f"🎯 Confidence: {st.session_state.get('confidence_threshold', 0.3):.2f}")
        st.sidebar.caption("💡 Re-initialize to change model or confidence")
        
        st.sidebar.markdown("---")
        
        # Speed presets
        speed_preset = st.sidebar.radio(
            "⚡ Speed Preset",
            options=["Balanced ⭐", "Fast", "Quality"],
            index=0,
            help="Quick presets for common scenarios"
        )
        
        if speed_preset == "Fast":
            default_interval = 3
            default_resolution = 0  # 640x480
            st.sidebar.caption("🚀 Optimized for speed (~2-3x faster)")
        elif speed_preset == "Quality":
            default_interval = 1
            default_resolution = 2  # 1280x720
            st.sidebar.caption("🎯 Optimized for accuracy (slower)")
        else:  # Balanced
            default_interval = 2
            default_resolution = 1  # 800x600
            st.sidebar.caption("⚖️ Good balance of speed and accuracy")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Advanced Settings**")
        
        detection_interval = st.sidebar.slider(
            "Detection Interval",
            min_value=1,
            max_value=5,
            value=default_interval,
            help="Detect vehicles every N frames (higher = faster, lower accuracy)"
        )
        
        max_resolution = st.sidebar.selectbox(
            "Max Resolution",
            options=["640x480 (Fast)", "800x600 (Balanced)", "1280x720 (Quality)"],
            index=default_resolution,
            help="Lower resolution = faster processing"
        )
        
        # Store settings in session state
        st.session_state.detection_interval = detection_interval
        resolution_map = {
            "640x480 (Fast)": (640, 480),
            "800x600 (Balanced)": (800, 600),
            "1280x720 (Quality)": (1280, 720)
        }
        st.session_state.max_resolution = resolution_map[max_resolution]
        
        # Show estimated FPS based on settings
        current_model_file = st.session_state.get('model_name', 'yolo26s.pt')
        
        # YOLO26 models (fastest - 43% faster than previous)
        if 'yolo26n' in current_model_file:
            base_fps = 25  # Very fast!
        elif 'yolo26s' in current_model_file:
            base_fps = 18  # Fast
        elif 'yolo26m' in current_model_file:
            base_fps = 12  # Balanced
        elif 'yolo26l' in current_model_file:
            base_fps = 8   # Accurate
        
        # YOLOv12 models (attention-based)
        elif 'yolo12n' in current_model_file:
            base_fps = 20
        elif 'yolo12s' in current_model_file:
            base_fps = 15
        elif 'yolo12m' in current_model_file:
            base_fps = 10
        
        # Legacy YOLOv8 models
        elif 'yolov8n' in current_model_file or 'v8n' in current_model_file:
            base_fps = 15
        elif 'yolov8s' in current_model_file or 'v8s' in current_model_file:
            base_fps = 10
        elif 'yolov8m' in current_model_file or 'v8m' in current_model_file:
            base_fps = 6
        elif 'n.pt' in current_model_file:
            base_fps = 15
        elif 's.pt' in current_model_file:
            base_fps = 10
        elif 'm.pt' in current_model_file:
            base_fps = 6
        elif 'l.pt' in current_model_file:
            base_fps = 3
        else:
            base_fps = 2
        
        # Adjust based on settings
        estimated_fps = base_fps * detection_interval
        if default_resolution == 0:  # 640x480
            estimated_fps *= 1.3
        elif default_resolution == 2:  # 1280x720
            estimated_fps *= 0.7
        
        st.sidebar.caption(f"📊 Estimated FPS: ~{int(estimated_fps)} fps")

    # Counting line configuration
    st.sidebar.header("📏 Counting Line")
    st.sidebar.markdown("*Set where vehicles will be counted*")
    
    if st.session_state.get('initialized', False):
        # Initialize default values if not set
        if 'line_y_position' not in st.session_state:
            st.session_state.line_y_position = 50  # Percentage from top (50% = center)
        if 'line_start_x' not in st.session_state:
            st.session_state.line_start_x = 10  # Percentage from left
        if 'line_end_x' not in st.session_state:
            st.session_state.line_end_x = 90  # Percentage from right
        
        # Line position controls
        st.sidebar.markdown("**Line Position (as % of frame)**")
        st.sidebar.markdown("*Adjust sliders to position the yellow counting line*")

        def update_line_y():
            st.session_state.line_y_position = st.session_state.line_y_slider
            processor.line_y_position = st.session_state.line_y_slider

        line_y = st.sidebar.slider(
            "Vertical Position (Y)",
            min_value=10,
            max_value=90,
            value=st.session_state.line_y_position,
            key="line_y_slider",
            on_change=update_line_y,
            help="Vertical position of counting line (50% = center, lower = top, higher = bottom)"
        )

        def update_line_start_x():
            st.session_state.line_start_x = st.session_state.line_start_x_slider
            processor.line_start_x = st.session_state.line_start_x_slider

        def update_line_end_x():
            st.session_state.line_end_x = st.session_state.line_end_x_slider
            processor.line_end_x = st.session_state.line_end_x_slider

        col_x1, col_x2 = st.sidebar.columns(2)
        with col_x1:
            line_start_x = st.sidebar.slider(
                "Start X (%)",
                min_value=0,
                max_value=80,
                value=st.session_state.line_start_x,
                key="line_start_x_slider",
                on_change=update_line_start_x,
                help="Left edge of line"
            )
        with col_x2:
            line_end_x = st.sidebar.slider(
                "End X (%)",
                min_value=20,
                max_value=100,
                value=st.session_state.line_end_x,
                key="line_end_x_slider",
                on_change=update_line_end_x,
                help="Right edge of line"
            )
        
        # Ensure end > start
        if line_end_x <= line_start_x:
            line_end_x = line_start_x + 10
            st.sidebar.warning("⚠️ End X must be greater than Start X")
        
        # Sync processor instance variables with current slider values (for thread access)
        processor.line_y_position = line_y
        processor.line_start_x = line_start_x
        processor.line_end_x = line_end_x
        
        # Show current position
        st.sidebar.info(f"📍 Line: Y={line_y}%, X={line_start_x}%-{line_end_x}%")
        st.sidebar.caption("💡 Position updates automatically when video is processing")
        
        st.sidebar.markdown("---")
        
        if st.sidebar.button("🔄 Reset Counts"):
            if processor.counter:
                processor.counter.reset_counts()
                st.success("Counts reset to zero")
        
        if st.sidebar.button("🔄 Reset Line to Center"):
            st.session_state.line_y_position = 50
            st.session_state.line_start_x = 10
            st.session_state.line_end_x = 90
            processor.line_y_position = 50
            processor.line_start_x = 10
            processor.line_end_x = 90
            st.session_state.line_applied = True
            st.sidebar.success("Line reset to center")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📹 Video Feed")

        # Video display area
        video_placeholder = st.empty()

        if processor.current_frame is not None:
            video_placeholder.image(processor.current_frame, channels="RGB", width='stretch')
        else:
            video_placeholder.markdown("""
            ### 📋 Instructions
            1. Click **"Initialize Components"** in the sidebar
            2. Upload a video file
            3. Click **"Start"** to begin processing
            4. Watch vehicles being detected and counted in real-time
            """)

    with col2:
        st.header("📊 Statistics")

        # Real-time counts
        if processor.counter:
            counts = processor.counter.get_counts()

            # Display counts in a nice format
            st.metric("🚗 Cars", counts.get('car', 0))
            st.metric("🏍️ Motorcycles", counts.get('motorcycle', 0))
            st.metric("🚌 Buses", counts.get('bus', 0))
            st.metric("🚚 Trucks", counts.get('truck', 0))
            st.metric("📈 Total Vehicles", counts.get('total', 0), delta=counts.get('total', 0))

            # Directional counts for two-way traffic
            direction_counts = processor.counter.get_direction_counts()
            if direction_counts['up'] > 0 or direction_counts['down'] > 0:
                st.markdown("---")
                st.subheader("🔄 Two-Way Traffic")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("⬆️ Upward", direction_counts.get('up', 0))
                with col2:
                    st.metric("⬇️ Downward", direction_counts.get('down', 0))

            # Status indicators
            if processor.is_processing:
                if processor.is_paused:
                    st.warning("⏸️ Processing Paused")
                else:
                    st.success("▶️ Processing Active")
            else:
                st.info("⏹️ Ready to Process")

        # Video info
        if 'video_path' in st.session_state:
            try:
                video_info = get_video_info(st.session_state.video_path)
                if video_info:
                    st.subheader("📋 Processing Information")
                    if input_method == "Upload File (<2GB)" and uploaded_file:
                        st.write(f"**File:** {uploaded_file.name}")
                    else:
                        st.write(f"**File:** {os.path.basename(st.session_state.video_path)}")
                    st.write(f"**Duration:** {video_info.get('duration', 0):.1f} seconds")
                    st.write(f"**FPS:** {video_info.get('fps', 0):.1f}")
                    st.write(f"**Resolution:** {video_info.get('width', 0)}x{video_info.get('height', 0)}")
            except:
                pass

    # Footer
    st.markdown("---")
    st.markdown("*Built By Devotic Labs*")

    # Auto-refresh for live updates
    if processor.is_processing:
        time.sleep(0.1)  # Small delay to prevent excessive CPU usage
        st.rerun()


if __name__ == "__main__":
    main()
