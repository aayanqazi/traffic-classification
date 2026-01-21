You are a senior Python computer vision engineer.

I want to build an OFFLINE desktop-style application using Streamlit.

PHASE-1 REQUIREMENTS:
- Offline application (no internet dependency)
- Simple UI
- User uploads a video file
- User clicks "Start Processing"
- Video starts playing frame-by-frame
- Show bounding boxes and tracking IDs on vehicles
- Show LIVE vehicle counts on the screen
- Count vehicles by class: car, motorcycle, bus, truck
- Use line-crossing logic to avoid double counting
- Video can be paused/stopped
- No authentication
- No cloud services

TECH STACK (MANDATORY):
- Python 3.10+
- Streamlit
- OpenCV
- YOLOv8 (Ultralytics)
- ByteTrack
- PyTorch
- CPU-first

DESIGN REQUIREMENTS:
1. Streamlit UI in app.py
2. Modular backend:
   - detector.py
   - tracker.py
   - counter.py
3. Live frame rendering using st.image
4. Sidebar for controls
5. Real-time statistics panel
6. Configurable counting line
7. Clean readable code
8. Logging instead of print

OUTPUT:
- Live stats on UI
- Final JSON + CSV saved locally

DO NOT:
- Use Flask/FastAPI
- Use React or JS
- Use cloud or web APIs

TASK:
1. Create project structure
2. Implement Streamlit UI
3. Implement video processing loop
4. Integrate YOLO + ByteTrack
5. Show live detection & counts
6. Provide run instructions

Start by generating the full project structure and then app.py.
