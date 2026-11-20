# Virtual Mouse - Hand Gesture & Face Detection Project

A Python-based computer vision project that enables hands-free control of your mouse and detect faces using your webcam. This project uses MediaPipe for hand detection and OpenCV for video processing.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Gesture Controls](#gesture-controls)
- [Troubleshooting](#troubleshooting)

---

## Features

✨ **Hand Gesture Control:**
- Move your mouse cursor using your index finger
- Left-click by pinching your index finger and thumb together
- Right-click by pinching your middle finger and thumb together
- Real-time hand pose detection and visualization

🎯 **Face Detection:**
- Detect and track faces in real-time
- Target locking system with visual feedback
- Multiple target detection capability

📹 **Real-time Webcam Processing:**
- Live video feed with skeleton visualization
- Instant gesture recognition
- Smooth cursor movement

---

## Project Structure

```
vitualmouse/
├── vrmouse.py          # Advanced virtual mouse with hand gestures
├── mouse.py            # Basic virtual mouse implementation
├── face.py             # Face detection with target lock
├── faceshoot.py        # Headshot tracking and face detection
└── README.md           # This file
```

### File Descriptions

#### **vrmouse.py** (Advanced - Recommended)
The main virtual mouse application with advanced gesture recognition:
- **Cursor Movement:** Index finger controls mouse position
- **Left Click:** Index & thumb pinch (distance < 0.05)
- **Right Click:** Middle finger & thumb pinch (distance < 0.05)
- **Finger Detection:** Identifies individual finger positions
- **Visual Feedback:** Shows gestures on the video feed

#### **mouse.py** (Basic)
Simplified version of the virtual mouse:
- Basic finger tracking
- Left and right-click detection
- Minimal gesture recognition

#### **face.py**
Face detection module using Haar Cascade:
- Detects faces in the webcam feed
- Displays target locks when faces are close
- Shows target count on screen

#### **faceshoot.py**
Advanced face detection for headshot tracking:
- Tracks face center and calculates headshot position
- Shows target locks with custom visuals
- Tracks multiple targets

---

## Requirements

- **Python 3.7+**
- **OpenCV** (cv2)
- **MediaPipe** (for hand detection)
- **PyAutoGUI** (for mouse control)
- **NumPy**
- **Webcam** (integrated or external)

---

## Installation

### Step 1: Clone or Download the Project
```bash
cd d:\jetson\vitualmouse
```

### Step 2: Install Required Libraries
```bash
pip install opencv-python mediapipe pyautogui numpy
```

Or install from requirements (if available):
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
Test if all imports work:
```bash
python -c "import cv2; import mediapipe; import pyautogui; print('All libraries installed!')"
```

---

## Usage

### Running the Virtual Mouse (vrmouse.py)

```bash
python vrmouse.py
```

**Controls:**
- Press `Q` to quit the application
- Point your index finger at the screen to move the cursor
- Pinch index + thumb to left-click
- Pinch middle + thumb to right-click

### Running Basic Mouse (mouse.py)

```bash
python mouse.py
```

**Controls:**
- Press `Q` to quit
- Same gesture controls as vrmouse.py

### Running Face Detection (face.py)

```bash
python face.py
```

**Controls:**
- Press `H` to quit
- Faces are marked with red rectangles
- "LOCKED" text appears when face is close enough

### Running Headshot Tracking (faceshoot.py)

```bash
python faceshoot.py
```

**Controls:**
- Press `Q` to quit
- Tracks face position for headshot aiming

---

## How It Works

### Hand Gesture Recognition (vrmouse.py)

1. **Initialization:**
   - MediaPipe Hands detector initializes with confidence threshold of 0.7
   - OpenCV captures video from webcam (camera index 0)
   - PyAutoGUI gets screen resolution

2. **Frame Processing Loop:**
   ```
   Capture Frame → Flip Horizontally → Convert to RGB → 
   Hand Detection → Extract Landmarks → Gesture Recognition → 
   Mouse Control → Display Video
   ```

3. **Landmark Detection:**
   - MediaPipe provides 21 hand landmarks (finger tips, knuckles, palm, etc.)
   - Key landmarks used:
     - Index Finger Tip (cursor position)
     - Middle Finger Tip (right-click)
     - Thumb Tip (click detection)

4. **Gesture Recognition:**
   - **Cursor Movement:** Index finger position maps to screen coordinates
   - **Left Click:** Distance between index & thumb < 0.05 triggers click
   - **Right Click:** Distance between middle & thumb < 0.05 triggers right-click

### Face Detection (face.py & faceshoot.py)

1. **Frame Capture:** Video captured from webcam
2. **Grayscale Conversion:** Frame converted to grayscale for processing
3. **Cascade Detection:** Haar Cascade classifier detects faces
4. **Visualization:** 
   - Red/Green rectangles drawn around faces
   - "LOCKED" text shown when face width > 100 pixels
5. **Target Tracking:** Calculates face center and dimensions

---

## Gesture Controls

### Virtual Mouse Gestures

| Gesture | Action |
|---------|--------|
| **Index Finger Up** | Move cursor to finger position |
| **Index + Thumb Pinch** | Left click |
| **Middle + Thumb Pinch** | Right click |
| **All Fingers Down** | No movement |

### Pinch Distance Threshold
- **Default:** 0.05 (normalized distance)
- Controls sensitivity of click detection
- Lower value = more sensitive click
- Adjust in code if needed: `if distance_index_thumb < 0.05:`

---

## Technical Details

### Coordinate Mapping

The hand landmarks are normalized (0-1 range) and mapped to screen coordinates:

```python
screen_x = int(index_tip.x * screen_width)
screen_y = int(index_tip.y * screen_height)
```

### Distance Calculation

Euclidean distance is used for pinch detection:

```python
distance = sqrt((point1.x - point2.x)² + (point1.y - point2.y)²)
```

### Frame Flipping

The video feed is horizontally flipped to create a mirror effect (more intuitive for users):

```python
frame = cv2.flip(frame, 1)  # 1 = horizontal flip
```

---

## Troubleshooting

### 1. **Webcam Not Found**
- Check if another application is using the camera
- Try changing camera index: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`
- Ensure camera permissions are granted

### 2. **Hand Not Detected**
- Ensure good lighting conditions
- Keep your hand in full view
- Hand should be at least 30 pixels wide
- Check camera angle and distance

### 3. **Cursor Jittering**
- The system naturally has some jitter due to hand tremor
- Increase the `min_detection_confidence` value (currently 0.7)
- Add smoothing filter to cursor movement if needed

### 4. **Clicks Not Registering**
- Adjust the pinch distance threshold (currently 0.05)
- Ensure index and thumb tips are clearly visible
- Try pinching slower or more deliberately

### 5. **Low Performance/Lag**
- Reduce frame resolution in OpenCV
- Lower the detection confidence threshold
- Close other applications consuming CPU/GPU

### 6. **Module Import Errors**
- Reinstall libraries: `pip install --upgrade opencv-python mediapipe pyautogui`
- Check Python version: `python --version` (should be 3.7+)

---

## Performance Optimization Tips

1. **Frame Skipping:** Process every 2nd or 3rd frame if speed is needed
2. **Resolution:** Capture at lower resolution if applicable
3. **ROI (Region of Interest):** Process only relevant areas of the video
4. **Threading:** Use threading for camera capture and gesture processing

---

## Future Enhancements

- [ ] Add gesture customization options
- [ ] Implement scroll wheel control (using hand tilt)
- [ ] Add drag functionality
- [ ] Create GUI for configuration
- [ ] Add gesture recording/playback
- [ ] Optimize for mobile devices (via screen streaming)
- [ ] Multi-hand support for advanced gestures

---

## License

This project is open-source and available for educational and personal use.

---

## Support & Issues

If you encounter issues:
1. Check the **Troubleshooting** section
2. Verify all dependencies are installed correctly
3. Try different lighting conditions
4. Adjust detection parameters in the code

---

**Created:** February 2026  
**Language:** Python 3  
**Main Libraries:** OpenCV, MediaPipe, PyAutoGUI
