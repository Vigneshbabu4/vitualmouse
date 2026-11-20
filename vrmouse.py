"""
Virtual Drawing Board - Hand Gesture Drawing System
Draw with index finger, erase with all fingers combined.
"""

import cv2
import mediapipe as mp
import numpy as np

# ===== CONFIGURATION =====
MIN_DETECTION_CONFIDENCE = 0.7
DRAW_COLOR = (0, 255, 0)  # Green
ERASE_COLOR = (0, 0, 0)    # Black
BRUSH_SIZE = 5
ERASE_SIZE = 50
FPS_DISPLAY = True

# ===== INITIALIZE MEDIAPIPE =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE
)
mp_drawing = mp.solutions.drawing_utils

# ===== INITIALIZE VIDEO CAPTURE =====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ===== CREATE DRAWING CANVAS =====
canvas = None
drawing = False
erasing = False
last_pos = None

# ===== HELPER FUNCTIONS =====

def fingers_up(hand_landmarks):
    """
    Detect which fingers are up (extended).
    Returns: List of 5 integers (1=up, 0=closed) for [thumb, index, middle, ring, pinky]
    """
    tips = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    ]
    
    mcps = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP],
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    ]
    
    fingers = []
    for i in range(5):
        if tips[i].y < mcps[i].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    # Special case for thumb
    wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
    if wrist_x < tips[0].x:
        fingers[0] = 1 if tips[0].x > mcps[0].x else 0
    else:
        fingers[0] = 1 if tips[0].x < mcps[0].x else 0
    
    return fingers

def draw_info_text(frame, text, position, color, font_scale=1, thickness=2):
    """Draw text with background for better visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height) = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    cv2.rectangle(frame, 
                  (position[0] - 5, position[1] - text_height - 5),
                  (position[0] + text_width + 5, position[1] + 5),
                  (0, 0, 0), -1)
    
    cv2.putText(frame, text, position, font, font_scale, color, thickness)

# ===== MAIN LOOP =====

frame_count = 0
fps = 0
fps_start = cv2.getTickCount()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    frame_count += 1
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Initialize canvas if not created
    if canvas is None:
        canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # ===== GESTURE DETECTION =====
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        
        # Get index finger position
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pos = (int(index_tip.x * frame_width), int(index_tip.y * frame_height))
        
        # Detect which fingers are up
        fingers = fingers_up(hand_landmarks)
        
        # ===== DRAWING MODE =====
        # Only index finger up -> Draw
        if fingers[1] == 1 and sum(fingers) == 1:
            drawing = True
            erasing = False
            draw_info_text(frame, "DRAWING MODE", (50, 50), (0, 255, 0))
            
            # Draw on canvas
            if last_pos is not None:
                cv2.line(canvas, last_pos, index_pos, DRAW_COLOR, BRUSH_SIZE)
            last_pos = index_pos
            
            # Draw indicator
            cv2.circle(frame, index_pos, BRUSH_SIZE + 5, (0, 255, 0), 2)
        
        # ===== ERASING MODE =====
        # All fingers up -> Erase
        elif sum(fingers) == 5:
            drawing = False
            erasing = True
            draw_info_text(frame, "ERASING MODE", (50, 50), (0, 0, 255))
            
            # Erase on canvas
            cv2.circle(canvas, index_pos, ERASE_SIZE, ERASE_COLOR, -1)
            last_pos = None
            
            # Draw eraser indicator
            cv2.circle(frame, index_pos, ERASE_SIZE, (0, 0, 255), 2)
        
        # ===== NEUTRAL MODE =====
        else:
            drawing = False
            erasing = False
            last_pos = None
            draw_info_text(frame, "NEUTRAL - Show 1 finger to draw, all 5 to erase", (50, 50), (255, 255, 0))
        
        # Draw hand skeleton
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    else:
        drawing = False
        erasing = False
        last_pos = None
        draw_info_text(frame, "Show hand to start drawing", (50, 50), (255, 255, 255))

    # ===== DISPLAY FPS =====
    if FPS_DISPLAY:
        if frame_count % 30 == 0:
            fps_end = cv2.getTickCount()
            fps = 30 / ((fps_end - fps_start) / cv2.getTickFrequency())
            fps_start = fps_end
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame_width - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display instructions
    cv2.putText(frame, "Press C to clear canvas, Q to exit", (10, frame_height - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Blend frame with canvas
    combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    
    # Show frames
    cv2.imshow("Virtual Drawing Board", combined)
    cv2.imshow("Canvas", canvas)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        print("Canvas cleared!")

# ===== CLEANUP =====
print("Shutting down...")
cap.release()
cv2.destroyAllWindows()
print("Done!")