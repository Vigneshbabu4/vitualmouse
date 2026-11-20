"""
Enhanced Virtual Mouse - Hand Gesture Control System
Controls your mouse cursor using hand gestures detected via webcam.
"""

import cv2
import mediapipe as mp
import pyautogui
import math

# ===== CONFIGURATION =====
MIN_DETECTION_CONFIDENCE = 0.7
CLICK_DISTANCE_THRESHOLD = 0.05
SCROLL_DISTANCE_THRESHOLD = 0.1
VOLUME_CHANGE_THRESHOLD = 0.05
TWO_HAND_VOLUME_SENSITIVITY = 0.01  # Very sensitive to small thumb movements
FPS_DISPLAY = True

# ===== INITIALIZE MEDIAPIPE =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,  # Enable two-hand detection
    min_detection_confidence=MIN_DETECTION_CONFIDENCE
)
mp_drawing = mp.solutions.drawing_utils

# ===== GET SCREEN DIMENSIONS =====
screen_width, screen_height = pyautogui.size()

# ===== INITIALIZE VIDEO CAPTURE =====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ===== STATE VARIABLES =====
last_volume_change = 'neutral'
last_thumbs_distance = None  # For two-hand volume control
last_click_time = 0
click_cooldown = 0.1  # seconds

# ===== HELPER FUNCTIONS =====

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def fingers_up(hand_landmarks):
    """
    Detect which fingers are up (extended).
    Returns: List of 5 integers (1=up, 0=closed) for [thumb, index, middle, ring, pinky]
    """
    # Get finger tips and MCP joints
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
    
    # Detect if each finger is up
    fingers = []
    for i in range(5):
        if tips[i].y < mcps[i].y:  # Tip is above MCP (finger is extended)
            fingers.append(1)
        else:
            fingers.append(0)
    
    # Special case for thumb (compare x coordinate)
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
    
    # Draw background rectangle
    cv2.rectangle(frame, 
                  (position[0] - 5, position[1] - text_height - 5),
                  (position[0] + text_width + 5, position[1] + 5),
                  (0, 0, 0), -1)
    
    # Draw text
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

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # ===== GESTURE DETECTION =====
    if result.multi_hand_landmarks:
        hand_count = len(result.multi_hand_landmarks)
        
        # ===== TWO-HAND VOLUME CONTROL & SCROLLING =====
        if hand_count == 2:
            # Identify hands by their position (left side vs right side of screen)
            hands_data = []
            for idx in range(hand_count):
                hand_landmarks = result.multi_hand_landmarks[idx]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                hands_data.append({
                    'idx': idx,
                    'wrist': wrist,
                    'hand': hand_landmarks,
                    'x_pos': wrist.x  # X position to determine left/right
                })
            
            # Sort by x position: left hand (smaller x) first
            hands_data.sort(key=lambda h: h['x_pos'])
            left_data = hands_data[0]  # Left side of screen
            right_data = hands_data[1]  # Right side of screen
            
            left_wrist = left_data['wrist']
            right_wrist = right_data['wrist']
            
            # ===== HAND POSITION SCROLLING =====
            # Only Right hand up (small wrist y) → Scroll DOWN
            if right_wrist.y < 0.3:
                pyautogui.scroll(-20)  # Scroll down faster
                draw_info_text(frame, "Scroll DOWN (Right Hand Only)", (50, 320), (0, 255, 128))
            
            # ===== VOLUME CONTROL =====
            # Get both thumbs for volume control
            thumb1 = result.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb2 = result.multi_hand_landmarks[1].landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            # Calculate distance between two thumbs
            thumbs_distance = calculate_distance(thumb1, thumb2)
            
            # Display thumb distance for debugging
            draw_info_text(frame, f"Thumb Distance: {thumbs_distance:.3f}", (50, 230), (255, 255, 255))
            
            if last_thumbs_distance is not None:
                distance_change = thumbs_distance - last_thumbs_distance
                
                # Thumbs getting CLOSER → DECREASE volume
                if distance_change < -TWO_HAND_VOLUME_SENSITIVITY:
                    pyautogui.press('volumedown')
                    draw_info_text(frame, "Volume DOWN (Thumbs Close)", (50, 250), (128, 0, 255))
                
                # Thumbs getting FARTHER → INCREASE volume
                elif distance_change > TWO_HAND_VOLUME_SENSITIVITY:
                    pyautogui.press('volumeup')
                    draw_info_text(frame, "Volume UP (Thumbs Apart)", (50, 250), (255, 0, 255))
            
            last_thumbs_distance = thumbs_distance
            
            # Draw both thumbs
            thumb1_pos = (int(thumb1.x * frame_width), int(thumb1.y * frame_height))
            thumb2_pos = (int(thumb2.x * frame_width), int(thumb2.y * frame_height))
            cv2.circle(frame, thumb1_pos, 15, (255, 0, 255), -1)
            cv2.circle(frame, thumb2_pos, 15, (255, 0, 255), -1)
            cv2.line(frame, thumb1_pos, thumb2_pos, (255, 0, 255), 2)
            
            # Draw all landmarks for both hands
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # ===== SINGLE HAND GESTURES =====
        elif hand_count == 1:
            last_thumbs_distance = None  # Reset for next two-hand gesture
            
            hand_landmarks = result.multi_hand_landmarks[0]
            # Get key landmark positions
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            
            # Detect which fingers are up
            fingers = fingers_up(hand_landmarks)
            
            # ===== CURSOR MOVEMENT =====
            # Move cursor only when index finger is up
            if fingers[1] == 1 and fingers[2] == 0:
                screen_x = int(index_tip.x * screen_width)
                screen_y = int(index_tip.y * screen_height)
                pyautogui.moveTo(screen_x, screen_y)
                
                # Draw cursor indicator on frame
                cv2.circle(frame, 
                          (int(index_tip.x * frame_width), int(index_tip.y * frame_height)), 
                          15, (0, 255, 0), -1)
            
            # ===== LEFT CLICK =====
            # Index finger and thumb pinch
            distance_index_thumb = calculate_distance(index_tip, thumb_tip)
            if distance_index_thumb < CLICK_DISTANCE_THRESHOLD and fingers[1] == 1:
                pyautogui.click()
                draw_info_text(frame, "Left Click", (50, 50), (0, 255, 0))
            
            # ===== RIGHT CLICK =====
            # Middle finger and thumb pinch
            distance_middle_thumb = calculate_distance(middle_tip, thumb_tip)
            if distance_middle_thumb < CLICK_DISTANCE_THRESHOLD and fingers[2] == 1:
                pyautogui.click(button='right')
                draw_info_text(frame, "Right Click", (50, 100), (0, 0, 255))
            
            # ===== DOUBLE CLICK =====
            # Ring finger and thumb pinch
            distance_ring_thumb = calculate_distance(ring_tip, thumb_tip)
            if distance_ring_thumb < CLICK_DISTANCE_THRESHOLD and fingers[3] == 1:
                pyautogui.doubleClick()
                draw_info_text(frame, "Double Click", (50, 150), (255, 0, 0))
            
            # ===== SCROLLING =====
            # Index and middle fingers up and close together
            distance_index_middle = calculate_distance(index_tip, middle_tip)
            if (fingers[1] == 1 and fingers[2] == 1 and 
                distance_index_middle < SCROLL_DISTANCE_THRESHOLD):
                # Calculate scroll direction
                scroll_direction = (middle_tip.y - index_tip.y) * 20
                pyautogui.scroll(int(scroll_direction * 200))
                
                scroll_text = "Scroll Down" if scroll_direction > 0 else "Scroll Up"
                draw_info_text(frame, scroll_text, (50, 200), (255, 255, 0))
            
            # ===== VOLUME CONTROL =====
            # Thumb extended, other fingers closed
            if fingers[0] == 1 and sum(fingers[1:]) == 0:
                thumb_relative_y = thumb_tip.y - wrist.y
                
                # Thumb pointing UP (above wrist) - increase volume
                if thumb_relative_y < -VOLUME_CHANGE_THRESHOLD and last_volume_change != 'up':
                    pyautogui.press('volumeup')
                    draw_info_text(frame, "Volume UP", (50, 250), (255, 0, 255))
                    last_volume_change = 'up'
                
                # Thumb pointing DOWN (below wrist) - decrease volume
                elif thumb_relative_y > VOLUME_CHANGE_THRESHOLD and last_volume_change != 'down':
                    pyautogui.press('volumedown')
                    draw_info_text(frame, "Volume DOWN", (50, 250), (128, 0, 255))
                    last_volume_change = 'down'
                
                # Thumb in neutral zone - reset state for next gesture
                elif -VOLUME_CHANGE_THRESHOLD <= thumb_relative_y <= VOLUME_CHANGE_THRESHOLD:
                    last_volume_change = 'neutral'
            
            # ===== DRAW FINGERTIPS =====
            fingertips = [
                (int(index_tip.x * frame_width), int(index_tip.y * frame_height)),
                (int(middle_tip.x * frame_width), int(middle_tip.y * frame_height)),
                (int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height))
            ]
            
            for point in fingertips:
                cv2.circle(frame, point, 12, (255, 0, 0), -1)  # Blue dots
            
            # Draw hand skeleton
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # ===== DISPLAY FPS =====
    if FPS_DISPLAY:
        if frame_count % 30 == 0:
            fps_end = cv2.getTickCount()
            fps = 30 / ((fps_end - fps_start) / cv2.getTickFrequency())
            fps_start = fps_end
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame_width - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display instructions
    cv2.putText(frame, "Press Q to exit", (10, frame_height - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Show frame
    cv2.imshow("Enhanced Virtual Mouse", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== CLEANUP =====
print("Shutting down...")
cap.release()
cv2.destroyAllWindows()
print("Done!")