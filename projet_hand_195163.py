import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to detect the number of fingers raised
def detect_fingers(hand_landmarks):
    # Initialize finger count
    finger_count = 0

    # Landmarks for the thumb
    thumb_base = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y
    thumb_joint = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

    # Calculate the thumb angle
    thumb_angle = calculate_angle(thumb_base, thumb_joint, thumb_tip)

    # Logic for the thumb
    if thumb_angle > 160:
        finger_count += 1

    # Tips of the other fingers
    finger_tips = [8, 12, 16, 20]

    # Logic for the other fingers
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            finger_count += 1

    return finger_count

# Function to detect letters
def detect_letter(hand_landmarks,handedness):
    # Déterminer si la main est droite ou gauche
    is_left_hand = handedness.classification[0].label == 'Left'
    
    # Vérifier si les doigts autre que le pouce et l'index(majeur, annulaire et auriculaire) sont repliés
    fingers_folded_except_index = all(hand_landmarks.landmark[finger_tip].y > hand_landmarks.landmark[finger_tip - 2].y
                                      for finger_tip in [12, 16, 20])
    
    # Déterminer si les doigts autres que le pouce sont repliés
    fingers_folded = all(hand_landmarks.landmark[finger_tip].y > hand_landmarks.landmark[finger_tip - 2].y
                         for finger_tip in [8, 12, 16, 20])

    # Position du pouce et de l'index
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]  # Joint interphalangien du pouce
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]    # Lettre A: Pouce étendu, autres doigts repliés
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    # Lettre A : tout les doigts sont repliés sauf le pouce 
    if all(hand_landmarks.landmark[finger_tip].y > hand_landmarks.landmark[finger_tip - 2].y for finger_tip in [8, 12, 16, 20]):
        if (is_left_hand and thumb_tip.x > thumb_ip.x) or \
           (not is_left_hand and thumb_tip.x < thumb_ip.x):
            return "A"

    # Lettre T : tout les doigts sont repliés sauf le pouce et l'index
    if fingers_folded_except_index and \
       ((is_left_hand and index_tip.x < index_mcp.x and thumb_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y) or \
        (not is_left_hand and index_tip.x > index_mcp.x and thumb_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y)):
        return "T"
    # Lettre S: Pouce replié sur la paume, autres doigts repliés
    is_S = fingers_folded and ((is_left_hand and thumb_tip.x < thumb_ip.x) or (not is_left_hand and thumb_tip.x > thumb_ip.x))

    if is_S:
        return "S" 
    # Lettre I: Index levé, autres doigts repliés
    is_I = (hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and
            all(hand_landmarks.landmark[finger_tip].y > hand_landmarks.landmark[finger_tip - 2].y
                for finger_tip in [12, 16, 20]))

    if is_I:
        return "I"

    # Lettre B: Tous les doigts sont étendus
    is_B = all(hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y
               for tip_index in range(4, 21, 4))

    if is_B:
        return "B"
    
    # Lettre M: Pouce sous le petit doigt
    is_M = hand_landmarks.landmark[4].y > hand_landmarks.landmark[18].y

    if is_M:
        return "M"

    return "Unknown"

# Variables for hand gesture history
hand_path = []
history_length = 30

# Function to detect horizontal swipe gesture
def detect_horizontal_swipe(hand_landmarks):
    wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
    hand_path.append(wrist_x)
    if len(hand_path) > history_length:
        hand_path.pop(0)
        average_x = sum(hand_path) / history_length
        if wrist_x < average_x - 0.03:  # Threshold can be adjusted
            return "Swipe Left"
        elif wrist_x > average_x + 0.03:
            return "Swipe Right"
    return ""

# Initialize the camera
cap = cv2.VideoCapture(0)

# Mode variable (fingers, letters, gesture)
mode = 'fingers'  # Default mode

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    output_text = "None"
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx] 
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if mode == 'fingers':
                output_text = f"Fingers: {detect_fingers(hand_landmarks)}"
            elif mode == 'letters':
                output_text = f"Letter: {detect_letter(hand_landmarks, handedness)}"
            elif mode == 'gesture':
                output_text = detect_horizontal_swipe(hand_landmarks)

    cv2.putText(frame, output_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Hand Gesture Recognition", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('c'):
        mode = 'fingers'
    elif key & 0xFF == ord('l'):
        mode = 'letters'
    elif key & 0xFF == ord('g'):
        mode = 'gesture'
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
