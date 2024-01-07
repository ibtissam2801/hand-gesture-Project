import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialisation de MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

def calculate_angle(a, b, c):
    # Calcul de l'angle entre trois points
    a = np.array(a)  # Premier
    b = np.array(b)  # Milieu
    c = np.array(c)  # Dernier

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def detect_fingers(frame, hand_landmarks):
    # Initialiser le comptage des doigts
    finger_count = 0

    # Points de repère pour le pouce
    thumb_base = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y
    thumb_joint = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

    # Calculer l'angle du pouce
    thumb_angle = calculate_angle(thumb_base, thumb_joint, thumb_tip)

    # Logique pour le pouce
    if thumb_angle > 160:  
        finger_count += 1

    # Index des extrémités des autres doigts
    finger_tips = [8, 12, 16, 20] 

    # Logique pour les autres doigts
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            finger_count += 1

    return finger_count, frame
#genere choix random pour l'ordinateur
def get_computer_choice():
    return random.choice(["Pierre", "Papier", "Ciseaux"])
#determine le gagnant
def determine_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        return "Egalite!"
    elif (user_choice == "Pierre" and computer_choice == "Ciseaux") or \
         (user_choice == "Ciseaux" and computer_choice == "Papier") or \
         (user_choice == "Papier" and computer_choice == "Pierre"):
        return "Victoire!"
    else:
        return "defaite!"

# Initialiser la caméra
cap = cv2.VideoCapture(0)

# Paramètres pour la moyenne mobile
window_size = 10
finger_count_buffer = []

# Initialiser le choix de l'ordinateur
computer_choice = get_computer_choice()
start_time=time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Calculer le temps restant
    time_remaining = max(0, 3 - int(time.time() - start_time))
    # Afficher le compteur
    cv2.putText(frame, f"Temps restant : {time_remaining}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    frame = cv2.resize(frame, (640, 480))
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Initialiser finger_count à chaque itération
    finger_count = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_count, frame = detect_fingers(frame, hand_landmarks)

    finger_count_buffer.append(finger_count)
    if len(finger_count_buffer) > window_size:
        finger_count_buffer.pop(0)
    avg_finger_count = int(np.mean(finger_count_buffer))

    # Interpréter le geste du joueur
    if avg_finger_count == 0:
        user_choice = "Pierre"
    elif avg_finger_count == 2:
        user_choice = "Ciseaux"
    elif avg_finger_count == 5:
        user_choice = "Papier"
    else:
        user_choice = "unknown"

    
    # Afficher les informations seulement après le délai de 3 secondes
    if time.time() - start_time >= 3:
        # Affichage des choix et du resultat
        cv2.putText(frame, f"Votre choix: {user_choice}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Choix ordi: {computer_choice}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        result = determine_winner(user_choice, computer_choice) if user_choice != "unknown" else ""
        cv2.putText(frame, result, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Pierre-Papier-Ciseaux", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Réinitialiser le choix de l'ordinateur si le joueur appuie sur 'r'
        computer_choice = get_computer_choice()
        start_time = time.time()
cap.release()
cv2.destroyAllWindows()