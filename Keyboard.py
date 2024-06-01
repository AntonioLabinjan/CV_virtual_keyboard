import cv2
import mediapipe as mp
import numpy as np
import pyautogui

cap = cv2.VideoCapture(0)

hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Virtual keyboard layout
keys = [
    ['Q', 'W', 'E', 'R', 'T', 'Z', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Y', 'X', 'C', 'V', 'B', 'N', 'M']
]

key_width = 70
key_height = 70

# Function to draw virtual keyboard
def draw_keyboard(img, keys, pressed_key=None):
    keyboard_img = np.zeros((300, 800, 3), np.uint8)
    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            x = j * key_width + 50
            y = i * key_height + 50
            color = (200, 200, 200) if key != pressed_key else (0, 255, 0)
            cv2.rectangle(keyboard_img, (x, y), (x + key_width, y + key_height), color, -1)
            cv2.putText(keyboard_img, key, (x + 20, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return keyboard_img

# Main loop
click_flag = False

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    pressed_key = None
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand.landmark
            index_x, index_y = None, None
            thumb_x, thumb_y = None, None
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                if id == 8:  # Index finger tip
                    index_x, index_y = x, y
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)
                if id == 4:  # Thumb tip
                    thumb_x, thumb_y = x, y
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)

            if index_x and thumb_x:
                if abs(index_x - thumb_x) < 50 and abs(index_y - thumb_y) < 50:
                    if not click_flag:
                        for i, row in enumerate(keys):
                            for j, key in enumerate(row):
                                x = j * key_width + 50
                                y = i * key_height + 50
                                if x < index_x < x + key_width and y < index_y < y + key_height:
                                    pyautogui.typewrite(key)
                                    pressed_key = key
                                    click_flag = True
                                    break
                        else:
                            click_flag = False
                else:
                    click_flag = False
    keyboard_img = draw_keyboard(frame, keys, pressed_key)
    cv2.imshow('Virtual Keyboard', keyboard_img)
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
