import cv2
import mediapipe as mp
import time
import pyautogui


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def count_fingers(hand_landmarks, hand_label):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    if hand_label == 'Right':
        fingers.append(int(hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x))
    else:
        fingers.append(int(hand_landmarks.landmark[tips_ids[0]].x > hand_landmarks.landmark[tips_ids[0] - 1].x))

    for i in range(1, 5):
        fingers.append(int(hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i] - 2].y))

    return sum(fingers)

cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()
prev_click = None

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = frame.shape

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            label = results.multi_handedness[0].classification[0].label

            finger_count = count_fingers(hand_landmarks, label)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            screen_x = (x / w) * screen_w
            screen_y = (y / h) * screen_h

            if finger_count == 1:
                pyautogui.moveTo(screen_x, screen_y, duration=0.01)
                prev_click = None
            elif finger_count == 2:
                if prev_click != 'left':
                    pyautogui.click(button='left')
                    prev_click = 'left'
            elif finger_count == 3:
                if prev_click != 'right':
                    pyautogui.click(button='right')
                    prev_click = 'right'
            else:
                prev_click = None

        cv2.imshow("Virtual Drawing", image)

        if cv2.getWindowProperty("Virtual Drawing", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
