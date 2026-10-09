import cv2
import mediapipe as mp
import pyautogui
import math
import time

print("Final Fast Gesture Control Started...")

# 🔥 Remove all internal delays
pyautogui.PAUSE = 0
pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.FAILSAFE = False

screen_w, screen_h = pyautogui.size()

# MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(3, 320)
cap.set(4, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Movement
prev_x, prev_y = 0, 0
smoothening = 3
scale = 1.5   # 👈 adjust 1.2–2.0 for comfort

# Click control
clicking = False

# FPS control (faster loop)
prev_time = 0

while True:
    success, img = cap.read()

    # ✅ Camera stability fix
    if not success or img is None:
        print("Reconnecting camera...")
        cap.release()
        time.sleep(0.3)
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        cap.set(3, 320)
        cap.set(4, 240)
        continue

    # 🔥 Faster FPS
    if time.time() - prev_time < 0.01:
        continue
    prev_time = time.time()

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]

        h, w, c = img.shape

        # Finger detection
        fingers = []

        # Thumb
        if handLms.landmark[4].x > handLms.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for tip in [8, 12, 16, 20]:
            if handLms.landmark[tip].y < handLms.landmark[tip-2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        # Index finger
        x = int(handLms.landmark[8].x * w)
        y = int(handLms.landmark[8].y * h)

        # 🔥 Distance-friendly mapping
        screen_x = screen_w * (x / w) * scale
        screen_y = screen_h * (y / h) * scale

        # Keep inside screen
        screen_x = min(screen_w, max(0, screen_x))
        screen_y = min(screen_h, max(0, screen_y))

        # 🖱️ MOVE (only index finger)
        if fingers == [0,1,0,0,0]:
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

        # 🖱️ CLICK (index + middle)
        if fingers[1] == 1 and fingers[2] == 1:
            x2 = int(handLms.landmark[12].x * w)
            y2 = int(handLms.landmark[12].y * h)

            distance = math.hypot(x2 - x, y2 - y)

            if distance < 50:
                if not clicking:
                    # 🔥 FAST CLICK (no lag)
                    pyautogui.mouseDown()
                    pyautogui.mouseUp()
                    clicking = True
            else:
                clicking = False

        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Final Fast Control", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()