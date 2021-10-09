import cv2
import mediapipe as mp


# 0 for webcam, "url/video" for ipwebcam
cap = cv2.VideoCapture(0)

# mediapipe solutions for hand detection and tracking!
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.8
)
mpDraw = mp.solutions.drawing_utils

# webcam reader!
while True:
    _, img = cap.read()
    cv2.flip(img, 1, img)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = hands.process(imgRGB)
    img.flags.writeable = True
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cv2.imshow("Live Hand Tracking", img)
    if cv2.waitKey(33) == ord('q'):
        break
