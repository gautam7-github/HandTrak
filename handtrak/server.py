from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

# "http://user:password@192.168.29.106:8080/video"
camera = cv2.VideoCapture(0)  # use 0 for web camera

# mediapipe solutions for hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.8
)
mpDraw = mp.solutions.drawing_utils


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, img = camera.read()  # read the camera frame
        if not success:
            break
        else:
            cv2.flip(img, 1, img)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            results = hands.process(imgRGB)
            img.flags.writeable = True
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(
                        img, handLms, mpHands.HAND_CONNECTIONS)
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
