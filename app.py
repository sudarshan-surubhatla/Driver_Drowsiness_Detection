from flask import Flask, render_template, Response
import os
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import threading

app = Flask(__name__)

# Variables
latest_alert = "Driver Drowsiness Monitoring"
lock = threading.Lock()

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True
)

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
EYE_AR_THRESH = 0.21
MOUTH_AR_THRESH = 0.75


def calculate_ear(landmarks, left_eye_indices, right_eye_indices):
    def eye_aspect_ratio(eye):
        p1 = distance.euclidean(eye[1], eye[5])
        p2 = distance.euclidean(eye[2], eye[4])
        p3 = distance.euclidean(eye[0], eye[3])
        return (p1 + p2) / (2.0 * p3)

    left_eye = np.array([[landmarks[i].x, landmarks[i].y] for i in left_eye_indices])
    right_eye = np.array([[landmarks[i].x, landmarks[i].y] for i in right_eye_indices])
    return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0


def calculate_mar(landmarks):
    horizontal_distance = distance.euclidean(
        [landmarks[61].x, landmarks[61].y],
        [landmarks[291].x, landmarks[291].y]
    )
    vertical_distance = distance.euclidean(
        [landmarks[13].x, landmarks[13].y],
        [landmarks[14].x, landmarks[14].y]
    )
    return vertical_distance / horizontal_distance


def generate_frames():
    global latest_alert
    cap = cv2.VideoCapture(0)
    counter = 0
    yawn_count = 0
    yawn_prev = False

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        drowsiness_active = False  # Flag for EAR-based alert
        yawning_active = False     # Flag for MAR-based alert

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Calculate EAR and MAR
                ear = calculate_ear(landmarks.landmark, LEFT_EYE_INDICES, RIGHT_EYE_INDICES)
                mar = calculate_mar(landmarks.landmark)

                # Display EAR and MAR on the frame
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # EAR logic (Drowsiness detection)
                if ear < EYE_AR_THRESH:
                    counter += 1
                    if counter >= 10:  # Threshold for consecutive frames
                        with lock:
                            latest_alert = "Drowsiness Alert: Eyes Closed!"
                        drowsiness_active = True
                else:
                    counter = 0

                # MAR logic (Yawning detection)
                if mar > MOUTH_AR_THRESH:
                    if not yawn_prev:
                        yawn_count += 1
                        with lock:
                            latest_alert = f"Yawning Alert! Total yawns: {yawn_count}"
                    yawning_active = True
                    yawn_prev = True
                else:
                    yawn_prev = False

        # Set "No alerts" only if no drowsiness or yawning alert is active
        if not drowsiness_active and not yawning_active:
            with lock:
                latest_alert = "No alerts, Drive Safe !!!"

        # Encode the frame and yield it for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')
    
def home():
    return render_template('index.html')  # This will look for index.html inside the templates folder

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/get_alerts')
def get_alerts():
    global latest_alert
    with lock:
        return latest_alert


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
