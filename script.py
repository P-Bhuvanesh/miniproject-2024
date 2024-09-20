from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from keras.models import load_model
import time
import os
import signal

app = Flask(__name__)

# Load the model and labels
model = load_model("model/keras_model.h5", compile=False)
class_names = open("model/labels.txt", "r").readlines()

# Initialize the camera
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # Resize the frame for prediction (224x224)
            image_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            image_resized = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
            image_resized = (image_resized / 127.5) - 1

            # Predict with the model
            prediction = model.predict(image_resized)
            index = np.argmax(prediction)
            class_name = class_names[index].strip()
            confidence_score = prediction[0][index]

            # Overlay prediction on the frame
            cv2.putText(frame, f"Class: {class_name}, Confidence: {confidence_score*100:.2f}%",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame for the video stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Home page
    return render_template('index.html')
def video_feed():
    # Video feed route for the webcam
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running the Werkzeug Server')
    func()  # Shutdown the Flask server
    return "Server shutting down..."


# @app.route('/webcam')
# def webcam():
#     # Webcam page
#     return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    # Video feed route for the webcam
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed')
def stop_feed():
    # Release the camera and stop video feed
    global camera
    camera.release()
    return "Camera Stopped"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port="3000", debug=True)
