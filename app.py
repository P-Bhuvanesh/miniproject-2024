import os
import cv2
import numpy as np
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.responses import StreamingResponse
from keras.models import load_model
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Set up the Jinja2 templates (assuming the templates folder exists)
templates = Jinja2Templates(directory="templates")

# Load the model and labels
model_path = os.path.join(os.path.dirname(__file__), "model/keras_model.h5")
model = load_model(model_path, compile=False)
label_path = os.path.join(os.path.dirname(__file__), "model/labels.txt")
class_names = open(label_path, "r").readlines()

# Initialize the camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")


# Generator function to capture frames from the camera
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            image_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            image_resized = np.asarray(image_resized, dtype=np.float32).reshape(
                1, 224, 224, 3
            )
            image_resized = (image_resized / 127.5) - 1

            prediction = model.predict(image_resized)
            index = np.argmax(prediction)
            class_name = class_names[index].strip()
            confidence_score = prediction[0][index]

            cv2.putText(
                frame,
                f"Class: {class_name}, Confidence: {confidence_score*100:.2f}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/shutdown")
async def shutdown():
    global camera
    camera.release()
    os._exit(0)  # This will forcefully stop the FastAPI server
    return {"message": "Server shutting down..."}


@app.get("/stop_feed")
async def stop_feed():
    global camera
    camera.release()
    return {"message": "Camera Stopped"}


# Use background tasks to stop feed asynchronously
@app.get("/stop_feed_async")
async def stop_feed_async(background_tasks: BackgroundTasks):
    background_tasks.add_task(camera.release)
    return {"message": "Stopping Camera in Background..."}
