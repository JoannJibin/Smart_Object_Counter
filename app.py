from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import os

app = FastAPI()

model = YOLO("yolov8n.pt")

templates = Jinja2Templates(directory="templates")

os.makedirs("static/results", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")


# HOME PAGE
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


# UPLOAD PAGE
@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


# WEBCAM PAGE
@app.get("/webcam", response_class=HTMLResponse)
async def webcam_page(request: Request):
    return templates.TemplateResponse("webcam.html", {"request": request})


# IMAGE DETECTION
@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    contents = await file.read()

    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(img)

    detections = results[0].boxes

    counts = defaultdict(int)

    for box in detections:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        counts[class_name] += 1

    annotated = results[0].plot()

    output_path = "static/results/output.jpg"
    cv2.imwrite(output_path, annotated)

    return {"counts": counts, "image": output_path}


# WEBCAM FRAME DETECTION
@app.post("/detect_frame")
async def detect_frame(file: UploadFile = File(...)):

    contents = await file.read()

    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(frame)

    detections = results[0].boxes

    counts = defaultdict(int)

    for box in detections:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        counts[class_name] += 1

    annotated = results[0].plot()

    _, buffer = cv2.imencode(".jpg", annotated)

    return {
        "counts": counts,
        "image": buffer.tobytes().hex()
    }