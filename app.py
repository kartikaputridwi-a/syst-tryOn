from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi import Form
from ultralytics import YOLO
from PIL import Image
import io
import shutil
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from summarizer import summarize_text


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

model = YOLO("models/yolov8n.pt")


@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    results = model(image)

    names = results[0].names  # class ID → name mapping
    labels = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        labels.append(names[cls_id])

    # avoid duplicates, optional
    labels = list(set(labels))

    return JSONResponse(content={"detected_objects": labels})

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def main_page():
    return FileResponse("static/index.html")


@app.get("/summarize")
async def summarize(text: str):
    from summarizer import summarize_text
    
    summary = summarize_text(text)

    print("SUMMARY:", summary)   # 👈 TEMP DEBUG
    
    return {"summary": summary}


