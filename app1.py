# app1.py
import pickle
import numpy as np
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from deepface import DeepFace
from io import BytesIO
from PIL import Image
import os
import logging

# ---------------- Settings ----------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU
logging.getLogger("tensorflow").setLevel(logging.ERROR)

app = FastAPI(title="Fast Face Attendance API")

# ---------------- Static Files ----------------
# Serve frontend folder (index.html, js, css)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------- Face DB ----------------
DB_FILE = "face_fast_db.pkl"

try:
    with open(DB_FILE, "rb") as f:
        face_db = pickle.load(f)
        print(f"[+] Loaded {len(face_db)} faces from database")
except FileNotFoundError:
    face_db = {}
    print("[!] Face database not found. Create 'face_fast_db.pkl' first.")

# ---------------- Load Model ----------------
print("[*] Loading Facenet model...")
model = DeepFace.build_model("Facenet")
print("[*] Facenet model loaded.")

# ---------------- Face Recognition ----------------
def recognize_face_bytes(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)

    try:
        embedding = DeepFace.represent(
            img_path=img_array,
            model_name="Facenet",
            model=model,
            detector_backend="opencv",
            enforce_detection=False
        )[0]["embedding"]
    except:
        return "Unknown", 0.0

    min_dist = float("inf")
    identity = "Unknown"
    for reg, db_emb in face_db.items():
        dist = np.linalg.norm(np.array(embedding) - np.array(db_emb))
        if dist < min_dist:
            min_dist = dist
            identity = reg

    threshold = 10  # tweak if needed
    similarity = 1 / (1 + min_dist)
    if min_dist > threshold:
        identity = "Unknown"

    return identity, similarity

# ---------------- API Endpoints ----------------
@app.get("/", response_class=HTMLResponse)
async def root():
    # Serve index.html from static folder
    return FileResponse("static/index.html")

@app.post("/verify-attendance")
async def verify_attendance(registerNumber: str = Form(...), file: UploadFile = None):
    if not file:
        return JSONResponse({"message": "❌ No image uploaded"}, status_code=400)

    image_bytes = await file.read()
    identity, similarity = recognize_face_bytes(image_bytes)
    similarity_percent = round(similarity * 100, 2)

    if identity == registerNumber:
        return JSONResponse({"message": f"✅ Attendance marked for {registerNumber} (Similarity: {similarity_percent}%)"})
    else:
        return JSONResponse({"message": f"❌ Face does not match register number (Similarity: {similarity_percent}%)"})
