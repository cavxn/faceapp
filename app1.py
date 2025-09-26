import pickle, numpy as np, os, logging
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from io import BytesIO
from PIL import Image

# Google Sheets
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ---------------- Settings ----------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

app = FastAPI(title="Fast Face Attendance API")

# CORS for frontend
origins = ["https://YOUR_FRONTEND_URL"]
app.add_middleware(
    CORSMiddleware, allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------- Face Database ----------------
DB_FILE = "face_fast_db.pkl"
try:
    with open(DB_FILE, "rb") as f: face_db = pickle.load(f)
except FileNotFoundError:
    face_db = {}

# ---------------- Load Facenet ----------------
model = DeepFace.build_model("Facenet")

# ---------------- Google Sheets ----------------
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SERVICE_ACCOUNT_FILE = "service_account.json"
SPREADSHEET_ID = "YOUR_SPREADSHEET_ID"
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
sheets_service = build("sheets", "v4", credentials=credentials)

def append_to_sheets(register_number, code):
    sheets_service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range="Sheet1!A:B",
        valueInputOption="RAW",
        body={"values": [[register_number, code]]}
    ).execute()

# ---------------- Face Recognition ----------------
def recognize_face_bytes(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)
    try:
        embedding = DeepFace.represent(
            img_path=img_array, model_name="Facenet", model=model,
            detector_backend="opencv", enforce_detection=False
        )[0]["embedding"]
    except:
        return "Unknown", 0.0

    identity, min_dist = "Unknown", float("inf")
    for reg, db_emb in face_db.items():
        dist = np.linalg.norm(np.array(embedding) - np.array(db_emb))
        if dist < min_dist: min_dist, identity = dist, reg

    threshold = 10
    similarity = 1 / (1 + min_dist)
    if min_dist > threshold: identity = "Unknown"
    return identity, similarity

# ---------------- API ----------------
@app.get("/")
async def root(): return {"message": "Face Attendance API running"}

@app.post("/verify-attendance")
async def verify_attendance(registerNumber: str = Form(...), file: UploadFile = None):
    if not file: return JSONResponse({"message": "❌ No image uploaded"}, status_code=400)
    image_bytes = await file.read()
    identity, similarity = recognize_face_bytes(image_bytes)
    similarity_percent = round(similarity * 100, 2)

    if identity == registerNumber:
        attendance_code = f"CODE-{np.random.randint(1000,9999)}"
        append_to_sheets(registerNumber, attendance_code)
        return JSONResponse({
            "message": f"✅ Attendance marked for {registerNumber} (Similarity: {similarity_percent}%)",
            "code": attendance_code
        })
    else:
        return JSONResponse({
            "message": f"❌ Face does not match register number (Similarity: {similarity_percent}%)"
        })

# ---------------- Run Server ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
