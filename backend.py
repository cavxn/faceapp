from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from deepface import DeepFace
import pickle
import os
from PIL import Image
import io
from datetime import datetime
import uvicorn

# ================= CONFIG =================
MODEL_NAME = "ArcFace"
DETECTOR = "mtcnn"
DB_FILE = "face_fast_db.pkl"
# For cosine similarity, higher is better (1.0 = identical)
COSINE_THRESHOLD = 0.25  # Lowered for better matching

app = FastAPI(title="Face Recognition Attendance API", version="1.0.0")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= Load face database =================
face_db = {}
if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        face_db = pickle.load(f)
    print(f"[+] Loaded {len(face_db)} faces from database")
else:
    print("[!] No face database found. Please run create_db.py first.")
    face_db = {}

# ================= Helper Functions =================
def preprocess_image(image_bytes):
    """Convert uploaded image bytes to numpy array for face recognition"""
    try:
        # Convert bytes to PIL Image, force RGB (handles RGBA/LA/PNG)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Convert PIL to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        return image_cv
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def get_face_embedding(image):
    """Extract face embedding from image. Returns None if no face detected."""
    try:
        reps = DeepFace.represent(
            image,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=False
        )
        if not reps or not isinstance(reps, list):
            return None
        embedding = reps[0].get("embedding")
        if embedding is None:
            return None
        return np.array(embedding)
    except Exception:
        # Treat any detection/representation error as no face found
        return None

def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector) + 1e-10
    return vector / norm

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    vec_a = l2_normalize(vec_a)
    vec_b = l2_normalize(vec_b)
    return float(np.dot(vec_a, vec_b))

def find_best_match(embedding):
    """Find the best matching enrollment number using cosine similarity."""
    if not face_db:
        return None, -1.0

    best_score = -1.0
    best_match = None

    for enroll_no, db_embedding in face_db.items():
        db_embedding = np.array(db_embedding)

        if embedding.shape != db_embedding.shape:
            continue

        score = cosine_similarity(embedding, db_embedding)
        if score > best_score:
            best_score = score
            best_match = enroll_no

    return best_match, best_score

# ================= API Endpoints =================
@app.get("/")
async def root():
    return {"message": "Face Recognition Attendance API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "registered_faces": len(face_db)}

@app.post("/verify-attendance")
async def verify_attendance(
    registerNumber: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Verify attendance by comparing uploaded face image with registered enrollment number
    """
    try:
        # Validate enrollment number
        if not registerNumber:
            raise HTTPException(status_code=400, detail="Enrollment number is required")
        
        # Check if enrollment number exists in database
        if registerNumber not in face_db:
            return {
                "success": False,
                "message": f"Enrollment number {registerNumber} not found in database",
                "status": "ABSENT",
                "enrollment_number": registerNumber,
                "timestamp": datetime.now().isoformat()
            }
        
        # Read and preprocess uploaded image
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)
        
        # Extract face embedding
        face_embedding = get_face_embedding(image)

        if face_embedding is None:
            return {
                "success": False,
                "message": "No detectable face in the image. Please try again with your face centered and well-lit.",
                "status": "ABSENT",
                "enrollment_number": registerNumber,
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }

        # Find best match (cosine similarity)
        best_match, similarity = find_best_match(face_embedding)

        # Determine if it's a match using cosine threshold
        is_match = best_match == registerNumber and similarity >= COSINE_THRESHOLD

        # Confidence is similarity clipped to [0, 1]
        confidence = float(np.clip(similarity, 0.0, 1.0))

        if is_match:
            return {
                "success": True,
                "message": f"Attendance verified for {registerNumber}",
                "status": "PRESENT",
                "enrollment_number": registerNumber,
                "confidence": round(confidence, 3),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "message": f"Face does not match enrollment number {registerNumber}",
                "status": "ABSENT",
                "enrollment_number": registerNumber,
                "confidence": round(confidence, 3) if best_match else 0.0,
                "timestamp": datetime.now().isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/registered-students")
async def get_registered_students():
    """Get list of all registered students"""
    return {
        "students": list(face_db.keys()),
        "count": len(face_db)
    }

@app.post("/register-student")
async def register_student(
    registerNumber: str = Form(...),
    file: UploadFile = File(...)
):
    """Register a new student's face"""
    try:
        if not registerNumber:
            raise HTTPException(status_code=400, detail="Enrollment number is required")
        
        # Read and preprocess image
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)
        
        # Extract face embedding
        face_embedding = get_face_embedding(image)
        
        # Add to database
        face_db[registerNumber] = face_embedding.tolist()
        
        # Save to file
        with open(DB_FILE, "wb") as f:
            pickle.dump(face_db, f)
        
        return {
            "success": True,
            "message": f"Student {registerNumber} registered successfully",
            "enrollment_number": registerNumber
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

if __name__ == "__main__":
    print("Starting Face Recognition Attendance API...")
    print(f"Registered students: {list(face_db.keys())}")
    uvicorn.run(app, host="127.0.0.1", port=8001)
