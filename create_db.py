# create_fast_db.py
import pickle
from deepface import DeepFace
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TensorFlow warnings

# ================= Enrollment data =================
enrollments = [
    ("23BCE1745", "kan.png"),
    ("23BCE1803", "cav.png"),
    ("23BCE1812", "bha.png"),
    ("23BCE1766", "jey.png"),
    ("23BCE1864", "adhi.png")
]

DB_FILE = "face_fast_db.pkl"
face_db = {}

# ================= Model and detector =================
MODEL_NAME = "ArcFace"    # Can also use "Facenet", "VGG-Face"
DETECTOR = "mtcnn"        # Better face detection than opencv

# ================= Build database =================
for enroll_no, img_path in enrollments:
    try:
        # Use file path directly; DeepFace handles loading/resizing internally
        embedding = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True
        )[0]["embedding"]

        face_db[enroll_no] = embedding
        print(f"[+] Registered {enroll_no}")
    except Exception as e:
        print(f"[!] Failed to register {enroll_no}: {e}")

# ================= Save DB =================
with open(DB_FILE, "wb") as f:
    pickle.dump(face_db, f)

print("✅ Fast face database created successfully!")
