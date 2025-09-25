# app.py
import cv2
import numpy as np
from deepface import DeepFace
import pickle
import os

# ================= CONFIG =================
MODEL_NAME = "ArcFace"         # Must match your database embeddings
DETECTOR = "opencv"            # Fast & stable
DB_FILE = "face_fast_db.pkl"

# ================= STEP 1: Load known embeddings =================
if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        face_db = pickle.load(f)  # {"name": embedding}
    print(f"[+] Loaded {len(face_db)} faces from database")
else:
    face_db = {}
    print("[!] No face database found. Create one first.")

# ================= STEP 2: Register new faces =================
def register_face(name, img_path):
    try:
        embedding = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True
        )[0]["embedding"]

        face_db[name] = embedding
        with open(DB_FILE, "wb") as f:
            pickle.dump(face_db, f)
        print(f"[+] Registered {name}")
    except Exception as e:
        print(f"[!] Failed to register face: {e}")

# ================= STEP 3: Realtime Recognition =================
def recognize_face(frame):
    try:
        result = np.array(DeepFace.represent(
            frame,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=False
        )[0]["embedding"])
    except:
        return "Unknown", frame

    min_dist = float("inf")
    identity = "Unknown"

    for name, db_emb in face_db.items():
        db_emb = np.array(db_emb)  # convert list to array
        # Skip if embedding shapes do not match
        if result.shape != db_emb.shape:
            continue

        dist = np.linalg.norm(result - db_emb)
        if dist < min_dist:
            min_dist = dist
            identity = name

    # Threshold for ArcFace embeddings (typical ~0.6)
    if min_dist > 0.6:
        identity = "Unknown"

    return identity, frame


# ================= STEP 4: Main =================
if __name__ == "__main__":
    print("Press 'q' to quit the camera window.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[!] Could not open webcam")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        name, _ = recognize_face(frame)
        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0) if name != "Unknown" else (0, 0, 255),
                    2, cv2.LINE_AA)
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
