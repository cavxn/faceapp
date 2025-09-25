# register_faces.py
import os
import pickle
from deepface import DeepFace

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

face_db = {}

students = {
    "23BCE1745": "kanishk.jpg",
    "23BCE1803": "cavin.PNG",
    "23BCE1812": "bharath.PNG",
    "23BCE1766": "jeya.PNG",
    "23BCE1864": "adithya.jpg"
}

for regno, img_path in students.items():
    try:
        embedding = DeepFace.represent(img_path=img_path, model_name="Facenet", detector_backend="mtcnn")[0]["embedding"]
        face_db[regno] = embedding
        print(f"[+] Registered {regno}")
    except Exception as e:
        print(f"[!] Failed to register {regno}: {e}")

with open("face_db.pkl", "wb") as f:
    pickle.dump(face_db, f)

print("✅ All faces registered and pickle file created.")
