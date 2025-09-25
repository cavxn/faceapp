# Face Recognition Attendance System

A complete face recognition-based attendance system using FastAPI backend and Streamlit frontend.

## Features

- ✅ Face recognition using DeepFace with ArcFace model
- ✅ Enrollment number-based attendance verification
- ✅ Real-time camera capture in Streamlit
- ✅ FastAPI backend with RESTful endpoints
- ✅ Present/Absent status tracking
- ✅ Confidence scoring for face matches
- ✅ Easy-to-use web interface

## Prerequisites

- Python 3.8 or higher
- Webcam for face capture
- Good lighting conditions for better face detection

## Installation

1. **Clone or download the project files**

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create the face database:**
   ```bash
   python create_db.py
   ```

## Quick Start

### Option 1: Use the startup script (Recommended)
```bash
python start_system.py
```

### Option 2: Manual startup

1. **Start the backend server:**
   ```bash
   python backend.py
   ```
   The backend will run on `http://127.0.0.1:8000`

2. **Start the frontend (in a new terminal):**
   ```bash
   streamlit run streamlit_app.py
   ```
   The frontend will run on `http://localhost:8501`

## How to Use

1. **Open the Streamlit app** in your browser (usually `http://localhost:8501`)

2. **Enter your enrollment number** (e.g., 23BCE1745)

3. **Capture your photo** using the camera interface

4. **Wait for verification** - the system will:
   - Extract face features from your photo
   - Compare with the registered face for that enrollment number
   - Return PRESENT if faces match, ABSENT if they don't

## API Endpoints

The FastAPI backend provides these endpoints:

- `GET /` - Health check
- `GET /health` - Server status and registered faces count
- `POST /verify-attendance` - Verify attendance with enrollment number and photo
- `GET /registered-students` - List all registered students
- `POST /register-student` - Register a new student

## File Structure

```
project/
├── backend.py              # FastAPI backend server
├── streamlit_app.py        # Streamlit frontend
├── create_db.py           # Create face database from images
├── register.py            # Alternative registration script
├── app.py                 # Original face recognition script
├── start_system.py        # Startup helper script
├── requirements.txt       # Python dependencies
├── face_fast_db.pkl      # Face embeddings database
├── *.jpg, *.PNG          # Student photos
└── README.md             # This file
```

## Registered Students

The system comes pre-configured with these students:
- 23BCE1745 (kanishk.jpg)
- 23BCE1803 (cavin.PNG)
- 23BCE1812 (bharath.PNG)
- 23BCE1766 (jeya.PNG)
- 23BCE1864 (adithya.jpg)

## Troubleshooting

### Common Issues:

1. **"Cannot connect to backend server"**
   - Make sure the backend is running on port 8000
   - Check if another process is using port 8000

2. **"Face detection failed"**
   - Ensure good lighting
   - Keep your face centered in the camera
   - Try capturing the image again

3. **"Enrollment number not found"**
   - Check if the enrollment number is registered
   - Use the correct format (e.g., 23BCE1745)

4. **Import errors**
   - Install all requirements: `pip install -r requirements.txt`
   - Make sure you're using Python 3.8+

### Performance Tips:

- Use good lighting for better face detection
- Keep your face centered and look directly at the camera
- Avoid wearing hats or sunglasses that might obscure your face
- The system works best with clear, well-lit photos

## Technical Details

- **Face Recognition Model**: ArcFace (via DeepFace)
- **Face Detection**: OpenCV
- **Backend**: FastAPI with CORS enabled
- **Frontend**: Streamlit with camera input
- **Database**: Pickle file with face embeddings
- **Similarity Threshold**: 0.6 (configurable)

## License

This project is for educational purposes. Make sure to comply with privacy laws when using face recognition technology.
