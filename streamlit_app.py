# streamlit_app.py
import streamlit as st
import requests
from PIL import Image
import io
import time
from datetime import datetime

API_URL = "http://127.0.0.1:8001/verify-attendance"
HEALTH_URL = "http://127.0.0.1:8001/health"
STUDENTS_URL = "http://127.0.0.1:8001/registered-students"

st.set_page_config(page_title="Face Attendance System", page_icon="📸", layout="wide")
st.title("📸 Face Recognition Attendance System")

# Sidebar for server status
with st.sidebar:
    st.header("Server Status")
    try:
        health_response = requests.get(HEALTH_URL, timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success("✅ Backend Connected")
            st.info(f"Registered Students: {health_data.get('registered_faces', 0)}")
        else:
            st.error("❌ Backend Error")
    except requests.exceptions.ConnectionError:
        st.error("❌ Backend Offline")
        st.warning("Please start the backend server first!")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Attendance Verification")
    
    # Get list of registered students
    try:
        students_response = requests.get(STUDENTS_URL, timeout=5)
        if students_response.status_code == 200:
            students_data = students_response.json()
            registered_students = students_data.get('students', [])
            
            if registered_students:
                st.info(f"Available Students: {', '.join(registered_students)}")
            else:
                st.warning("No students registered yet!")
        else:
            registered_students = []
    except:
        registered_students = []
    
    enroll_number = st.text_input(
        "Enter your Enrollment Number:", 
        placeholder="e.g., 23BCE1745",
        help="Enter your enrollment number to verify attendance"
    )
    
    if enroll_number:
        st.write("📷 **Camera Instructions:**")
        st.write("1. Look directly at the camera")
        st.write("2. Ensure good lighting")
        st.write("3. Keep your face centered")
        
        captured_image = st.camera_input("Look at the camera!", key="attendance_camera")
        
        if captured_image:
            st.info("🔄 Processing image...")
            
            # Show captured image
            image_bytes = captured_image.getvalue()
            image = Image.open(io.BytesIO(image_bytes))
            
            with st.spinner("Verifying attendance..."):
                files = {"file": ("captured.jpg", image_bytes, "image/jpeg")}
                data = {"registerNumber": enroll_number}
                
                try:
                    response = requests.post(API_URL, data=data, files=files, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display result
                        if result.get("success"):
                            st.success(f"✅ {result.get('message')}")
                            st.balloons()
                        else:
                            st.error(f"❌ {result.get('message')}")
                        
                        # Show additional details
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Status", result.get("status", "UNKNOWN"))
                        with col_b:
                            if "confidence" in result:
                                st.metric("Confidence", f"{result['confidence']:.1%}")
                        
                        # Show timestamp
                        if "timestamp" in result:
                            timestamp = datetime.fromisoformat(result["timestamp"].replace('Z', '+00:00'))
                            st.caption(f"Verified at: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    else:
                        error_data = response.json()
                        st.error(f"❌ Error: {error_data.get('message', 'Unknown error')}")
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend server!")
                    st.warning("Please make sure the backend is running at http://127.0.0.1:8001")
                except requests.exceptions.Timeout:
                    st.error("⏰ Request timed out. Please try again.")
                except Exception as e:
                    st.error(f"❌ Server error: {str(e)}")

with col2:
    st.header("Captured Image")
    if 'captured_image' in locals() and captured_image:
        st.image(image, caption="Your captured image", use_container_width=True)
    else:
        st.info("📸 Capture an image to see it here")
    
    st.header("Instructions")
    st.markdown("""
    ### How to use:
    1. **Enter Enrollment Number**: Type your enrollment number in the text field
    2. **Capture Photo**: Click the camera button and look at the camera
    3. **Wait for Verification**: The system will process your image
    4. **Check Result**: You'll see if you're marked as PRESENT or ABSENT
    
    ### Troubleshooting:
    - Make sure the backend server is running
    - Ensure good lighting for better face detection
    - Keep your face centered in the camera view
    - If you get an error, try capturing the image again
    """)

# Footer
st.markdown("---")
st.caption("Face Recognition Attendance System | Powered by DeepFace & FastAPI")
