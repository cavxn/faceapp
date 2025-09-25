#!/usr/bin/env python3
"""
Face Recognition Attendance System Startup Script
This script helps you start both the backend and frontend components
"""

import subprocess
import sys
import time
import os
import threading
import webbrowser
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import fastapi
        import streamlit
        import deepface
        import cv2
        import numpy
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    try:
        subprocess.run([sys.executable, "backend.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")

def start_frontend():
    """Start the Streamlit frontend"""
    print("🚀 Starting Streamlit frontend...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")

def main():
    print("=" * 60)
    print("🎯 Face Recognition Attendance System")
    print("=" * 60)
    
    # Check if database exists
    if not os.path.exists("face_fast_db.pkl"):
        print("⚠️  Face database not found!")
        print("Creating database from existing images...")
        try:
            subprocess.run([sys.executable, "create_db.py"], check=True)
            print("✅ Database created successfully!")
        except Exception as e:
            print(f"❌ Error creating database: {e}")
            return
    
    # Check requirements
    if not check_requirements():
        return
    
    print("\nChoose an option:")
    print("1. Start Backend Only (FastAPI)")
    print("2. Start Frontend Only (Streamlit)")
    print("3. Start Both (Recommended)")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        start_backend()
    elif choice == "2":
        start_frontend()
    elif choice == "3":
        print("\n🚀 Starting both servers...")
        print("Backend will run on: http://127.0.0.1:8000")
        print("Frontend will run on: http://localhost:8501")
        print("\nPress Ctrl+C to stop both servers")
        
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=start_backend, daemon=True)
        backend_thread.start()
        
        # Wait a moment for backend to start
        time.sleep(3)
        
        # Start frontend (this will block)
        start_frontend()
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
