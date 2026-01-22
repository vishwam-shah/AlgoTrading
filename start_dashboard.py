"""
Start script for running both FastAPI backend and Next.js frontend.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def main():
    root_dir = Path(__file__).parent
    
    print("=" * 60)
    print("AI Trading System - Dashboard Startup")
    print("=" * 60)
    
    # Start FastAPI backend
    print("\n[1/2] Starting FastAPI backend on http://localhost:8000...")
    backend_cmd = [
        sys.executable, "-m", "uvicorn", 
        "backend.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000",
        "--reload"
    ]
    
    backend_process = subprocess.Popen(
        backend_cmd,
        cwd=root_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    time.sleep(2)  # Wait for backend to start
    
    # Check if backend started
    if backend_process.poll() is None:
        print("✓ Backend started successfully!")
    else:
        print("✗ Backend failed to start")
        return
    
    # Start Next.js frontend
    print("\n[2/2] Starting Next.js frontend on http://localhost:3000...")
    frontend_dir = root_dir / "frontend"
    
    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        print("Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, shell=True)
    
    frontend_cmd = ["npm", "run", "dev"]
    frontend_process = subprocess.Popen(
        frontend_cmd,
        cwd=frontend_dir,
        shell=True
    )
    
    print("\n" + "=" * 60)
    print("Dashboard is starting up!")
    print("=" * 60)
    print("\n📊 Frontend: http://localhost:3000")
    print("🔌 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop both servers")
    print("=" * 60)
    
    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        backend_process.terminate()
        frontend_process.terminate()
        print("Servers stopped.")


if __name__ == "__main__":
    main()
