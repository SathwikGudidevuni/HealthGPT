# Main script for HealthGPT project
"""
Entry point to start the FastAPI server for HealthGPT.

Run:
    python main.py

This will start uvicorn serving app.api:app on 0.0.0.0:8000
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, log_level="info")
