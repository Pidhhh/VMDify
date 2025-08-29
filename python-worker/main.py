from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import asyncio
import json
import os
import tempfile
from typing import Dict, List
import uuid
from pipeline import pipeline
from typing import Optional

# Create the FastAPI application
app = FastAPI(
    title="VMDify ML Worker",
    description="Handles AI-based pose estimation and motion generation.",
    version="0.1.0",
)

# Enable CORS to allow Electron frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Electron app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
active_connections: List[WebSocket] = []

@app.get("/probe")
def probe_system_health():
    """
    Checks system health, including Python environment and GPU availability.
    """
    try:
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
        gpu_count = torch.cuda.device_count() if gpu_available else 0
    except Exception as e:
        gpu_available = False
        gpu_name = f"Error checking GPU: {e}"
        gpu_count = 0
    
    return {
        "status": "ok",
        "message": "VMDify ML Worker is running.",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "torch_version": torch.__version__,
    }

@app.post("/process_video")
async def process_video(video: UploadFile = File(...)):
    """
    Start processing a video file through the AI pipeline.
    This will be implemented with the full pipeline later.
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create temporary file for uploaded video
        temp_dir = tempfile.gettempdir()
        file_extension = "mp4"  # Default extension
        if video.filename and "." in video.filename:
            file_extension = video.filename.split('.')[-1]
        temp_file_path = os.path.join(temp_dir, f"vmdify_input_{job_id}.{file_extension}")
        
        # Save uploaded file
        with open(temp_file_path, "wb") as temp_file:
            content = await video.read()
            temp_file.write(content)
        
        # TODO: Start actual AI processing pipeline here
        print(f"Received video: {video.filename} ({len(content)} bytes)")
        print(f"Saved to: {temp_file_path}")
        print(f"Job ID: {job_id}")
        
        # Start the AI pipeline (placeholder - will run in background)
        # For now, just simulate processing
        pipeline_result = await pipeline.process_video(temp_file_path, job_id)
        
        # Optionally broadcast a completion message over any active websockets
        try:
            payload = {
                "type": "job_complete",
                "job_id": job_id,
                "output_path": pipeline_result.get("output_path"),
                "debug_json_path": pipeline_result.get("debug_json_path"),
                "preview_trajectory": pipeline_result.get("preview_trajectory"),
                "total_frames": pipeline_result.get("total_frames", 0)
            }
            for ws in list(active_connections):
                await ws.send_text(json.dumps(payload))
        except Exception as _:
            pass

        return {
            "status": "completed",
            "job_id": job_id,
            "filename": video.filename,
            "file_size": len(content),
            "pipeline_result": pipeline_result,
            "message": "Video processed through AI pipeline (placeholder implementation)"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to process video: {str(e)}"
        }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time communication with the frontend.
    This will stream bone animation data during processing.
    """
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_text(json.dumps({
            "type": "connection",
            "message": "Connected to VMDify ML Worker"
        }))
        
        # Keep connection alive and listen for messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Echo back for now (will implement real processing later)
            await websocket.send_text(json.dumps({
                "type": "response",
                "original_message": message,
                "timestamp": str(asyncio.get_event_loop().time())
            }))
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print("WebSocket connection closed")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the VMDify ML Worker",
        "endpoints": {
            "health_check": "/probe",
            "process_video": "/process_video",
            "websocket": "/ws"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
