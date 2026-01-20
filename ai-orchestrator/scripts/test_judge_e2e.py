import os
import time
import json
import threading
import http.server
import socketserver
import requests

API_URL = "http://localhost:8000/api/missions/judge"
VIDEO_PORT = 9000
VIDEO_FILENAME = "test_data/sample_sit.mp4"
LOCAL_IP = "localhost"

def run_video_server():
    handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", VIDEO_PORT), handler) as httpd:
        print(f"[Video Server] Serving directory at port {VIDEO_PORT}")
        httpd.serve_forever()

def test_judge():    
    if not os.path.exists(VIDEO_FILENAME):
        print(f"[Error] Video file not found: {VIDEO_FILENAME}")
        print("Please create 'test_data' folder and put a 'sample_sit.mp4' file in it.")
        return

    server_thread = threading.Thread(target=run_video_server, daemon=True)
    server_thread.start()
    time.sleep(2)

    video_url = f"http://{LOCAL_IP}:{VIDEO_PORT}/{VIDEO_FILENAME}"
    print(f"[Test] Using Video URL: {video_url}")

    payload = {
        "analysis_id": "test-analysis-001",
        "walk_id": "test-walk-001",
        "missions": [
            {
                "mission_id": "m_001",
                "mission_type": "SIT",
                "video_url": video_url
            }
        ]
    }

    print(f"[Test] Sending Request to {API_URL}...")
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        
        print("\n[Result] Success!")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        
    except requests.exceptions.ConnectionError:
        print("\n[Error] Could not connect to API Server.")
        print("Make sure you are running 'uvicorn app.main:app' in another terminal.")
    except Exception as e:
        print(f"\n[Error] Request failed: {e}")
        
        if 'response' in locals():
            print(f"Server Response: {response.text}")

if __name__ == "__main__":
    test_judge()
