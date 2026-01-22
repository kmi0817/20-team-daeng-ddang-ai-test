import os
import sys
import time
import json
import threading
import http.server
import socketserver
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from app.services.gemini_client import GeminiClient
except Exception as import_err:  # pragma: no cover - only for local dev without deps
    GeminiClient = None
    GEMINI_IMPORT_ERROR = import_err
else:
    GEMINI_IMPORT_ERROR = None

REASON_PROMPT_TEMPLATE = """
당신은 반려견 훈련 임무의 최종 판정을 이미 확인했습니다.

- Mission Type: {mission_type}
- Final Judgment: {judgment}

영상에서 관찰된 가장 결정적인 시각적 단서를 1~2문장으로 설명하세요.
자세/지속시간/시야 가림/핸들러 상호작용 등 구체적인 요소를 언급하고,
불분명할 경우 그 이유를 명확히 기술하세요.
""".strip()

_gemini_client = None

def ensure_gemini_client() -> GeminiClient:
    if GeminiClient is None:
        raise RuntimeError(f"GeminiClient import 실패: {GEMINI_IMPORT_ERROR}")

    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client

def build_reason_prompt(mission_type: str, success: bool) -> str:
    judgment = "SUCCESS" if success else "FAIL"
    return REASON_PROMPT_TEMPLATE.format(mission_type=mission_type, judgment=judgment)

def fetch_reason_from_gemini(video_url: str, mission_type: str, success: bool) -> str:
    client = ensure_gemini_client()
    prompt = build_reason_prompt(mission_type, success)
    reason_text = client.generate_text_from_video_url(
        video_url = video_url,
        prompt_text = prompt,
    )
    return reason_text.strip()

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
    test_cases = [
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/down.mp4", "mission": "DOWN"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/jump.mp4", "mission": "JUMP"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/paw.mp4", "mission": "PAW"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/sit_small.mp4", "mission": "SIT"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/spin.mp4", "mission": "TURN"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/IMG_6135.mov", "mission": "DOWN"}, #실패케이스
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/IMG_6137.mov", "mission": "DOWN"}, #실패케이스
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/DOWN_02.mp4", "mission": "DOWN"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/DOWN_03.mp4", "mission": "DOWN"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/DOWN_04.mp4", "mission": "DOWN"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/DOWN_05.mp4", "mission": "DOWN"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/DOWN_06.mp4", "mission": "DOWN"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/JUMP_02.mp4", "mission": "JUMP"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/JUMP_03.mp4", "mission": "JUMP"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/JUMP_04.mp4", "mission": "JUMP"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/JUMP_05.mp4", "mission": "JUMP"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/JUMP_06.mp4", "mission": "JUMP"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/PAW_02.mp4", "mission": "PAW"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/PAW_03.mp4", "mission": "PAW"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/PAW_04.mp4", "mission": "PAW"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/PAW_05.mp4", "mission": "PAW"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/PAW_06.mp4", "mission": "PAW"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/SIT_02.mp4", "mission": "SIT"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/SIT_03.mp4", "mission": "SIT"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/SIT_04.mp4", "mission": "SIT"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/SIT_05.mp4", "mission": "SIT"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/SIT_06.mp4", "mission": "SIT"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/TURN_02.mp4", "mission": "TURN"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/TURN_03.mp4", "mission": "TURN"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/TURN_04.mp4", "mission": "TURN"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/TURN_05.mp4", "mission": "TURN"},
        {"url": "https://daeng-map.s3.ap-northeast-2.amazonaws.com/test_set/TURN_06.mp4", "mission": "TURN"}
        
    ]

    print(f"[Test] Starting sequential test for {len(test_cases)} videos...\n")

    for i, case in enumerate(test_cases):
        video_url = case["url"]
        mission_type = case["mission"]
        
        print(f"=== [Test Case {i+1}] Mission: {mission_type} ===")
        print(f"URL: {video_url}")

        payload = {
            "analysis_id": f"test-analysis-{i+1}",
            "walk_id": f"test-walk-{i+1}",
            "missions": [
                {
                    "mission_id": f"m_{i+1}",
                    "mission_type": mission_type,
                    "video_url": video_url
                }
            ]
        }

        print(f"Sending Request to {API_URL}...", end=" ", flush=True)
        try:
            start_time = time.time()
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            elapsed = time.time() - start_time
            
            result = response.json()
            mission_res = result["missions"][0]
            status = "SUCCESS" if mission_res["success"] else "FAIL"
            
            print(f"DONE ({elapsed:.2f}s)")
            print(f"Result: {status} (Confidence: {mission_res['confidence']:.2f})")
            
            reason_text = (mission_res.get("reason") or "").strip()
            if not reason_text:
                try:
                    reason_text = fetch_reason_from_gemini(
                        video_url = video_url,
                        mission_type = mission_type,
                        success = mission_res["success"],
                    )
                except Exception as reason_error:
                    reason_text = f"[Reason 생성 실패: {reason_error}]"

            print(f"Reason: {reason_text}")
            
        except requests.exceptions.HTTPError as e:
            print(f"FAILED (HTTP {response.status_code})")
            print(f"Response: {response.text}")
        except Exception as e:
            print(f"ERROR: {e}")
        
        print("-" * 50)
        time.sleep(1)

if __name__ == "__main__":
    test_judge()
