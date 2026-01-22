import os
import requests
from typing import Optional

from google import genai
from google.genai import types

class GeminiClient:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set in environment variables.")

        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        self._judgment_schema = types.Schema(
            type = types.Type.OBJECT,
            required = ["success", "confidence", "reason"],
            properties = {
                "success": types.Schema(
                    type = types.Type.BOOLEAN,
                    description = "Whether the mission succeeded based on the video."
                ),
                "confidence": types.Schema(
                    type = types.Type.NUMBER,
                    description = "Confidence score between 0 and 1."
                ),
                "reason": types.Schema(
                    type = types.Type.STRING,
                    description = "1-2 sentence explanation referencing decisive visual cues."
                ),
            },
        )

    def generate_from_video_url(
        self,
        video_url: str,
        prompt_text: str,
        *,
        response_mime_type: str = "application/json",
        response_schema: Optional[types.Schema] = None,
    ) -> str:
        """
        Generates content using Gemini.
        - If URL is local (localhost/127.0.0.1), downloads bytes and sends inline.
        - If URL is public, sends URL directly to Gemini.
        """
        is_local = "localhost" in video_url or "127.0.0.1" in video_url
        
        parts = []
        
        if is_local:
            try:
                r = requests.get(video_url, timeout=30)
                r.raise_for_status()
                parts = [
                    types.Part.from_bytes(data=r.content, mime_type="video/mp4"),
                    types.Part.from_text(text=prompt_text),
                ]
            except Exception as e:
                raise RuntimeError(f"Failed to fetch local video: {e}")
        else:
            parts = [
                types.Part.from_uri(
                    file_uri=video_url,
                    mime_type="video/mp4"
                ),
                types.Part.from_text(text=prompt_text),
            ]

        config_kwargs = {"response_mime_type": response_mime_type}
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema

        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=[types.Content(parts=parts)],
            config=types.GenerateContentConfig(**config_kwargs)
        )
        return resp.text or ""

    def generate_judgment_from_video_url(self, video_url: str, prompt_text: str) -> str:
        return self.generate_from_video_url(
            video_url = video_url,
            prompt_text = prompt_text,
            response_schema = self._judgment_schema,
        )

    def generate_text_from_video_url(self, video_url: str, prompt_text: str) -> str:
        return self.generate_from_video_url(
            video_url = video_url,
            prompt_text = prompt_text,
            response_mime_type = "text/plain",
        )
