# app/schemas/face_schema.py
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class FaceAnalyzeRequest(BaseModel):
    analysis_id: str
    video_url: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)

class FaceAnalyzeResponse(BaseModel):
    analysis_id: str
    analyze_at: str
    processing: Dict[str, Any]
    result: Dict[str, Any] # "emotion": { ... }

class FaceAnalyzeResult(BaseModel):
    emotion: Dict[str, Any]

class FaceErrorResponse(BaseModel):
    request_id: str
    error_code: str
    message: str
    debug: Optional[Dict[str, Any]] = None