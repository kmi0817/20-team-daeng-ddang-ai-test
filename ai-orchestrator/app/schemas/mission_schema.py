from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class ApiResponse(BaseModel):
    message: str
    data: Optional[dict] = None
    errorCode: Optional[str] = None


class MissionType(str, Enum):
    SIT = "SIT"
    DOWN = "DOWN"
    PAW = "PAW"
    TURN = "TURN"
    JUMP = "JUMP"


class MissionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"


class MissionInput(BaseModel):
    missionRecordId: int = Field(..., ge = 1)
    missionId: int = Field(..., ge = 1)
    missionType: MissionType
    missionVideoUrl: HttpUrl
    title: Optional[str] = None


class MissionAnalysisRequest(BaseModel):
    missions: List[MissionInput] = Field(..., min_length = 1)


class MissionResult(BaseModel):
    missionRecordId: int
    missionId: int
    title: str
    status: MissionStatus
    confidence: float = Field(..., ge = 0.0, le = 1.0)
    message: str


class MissionAnalysisData(BaseModel):
    walkId: int
    analyzedAt: str
    missions: List[MissionResult]
