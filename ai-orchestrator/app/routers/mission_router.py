from __future__ import annotations

from fastapi import APIRouter, Path

from app.schemas.mission_schema import ApiResponse, MissionAnalysisData, MissionAnalysisRequest
from app.services.mission_service import analyze_sync_mock, now_iso

router = APIRouter(prefix = "/internal/v1", tags = ["mission"])


@router.post("/walks/{walkId}/missions/analysis", response_model = ApiResponse)
def analyze_missions_sync(
    req: MissionAnalysisRequest,
    walkId: int = Path(..., ge=1),
):
    results = analyze_sync_mock(req.missions)

    data = MissionAnalysisData(
        walkId = walkId,
        analyzedAt = now_iso(),
        missions = results,
    )

    return ApiResponse(
        message = "mission analysis completed",
        data = data.model_dump(),
        errorCode = None,
    )
