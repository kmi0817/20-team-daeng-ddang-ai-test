from __future__ import annotations

from datetime import datetime
from typing import List

from app.schemas.mission_schema import MissionResult, MissionStatus, MissionType, MissionInput

def default_title(mission_type: MissionType) -> str:
    return {
        MissionType.SIT: "Sit",
        MissionType.DOWN: "Down",
        MissionType.PAW: "Paw",
        MissionType.TURN: "Turn",
        MissionType.JUMP: "Jump",
    }[mission_type]

def analyze_sync_mock(missions: List[MissionInput]) -> List[MissionResult]:
    results: List[MissionResult] = []

    for m in missions:
        title = m.title or default_title(m.missionType)

        if m.missionType in {MissionType.SIT, MissionType.PAW}:
            results.append(
                MissionResult(
                    missionRecordId = m.missionRecordId,
                    missionId = m.missionId,
                    title = title,
                    status = MissionStatus.SUCCESS,
                    confidence = 0.92,
                    message = "지시에 맞춰 동작을 수행했습니다. (mock)",
                )
            )
        else:
            results.append(
                MissionResult(
                    missionRecordId = m.missionRecordId,
                    missionId = m.missionId,
                    title = title,
                    status = MissionStatus.FAIL,
                    confidence = 0.61,
                    message="요구된 동작으로 보기에는 부족했습니다. (mock)",
                )
            )

    return results

def now_iso() -> str:
    return datetime.now().astimezone().isoformat()
