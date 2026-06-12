from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.trajectory import analyze_trajectory
from app.services.emotional_memory import get_conversation_memory

router = APIRouter(prefix="/trajectory", tags=["Emotional Trajectory"])


class TrajectoryInput(BaseModel):
    conversation_id: str
    current_emotions: dict


@router.post("/analyze")
async def analyze_emotional_trajectory(input: TrajectoryInput):
    """
    Analyze emotional trajectory for a conversation.
    Detects escalation, de-escalation, stability and risk levels.
    """
    if not input.conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id is required")

    # Get conversation memory
    memory = get_conversation_memory(input.conversation_id)
    emotional_history = memory.get("emotional_history", [])

    if not emotional_history:
        raise HTTPException(
            status_code=404,
            detail="No emotional history found for this conversation. Send some messages first."
        )

    # Analyze trajectory
    result = analyze_trajectory(emotional_history, input.current_emotions)

    return {
        "conversation_id": input.conversation_id,
        "message_count": memory.get("message_count", 0),
        "trajectory_analysis": result,
        "emotional_history_length": len(emotional_history)
    }