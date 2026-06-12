from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.contextual_interpretation import interpret_contextual_signals
from app.services.emotional_memory import get_conversation_memory

router = APIRouter(prefix="/contextual", tags=["Contextual Interpretation"])


class ContextualInput(BaseModel):
    conversation_id: str
    current_emotions: dict
    primary_trigger: dict


@router.post("/interpret")
async def interpret_context(input: ContextualInput):
    """
    Interpret deeper psychological meaning beneath surface emotions.
    Distinguishes surface emotions from deeper emotional drivers.
    Detects cognitive patterns.
    """
    if not input.conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id is required")

    # Get conversation memory to access history
    memory = get_conversation_memory(input.conversation_id)
    emotional_history = memory.get("emotional_history", [])
    
    # Extract recent messages from history
    recent_messages = [
        entry.get("message", "") for entry in emotional_history[-5:]
    ]

    # Get contextual interpretation
    result = await interpret_contextual_signals(
        current_emotions=input.current_emotions,
        primary_trigger=input.primary_trigger,
        emotional_history=emotional_history,
        recent_messages=recent_messages
    )

    return {
        "conversation_id": input.conversation_id,
        "conversation_interpretation": result,
        "current_emotions": input.current_emotions,
        "primary_trigger": input.primary_trigger
    }