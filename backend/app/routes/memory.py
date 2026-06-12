from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.emotional_memory import (
    update_emotional_memory,
    get_emotional_summary,
    initialize_conversation
)

router = APIRouter(prefix="/memory", tags=["Emotional Memory"])


class MemoryUpdateInput(BaseModel):
    conversation_id: str
    message: str
    emotion_scores: dict
    trigger: dict


class ConversationInput(BaseModel):
    conversation_id: str


@router.post("/update")
async def update_memory(input: MemoryUpdateInput):
    """
    Update emotional memory for a conversation.
    Uses exponential moving average to smooth emotional states.
    """
    if not input.conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id is required")

    result = update_emotional_memory(
        conversation_id=input.conversation_id,
        emotion_scores=input.emotion_scores,
        trigger=input.trigger,
        message=input.message
    )

    return result


@router.get("/summary/{conversation_id}")
async def get_memory_summary(conversation_id: str):
    """
    Get full emotional memory summary for a conversation.
    """
    result = get_emotional_summary(conversation_id)
    return result


@router.post("/initialize")
async def initialize_new_conversation(input: ConversationInput):
    """
    Initialize a new conversation memory.
    """
    if not input.conversation_id:
        raise HTTPException(status_code=400, detail="conversation_id is required")

    result = initialize_conversation(input.conversation_id)
    return {
        "message": f"Conversation {input.conversation_id} initialized",
        "conversation_id": input.conversation_id
    }