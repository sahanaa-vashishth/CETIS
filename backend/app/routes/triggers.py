from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.trigger_extraction import extract_triggers

router = APIRouter(prefix="/triggers", tags=["Trigger Extraction"])

class TextInput(BaseModel):
    text: str
    conversation_id: str = "default"

@router.post("/extract")
async def extract_trigger(input: TextInput):
    """
    Extract emotional triggers from text using AI.
    Understands any situation without fixed keyword lists.
    """
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    result = await extract_triggers(input.text)

    return {
        "conversation_id": input.conversation_id,
        "input_text": input.text,
        "primary_trigger": result.get("primary_trigger", {}),
        "secondary_triggers": result.get("secondary_triggers", []),
        "trigger_chain": result.get("trigger_chain", [])
    }