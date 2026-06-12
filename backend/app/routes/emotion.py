from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.emotion_detection import detect_emotions, detect_emotions_batch

router = APIRouter(prefix="/emotion", tags=["Emotion Detection"])

class TextInput(BaseModel):
    text: str
    conversation_id: str = "default"

class BatchTextInput(BaseModel):
    texts: list[str]
    conversation_id: str = "default"

@router.post("/detect")
async def detect_emotion(input: TextInput):
    """
    Detect emotions from a single text message.
    Returns probability scores for each emotion.
    """
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = detect_emotions(input.text)
    
    return {
        "conversation_id": input.conversation_id,
        "input_text": input.text,
        "current_emotional_state": result["emotion_scores"],
        "dominant_emotions": result["dominant_emotions"],
        "intensity": result["intensity"],
        "top_emotion": result["top_emotion"]
    }

@router.post("/detect-batch")
async def detect_emotion_batch(input: BatchTextInput):
    """
    Detect emotions from multiple text messages.
    """
    if not input.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    results = detect_emotions_batch(input.texts)
    
    return {
        "conversation_id": input.conversation_id,
        "results": [
            {
                "text": text,
                "current_emotional_state": result["emotion_scores"],
                "dominant_emotions": result["dominant_emotions"],
                "intensity": result["intensity"],
                "top_emotion": result["top_emotion"]
            }
            for text, result in zip(input.texts, results)
        ]
    }