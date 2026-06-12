from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.llm_reasoning import generate_emotional_reasoning
from app.services.emotional_memory import get_conversation_memory
from app.services.trajectory import analyze_trajectory
from app.services.contextual_interpretation import interpret_contextual_signals
from app.services.trigger_extraction import extract_triggers
from app.services.emotion_detection import detect_emotions

router = APIRouter(prefix="/reasoning", tags=["LLM Reasoning"])


class ReasoningInput(BaseModel):
    conversation_id: str
    message: str


@router.post("/analyze")
async def full_emotional_analysis(input: ReasoningInput):
    """
    Full CETIS pipeline — runs all modules and returns the master output format.
    Single endpoint that combines emotion detection, trigger extraction,
    memory, trajectory, contextual interpretation and LLM reasoning.
    """
    if not input.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Step 1 — Detect emotions
    emotion_result = detect_emotions(input.message)
    current_emotional_state = emotion_result["emotion_scores"]
    dominant_emotions = emotion_result["dominant_emotions"]

    # Step 2 — Extract triggers
    trigger_result = await extract_triggers(input.message)
    primary_trigger = trigger_result.get("primary_trigger", {})
    secondary_triggers = trigger_result.get("secondary_triggers", [])
    trigger_chain = trigger_result.get("trigger_chain", [])

    # Step 3 — Update emotional memory
    from app.services.emotional_memory import update_emotional_memory
    memory_result = update_emotional_memory(
        conversation_id=input.conversation_id,
        emotion_scores=current_emotional_state,
        trigger=primary_trigger,
        message=input.message
    )

    # Step 4 — Get full memory
    memory = get_conversation_memory(input.conversation_id)
    emotional_history = memory.get("emotional_history", [])

    # Step 5 — Analyze trajectory
    trajectory_result = analyze_trajectory(emotional_history, current_emotional_state)

    # Step 6 — Contextual interpretation
    recent_messages = [entry.get("message", "") for entry in emotional_history[-5:]]
    interpretation_result = await interpret_contextual_signals(
        current_emotions=current_emotional_state,
        primary_trigger=primary_trigger,
        emotional_history=emotional_history,
        recent_messages=recent_messages
    )

    # Step 7 — LLM Reasoning
    persistent_memory = {
        "historical_patterns": memory_result.get("historical_patterns", []),
        "baseline_emotional_state": memory_result.get("baseline_emotional_state", {}),
        "recurring_triggers": memory_result.get("recurring_triggers", [])
    }

    reasoning_result = await generate_emotional_reasoning(
        current_emotional_state=current_emotional_state,
        dominant_emotions=dominant_emotions,
        primary_trigger=primary_trigger,
        secondary_triggers=secondary_triggers,
        trigger_chain=trigger_chain,
        trajectory_analysis=trajectory_result,
        conversation_interpretation=interpretation_result,
        persistent_memory=persistent_memory,
        message_count=memory_result.get("message_count", 1)
    )

    # Final master output format
    return {
        "current_emotional_state": current_emotional_state,
        "dominant_emotions": dominant_emotions,
        "primary_trigger": primary_trigger,
        "secondary_triggers": secondary_triggers,
        "trigger_chain": trigger_chain,
        "trajectory_analysis": trajectory_result,
        "detected_contextual_signals": reasoning_result.get("detected_contextual_signals", []),
        "conversation_interpretation": {
            "surface_state": interpretation_result.get("surface_state", ""),
            "deeper_state": interpretation_result.get("deeper_state", ""),
            "cognitive_patterns": interpretation_result.get("cognitive_patterns", [])
        },
        "persistent_emotional_memory": persistent_memory,
        "confidence_analysis": reasoning_result.get("confidence_analysis", {}),
        "emotional_summary": reasoning_result.get("emotional_summary", "")
    }