import os
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

ALPHA = float(os.getenv("ALPHA", "0.3"))

# In-memory storage for conversations
# In production this would be PostgreSQL + Redis
conversation_memory = {}


def initialize_conversation(conversation_id: str) -> dict:
    """Initialize a new conversation memory."""
    memory = {
        "conversation_id": conversation_id,
        "message_count": 0,
        "emotional_history": [],
        "smoothed_emotional_state": {},
        "baseline_emotional_state": {},
        "recurring_triggers": [],
        "emotional_volatility": 0.0,
        "dominant_pattern": "neutral"
    }
    conversation_memory[conversation_id] = memory
    return memory


def get_conversation_memory(conversation_id: str) -> dict:
    """Get existing conversation memory or create new one."""
    if conversation_id not in conversation_memory:
        return initialize_conversation(conversation_id)
    return conversation_memory[conversation_id]


def update_emotional_memory(
    conversation_id: str,
    emotion_scores: dict,
    trigger: dict,
    message: str
) -> dict:
    """
    Update emotional memory using exponential moving average.
    Formula: E_t = alpha * X_t + (1 - alpha) * E_(t-1)
    """
    memory = get_conversation_memory(conversation_id)

    # Get previous smoothed state
    previous_state = memory["smoothed_emotional_state"]

    # Calculate new smoothed emotional state
    if not previous_state:
        # First message - use current emotions directly
        new_smoothed_state = emotion_scores.copy()
    else:
        # Apply exponential moving average
        new_smoothed_state = {}
        all_emotions = set(list(emotion_scores.keys()) + list(previous_state.keys()))
        for emotion in all_emotions:
            current = emotion_scores.get(emotion, 0.0)
            previous = previous_state.get(emotion, 0.0)
            new_smoothed_state[emotion] = round(
                ALPHA * current + (1 - ALPHA) * previous, 4
            )

    # Update message count
    memory["message_count"] += 1

    # Add to emotional history
    history_entry = {
        "message_index": memory["message_count"],
        "message": message[:100],
        "raw_emotions": emotion_scores,
        "smoothed_emotions": new_smoothed_state,
        "trigger": trigger.get("event", "unknown")
    }
    memory["emotional_history"].append(history_entry)

    # Keep only last 20 messages in memory
    if len(memory["emotional_history"]) > 20:
        memory["emotional_history"] = memory["emotional_history"][-20:]

    # Update smoothed state
    memory["smoothed_emotional_state"] = new_smoothed_state

    # Set baseline from first 3 messages
    if memory["message_count"] <= 3:
        memory["baseline_emotional_state"] = new_smoothed_state.copy()

    # Track recurring triggers
    trigger_event = trigger.get("event", "")
    if trigger_event and trigger_event != "none":
        if trigger_event not in memory["recurring_triggers"]:
            memory["recurring_triggers"].append(trigger_event)

    # Calculate emotional volatility
    if len(memory["emotional_history"]) >= 2:
        prev_emotions = memory["emotional_history"][-2]["raw_emotions"]
        curr_emotions = emotion_scores
        all_emotions = set(list(prev_emotions.keys()) + list(curr_emotions.keys()))
        total_change = sum(
            abs(curr_emotions.get(e, 0) - prev_emotions.get(e, 0))
            for e in all_emotions
        )
        memory["emotional_volatility"] = round(total_change / len(all_emotions), 4)

    # Detect dominant pattern
    if new_smoothed_state:
        memory["dominant_pattern"] = max(
            new_smoothed_state, key=new_smoothed_state.get
        )

    # Save back to memory store
    conversation_memory[conversation_id] = memory

    return {
        "conversation_id": conversation_id,
        "message_count": memory["message_count"],
        "smoothed_emotional_state": new_smoothed_state,
        "baseline_emotional_state": memory["baseline_emotional_state"],
        "recurring_triggers": memory["recurring_triggers"],
        "emotional_volatility": memory["emotional_volatility"],
        "dominant_pattern": memory["dominant_pattern"],
        "historical_patterns": [
            h["trigger"] for h in memory["emotional_history"]
        ]
    }


def get_emotional_summary(conversation_id: str) -> dict:
    """Get full emotional memory summary for a conversation."""
    memory = get_conversation_memory(conversation_id)
    return {
        "conversation_id": conversation_id,
        "message_count": memory["message_count"],
        "smoothed_emotional_state": memory["smoothed_emotional_state"],
        "baseline_emotional_state": memory["baseline_emotional_state"],
        "recurring_triggers": memory["recurring_triggers"],
        "emotional_volatility": memory["emotional_volatility"],
        "dominant_pattern": memory["dominant_pattern"],
        "emotional_history": memory["emotional_history"]
    }