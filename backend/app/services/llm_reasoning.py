import os
import json
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

client = AsyncGroq(api_key=GROQ_API_KEY)


async def generate_emotional_reasoning(
    current_emotional_state: dict,
    dominant_emotions: list,
    primary_trigger: dict,
    secondary_triggers: list,
    trigger_chain: list,
    trajectory_analysis: dict,
    conversation_interpretation: dict,
    persistent_memory: dict,
    message_count: int
) -> dict:
    """
    Generate psychologically coherent emotional reasoning and summary.
    Synthesizes all modules into a final structured emotional output.
    """

    prompt = f"""You are a psychologically-aware emotional analyst.

You have been given a complete emotional profile of a person from a conversation. Your job is to synthesize all of this into a coherent psychological summary.

EMOTIONAL DATA:
Current Emotional State: {json.dumps(current_emotional_state)}
Dominant Emotions: {dominant_emotions}
Primary Trigger: {json.dumps(primary_trigger)}
Secondary Triggers: {json.dumps(secondary_triggers)}
Trigger Chain: {trigger_chain}
Trajectory Analysis: {json.dumps(trajectory_analysis)}
Surface State: {conversation_interpretation.get("surface_state", "")}
Deeper State: {conversation_interpretation.get("deeper_state", "")}
Cognitive Patterns: {conversation_interpretation.get("cognitive_patterns", [])}
Recurring Triggers: {persistent_memory.get("recurring_triggers", [])}
Messages Analyzed: {message_count}

Your task:
1. Write a psychologically coherent emotional summary (2-3 sentences)
2. Identify detected contextual signals (list of psychological signals present)
3. Assign an overall confidence score (0.0 to 1.0) for this analysis
4. List any uncertainty flags if the analysis is unclear

Rules:
- Do NOT use shallow sentiment labels like "the user is sad"
- DO reason about causality, patterns, and emotional evolution
- BE specific and psychologically meaningful
- Reference the trigger and trajectory in your summary

Respond ONLY in this exact JSON format, nothing else:
{{
    "emotional_summary": "your 2-3 sentence psychologically coherent summary here",
    "detected_contextual_signals": [
        "signal 1",
        "signal 2",
        "signal 3"
    ],
    "confidence_analysis": {{
        "overall_confidence": 0.0,
        "uncertainty_flags": []
    }}
}}"""

    try:
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert psychologist. Always respond in valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_tokens=800
        )

        raw_text = response.choices[0].message.content

        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start != -1 and end != 0:
            json_str = raw_text[start:end]
            parsed = json.loads(json_str)
            return parsed
        else:
            return _fallback_reasoning()

    except Exception as e:
        import traceback
        print(f"LLM reasoning error: {e}")
        print(traceback.format_exc())
        return _fallback_reasoning()


def _fallback_reasoning() -> dict:
    """Return a safe fallback if Groq fails."""
    return {
        "emotional_summary": "Emotional analysis could not be completed at this time.",
        "detected_contextual_signals": [],
        "confidence_analysis": {
            "overall_confidence": 0.0,
            "uncertainty_flags": ["llm_reasoning_failed"]
        }
    }