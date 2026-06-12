import os
import json
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

client = AsyncGroq(api_key=GROQ_API_KEY)


async def interpret_contextual_signals(
    current_emotions: dict,
    primary_trigger: dict,
    emotional_history: list,
    recent_messages: list
) -> dict:
    """
    Interpret deeper psychological meaning beneath surface emotions.
    Distinguishes surface emotions from deeper emotional drivers.
    Detects cognitive patterns like catastrophic thinking, abandonment fear, etc.
    """
    if not current_emotions:
        return _neutral_interpretation()

    # Build conversation context
    conversation_text = " ".join([msg[:100] for msg in recent_messages[-5:]])

    prompt = f"""You are an expert psychologist and emotional analyst.

Analyze the following emotional and conversational data to identify deeper psychological patterns.

Current Emotions: {json.dumps(current_emotions)}
Primary Trigger: {json.dumps(primary_trigger)}
Recent Conversation: "{conversation_text}"

Your task is to:
1. Identify the SURFACE EMOTION - what emotion is directly visible
2. Identify the DEEPER EMOTIONAL DRIVER - what underlying fear, need, or pattern might be driving this
3. Identify COGNITIVE PATTERNS - psychological patterns like catastrophic thinking, abandonment fear, self-worth collapse, emotional dependency, validation-seeking, identity insecurity, emotional suppression, social comparison sensitivity

Rules:
- Surface emotion is what we detect directly from the text
- Deeper driver is what psychologically motivates that emotion
- Cognitive patterns are recurring psychological themes

Examples:
- Surface: Anger → Deeper: Fear of rejection
- Surface: Hopelessness → Deeper: Self-worth collapse
- Surface: Withdrawal → Deeper: Fear of abandonment

Respond ONLY in this exact JSON format, nothing else:
{{
    "surface_state": "describe the directly observable emotion",
    "deeper_state": "describe the underlying psychological driver",
    "cognitive_patterns": [
        "cognitive pattern 1",
        "cognitive pattern 2",
        "cognitive pattern 3"
    ],
    "pattern_confidence": 0.0
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
            max_tokens=600
        )

        raw_text = response.choices[0].message.content

        # Extract JSON from response
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start != -1 and end != 0:
            json_str = raw_text[start:end]
            parsed = json.loads(json_str)
            return parsed
        else:
            return _neutral_interpretation()

    except Exception as e:
        import traceback
        print(f"Contextual interpretation error: {e}")
        print(traceback.format_exc())
        return _neutral_interpretation()


def _neutral_interpretation() -> dict:
    """Return a safe fallback if Groq fails."""
    return {
        "surface_state": "emotional state detected",
        "deeper_state": "underlying emotional drivers present",
        "cognitive_patterns": [],
        "pattern_confidence": 0.0
    }