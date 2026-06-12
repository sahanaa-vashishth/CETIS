import os
import json
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

client = AsyncGroq(api_key=GROQ_API_KEY)


async def extract_triggers(text: str) -> dict:
    """
    Extract emotional triggers from text using Groq LLM.
    Understands any situation without fixed keyword lists.
    """
    if not text or not text.strip():
        return {
            "primary_trigger": {"event": "none", "confidence": 0.0},
            "secondary_triggers": [],
            "trigger_chain": []
        }

    prompt = f"""You are an expert psychologist and emotional analyst.

Analyze the following text and identify the emotional triggers present.

Text: "{text}"

Your job is to identify:
1. What life event or situation is causing emotional distress?
2. Are there any secondary contributing factors?
3. What is the chain of events leading to this emotional state?

Rules:
- Do NOT use fixed categories. Understand the situation naturally.
- Consider explicit triggers (things directly stated) and implicit triggers (things implied).
- Assign a confidence score between 0.0 and 1.0 for each trigger.
- Be specific and psychologically meaningful.

Respond ONLY in this exact JSON format, nothing else, no extra text:
{{
    "primary_trigger": {{
        "event": "describe the main emotional trigger here",
        "confidence": 0.0
    }},
    "secondary_triggers": [
        {{
            "event": "describe secondary trigger here",
            "confidence": 0.0
        }}
    ],
    "trigger_chain": ["first event", "second event", "resulting emotion"]
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
            temperature=0.3,
            max_tokens=500
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
            return _fallback_response()

    except Exception as e:
        import traceback
        print(f"Trigger extraction error: {e}")
        print(traceback.format_exc())
        return _fallback_response()


def _fallback_response() -> dict:
    """Return a safe fallback if Groq fails."""
    return {
        "primary_trigger": {
            "event": "unspecified emotional distress",
            "confidence": 0.4
        },
        "secondary_triggers": [],
        "trigger_chain": []
    }