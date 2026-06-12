import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

client = Groq(api_key=GROQ_API_KEY)

INTENSITY_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.5,
    "high": 0.7
}


def detect_emotions(text: str) -> dict:
    """
    Detect emotions using Groq LLM - pure contextual understanding.
    No keywords, understands nuance and context naturally.
    """
    if not text or not text.strip():
        return {
            "emotion_scores": {"calm": 1.0},
            "dominant_emotions": ["calm"],
            "intensity": "low",
            "top_emotion": "calm"
        }

    prompt = f"""You are an expert psychologist analyzing emotions from text.

Text: "{text}"

Analyze the emotional state and respond ONLY with this JSON format, nothing else:
{{
    "sadness": 0.0,
    "anger": 0.0,
    "fear": 0.0,
    "joy": 0.0,
    "calm": 0.0,
    "frustration": 0.0,
    "anxiety": 0.0,
    "hopelessness": 0.0,
    "confusion": 0.0
}}

Rules:
- Each emotion is a probability (0.0 to 1.0)
- All emotions must sum to approximately 1.0
- Understand CONTEXT and NUANCE, not keywords
- If someone says "i dont feel worth it", recognize hopelessness and sadness, NOT joy
- Be psychologically accurate"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=300
        )

        raw_text = response.choices[0].message.content

        # Extract JSON
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start != -1 and end != 0:
            json_str = raw_text[start:end]
            emotion_scores = json.loads(json_str)

            # Normalize to sum to 1.0
            total = sum(emotion_scores.values())
            if total > 0:
                emotion_scores = {k: round(v / total, 4) for k, v in emotion_scores.items()}

            # Get dominant emotions
            dominant_emotions = [
                emotion for emotion, score in emotion_scores.items()
                if score > 0.1
            ]
            dominant_emotions.sort(key=lambda x: emotion_scores[x], reverse=True)

            # Detect intensity
            top_score = max(emotion_scores.values()) if emotion_scores else 0
            if top_score >= INTENSITY_THRESHOLDS["high"]:
                intensity = "high"
            elif top_score >= INTENSITY_THRESHOLDS["medium"]:
                intensity = "medium"
            else:
                intensity = "low"

            return {
                "emotion_scores": emotion_scores,
                "dominant_emotions": dominant_emotions[:3],
                "intensity": intensity,
                "top_emotion": max(emotion_scores, key=emotion_scores.get) if emotion_scores else "calm"
            }
        else:
            return _fallback_response()

    except Exception as e:
        print(f"Emotion detection error: {e}")
        import traceback
        print(traceback.format_exc())
        return _fallback_response()


def _fallback_response() -> dict:
    """Safe fallback if Groq fails."""
    return {
        "emotion_scores": {"calm": 1.0},
        "dominant_emotions": ["calm"],
        "intensity": "low",
        "top_emotion": "calm"
    }


def detect_emotions_batch(texts: list) -> list:
    """
    Detect emotions for a batch of texts.
    """
    return [detect_emotions(text) for text in texts]