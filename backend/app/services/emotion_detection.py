from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()

EMOTION_MODEL = os.getenv("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")

# Load the emotion detection pipeline once at startup
print(f"Loading emotion model: {EMOTION_MODEL}")
emotion_classifier = pipeline(
    "text-classification",
    model=EMOTION_MODEL,
    top_k=None
)
print("Emotion model loaded successfully!")

# Emotion mapping to our required emotions
EMOTION_MAP = {
    "joy": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "fear": "fear",
    "disgust": "frustration",
    "surprise": "confusion",
    "neutral": "calm"
}

INTENSITY_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.5,
    "high": 0.7
}

def detect_emotions(text: str) -> dict:
    """
    Detect fine-grained emotions from text.
    Returns probability scores for each emotion.
    """
    if not text or not text.strip():
        return {"calm": 1.0}

    try:
        # Run emotion classification
        results = emotion_classifier(text)

        # Build emotion scores dictionary
        emotion_scores = {}
        for result in results[0]:
            label = result["label"].lower()
            score = round(result["score"], 4)
            mapped_label = EMOTION_MAP.get(label, label)
            emotion_scores[mapped_label] = score

        # Get dominant emotions (score > 0.1)
        dominant_emotions = [
            emotion for emotion, score in emotion_scores.items()
            if score > 0.1
        ]
        dominant_emotions.sort(key=lambda x: emotion_scores[x], reverse=True)

        # Detect intensity of top emotion
        top_score = max(emotion_scores.values())
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
            "top_emotion": max(emotion_scores, key=emotion_scores.get)
        }

    except Exception as e:
        print(f"Emotion detection error: {e}")
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