from typing import Optional


def analyze_trajectory(emotional_history: list, current_emotions: dict) -> dict:
    """
    Analyze emotional trajectory from conversation history.
    Detects escalation, de-escalation, stability and risk levels.
    """
    if not emotional_history:
        return _neutral_trajectory()

    # Need at least 2 messages for trajectory
    if len(emotional_history) < 2:
        return {
            "direction": "insufficient data",
            "stability": "unknown",
            "risk_level": "unknown",
            "trajectory_pattern": "neutral",
            "emotional_momentum": 0.0
        }

    # Get negative emotion scores over time
    negative_emotions = ["sadness", "anger", "fear", "frustration",
                        "hopelessness", "anxiety", "shame", "guilt",
                        "loneliness", "confusion"]

    positive_emotions = ["joy", "calm"]

    # Calculate negative emotion intensity per message
    negative_scores = []
    positive_scores = []

    for entry in emotional_history:
        raw = entry.get("raw_emotions", {})
        neg_score = sum(raw.get(e, 0) for e in negative_emotions)
        pos_score = sum(raw.get(e, 0) for e in positive_emotions)
        negative_scores.append(neg_score)
        positive_scores.append(pos_score)

    # Calculate trend
    first_half_neg = sum(negative_scores[:len(negative_scores)//2 or 1])
    second_half_neg = sum(negative_scores[len(negative_scores)//2 or 1:] or [negative_scores[-1]])

    first_half_pos = sum(positive_scores[:len(positive_scores)//2 or 1])
    second_half_pos = sum(positive_scores[len(positive_scores)//2 or 1:] or [positive_scores[-1]])

    neg_change = second_half_neg - first_half_neg
    pos_change = second_half_pos - first_half_pos

    # Calculate emotional momentum
    recent = negative_scores[-1] if negative_scores else 0
    previous = negative_scores[-2] if len(negative_scores) >= 2 else 0
    momentum = round(recent - previous, 4)

    # Determine direction
    if neg_change > 0.3:
        direction = "rapid emotional escalation"
    elif neg_change > 0.1:
        direction = "gradual emotional escalation"
    elif neg_change < -0.3:
        direction = "rapid emotional recovery"
    elif neg_change < -0.1:
        direction = "gradual emotional recovery"
    elif abs(neg_change) <= 0.1 and recent > 0.5:
        direction = "emotionally persistent distress"
    else:
        direction = "emotionally stable"

    # Determine stability
    score_variance = max(negative_scores) - min(negative_scores)
    if score_variance > 0.5:
        stability = "low"
    elif score_variance > 0.2:
        stability = "moderate"
    else:
        stability = "high"

    # Determine risk level
    current_neg = sum(current_emotions.get(e, 0) for e in negative_emotions)
    if current_neg > 1.5 or (neg_change > 0.3 and recent > 0.8):
        risk_level = "high emotional distress"
    elif current_neg > 0.8 or neg_change > 0.1:
        risk_level = "moderate emotional distress"
    elif current_neg > 0.3:
        risk_level = "mild emotional distress"
    else:
        risk_level = "low emotional distress"

    # Detect trajectory pattern
    if len(negative_scores) >= 3:
        if all(negative_scores[i] <= negative_scores[i+1]
               for i in range(len(negative_scores)-1)):
            pattern = "continuous escalation"
        elif all(negative_scores[i] >= negative_scores[i+1]
                 for i in range(len(negative_scores)-1)):
            pattern = "continuous recovery"
        elif negative_scores[-1] > negative_scores[0]:
            pattern = "net escalation with fluctuations"
        else:
            pattern = "net recovery with fluctuations"
    else:
        pattern = "escalating" if neg_change > 0 else "recovering"

    return {
        "direction": direction,
        "stability": stability,
        "risk_level": risk_level,
        "trajectory_pattern": pattern,
        "emotional_momentum": momentum
    }


def _neutral_trajectory() -> dict:
    """Return neutral trajectory for empty history."""
    return {
        "direction": "emotionally stable",
        "stability": "high",
        "risk_level": "low emotional distress",
        "trajectory_pattern": "neutral",
        "emotional_momentum": 0.0
    }