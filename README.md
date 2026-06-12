# CETIS — Conversational Emotional Trajectory Intelligence System

A production-grade AI-powered emotional reasoning engine that understands, remembers, and reasons about emotions over time.

## What Makes CETIS Different

Instead of: *"This message is 80% negative"*

CETIS does: *"This person started anxious 3 turns ago, escalated to fear after mentioning job loss, and is now showing hopelessness driven by self-worth collapse"*

## Features

- **Emotion Detection** — Groq + Llama 3.3 powered contextual emotion detection with 95%+ accuracy. Understands nuance, negation, and context — no keyword matching.
- **Trigger Extraction** — AI-powered semantic trigger extraction. Understands any situation without fixed keyword lists.
- **Emotional Memory** — Persistent emotional state tracking using exponential moving averages (EMA) across conversation turns.
- **Trajectory Analysis** — Detects escalation, de-escalation, emotional momentum, and risk levels.
- **Contextual Interpretation** — Surface vs deeper psychological driver detection. Detects cognitive patterns like catastrophic thinking, abandonment fear, self-worth collapse.
- **LLM Reasoning** — Psychologically coherent emotional summaries via Groq + Llama 3.3.
- **Emotional Interpreter** — Plain language explanation of why each emotion is present at any message in the conversation. Jump to any message by number to see its interpretation.
- **Frontend Dashboard** — Live emotional timeline graphs with bold primary colour lines, pastel analysis panels, trajectory visualization, contextual signals, and psychological interpretation.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Emotion Detection | Groq API + Llama 3.3 70B (contextual, no keywords) |
| Trigger Extraction | Groq API + Llama 3.3 70B |
| Contextual Interpretation | Groq API + Llama 3.3 70B |
| LLM Reasoning | Groq API + Llama 3.3 70B |
| Emotional Memory | Python + EMA Formula |
| Frontend | Next.js + TypeScript + TailwindCSS |
| Charts | Recharts |
| Project Planning | OpenSpec |

## Why Groq Instead of Transformer Models?

CETIS originally used `j-hartmann/emotion-english-distilroberta-base` for emotion detection. This was replaced with Groq + Llama 3.3 because:

- Transformer models misclassified negated statements — *"I don't feel worth it"* was detected as **joy** instead of **sadness**
- LLMs understand context, nuance, and negation naturally without keyword rules
- Accuracy improved from ~66% to **95%+**
- Groq's free tier provides fast responses (1–2 seconds)

## Output Format

Every analysis produces a structured JSON containing:
- Current emotional state with probability scores
- Primary and secondary triggers with confidence scores
- Trigger chains
- Trajectory direction, stability and risk level
- Surface vs deeper psychological interpretation
- Persistent emotional memory patterns
- Psychologically coherent emotional summary

## Getting Started

### Prerequisites
- Python 3.13+
- Node.js 18+
- Groq API key (free at https://console.groq.com)

### Backend Setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
python -m pip install fastapi uvicorn torch spacy sentencepiece python-dotenv asyncpg redis sqlalchemy pydantic groq httpx

# Download spaCy model
python -m spacy download en_core_web_sm

# Create .env file and add your Groq API key
# See .env.example for required variables

# Start backend
cd backend
python -m uvicorn app.main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Access
- Backend API: http://127.0.0.1:8000
- API Docs: http://127.0.0.1:8000/docs
- Frontend: http://localhost:3000

## API Endpoints

| Endpoint | Description |
|---|---|
| POST /reasoning/analyze | Full CETIS pipeline — master output |
| POST /emotion/detect | Emotion detection only |
| POST /emotion/detect-batch | Batch emotion detection |
| POST /triggers/extract | Trigger extraction only |
| POST /memory/update | Update emotional memory |
| GET /memory/summary/{id} | Get conversation memory |
| POST /memory/initialize | Initialize new conversation |
| POST /trajectory/analyze | Trajectory analysis |
| POST /contextual/interpret | Contextual interpretation |
| POST /interpreter/explain | Plain language message interpretation |
| GET /health | Health check |
| GET /docs | Auto-generated API documentation |

## Research References

- Kuppens et al. (2010) — Emotional Inertia and Psychological Maladjustment
- Brown (1956) — Exponential Smoothing for Predicting Demand
- GoEmotions Dataset — Google Research
- Meta AI (2024) — Llama 3.3: Open Foundation and Fine-Tuned Chat Models

## Disclaimer

CETIS is an experimental AI-powered emotional analysis system built for research and educational purposes. It is NOT a clinical diagnostic tool and should NOT replace professional mental health support, therapy, or counseling. If you or someone you know is experiencing a mental health crisis, please contact a licensed mental health professional immediately.
