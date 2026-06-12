# CETIS — Conversational Emotional Trajectory Intelligence System

A production-grade AI-powered emotional reasoning engine that understands, remembers, and reasons about emotions over time.

## What Makes CETIS Different

Instead of: *"This message is 80% negative"*

CETIS does: *"This person started anxious 3 turns ago, escalated to fear after mentioning job loss, and is now showing hopelessness driven by self-worth collapse"*

## Features

- **Emotion Detection** — Fine-grained multi-label emotion detection using transformer models
- **Trigger Extraction** — AI-powered semantic trigger extraction (no keyword lists)
- **Emotional Memory** — Persistent emotional state tracking using exponential moving averages
- **Trajectory Analysis** — Detects escalation, de-escalation, and emotional patterns
- **Contextual Interpretation** — Surface vs deeper psychological driver detection
- **LLM Reasoning** — Psychologically coherent emotional summaries via Groq + Llama 3.3
- **Frontend Dashboard** — Live emotional timeline graphs and analysis panels

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Emotion Detection | HuggingFace Transformers |
| AI Reasoning | Groq API + Llama 3.3 70B |
| Frontend | Next.js + TypeScript + TailwindCSS |
| Charts | Recharts |
| Project Planning | OpenSpec |

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
python -m pip install fastapi uvicorn transformers torch spacy sentencepiece python-dotenv asyncpg redis sqlalchemy pydantic anthropic groq

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
| POST /triggers/extract | Trigger extraction only |
| POST /memory/update | Update emotional memory |
| GET /memory/summary/{id} | Get conversation memory |
| POST /trajectory/analyze | Trajectory analysis |
| POST /contextual/interpret | Contextual interpretation |

## Research References

- Kuppens et al. (2010) — Emotional Inertia and Psychological Maladjustment
- Brown (1956) — Exponential Smoothing for Predicting Demand
- GoEmotions Dataset — Google Research