from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.routes.emotion import router as emotion_router
from app.routes.triggers import router as triggers_router
from app.routes.memory import router as memory_router
from app.routes.trajectory import router as trajectory_router
from app.routes.contextual import router as contextual_router
from app.routes.reasoning import router as reasoning_router

app = FastAPI(
    title="CETIS - Conversational Emotional Trajectory Intelligence System",
    description="Production-grade AI-powered emotional reasoning engine",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(emotion_router)
app.include_router(triggers_router)
app.include_router(memory_router)
app.include_router(trajectory_router)
app.include_router(contextual_router)
app.include_router(reasoning_router)

@app.get("/")
async def root():
    return {
        "system": "CETIS",
        "version": "1.0.0",
        "status": "running",
        "modules": [
            "emotion_detection",
            "trigger_extraction",
            "emotional_memory",
            "trajectory_analysis",
            "contextual_interpretation",
            "llm_reasoning"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)