from fastapi import APIRouter
from pydantic import BaseModel, Field
from core.teo_engine import engine

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[dict] = []   # ignored until Phase 2 (Memory Layer) — contract exists now


class ChatResponse(BaseModel):
    response: str


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    response = engine.get_response(req.message, req.history)
    return ChatResponse(response=response)
