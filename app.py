# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import Optional, Dict, List

# ==============================
# ⚙️ CONFIGURACIÓN
# ==============================
load_dotenv()
app = FastAPI(title="AI Service", version="1.0")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ==============================
# 📦 MODELOS DE DATOS
# ==============================
class UserMessage(BaseModel):
    userMessage: str


class NluRequest(BaseModel):
    text: str
    locale: Optional[str] = "es-CL"
    tz: Optional[str] = "America/Santiago"
    hints: Optional[Dict] = None
    taxonomy: Optional[List[str]] = None


class NluResponse(BaseModel):
    intent: str
    slots: Dict = Field(default_factory=dict)
    confidence: float = 1.0
    raw_text: Optional[str] = None


class Persona(BaseModel):
    tone: Optional[str] = "neutral"
    intensity: Optional[int] = 1
    formality: Optional[str] = "neutral"
    verbosity: Optional[str] = "balanced"
    emojis: Optional[str] = "few"
    locale: Optional[str] = "es-CL"


class ReplyContext(BaseModel):
    action: str
    summary: str


class ReplyRequest(BaseModel):
    persona: Persona
    context: ReplyContext


class ReplyResponse(BaseModel):
    reply: str


# ==============================
# 💬 ENDPOINTS
# ==============================

@app.post("/ask", response_model=Dict)
def ask_ai(data: UserMessage):
    """
    Enruta cualquier mensaje del usuario directamente a GPT.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": data.userMessage}
            ],
            timeout=20,
        )
        reply = completion.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")


@app.post("/nlu/parse", response_model=NluResponse)
def nlu_parse(req: NluRequest):
    """
    NLU genérico (sin lógica de clasificación). Solo devuelve texto plano.
    """
    try:
        return NluResponse(
            intent="unknown",
            slots={},
            confidence=0.0,
            raw_text=req.text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLU error: {str(e)}")


@app.post("/style/reply", response_model=ReplyResponse)
def style_reply(req: ReplyRequest):
    """
    Genera una respuesta textual simple, sin estilo específico.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": req.context.summary}
            ],
            timeout=20,
        )
        reply = completion.choices[0].message.content
        return ReplyResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reply error: {str(e)}")


@app.get("/")
def root():
    return {"status": "ok", "service": "ai-service", "mode": "generic"}
