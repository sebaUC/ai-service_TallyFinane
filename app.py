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
    prompt = f"""
Eres un clasificador de intenciones para un asistente financiero personal.
Tu objetivo es interpretar el mensaje del usuario y devolver SOLO un JSON válido.

Posibles intenciones:
- greeting
- register_transaction
- ask_balance
- ask_budget_status
- ask_goal_status
- modify_persona
- smalltalk
- unknown

Extrae estos SLOTS cuando existan:
- amount (monto en CLP, como número puro)
- category (hogar, alimentación, transporte, salud, etc.)
- date (convertir “ayer”, “hoy”, “lunes pasado” a YYYY-MM-DD, usando tz: {req.tz})
- payment_method (efectivo, débito, crédito)
- goalId (si aplica)
- persona_property (tono, intensidad, etc.)

INSTRUCCIONES:
- Responde SOLO un JSON.
- No agregues comentarios ni texto fuera del JSON.
- Si no estás seguro de algo, deja el campo en null.

Mensaje del usuario: "{req.text}"
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en NLU financiero."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        raw = completion.choices[0].message.content.strip()

        # Validar JSON
        parsed = json.loads(raw)

        return NluResponse(
            intent=parsed.get("intent", "unknown"),
            slots=parsed.get("slots", {}),
            confidence=float(parsed.get("confidence", 1.0)),
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
