import os
import anthropic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Load knowledge base from file ─────────────────────────────────────────────
KB_PATH = "clove_knowledge_base.txt"
if os.path.exists(KB_PATH):
    with open(KB_PATH, "r", encoding="utf-8") as f:
        CLOVE_KNOWLEDGE = f.read()
    print(f"Knowledge base loaded: {len(CLOVE_KNOWLEDGE)} characters")
else:
    CLOVE_KNOWLEDGE = ""
    print("WARNING: clove_knowledge_base.txt not found!")

# ── System prompt with full knowledge base ────────────────────────────────────
SYSTEM_PROMPT = (
    "You are the Ubora AI Expert Agronomist Advisor, specialized in:\n"
    "- Zanzibar and Pemba clove farming (Syzygium aromaticum)\n"
    "- ZSTC official grading rules\n"
    "- Clove diseases, post-harvest handling, and market prices\n\n"
    "You assist farmers in BOTH Swahili and English.\n"
    "Always reply in the SAME language the farmer used.\n"
    "Keep answers short and practical for a small phone screen.\n"
    "Use numbered steps when giving instructions.\n"
    "Always prioritize the knowledge base below over your own knowledge.\n\n"
    "--- OFFICIAL KNOWLEDGE BASE ---\n"
    + CLOVE_KNOWLEDGE +
    "\n--- END OF KNOWLEDGE BASE ---"
)

# ── Anthropic client ──────────────────────────────────────────────────────────
# Key is read from environment variable — never hardcoded
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Ubora AI Clove Advisor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.get("/")
def health_check():
    return {
        "status": "Ubora AI running",
        "knowledge_base_chars": len(CLOVE_KNOWLEDGE)
    }

@app.post("/v1/clove-advisor")
async def chat_advisor(request: ChatRequest):
    try:
        messages = []
        for turn in request.history:
            role = "user" if turn.get("role") == "user" else "assistant"
            messages.append({
                "role": role,
                "content": turn.get("text", "")
            })
        messages.append({
            "role": "user",
            "content": request.message
        })

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=messages
        )

        return {"reply": response.content[0].text, "status": "success"}

    except Exception as e:
        return {
            "reply": "Samahani, kuna tatizo. / Sorry, error: " + str(e),
            "status": "error"
        }
