from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

# Load secret (you will replace this)
EXPECTED_SECRET = os.getenv("SECRET", "my_secret")


# Request body model
class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str


@app.post("/")
async def handle_quiz(req: QuizRequest):
    # 1. Check secret
    if req.secret != EXPECTED_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    # 2. Basic response (later we will solve the quiz)
    return {
        "message": "Secret verified. Quiz solver running...",
        "email": req.email,
        "received_url": req.url
    }
