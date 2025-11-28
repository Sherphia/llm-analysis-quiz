from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from playwright.sync_api import sync_playwright
import requests
import re

app = FastAPI()

SECRET = "SherphiaLLMQuiz2025"
SUBMIT_PATH = "/submit"

class QuizInput(BaseModel):
    email: str
    secret: str
    url: str

# ----------------------
# Override validation error handler
# ----------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Convert 422 → 400
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors(), "body": exc.body}
    )

# ----------------------
# / endpoint
# ----------------------
@app.post("/")
def handle_quiz(data: QuizInput):
    if data.secret != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    html_content = fetch_page_sync(data.url)
    return {
        "message": "Secret verified. Page fetched successfully!",
        "email": data.email,
        "received_url": data.url,
        "page_length": len(html_content),
        "sample_html": html_content[:500]
    }

# ----------------------
# /solve endpoint
# ----------------------
@app.post("/solve")
def solve_quiz(data: QuizInput):
    if data.secret != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    html_content = fetch_page_sync(data.url)
    match = re.search(r'<span class="origin">(.*?)</span>', html_content)
    if not match:
        raise HTTPException(status_code=400, detail="Could not find origin in HTML")
    origin = match.group(1)
    submit_url = f"{origin}{SUBMIT_PATH}"
    payload = {
        "email": data.email,
        "secret": data.secret,
        "url": data.url,
        "answer": "Sherphia solved it!"
    }
    response = requests.post(submit_url, json=payload)
    return {
        "submitted_to": submit_url,
        "status_code": response.status_code,
        "server_response": response.json() if response.content else "No response"
    }

# ----------------------
# Playwright fetch
# ----------------------
def fetch_page_sync(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        html = page.content()
        browser.close()
        return html
