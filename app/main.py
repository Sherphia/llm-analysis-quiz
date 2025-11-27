from fastapi import FastAPI
from pydantic import BaseModel
from playwright.sync_api import sync_playwright
import threading
import requests
import re

app = FastAPI()

SECRET = "SherphiaLLMQuiz2025"

class QuizInput(BaseModel):
    email: str
    secret: str
    url: str


@app.post("/")
def handle_quiz(data: QuizInput):
    # Validate secret
    if data.secret != SECRET:
        return {"error": "Invalid secret!"}

    # Fetch page using sync browser
    html_content = fetch_page_sync(data.url)

    return {
        "message": "Secret verified. Page fetched successfully!",
        "email": data.email,
        "received_url": data.url,
        "page_length": len(html_content),
        "sample_html": html_content[:500]
    }


def fetch_page_sync(url: str) -> str:
    # Run Playwright inside a normal synchronous function
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        html = page.content()
        browser.close()
        return html

SUBMIT_PATH = "/submit"

@app.post("/solve")
def solve_quiz(data: QuizInput):
    # Validate secret
    if data.secret != SECRET:
        return {"error": "Invalid secret!"}

    # Fetch the page HTML
    html_content = fetch_page_sync(data.url)

    # Extract origin from HTML 
    # Looks like: https://tds-llm-analysis.s-anand.net
    match = re.search(r'<span class="origin">(.*?)</span>', html_content)
    if not match:
        return {"error": "Could not find origin in the HTML"}

    origin = match.group(1)

    # Final submit URL
    submit_url = f"{origin}{SUBMIT_PATH}"

    # Prepare payload
    payload = {
        "email": data.email,
        "secret": data.secret,
        "url": data.url,
        "answer": "Sherphia solved it!"
    }

    # Call the quiz /submit endpoint
    response = requests.post(submit_url, json=payload)

    return {
        "submitted_to": submit_url,
        "status_code": response.status_code,
        "server_response": response.json() if response.content else "No response"
    }
