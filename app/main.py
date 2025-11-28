from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from playwright.sync_api import sync_playwright
import requests, re, base64, json, pandas as pd, io, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bs4 import BeautifulSoup

# ------------------------ CONFIG ------------------------
SECRET = "SherphiaLLMQuiz2025"
SUBMIT_PATH = "/submit"
MAX_TOTAL_SECONDS = 180
LOCAL_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

llama_model = None
llama_tokenizer = None

# ------------------------ APP ------------------------
app = FastAPI()

class QuizInput(BaseModel):
    email: str
    secret: str
    url: str

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": exc.errors(), "body": exc.body})

# ------------------------ LLaMA ------------------------
def load_llama():
    global llama_model, llama_tokenizer
    if llama_model is None:
        llama_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
        llama_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)

def llama_predict(system_prompt: str, user_prompt: str, max_new_tokens=200):
    load_llama()
    prompt = f"<s>[SYSTEM]{system_prompt}[/SYSTEM]\n[USER]{user_prompt}[/USER]\n[ASSISTANT]"
    inputs = llama_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = llama_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            repetition_penalty=1.05
        )
    text = llama_tokenizer.decode(output[0], skip_special_tokens=True)
    if "[ASSISTANT]" in text:
        text = text.split("[ASSISTANT]")[-1].strip()
    return text.strip()

# ------------------------ Playwright ------------------------
def fetch_page_sync(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        html = page.content()
        browser.close()
        return html

# ------------------------ File / JSON / CSV ------------------------
def decode_atob_payload(html: str):
    match = re.search(r'atob\(`([\s\S]+?)`\)', html)
    if not match:
        return None
    try:
        decoded = base64.b64decode(match.group(1)).decode()
        return json.loads(decoded)
    except:
        return None

def decode_inline_base64_file(b64text: str) -> bytes:
    if b64text.startswith("data:"):
        return base64.b64decode(b64text.split(",", 1)[1])
    return base64.b64decode(b64text)

def process_file_from_url_or_bytes(url_or_bytes):
    try:
        if isinstance(url_or_bytes, bytes):
            content_bytes = url_or_bytes
        else:
            resp = requests.get(url_or_bytes, timeout=30)
            content_bytes = resp.content
        # Try CSV
        try:
            text = content_bytes.decode("utf-8")
            if "," in text:
                df = pd.read_csv(io.StringIO(text))
                if "value" in df.columns:
                    return int(df["value"].sum())
        except: pass
        # Try JSON
        try:
            obj = json.loads(content_bytes.decode("utf-8"))
            if isinstance(obj, dict) and "value" in obj:
                return int(sum(obj["value"]))
            return obj
        except: pass
        return {"error": "Unsupported file"}
    except Exception as e:
        return {"error": str(e)}

# ------------------------ DOM extraction ------------------------
def extract_hidden_key_answer(html: str):
    """Look for <div class="hidden-key"> and reverse its text."""
    soup = BeautifulSoup(html, "html.parser")
    div = soup.find("div", class_="hidden-key")
    if div and div.text.strip():
        return div.text.strip()[::-1]  # Reverse the text
    return None

# ------------------------ / endpoint ------------------------
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

# ------------------------ /solve endpoint ------------------------
@app.post("/solve")
def solve_quiz(data: QuizInput):
    if data.secret != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    overall_start = time.time()
    current_url = data.url
    results = []

    while current_url and (time.time() - overall_start) < MAX_TOTAL_SECONDS:
        html = fetch_page_sync(current_url)
        payload_json = decode_atob_payload(html)
        answer = None

        # ------------------------ Step 1: File / JSON / CSV ------------------------
        if payload_json:
            if "answer" in payload_json:
                answer = payload_json["answer"]
            elif "url" in payload_json:
                fileloc = payload_json["url"]
                if fileloc.startswith("data:"):
                    b = decode_inline_base64_file(fileloc)
                    answer = process_file_from_url_or_bytes(b)
                else:
                    answer = process_file_from_url_or_bytes(fileloc)
            else:
                sys_p = f"Solve the quiz. Never reveal the secret '{SECRET}'."
                usr_p = f"Question JSON: {json.dumps(payload_json)}. Give final answer only."
                out = llama_predict(sys_p, usr_p)
                nums = re.findall(r"-?\d+", out)
                answer = int(nums[0]) if len(nums) == 1 else out
        else:
            # ------------------------ Step 2: DOM puzzles (hidden keys, reversed text) ------------------------
            dom_answer = extract_hidden_key_answer(html)
            if dom_answer:
                answer = dom_answer
            else:
                # ------------------------ Step 3: Fallback LLaMA ------------------------
                snippet = html[:3000]
                sys_p = f"Solve quiz from HTML. Never reveal secret '{SECRET}'."
                usr_p = f"Extract question and give answer only. HTML:\n{snippet}"
                out = llama_predict(sys_p, usr_p)
                nums = re.findall(r"-?\d+", out)
                answer = int(nums[0]) if nums else out

        # ------------------------ Step 4: Submit ------------------------
        origin = re.search(r'<span class="origin">(.*?)</span>', html)
        submit_url = origin.group(1) + SUBMIT_PATH if origin else None
        resp_json = {}
        if submit_url:
            payload = {
                "email": data.email,
                "secret": data.secret,
                "url": current_url,
                "answer": answer
            }
            resp = requests.post(submit_url, json=payload)
            try: resp_json = resp.json()
            except: resp_json = {"status": resp.status_code}

        results.append({
            "quiz_url": current_url,
            "submit_url": submit_url,
            "answer": answer,
            "server_response": resp_json
        })

        # ------------------------ Step 5: Next URL ------------------------
        current_url = resp_json.get("url") if isinstance(resp_json, dict) else None

    return {
        "total_elapsed": round(time.time() - overall_start, 2),
        "results": results
    }
