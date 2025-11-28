import requests
import json
import time

BASE = "http://127.0.0.1:8000"

TEST_URLS = [
    "https://tds-llm-analysis.s-anand.net/demo",
    "https://tds-llm-analysis.s-anand.net/demo2",
    "https://tdsbasictest.vercel.app/quiz/1",
    "https://p2testingone.vercel.app/q1.html",
    "https://p2testingone.vercel.app/q2.html",
    "https://p2testingone.vercel.app/q3.html",
    "https://p2testingone.vercel.app/q4.html",
    "https://p2testingone.vercel.app/q5.html",
    "https://p2testingone.vercel.app/q6.html",
    "https://p2testingone.vercel.app/q7.html",
    "https://p2testingone.vercel.app/q8.html",
    "https://p2testingone.vercel.app/q9.html",
    "https://p2testingone.vercel.app/q10.html",
]

EMAIL = "22f2001145@ds.study.iitm.ac.in"
SECRET = "SherphiaLLMQuiz2025"


def run_test(url):
    print("\n" + "="*80)
    print(f"TESTING: {url}")

    payload = {
        "email": EMAIL,
        "secret": SECRET,
        "url": url
    }

    start = time.time()
    r = requests.post(f"{BASE}/solve", json=payload)
    end = time.time()

    try:
        j = r.json()
        print("STATUS :", r.status_code)
        print("TIME   :", round(end - start, 3), "sec")
        print("RESULT :")
        print(json.dumps(j, indent=2)[:1500])   # truncate so terminal not flooded
    except:
        print("Invalid JSON returned.")
        print(r.text[:1000])


if __name__ == "__main__":
    for u in TEST_URLS:
        run_test(u)
