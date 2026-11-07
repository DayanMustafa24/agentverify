import os, json, urllib.request as u
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
d = json.dumps({"contents": [{"parts": [{"text": "Explain how AI works in a few words"}]}]}).encode()
r = u.Request(url, d, {"Content-Type": "application/json", "X-goog-api-key": os.environ["OPENAI_API_KEY"]})
try:
    print(json.load(u.urlopen(r))["candidates"][0]["content"]["parts"][0]["text"])
except Exception as e:
    print("FAIL", e, getattr(e, "read", lambda: b"")().decode()[:300])
