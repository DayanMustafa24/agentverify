import os, json, urllib.request as u
base = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
for m in ["gemini-3-flash-preview", "gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "gemma-4-31b-it", "gemma-4-26b-a4b-it"]:
    d = json.dumps({"model": m, "messages": [{"role": "user", "content": "say ok"}], "max_tokens": 20}).encode()
    r = u.Request(base, d, {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"})
    try:
        print(m, "->", json.load(u.urlopen(r))["choices"][0]["message"]["content"])
    except Exception as e:
        print(m, "-> FAIL", e)
