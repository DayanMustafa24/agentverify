import os, json, urllib.request as u
base = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
for m in ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-latest"]:
    d = json.dumps({"model": m, "messages": [{"role": "user", "content": "say ok"}], "max_tokens": 20}).encode()
    r = u.Request(base, d, {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"})
    try:
        print(m, "->", json.load(u.urlopen(r))["choices"][0]["message"]["content"])
    except Exception as e:
        print(m, "-> FAIL", e)
