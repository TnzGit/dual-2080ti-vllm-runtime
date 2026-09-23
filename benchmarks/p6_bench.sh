#!/bin/bash
# P6 正式基准：3x distinct 4096-token prompt, 128 output
source ~/.venv-vllm-dflash2-upstream/bin/activate
python3 <<'PYEOF'
import json, time, urllib.request

BASE = "http://127.0.0.1:8002"
TOPICS = ["quantum computing", "marine biology", "Renaissance art", "space exploration"]

def one_run(i):
    prompt = f"The history of {TOPICS[i%4]} is long and fascinating. " * 455
    body = json.dumps({"text": prompt,
        "sampling_params": {"temperature": 0, "max_new_tokens": 128}, "stream": True}).encode()
    req = urllib.request.Request(BASE + "/generate", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; meta = None
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            d = json.loads(payload)
            if d.get("text") and ttft is None: ttft = time.time() - t0
            if d.get("meta_info", {}).get("finish_reason"): meta = d["meta_info"]
    e2e = time.time() - t0; mi = meta or {}
    pt, ct = mi.get("prompt_tokens",0), mi.get("completion_tokens",0)
    return pt/ttft if ttft else 0, ct/(e2e-ttft) if e2e>ttft else 0, ttft

rs = [one_run(i) for i in range(3)]
avg = lambda xs: sum(xs)/len(xs)
print(f"PREFILL={avg([r[0] for r in rs]):.1f} DECODE={avg([r[1] for r in rs]):.2f} TTFT={avg([r[2] for r in rs]):.2f}")
PYEOF
