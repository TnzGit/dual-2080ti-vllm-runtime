import json, time, urllib.request

BASE = "http://127.0.0.1:8002"
PROMPT = "The history of computing is long and fascinating. " * 320  # ~4K tokens

def one_run(i):
    body = json.dumps({
        "text": PROMPT,
        "sampling_params": {"temperature": 0, "max_new_tokens": 128},
        "stream": True,
    }).encode()
    req = urllib.request.Request(BASE + "/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    text_parts = []
    meta = None
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("text"):
                if ttft is None:
                    ttft = time.time() - t0
                text_parts.append(d["text"])
            if d.get("meta_info", {}).get("finish_reason"):
                meta = d["meta_info"]
    e2e = time.time() - t0
    mi = meta or {}
    pt = mi.get("prompt_tokens", 0)
    ct = mi.get("completion_tokens", 0)
    prefill = pt / ttft if ttft else 0
    decode = ct / (e2e - ttft) if e2e > ttft else 0
    print(f"run{i+1}: prompt={pt} out={ct} TTFT={ttft:.2f}s prefill={prefill:.1f} tok/s decode={decode:.2f} tok/s")
    return prefill, decode, ttft

rs = [one_run(i) for i in range(3)]
avg = lambda xs: sum(xs)/len(xs)
print(f"AVG: prefill={avg([r[0] for r in rs]):.1f} decode={avg([r[1] for r in rs]):.2f} ttft={avg([r[2] for r in rs]):.2f}s")
print("TARGET: prefill=1408.1 decode=92.09 ttft=2.91s")
