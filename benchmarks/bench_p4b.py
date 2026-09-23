import json, time, urllib.request

BASE = "http://127.0.0.1:8002"
TOPICS = ["computing", "aviation", "oceanography"]

def one_run(i):
    prompt = f"The history of {TOPICS[i%3]} is long and fascinating. " * 455
    body = json.dumps({
        "text": prompt,
        "sampling_params": {"temperature": 0, "max_new_tokens": 128},
        "stream": True,
    }).encode()
    req = urllib.request.Request(BASE + "/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, meta, parts = None, None, []
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            d = json.loads(payload)
            if d.get("text"):
                if ttft is None: ttft = time.time() - t0
                parts.append(d["text"])
            if d.get("meta_info", {}).get("finish_reason"): meta = d["meta_info"]
    e2e = time.time() - t0
    mi = meta or {}
    pt, ct = mi.get("prompt_tokens", 0), mi.get("completion_tokens", 0)
    print(f"run{i+1}: prompt={pt} out={ct} TTFT={ttft:.2f}s "
          f"prefill={pt/ttft:.1f} decode={ct/(e2e-ttft):.2f} tok/s")
    return pt/ttft, ct/(e2e-ttft), ttft

rs = [one_run(i) for i in range(3)]
avg = lambda xs: sum(xs)/len(xs)
print(f"AVG : prefill={avg([r[0] for r in rs]):.1f} decode={avg([r[1] for r in rs]):.2f} ttft={avg([r[2] for r in rs]):.2f}s")
print("TARGET: prefill=1408.1 decode=92.09 ttft=2.91s")
