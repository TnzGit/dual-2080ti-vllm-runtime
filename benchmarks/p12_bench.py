#!/usr/bin/env python
"""P12b: robust same-domain bench via non-streaming requests.
decode = completion_tokens / (e2e_full - e2e_prefill_only)
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8002"
TOPICS = ["quantum computing", "marine biology", "Renaissance art", "space exploration"]


def run(i, max_new):
    prompt = f"The history of {TOPICS[i % 4]} is long and fascinating. " * 455
    body = json.dumps({"text": prompt,
                       "sampling_params": {"temperature": 0,
                                           "max_new_tokens": max_new}}).encode()
    req = urllib.request.Request(BASE + "/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req) as r:
        out = json.loads(r.read())
    e2e = time.time() - t0
    mi = out.get("meta_info", {})
    return e2e, mi.get("prompt_tokens", 0), mi.get("completion_tokens", 0)


rows = []
for i in range(3):
    e2e_s, pt, ct_s = run(i, 8)
    e2e_f, _, ct_f = run(i, 136)
    rate = (ct_f - ct_s) / (e2e_f - e2e_s)
    rows.append((pt / e2e_s, rate, e2e_s, e2e_f))

avg = lambda xs: sum(xs) / len(xs)
print("PREFILL=%.1f DECODE=%.2f (e2e_prefill=%.3fs e2e_full=%.3fs)" % (
    avg([r[0] for r in rows]), avg([r[1] for r in rows]),
    avg([r[2] for r in rows]), avg([r[3] for r in rows])))
