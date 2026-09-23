#!/usr/bin/env python
"""P7 DFlash2 A/B benchmark: fixed distinct prompts, greedy, sequential.

Usage: python3 p7_bench.py <label>
Appends a summary line to ~/p7_ab_summary.txt
"""
import json
import sys
import time

import requests

BASE = "http://127.0.0.1:8002"
MAX_NEW = 256

TOPICS = [
    "Explain how a transformer attention head allocates weights across a long document, then continue with a detailed numerical example involving a 4096-token context.",
    "Write a step-by-step recipe for sourdough bread, then explain the microbiology of wild yeast cultivation in detail.",
    "Describe the logistics of a container ship arriving at a busy port, including crane scheduling, customs, and rail handoff, in concrete detail.",
    "Walk through the derivation of the Black-Scholes equation, then explain each term's economic intuition.",
    "Explain how a modern CPU branch predictor works, then trace an example loop's execution with misprediction penalties.",
    "Give a detailed history of the Suez Canal and analyze its geopolitical importance for global trade.",
    "Explain the water cycle in a tropical rainforest, then describe how deforestation changes regional rainfall patterns.",
    "Describe how mRNA vaccines work at the molecular level, then discuss cold-chain logistics challenges.",
    "Explain the C10K problem and how epoll solves it, with a concrete server design walkthrough.",
    "Describe the manufacturing process of a lithium-ion battery cell from raw materials to finished pack, step by step.",
]


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    rows = []
    total_tokens = 0
    t_all0 = time.time()
    for i, text in enumerate(TOPICS):
        payload = {
            "text": text,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": MAX_NEW,
                "ignore_eos": True,
            },
        }
        t0 = time.time()
        r = requests.post(f"{BASE}/generate", json=payload, timeout=600)
        dt = time.time() - t0
        r.raise_for_status()
        out = r.json()
        comp = int(out.get("meta_info", {}).get("completion_tokens", 0))
        total_tokens += comp
        rows.append((i, comp, dt))
        print(f"[{label}] prompt {i}: {comp} tok in {dt:.2f}s -> {comp/dt:.1f} tok/s", flush=True)
    wall = time.time() - t_all0
    print(f"[{label}] TOTAL: {total_tokens} tok in {wall:.1f}s -> {total_tokens/wall:.1f} tok/s (incl. prefill)", flush=True)
    with open(f"<HOME>/p7_ab_summary.txt", "a") as f:
        f.write(f"{label} total_tok={total_tokens} wall={wall:.1f}s tokps={total_tokens/wall:.1f}\n")


if __name__ == "__main__":
    main()
