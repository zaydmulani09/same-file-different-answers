"""Scale reference: KLD of each quantized Qwen3-0.6B file against the F16 file (both at the KL-grid
reference config), on the same 12 chunks. Uses the F16 base logits saved by kl.py."""
import json
from kl import run, grab, WORK, ROOT
out = ROOT / "results" / "quantkl.jsonl"
with out.open("w", encoding="utf-8", newline="\n") as f:
    for lab, file in [("q06-q80", "Qwen3-0.6B-Q8_0.gguf"), ("q06-q40", "Qwen3-0.6B-Q4_0.gguf"), ("q06-q4km", "Qwen3-0.6B-Q4_K_M.gguf")]:
        t = run("alderlake", file, [], WORK / "q06-f16.kld", False)
        row = {"model": lab, "vs": "q06-f16", "mean_kld": grab(t, r"Mean\s+KLD:\s+([-\d.eE+]+)"),
               "same_top": grab(t, r"Same top p:\s+([\d.]+)")}
        print(json.dumps(row), flush=True); f.write(json.dumps(row) + "\n")
