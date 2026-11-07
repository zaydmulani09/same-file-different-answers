"""Addendum A test: does activation discretization amplify execution differences?
Qwen3-0.6B in six formats; reference alderlake/8 threads; perturbations ivybridge, sse42, t4.
Writes results/amp.jsonl. Usage: py -3 amp.py [chunks]"""
import json, sys
from pathlib import Path
from kl import ISA, M, ROOT, grab
import subprocess

WORK = Path(r"C:\Users\jymul\dev\swkl\amp")
FORMATS = {  # label: (file, extra args)
    "F32+f32KV": ("Qwen3-0.6B-F32.gguf", ["-ctk", "f32", "-ctv", "f32"]),
    "F32": ("Qwen3-0.6B-F32.gguf", []),
    "F16": ("Qwen3-0.6B-F16.gguf", []),
    "Q8_0": ("Qwen3-0.6B-Q8_0.gguf", []),
    "Q4_K_M": ("Qwen3-0.6B-Q4_K_M.gguf", []),
    "Q4_0": ("Qwen3-0.6B-Q4_0.gguf", []),
}
PERTURB = {"ivybridge": ("ivybridge", []), "sse42": ("sse42", []), "t4": ("alderlake", ["-t", "4", "-tb", "4"])}
CHUNKS = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def run(variant, model, extra, logits, base):
    cmd = [str(ISA / variant / "llama-perplexity.exe"), "-m", str(M / model), "-f", str(ROOT / "data" / "wiki_kl.txt"),
           "-c", "512", "-b", "512", "--chunks", str(CHUNKS), "-t", "8", "-tb", "8", "-fa", "off",
           "--kl-divergence-base", str(logits)] + ([] if base else ["--kl-divergence"]) + extra
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.stdout + p.stderr


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    out = ROOT / "results" / "amp.jsonl"
    with out.open("a", encoding="utf-8", newline="\n") as f:
        for lab, (file, fx) in FORMATS.items():
            base = WORK / f"{lab}.kld"
            if not base.exists():
                t = run("alderlake", file, fx, base, True)
                if not base.exists():
                    print(t[-1500:]); raise SystemExit("base failed")
            for pn, (v, px) in PERTURB.items():
                t = run(v, file, fx + px, base, False)
                row = {"format": lab, "perturb": pn, "chunks": CHUNKS,
                       "mean_kld": grab(t, r"Mean\s+KLD:\s+([-\d.eE+]+)"),
                       "max_kld": grab(t, r"Maximum KLD:\s+([-\d.eE+]+)"),
                       "same_top": grab(t, r"Same top p:\s+([\d.]+)")}
                print(json.dumps(row), flush=True)
                f.write(json.dumps(row) + "\n"); f.flush()
