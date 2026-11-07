"""Teacher-forced divergence grid with llama-perplexity: for each model, save reference logits
(alderlake, 8 threads, ub 512, fa off) on N chunks of 512 tokens of wikitext, then rerun each
execution-equivalent configuration against that file and record mean/max KLD, same-top-1 rate
and PPL. Writes results/kl.jsonl (one row per model x config). Skips rows already present."""
import json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISA = Path(r"C:\Users\jymul\dev\isa")
WORK = Path(r"C:\Users\jymul\dev\swkl")
M = Path(r"C:\Users\jymul\dev\swmodels")
MODELS = {
    "q06-q4km": "Qwen3-0.6B-Q4_K_M.gguf", "q06-q40": "Qwen3-0.6B-Q4_0.gguf",
    "q06-q80": "Qwen3-0.6B-Q8_0.gguf", "q06-f16": "Qwen3-0.6B-F16.gguf",
    "g3-1b-q40": "gemma-3-1b-it-qat-Q4_0.gguf", "q35-08-q40": "Qwen3.5-0.8B-Q4_0.gguf",
}
CONFIGS = {  # name: (variant, extra args)
    "ref-rerun": ("alderlake", []),
    "isa-haswell": ("haswell", []), "isa-ivybridge": ("ivybridge", []),
    "isa-sandybridge": ("sandybridge", []), "isa-sse42": ("sse42", []),
    "t4": ("alderlake", ["-t", "4", "-tb", "4"]), "t12": ("alderlake", ["-t", "12", "-tb", "12"]),
    "ub1": ("alderlake", ["-ub", "1"]), "ub7": ("alderlake", ["-ub", "7"]),
    "ub8": ("alderlake", ["-ub", "8"]), "ub64": ("alderlake", ["-ub", "64"]),
    "fa-on": ("alderlake", ["-fa", "on"]), "repack-off": ("alderlake", ["--no-repack"]),
}
CHUNKS = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def run(variant, model, extra, logits, base):
    cmd = [str(ISA / variant / "llama-perplexity.exe"), "-m", str(M / model), "-f", str(ROOT / "data" / "wiki_kl.txt"),
           "-c", "512", "-b", "512", "--chunks", str(CHUNKS), "-t", "8", "-tb", "8", "-fa", "off",
           "--kl-divergence-base", str(logits)] + (["--kl-divergence"] if not base else []) + extra
    # later flags win in llama.cpp's arg parser, so per-config overrides go last
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.stdout + p.stderr


def grab(txt, pat):
    m = re.search(pat, txt)
    return float(m.group(1)) if m else None


def main():
    WORK.mkdir(exist_ok=True)
    out = ROOT / "results" / "kl.jsonl"
    done = set()
    if out.exists():
        done = {(r["model"], r["config"]) for r in map(json.loads, out.read_text().splitlines())}
    with out.open("a", encoding="utf-8", newline="\n") as f:
        for m, file in MODELS.items():
            base = WORK / f"{m}.kld"
            if not base.exists():
                txt = run("alderlake", file, [], base, True)
                if not base.exists():
                    print(txt[-2000:]); raise SystemExit("base failed")
            for c, (v, extra) in CONFIGS.items():
                if (m, c) in done:
                    continue
                txt = run(v, file, extra, base, False)
                row = {"model": m, "config": c, "chunks": CHUNKS,
                       "mean_kld": grab(txt, r"Mean\s+KLD:\s+([-\d.eE+]+)"),
                       "max_kld": grab(txt, r"Maximum KLD:\s+([-\d.eE+]+)"),
                       "same_top": grab(txt, r"Same top p:\s+([\d.]+)"),
                       "rms_dp": grab(txt, r"RMS Δp\s*:\s+([\d.]+)"),
                       "ppl_q": grab(txt, r"Mean PPL\(Q\)\s*:\s+([\d.]+)")}
                if row["mean_kld"] is None:
                    (WORK / f"{m}__{c}.log").write_text(txt, encoding="utf-8")
                print(json.dumps(row), flush=True)
                f.write(json.dumps(row) + "\n"); f.flush()


if __name__ == "__main__":
    main()
