"""Run one execution configuration of one GGUF over the frozen eval set via llama-server.

Every configuration is meant to be *execution-equivalent* to the reference: same binary
release, same weights file, same prompts, greedy decoding. Only how the arithmetic is
executed changes (ISA backend, threads, ubatch, flash attention, repack, slots, prompt cache).

Usage:
  py -3 run.py --variant alderlake --model M.gguf --name qwen06-q4_0__ref [--threads 4]
       [--ub 512] [--fa off] [--repack on] [--np 1] [--conc 1] [--cache 0] [--port 8100]
       [--mmlu 1000] [--tqa 500] [--gsm 150]

Writes results/<name>.jsonl, one row per item:
  id, task, correct, pred, tokens (generated ids), lps (logprob of each chosen token, float32
  as printed by the server), top (first-step top-20 [id, logprob]) and for MMLU the letter
  logprobs and margin.
"""
import argparse, json, re, string, subprocess, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISA = Path(r"C:\Users\jymul\dev\isa")


def norm(s):  # SQuAD/TriviaQA-style answer normalization
    s = "".join(c for c in s.lower() if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def last_number(s):
    s = s.split("####")[-1] if "####" in s else s
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s)
    return nums[-1].replace(",", "").rstrip(".") if nums else ""


def letter_scores(top):
    best = {}
    for t in top:
        k = t["token"].strip()
        if k in ("A", "B", "C", "D"):
            best[k] = max(best.get(k, -1e30), t["logprob"])
    return best


def load_rows(n_mmlu, n_tqa, n_gsm):
    rows = [json.loads(l) for l in (ROOT / "data" / "evalset.jsonl").open(encoding="utf-8")]
    out = []
    for task, n in (("mmlu", n_mmlu), ("triviaqa", n_tqa), ("gsm8k", n_gsm)):
        out += [r for r in rows if r["task"] == task][:n]
    return out


class Client:
    def __init__(self, port, cache):
        self.url = f"http://127.0.0.1:{port}/completion"
        self.cache = bool(cache)

    def post(self, payload):
        payload = {"temperature": 0, "top_k": 1, "seed": 0, "cache_prompt": self.cache, **payload}
        req = urllib.request.Request(self.url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.load(r)

    def run(self, row):
        t = row["task"]
        if t == "mmlu":
            d = self.post({"prompt": row["prompt"], "n_predict": 1, "n_probs": 20})
        elif t == "triviaqa":
            d = self.post({"prompt": row["prompt"], "n_predict": 24, "n_probs": 20, "stop": ["\n"]})
        else:
            d = self.post({"prompt": row["prompt"], "n_predict": 256, "n_probs": 20,
                           "stop": ["\n\nQuestion:", "\nQuestion:"]})
        probs = d.get("completion_probabilities", [])
        rec = {"id": row["id"], "task": t,
               "tokens": [p["id"] for p in probs],
               "lps": [p["logprob"] for p in probs],
               "top": [[q["id"], q["logprob"]] for q in probs[0]["top_logprobs"]] if probs else []}
        out = d["content"]
        if t == "mmlu":
            ls = letter_scores(probs[0]["top_logprobs"]) if probs else {}
            if ls:
                order = sorted(ls, key=ls.get, reverse=True)
                rec["pred"] = order[0]
                rec["margin"] = ls[order[0]] - ls[order[1]] if len(order) > 1 else 99.0
            else:
                rec["pred"], rec["margin"] = None, None
            rec["letters"] = ls
            rec["correct"] = rec["pred"] == row["gold"]
        elif t == "gsm8k":
            pred = last_number(out)
            rec["pred"] = pred
            try:
                rec["correct"] = pred != "" and float(pred) == float(row["gold"])
            except ValueError:
                rec["correct"] = False
        else:
            rec["pred"] = out
            rec["correct"] = norm(out) in {norm(g) for g in row["gold"]}
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="alderlake")
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ub", type=int, default=512)
    ap.add_argument("--fa", default="off")
    ap.add_argument("--repack", default="on")
    ap.add_argument("--np", type=int, default=1)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--cache", type=int, default=0)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--mmlu", type=int, default=1000)
    ap.add_argument("--tqa", type=int, default=500)
    ap.add_argument("--gsm", type=int, default=150)
    a = ap.parse_args()

    out = ROOT / "results" / f"{a.name}.jsonl"
    if out.exists():
        print(f"skip {a.name} (exists)")
        return
    rows = load_rows(a.mmlu, a.tqa, a.gsm)
    exe = ISA / a.variant / "llama-server.exe"
    cmd = [str(exe), "-m", a.model, "--port", str(a.port), "-np", str(a.np), "-c", str(4096 * a.np),
           "-t", str(a.threads), "-tb", str(a.threads), "-b", "2048", "-ub", str(a.ub), "-fa", a.fa,
           "--no-webui", "--no-warmup"]
    if a.repack == "off":
        cmd.append("--no-repack")
    log_path = ROOT / "results" / "logs" / f"{a.name}.server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2)
                break
            except Exception:
                if proc.poll() is not None:
                    raise SystemExit(f"server died, see {log_path}")
                time.sleep(1)
        c = Client(a.port, a.cache)
        with ThreadPoolExecutor(a.conc) as ex:
            res = list(ex.map(c.run, rows))
    finally:
        proc.terminate()
        proc.wait()
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"meta": vars(a) | {"cmd": cmd, "seconds": round(time.time() - t0)}}) + "\n")
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out)
    summ = {t: f"{sum(r['correct'] for r in res if r['task'] == t)}/{sum(r['task'] == t for r in res)}"
            for t in ("mmlu", "triviaqa", "gsm8k")}
    print(json.dumps({"name": a.name, "s": round(time.time() - t0), **summ}), flush=True)


if __name__ == "__main__":
    assert last_number("so 5 + 37 = 42\n#### 42") == "42" and last_number("total is $1,234.") == "1234"
    assert norm("The Ghost!") == "ghost"
    assert letter_scores([{"token": " B", "logprob": -1.0}, {"token": "B", "logprob": -0.5},
                          {"token": " x", "logprob": 0.0}]) == {"B": -0.5}
    main()
