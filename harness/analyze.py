"""Compare every configuration of a model against its reference run.

Outputs results/summary.json and prints a table. Definitions:
  identical   : generated token ids AND every chosen-token logprob AND the first-step top-20
                (ids and logprobs) are exactly equal to the reference (bitwise on float32 text).
  same_output : generated token ids equal (the visible answer string is the same).
  flip        : correctness differs from the reference.
"""
import json, math, sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import binomtest

R = Path(__file__).resolve().parent.parent / "results"


def load(name):
    p = R / f"{name}.jsonl"
    if not p.exists():
        return None, None
    lines = p.read_text(encoding="utf-8").splitlines()
    meta = json.loads(lines[0])["meta"]
    return meta, {r["id"]: r for r in map(json.loads, lines[1:])}


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def max_top_dlp(a, b):
    """max |delta logprob| over token ids present in both first-step top-20 lists"""
    da = dict(map(tuple, a)); db = dict(map(tuple, b))
    common = set(da) & set(db)
    return max((abs(da[k] - db[k]) for k in common), default=float("nan"))


def compare(ref, oth):
    out = {}
    tasks = defaultdict(list)
    for k, r in ref.items():
        tasks[r["task"]].append((r, oth[k]))
    tasks["all"] = [p for t in ("mmlu", "triviaqa", "gsm8k") for p in tasks[t]]
    for t, pairs in tasks.items():
        n = len(pairs)
        ident = sum(a["tokens"] == b["tokens"] and a["lps"] == b["lps"] and a["top"] == b["top"] for a, b in pairs)
        same = sum(a["tokens"] == b["tokens"] for a, b in pairs)
        r2w = sum(a["correct"] and not b["correct"] for a, b in pairs)
        w2r = sum(b["correct"] and not a["correct"] for a, b in pairs)
        acc_r = sum(a["correct"] for a, _ in pairs) / n
        acc_o = sum(b["correct"] for _, b in pairs) / n
        dl = [max_top_dlp(a["top"], b["top"]) for a, b in pairs if a["top"] and b["top"]]
        dl = sorted(x for x in dl if not math.isnan(x))
        d = {"n": n, "identical": ident, "same_output": same, "flips": r2w + w2r, "r2w": r2w, "w2r": w2r,
             "acc_ref": acc_r, "acc": acc_o, "dacc": acc_o - acc_r,
             "mcnemar_p": binomtest(r2w, r2w + w2r).pvalue if r2w + w2r else 1.0,
             "dlp_median": dl[len(dl) // 2] if dl else None, "dlp_max": dl[-1] if dl else None}
        if t == "mmlu":
            d["pred_changed"] = sum(a["pred"] != b["pred"] for a, b in pairs)
            d["flip_ref_margins"] = sorted(a["margin"] for a, b in pairs if a["pred"] != b["pred"] and a["margin"] is not None)
        if t in ("gsm8k", "triviaqa"):
            divs = [first_div(a["tokens"], b["tokens"]) for a, b in pairs]
            divs = [x for x in divs if x is not None]
            d["div_positions"] = sorted(divs)
        out[t] = d
    return out


MODELS = ["q06-q4km", "q06-q40", "q06-q80", "q06-f16", "g3-1b-q40", "q35-08-q40"]
CONFIGS = ["ref-rerun", "isa-haswell", "isa-ivybridge", "isa-sandybridge", "isa-sse42", "t1", "t12",
           "ub7", "ub64", "fa-on", "repack-off", "cache-on", "server4-a", "server4-b"]


def main():
    summ = {}
    for m in MODELS:
        _, ref = load(f"{m}__ref")
        if ref is None:
            continue
        summ[m] = {"ref_acc": {t: sum(r["correct"] for r in ref.values() if r["task"] == t) /
                               sum(r["task"] == t for r in ref.values()) for t in ("mmlu", "triviaqa", "gsm8k")}}
        for c in CONFIGS:
            meta, oth = load(f"{m}__{c}")
            if oth is None:
                continue
            summ[m][c] = compare(ref, oth)
            summ[m][c]["seconds"] = meta["seconds"]
        # run-to-run under the parallel server: a vs b
        _, a = load(f"{m}__server4-a"); _, b = load(f"{m}__server4-b")
        if a and b:
            summ[m]["server4-a_vs_b"] = compare(a, b)
    # cross-model: quantization differences on the same reference config (the "effect" scale)
    refs = {m: load(f"{m}__ref")[1] for m in ("q06-q4km", "q06-q40", "q06-q80", "q06-f16")}
    refs = {k: v for k, v in refs.items() if v}
    summ["_quant_pairs"] = {f"{x}|{y}": compare(refs[x], refs[y]) for x in refs for y in refs if x < y}
    (R / "summary.json").write_text(json.dumps(summ, indent=1))
    for m, d in summ.items():
        if m.startswith("_"):
            continue
        print(f"\n== {m}  ref acc " + " ".join(f"{t}={v:.3f}" for t, v in d["ref_acc"].items()))
        print(f"{'config':18s} {'ident%':>7s} {'sameout%':>8s} {'flips':>5s} {'mmluPred':>8s} {'dAcc_mmlu':>9s} {'dAcc_tqa':>8s} {'dAcc_gsm':>8s} {'dlp_med':>9s}")
        for c, v in d.items():
            if c == "ref_acc":
                continue
            A = v["all"]
            print(f"{c:18s} {100*A['identical']/A['n']:7.1f} {100*A['same_output']/A['n']:8.1f} {A['flips']:5d} "
                  f"{v['mmlu'].get('pred_changed', 0):8d} {100*v['mmlu']['dacc']:+9.1f} {100*v['triviaqa']['dacc']:+8.1f} "
                  f"{100*v['gsm8k']['dacc']:+8.1f} {A['dlp_median'] if A['dlp_median'] is not None else float('nan'):9.2e}")


if __name__ == "__main__":
    main()
