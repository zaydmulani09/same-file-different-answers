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
        # |delta| of the reference's top-1 first-step logprob (the decision-relevant quantity)
        d1 = sorted(abs(a["top"][0][1] - dict(map(tuple, b["top"])).get(a["top"][0][0], float("nan")))
                    for a, b in pairs if a["top"] and b["top"])
        d1 = [x for x in d1 if not math.isnan(x)]
        d = {"n": n, "identical": ident, "same_output": same, "flips": r2w + w2r, "r2w": r2w, "w2r": w2r,
             "acc_ref": acc_r, "acc": acc_o, "dacc": acc_o - acc_r,
             "mcnemar_p": binomtest(r2w, r2w + w2r).pvalue if r2w + w2r else 1.0,
             "dlp_median": dl[len(dl) // 2] if dl else None, "dlp_max": dl[-1] if dl else None,
             "dlp1_median": sorted(d1)[len(d1) // 2] if d1 else None,
             "dlp1_nonzero": sum(x > 0 for x in d1)}
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


PAPER = Path(__file__).resolve().parent.parent / "paper"
NICE = {"ref-rerun": "ref-rerun", "isa-ivybridge": "ivybridge tier", "isa-sandybridge": "sandybridge tier",
        "isa-sse42": "sse42 tier", "ub7": "micro-batch 7", "ub64": "micro-batch 64", "fa-on": "flash attn on",
        "repack-off": "repack off", "cache-on": "prompt cache on", "server4-a": "server4 (run a)",
        "server4-b": "server4 (run b)", "server4-a_vs_b": "server4 a vs b"}


def pct(x, n):
    return f"{100 * x / n:.1f}"


def write_tables():
    s = json.loads((R / "summary.json").read_text())
    L = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"model & configuration & ident. & same out & MMLU ans. & flips & $\Delta$MMLU & $\Delta$TQA & $\Delta$GSM\\",
         r" & & \% & \% & changed & (r$\to$w / w$\to$r) & pts & pts & pts\\", r"\midrule"]
    for m, d in s.items():
        if m.startswith("_"):
            continue
        first = True
        for c in [*CONFIGS, "server4-a_vs_b"]:
            if c not in d:
                continue
            v, A = d[c], d[c]["all"]
            L.append(f"{m if first else ''} & {NICE[c]} & {pct(A['identical'], A['n'])} & {pct(A['same_output'], A['n'])} & "
                     f"{v['mmlu']['pred_changed']} & {A['flips']} ({A['r2w']}/{A['w2r']}) & "
                     f"{100*v['mmlu']['dacc']:+.1f} & {100*v['triviaqa']['dacc']:+.1f} & {100*v['gsm8k']['dacc']:+.1f}\\\\")
            first = False
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    (PAPER / "tab_eval.tex").write_text("\n".join(L) + "\n")

    q = s.get("_quant_pairs", {})
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"pair (both at $R$) & same out \% & MMLU ans. changed & flips & $\Delta$MMLU & $\Delta$GSM\\", r"\midrule"]
    for pair, v in q.items():
        A = v["all"]
        L.append(f"{pair.replace('|', ' vs ')} & {pct(A['same_output'], A['n'])} & {v['mmlu']['pred_changed']} & {A['flips']} & "
                 f"{100*v['mmlu']['dacc']:+.1f} & {100*v['gsm8k']['dacc']:+.1f}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (PAPER / "tab_quant.tex").write_text("\n".join(L) + "\n")

    amp = R / "amp.jsonl"
    if amp.exists():
        rows = [json.loads(l) for l in amp.read_text().splitlines()]
        fmts = list(dict.fromkeys(r["format"] for r in rows))
        per = ["ivybridge", "sse42", "t4"]
        L = [r"\begin{tabular}{l" + "rr" * len(per) + "}", r"\toprule",
             "format & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{p}}}" for p in per) + r"\\",
             " & " + " & ".join(["mean KLD & top-1 \\%"] * len(per)) + r"\\", r"\midrule"]
        for f_ in fmts:
            cells = []
            for p in per:
                r = next((x for x in rows if x["format"] == f_ and x["perturb"] == p), None)
                if r is None:
                    cells += ["--", "--"]
                    continue
                k = r["mean_kld"]
                cells += ["$\\le 10^{-6}$" if abs(k) <= 1e-6 else f"{k:.4f}", f"{r['same_top']:.2f}"]
            L.append(f"{f_.replace('_', chr(92) + '_')} & " + " & ".join(cells) + r"\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (PAPER / "tab_amp.tex").write_text("\n".join(L) + "\n")

    kl = R / "kl.jsonl"
    if kl.exists():
        rows = [json.loads(l) for l in kl.read_text().splitlines()]
        models = list(dict.fromkeys(r["model"] for r in rows))
        cfgs = list(dict.fromkeys(r["config"] for r in rows))
        L = [r"\begin{tabular}{l" + "r" * len(models) + "}", r"\toprule",
             "configuration & " + " & ".join(models) + r"\\", r"\midrule"]
        for c in cfgs:
            cells = []
            for m in models:
                r = next((x for x in rows if x["model"] == m and x["config"] == c), None)
                if r is None or r["mean_kld"] is None:
                    cells.append("--")
                elif abs(r["mean_kld"]) <= 1e-6 and r["same_top"] == 100.0:
                    cells.append("$=$")
                else:
                    cells.append(f"{r['mean_kld']:.1e} / {r['same_top']:.1f}".replace("e-0", "e-"))
            L.append(f"{c} & " + " & ".join(cells) + r"\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (PAPER / "tab_kl.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
    write_tables()
