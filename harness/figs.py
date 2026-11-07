"""Figures for the paper from results/*.jsonl, results/summary.json and results/kl.jsonl."""
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent.parent / "paper" / "fig"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
INK, ACC, MUT = "#222222", "#b5432a", "#9a9a9a"


def rows(name):
    p = R / f"{name}.jsonl"
    if not p.exists():
        return None
    ls = p.read_text(encoding="utf-8").splitlines()[1:]
    return {r["id"]: r for r in map(json.loads, ls)}


def fig_kl_grid():
    p = R / "kl.jsonl"
    if not p.exists():
        return
    kl = [json.loads(l) for l in p.read_text().splitlines()]
    models = list(dict.fromkeys(r["model"] for r in kl))
    configs = list(dict.fromkeys(r["config"] for r in kl))
    M = np.full((len(models), len(configs)), np.nan)
    for r in kl:
        if r["mean_kld"] is not None:
            M[models.index(r["model"]), configs.index(r["config"])] = r["mean_kld"]
    fig, ax = plt.subplots(figsize=(6.8, 2.3))
    disp = np.where(M > 0, np.log10(np.maximum(M, 1e-12)), np.nan)
    im = ax.imshow(disp, cmap="Oranges", vmin=-7, vmax=-2, aspect="auto")
    for i in range(len(models)):
        for j in range(len(configs)):
            v = M[i, j]
            txt = "—" if math.isnan(v) else ("0" if v == 0 else f"{v:.0e}".replace("e-0", "e-"))
            ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                    color=INK if (math.isnan(v) or v == 0 or disp[i, j] < -4) else "white")
    ax.set_xticks(range(len(configs)), configs, rotation=40, ha="right")
    ax.set_yticks(range(len(models)), models)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("log10 mean KLD", fontsize=7)
    ax.set_title("Teacher-forced mean KL divergence from the reference (0 = no difference at storage precision)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "kl_grid.pdf")


def fig_margins():
    ref = rows("q06-q4km__ref")
    if ref is None:
        return
    allm = [r["margin"] for r in ref.values() if r["task"] == "mmlu" and r["margin"] is not None]
    flipped = {}
    for c in ["isa-sandybridge", "isa-sse42", "isa-ivybridge", "ub7", "cache-on", "server4-a", "server4-b",
              "fa-on", "repack-off", "ub64"]:
        o = rows(f"q06-q4km__{c}")
        if o is None:
            continue
        for k, r in ref.items():
            if r["task"] == "mmlu" and r["pred"] != o[k]["pred"] and r["margin"] is not None:
                flipped[k] = r["margin"]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    bins = np.logspace(-6, 1, 36)
    ax.hist(np.clip(allm, 1e-6, None), bins=bins, color=MUT, label=f"all MMLU items (n={len(allm)})")
    ax.hist(np.clip(list(flipped.values()), 1e-6, None), bins=bins, color=ACC,
            label=f"answer changed in any config (n={len(flipped)})")
    ax.set_xscale("log")
    ax.set_xlabel("reference margin between top two letters (nats)")
    ax.set_ylabel("items")
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "margins.pdf")


def fig_divergence():
    ref = rows("q06-q4km__ref")
    if ref is None:
        return
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for c, col, ls in [("isa-sandybridge", ACC, "-"), ("server4-a", INK, "-"), ("cache-on", INK, "--"),
                       ("ub7", MUT, "-"), ("isa-sse42", ACC, ":")]:
        o = rows(f"q06-q4km__{c}")
        if o is None:
            continue
        g = [(r, o[k]) for k, r in ref.items() if r["task"] == "gsm8k"]
        pos = []
        for a, b in g:
            d = next((i for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y), None)
            if d is None and len(a["tokens"]) != len(b["tokens"]):
                d = min(len(a["tokens"]), len(b["tokens"]))
            pos.append(d)
        xs = np.arange(0, 257)
        frac = [sum(p is not None and p <= x for p in pos) / len(pos) for x in xs]
        ax.plot(xs, frac, color=col, ls=ls, lw=1.1, label=c)
    ax.set_xlabel("generated token index")
    ax.set_ylabel("GSM8K outputs diverged so far")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(OUT / "divergence.pdf")


def fig_noise_floor():
    s = json.loads((R / "summary.json").read_text())
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ys, labels = [], []
    for i, (m, d) in enumerate((k, v) for k, v in s.items() if not k.startswith("_")):
        vals = [100 * v["mmlu"]["dacc"] for c, v in d.items() if c not in ("ref_acc", "server4-a_vs_b")]
        ax.scatter(vals, [i] * len(vals), s=10, color=INK, zorder=3)
        labels.append(m)
    q = s.get("_quant_pairs", {})
    for j, (pair, v) in enumerate(q.items()):
        ax.scatter([100 * v["mmlu"]["dacc"]], [len(labels) + j], s=14, color=ACC, marker="D", zorder=3)
        labels.append(pair.replace("q06-", "").replace("|", " vs "))
    ax.axvline(0, color=MUT, lw=0.6)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    ax.set_xlabel("MMLU accuracy change (points)")
    fig.tight_layout()
    fig.savefig(OUT / "noise_floor.pdf")


if __name__ == "__main__":
    fig_kl_grid(); fig_margins(); fig_divergence(); fig_noise_floor()
    print("figures ->", OUT)
