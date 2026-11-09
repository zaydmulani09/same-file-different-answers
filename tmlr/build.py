"""Build the anonymized TMLR submission from paper/ sources.

  py -3 tmlr/build.py

Writes tmlr/out/main.pdf (anonymous, for review) and tmlr/out/supplementary.zip
(code + data without git history or author names). paper/ stays the source of truth.
"""
import pathlib, re, shutil, subprocess, zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAPER, OUT, STYLE = ROOT / "paper", ROOT / "tmlr" / "out", ROOT / "tmlr" / "_style"
REPO = r"\url{https://github.com/zaydmulani09/same-file-different-answers}"
ANON = "the supplementary material"
IDENT = re.compile(r"zayd|mulani|zenodo|jymul", re.I)

HEAD = r"""\documentclass[10pt]{article}
\usepackage{tmlr}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{url}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\setlength{\emergencystretch}{3em}
\title{Same File, Different Answers: Execution-Equivalent Settings Change Benchmark Results of Quantized LLMs on CPUs}
\author{\name Anonymous authors \\ \addr Paper under double-blind review}
\def\month{MM}\def\year{YYYY}\def\openreview{\url{https://openreview.net/forum?id=XXXX}}
\begin{document}
\maketitle
"""
BODY = ["sec_abstract", "sec_intro", "sec_background", "sec_related", "sec_method",
        "sec_results", "sec_mechanism", "sec_recommend", "sec_limits", "refs"]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for f in ("tmlr.sty", "fancyhdr.sty", "tmlr.bst"):
        shutil.copy(STYLE / f, OUT)
    shutil.copytree(PAPER / "fig", OUT / "fig")
    for tex in PAPER.glob("*.tex"):
        if tex.name == "main.tex":
            continue
        s = tex.read_text(encoding="utf-8").replace(REPO, ANON)
        if tex.name == "sec_abstract.tex":  # "are public at the supplementary material" reads badly
            s = s.replace("are\npublic at " + ANON, "are in " + ANON).replace("are public at " + ANON, "are in " + ANON)
        if tex.name == "sec_appendix.tex":
            s = s.replace("are in\n" + ANON + "; its commit history shows", "are in " + ANON + "; its git history (shared after review) shows")
        hit = IDENT.search(s)
        assert not hit, f"identifying text left in {tex.name}: {hit.group(0)}"
        (OUT / tex.name).write_text(s, encoding="utf-8")
    body = "".join(f"\\input{{{b}}}\n" for b in BODY)
    (OUT / "main.tex").write_text(HEAD + body + "\\appendix\n\\input{sec_appendix}\n\\end{document}\n", encoding="utf-8")
    subprocess.run(["tectonic", "-X", "compile", "main.tex"], cwd=OUT, check=True)
    supplementary()


def supplementary():
    """Code, data and results, minus anything naming the author."""
    keep = ["harness", "data", "results", "PREDICTIONS.md"]
    with zipfile.ZipFile(OUT / "supplementary.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for k in keep:
            p = ROOT / k
            for f in ([p] if p.is_file() else sorted(p.rglob("*"))):
                if f.is_dir() or "__pycache__" in f.parts or f.suffix == ".tmp" or "logs" in f.relative_to(ROOT).parts:
                    continue
                arc = f.relative_to(ROOT).as_posix()
                if f.suffix in (".py", ".md", ".txt", ".json", ".jsonl"):
                    t = f.read_text(encoding="utf-8").replace("jymul", "user")  # Windows username in model paths
                    assert not IDENT.search(t), f"identifying text in {arc}: {IDENT.search(t).group(0)}"
                    z.writestr(arc, t)
                else:
                    z.write(f, arc)
        z.writestr("README.md", README)


README = """# Supplementary material (anonymized for review)

harness/  scripts that produced every number (run.py, jobs.py, kl.py, amp.py, quantkl.py,
          amp_sim.py, ub_shape.py, analyze.py, figs.py)
data/     frozen MMLU / GSM8K / TriviaQA items and the WikiText text for the KL grids
results/  raw per-item outputs (<model>__<config>.jsonl), kl.jsonl, amp.jsonl, quantkl.jsonl,
          ub_shape.jsonl, summary.json
PREDICTIONS.md  predictions and Addendum A; the git history showing they were committed
          before the results will be linked after review.

Rebuild tables and figures: py -3 harness/analyze.py && py -3 harness/figs.py
Full reproduction steps are in the paper's appendix.
"""

if __name__ == "__main__":
    main()
