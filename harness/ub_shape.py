"""Follow-up probe (added after the ub64 benchmark run differed from R although the 512-token
KL grid showed ub64 identical): does micro-batch 64 change the logits when the context length
is not a multiple of 64? Q4_K_M, alderlake, 8 chunks per context length. Writes results/ub_shape.jsonl."""
import json
from kl import run, grab, WORK, ROOT, ISA, M
import kl

MODEL = "Qwen3-0.6B-Q4_K_M.gguf"
out = ROOT / "results" / "ub_shape.jsonl"
kl.CHUNKS = 8
with out.open("w", encoding="utf-8", newline="\n") as f:
    for ctx in (512, 500, 449, 450):
        base = WORK / f"ubshape_{ctx}.kld"
        orig = kl.subprocess.run

        def patched(cmd, **kw):  # swap the fixed -c/-b 512 for this context length
            cmd = [str(ctx) if (i > 0 and cmd[i - 1] in ("-c", "-b")) else a for i, a in enumerate(cmd)]
            return orig(cmd, **kw)

        kl.subprocess.run = patched
        run("alderlake", MODEL, [], base, True)
        for ub in ("64", "8"):
            txt = run("alderlake", MODEL, ["-ub", ub], base, False)
            row = {"ctx": ctx, "ub": int(ub), "mean_kld": grab(txt, r"Mean\s+KLD:\s+([-\d.eE+]+)"),
                   "same_top": grab(txt, r"Same top p:\s+([\d.]+)")}
            print(json.dumps(row), flush=True)
            f.write(json.dumps(row) + "\n")
        kl.subprocess.run = orig
