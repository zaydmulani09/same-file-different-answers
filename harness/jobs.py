"""The frozen run plan. `py -3 jobs.py <lane> <nlanes>` runs jobs[lane::nlanes] in order.
Reference config R = alderlake backend, 8 threads, ubatch 512, flash-attn off, repack on,
1 slot, sequential client, prompt cache off. Each other job changes exactly one factor.
isa-haswell, t4 and t12 were dropped from the eval queue after the KL probes showed them
bitwise identical to R; isa-ivybridge and isa-sse42 were dropped because the KL grid shows
them numerically equivalent to isa-sandybridge for this model and the CPU budget ran short.
The teacher-forced grid (kl.py) still covers all of them."""
import subprocess, sys
from pathlib import Path

M = Path(r"C:\Users\jymul\dev\swmodels")
MODELS = {
    "q06-q4km": M / "Qwen3-0.6B-Q4_K_M.gguf",
    "q06-q40": M / "Qwen3-0.6B-Q4_0.gguf",
    "q06-q80": M / "Qwen3-0.6B-Q8_0.gguf",
    "q06-f16": M / "Qwen3-0.6B-F16.gguf",
    "g3-1b-q40": M / "gemma-3-1b-it-qat-Q4_0.gguf",
    "q35-08-q40": M / "Qwen3.5-0.8B-Q4_0.gguf",
}
SERVER4 = ["--np", "4", "--conc", "4", "--cache", "1"]  # how a parallel eval client usually runs

FULL = [  # one-factor-at-a-time sweep, primary model
    ("ref", []), ("ref-rerun", []),
    ("isa-sandybridge", ["--variant", "sandybridge"]),
    ("ub7", ["--ub", "7"]), ("ub64", ["--ub", "64"]),
    ("fa-on", ["--fa", "on"]),  # repack-off: KL grid only (CPU budget)
    ("cache-on", ["--cache", "1"]),
    ("server4-a", SERVER4), ("server4-b", SERVER4),
]
SHORT = [("ref", []), ("isa-sandybridge", ["--variant", "sandybridge"]), ("server4-a", SERVER4)]

FIRST = {"ref", "ref-rerun", "isa-sandybridge", "server4-a", "server4-b", "cache-on", "ub7"}
JOBS = [("q06-q4km", n, a) for n, a in FULL if n in FIRST]  # priority order: most informative first
JOBS += [("q06-q40", n, a) for n, a in SHORT]
JOBS += [("q06-q80", "ref", []), ("q06-q80", "server4-a", SERVER4)]  # q80 ISA: KL grid only (CPU budget)
# CPU budget: F16 gets the FMA-only tier change (ivybridge: AVX+F16C, the tier used in amp.py)
# as a low-amplification control; the two other architectures get ref + server4 only.
JOBS += [("q06-f16", "ref", []), ("q06-f16", "isa-ivybridge", ["--variant", "ivybridge"]), ("q06-f16", "server4-a", SERVER4)]
for m in ("g3-1b-q40", "q35-08-q40"):
    JOBS += [(m, "ref", []), (m, "server4-a", SERVER4)]
JOBS += [("q06-q4km", n, a) for n, a in FULL if n not in FIRST]

if __name__ == "__main__":
    lane, nl = int(sys.argv[1]), int(sys.argv[2])
    here = Path(__file__).parent
    for m, n, a in JOBS[lane::nl]:
        cmd = [sys.executable, str(here / "run.py"), "--model", str(MODELS[m]), "--name", f"{m}__{n}",
               "--port", str(8100 + lane), "--threads", "8",
               "--mmlu", "600", "--tqa", "300", "--gsm", "80", *a]
        print(">>", m, n, flush=True)
        subprocess.run(cmd)
