"""Back-of-envelope for H-amp: perturb float32 activations by one ulp-scale relative error and
count how many Q8_0 codes change, and how big the resulting error is relative to the input
perturbation. Gaussian-with-outliers activations; seed fixed."""
import numpy as np
rng = np.random.default_rng(0)
n, B = 1 << 20, 32
x = rng.standard_normal(n).astype(np.float32)
x[rng.random(n) < 0.002] *= 20                           # a few outlier channels, as in LLM activations
eps = np.float32(2.0 ** -23)                              # ~1 ulp relative perturbation
xp = x * (1 + eps * rng.choice([-1, 1], n).astype(np.float32))

def q8_0(v):
    v = v.reshape(-1, B)
    d = np.abs(v).max(1, keepdims=True) / 127
    q = np.round(v / d)
    return (q * d).ravel(), q.ravel()

dx, qa = q8_0(x)
dp, qb = q8_0(xp)
rel_in = np.linalg.norm(xp - x) / np.linalg.norm(x)
rel_out = np.linalg.norm(dp - dx) / np.linalg.norm(dx)
print(f"codes changed: {np.mean(qa != qb):.2e} of activations")
print(f"relative perturbation in: {rel_in:.2e}  after Q8_0: {rel_out:.2e}  amplification: {rel_out/rel_in:.0f}x")
