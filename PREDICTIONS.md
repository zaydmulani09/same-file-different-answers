# Predictions (written 2026-11-07, before any result of the KL grid or the eval queue was read)

Derived from reading llama.cpp release b11696 (commit ceefd16da) source, CPU backend, on an
i7-1355U (AVX2 + FMA + F16C + AVX-VNNI, no AVX-512). "Identical" means bitwise-equal logits
(KLD exactly 0, same-top-1 100%, and in the server runs equal float32 logprob strings).

Relevant code facts:
- F1. Every output element of a CPU matmul is one dot product computed by one thread with a
  fixed reduction order; threads only change which rows a thread owns
  (`ggml-cpu.c` chunking, `repack.cpp` `forward_mul_mat`).
- F2. Integer parts of quantized dot products are exact on every ISA; ISA differences enter only
  through float epilogues: AVX2 builds use `_mm256_fmadd_ps` (one rounding), AVX builds use
  `_mm256_add_ps(_mm256_mul_ps(..))` (two roundings), SSE4.2 and plain x64 use other loops
  (`arch/x86/quants.c`, `simd-mappings.h`).
- F3. Q4_0 and Q4_K weights are repacked to 8x8 interleaved layouts only when the CPU backend
  reports AVX2 (`repack.cpp` `ggml_repack_get_optimal_repack_type`). Q8_0, Q6_K and F16 are not
  repacked on x86.
- F4. In the repacked matmul, the src1 rows of a ubatch are processed with `gemm` in groups of 4
  and the remaining `nrows % 4` rows with `gemv` (`forward_mul_mat_one_chunk`). So a token's
  logits depend on its row index within its ubatch modulo 4.
- F5. `llamafile_sgemm` (tinyBLAS) is used only when n >= 2 (prompt processing). For Q8_0/Q4_0
  it further requires n >= 8 or n == 4; F16 requires F16C. Otherwise the per-row `vec_dot`
  path runs (`sgemm.cpp`). Attention's KQ product reads the F16 K cache, so it also switches
  path with the number of query rows.

Predictions, KL grid (alderlake reference, 8 threads, ub 512, fa off, 512-token chunks):
- P1. Threads 4 / 8 / 12: identical for all six models (F1).
- P2. haswell vs alderlake: identical for all models (F2: both have FMA; VNNI only changes the
  exact integer part).
- P3. ivybridge vs alderlake: differs for all models (F2 no FMA, F3 no repack).
- P4. sandybridge vs ivybridge: differs for all models (F5: F16 KQ attention loses tinyBLAS).
  sse42 differs from all AVX variants.
- P5. ub 1 and ub 7 differ from ub 512 for all models (F4, F5). ub 8 and ub 64 are uncertain:
  for repacked types every row of a ub-8/64 chunk takes `gemm`, but attention row counts change;
  no confident prediction.
- P6. repack-off: differs for Q4_0, Q4_K_M (F3); identical for Q8_0 and F16.
- P7. flash attention on: differs for all models (different attention kernel).

Predictions, eval queue (server, greedy):
- P8. ref-rerun: 100% identical to ref (a fixed configuration is deterministic).
- P9. cache-on: not identical to ref for some items (prefix reuse changes which prompt rows are
  computed in which ubatch).
- P10. server4-a vs server4-b (4 slots, 4 concurrent clients, prompt cache on): not identical,
  i.e. the same command run twice gives different outputs, for every model (decode batch size
  1-4 depends on request timing; F4/F5).
- P11. Any divergence produces benchmark flips; flips concentrate on low-margin MMLU items.
