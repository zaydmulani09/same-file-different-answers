# Same File, Different Answers

Code, data and paper for *Same File, Different Answers: Execution-Equivalent Settings Change
Benchmark Results of Quantized LLMs on CPUs* (Zayd Mulani, 2026). The PDF is
[`paper/main.pdf`](paper/main.pdf); the archived preprint is on Zenodo,
[doi:10.5281/zenodo.24055876](https://doi.org/10.5281/zenodo.24055876).

The question: if you keep the llama.cpp release, the GGUF file, the prompts and greedy decoding
fixed, and only change settings that are supposed to be performance options (the CPU backend
tier the runtime picks, micro-batch size, flash attention, repacking, prompt cache, parallel
server slots), how much do benchmark answers change?

Short answer, for Qwen3-0.6B Q4_K_M on 980 items:

| change (everything else fixed) | items bitwise identical | MMLU answers changed (of 600) | accuracy change |
|---|---|---|---|
| rerun of the same configuration | 100% | 0 | 0 |
| AVX tier instead of AVX2 tier | 0% | 41 | at most 1.3 points |
| micro-batch 7 instead of 512 | 0% | 49 | at most 1.3 points |
| micro-batch 64 instead of 512 | 89.9% | 4 | 0 |
| flash attention on | 0% | 51 | at most 3.7 points (3 of 80 GSM8K) |
| parallel server, run twice, compared with itself | 45.4% | 13 | at most 1 point |

Thread count never changed anything. A pre-registered dose-response test traces the size of the
effect to activation quantization: the same tier change leaves an F32 model at the measurement
floor, gives a mean KL divergence of about 1e-3 for Q8_0, and about 1e-2 for Q4_K_M.

## Layout

- `PREDICTIONS.md`: predictions written from the llama.cpp source before the runs, plus
  Addendum A (the amplification test), committed before the corresponding results existed.
- `harness/run.py`: one benchmark run through `llama-server`, one JSON line per item with token
  ids, float32 logprobs and the first-step top-20.
- `harness/jobs.py`: the run queue, with the reason for every omitted run.
- `harness/kl.py`, `harness/amp.py`, `harness/quantkl.py`: teacher-forced KL grids via
  `llama-perplexity --kl-divergence`.
- `harness/amp_sim.py`: numpy simulation of int8 activation-quantization amplification.
- `harness/analyze.py`, `harness/figs.py`: rebuild every table and figure from `results/`.
- `results/`: raw per-item outputs (`<model>__<config>.jsonl`), `kl.jsonl`, `amp.jsonl`,
  `quantkl.jsonl`, `summary.json`.
- `data/evalset.jsonl`: the frozen MMLU / GSM8K / TriviaQA items; `data/wiki_kl.txt`: WikiText
  text for the KL grids.
- `paper/`: LaTeX source (`tectonic main.tex`).

## Reproducing

Needs Windows x86-64 with AVX2, Python 3 with numpy, scipy, matplotlib, and
`llama-b11696-bin-win-cpu-x64.zip` from the llama.cpp releases page, unpacked into one folder
per tier, each holding exactly one `ggml-cpu-*.dll`. Paths to the tier folders and model files
are at the top of `harness/run.py`, `harness/kl.py` and `harness/jobs.py`. Model hashes are in
Table 1 of the paper. Then:

```
py -3 harness/jobs.py 0 1      # benchmark queue (about 40-55 min per run)
py -3 harness/kl.py 12         # teacher-forced grid
py -3 harness/amp.py 8         # amplification test
py -3 harness/ub_shape.py      # micro-batch vs sequence-length probe (post hoc)
py -3 harness/analyze.py
py -3 harness/figs.py
```

Other CPUs load a different set of tiers. All results come from one i7-1355U laptop; see the
paper's limitations section.

## License

Code: MIT. Paper, figures and result data: CC BY 4.0. The evaluation items come from MMLU
(MIT), GSM8K (MIT) and TriviaQA (Apache 2.0); WikiText is CC BY-SA.
