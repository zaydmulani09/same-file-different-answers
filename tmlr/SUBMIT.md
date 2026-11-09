# TMLR submission sheet

Rebuild: `git clone --depth 1 https://github.com/JmlrOrg/tmlr-style-file tmlr/_style` (once), then
`py -3 tmlr/build.py`. Upload `tmlr/out/main.pdf` and `tmlr/out/supplementary.zip`.

Submit at https://openreview.net/group?id=TMLR (Author Console, "TMLR Submission").

| form field | answer |
|---|---|
| Title | Same File, Different Answers: Execution-Equivalent Settings Change Benchmark Results of Quantized LLMs on CPUs |
| Authors | yourself (OpenReview profile) |
| Abstract | the abstract in `paper/sec_abstract.tex`, with the last sentence changed to: "Code, raw per-item outputs and the pre-registration are in the supplementary material." |
| PDF | `tmlr/out/main.pdf` (anonymous, 11 pages main text + references + appendix) |
| Submission length | Regular submission (no more than 12 pages of main content) |
| Supplementary material | `tmlr/out/supplementary.zip` |
| Previous submission / changes | none (first submission) |
| Competing interests | none |
| Human subjects | none |

Notes
- Do not link the Zenodo DOI, SSRN page or GitHub repo anywhere in the submission. Posting
  preprints is allowed; linking them from the anonymous submission is not.
- If the form has a private comment to the action editor, it may say: "A non-anonymous preprint
  of this work exists on Zenodo and SSRN."
- After acceptance: switch to `\usepackage[accepted]{tmlr}`, restore the author block and the repo
  URL, and add the OpenReview link to Zenodo, SSRN and the README.
