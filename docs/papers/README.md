# Paper notes

Structured extraction notes for the papers this project builds on — enough
detail to implement from without re-reading PDFs. One file per paper.

| File | Paper | Role |
|------|-------|------|
| layden2023_qemcmc.md | Nature 619, 282 (2023) | THE algorithm — implementation spec |
| ferguson2025_coarse_graining.md | PRR 7, 013231 (2025) | scaling past qubit count |
| (see docs/BACKGROUND.md) | remaining bibliography with roles | context/differentiation |

PDFs are not committed (copyright + repo bloat). `./get_papers.sh` downloads
the open-access ones into docs/papers/pdf/ (gitignored). references.bib holds
citation entries for the eventual paper.

When you read a full PDF and learn something implementation-relevant, add it
to the paper's notes file — the notes are the project's working memory.
