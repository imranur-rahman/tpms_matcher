# Author retrieval is the main ground truth of the benchmark

No venue publishes its real reviewer assignments, so the benchmark needs a public proxy for "the right PC member". We take accepted papers from ICSE, FSE, ASE, and ISSTA (2022–2025) and treat the authors who sit on the PC of the same venue edition as the correct PC members. This proxy needs no manual work and tests the full pipeline (PC parsing, author identity, profiles, scorer). It measures topical closeness, not reviewer choice: the tool would remove these authors as conflicts of interest, so the benchmark runs with the conflict check off. The Stelmakh et al. (2023) gold-standard dataset is a second, scorer-only check, and manual expertise ratings are left for a later paper.

## Considered Options

- **Citation overlap** (PC members whose papers the submission cites): rejected as the main ground truth, because a citation is a weak signal of expertise.
- **Manual expertise ratings**: the most valid option, but too slow for tuning; kept for a later paper.
- **Real assignments or bids**: not public.

## Consequences

- Each profile keeps only the papers from before the submission deadline, and the submission itself (preprint or published version) is removed. Otherwise the authors find their own paper and the numbers are inflated.
- Parameters are tuned on the 2022–2023 editions and reported on the 2024–2025 editions.
