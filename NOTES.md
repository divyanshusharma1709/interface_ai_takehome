# NOTES

**Run.** From the repo root: `pip install -r requirements.txt` (optional; only the `anthropic` SDK), then
`python harness/run_eval.py --submission solution --eval eval/public_eval.jsonl`.
The submission is `solution/` (`build()` in `solution/solution.py`); the design is in `docs/DESIGN.md`.

**API key (optional).** The core system is standard-library Python and fully deterministic. With an
`ANTHROPIC_API_KEY` (in the environment, or a `.env` file in the repo root or `solution/`) it also uses a small
Claude model (`claude-haiku-5-5`) as a fallback for messy or uncertain questions: a parser that maps the question
onto the same structured query the rules use, and a reader that extracts an answer from the top retrieved
passages. Every number, name, phone and time the reader returns must appear verbatim in a quoted passage, or
the rule answer is kept; rates always come from the fact store, never from the model. Without the SDK or a key
the LLM layer switches itself off. `KS_LLM=off` forces rules only; `KS_LLM_BUDGET` (default 3.0 s) caps the LLM
time per answer.

**How "current" is decided.** Snapshots are ingested oldest first into a bi-temporal fact store: each fact
(product, attribute, tier/term qualifiers) records the snapshots and effective dates in which it held.
`answer(q)` uses the newest snapshot; `answer(q, as_of=D)` uses the latest snapshot taken on or before D. Within
a snapshot one value wins per fact by source authority (rate sheet > product page > other pages > blog/newsroom;
known stale landing pages are demoted), then by effective date. Losing values never appear in the answer text;
`conflicts()` reports them and picks each winner with the same rule. Tiers and terms are separate facts, not
conflicts. When nothing grounded answers the question, the system says it doesn't know, with no numbers.

**Performance.** Ingest of the 5 snapshots takes about 15–20 s. Rule-only answers take well under 0.5 s; with
the LLM, most questions skip the API and the rest add about 1–2.5 s.
