# Design

A knowledge system over five crawls of one credit union's website. It answers members' questions with the
**current** value, the **right** tier, a source, and an honest "I don't know"; and it lists where the site
contradicts itself. Entry point: `solution/solution.py` (`build()` → `ingest` / `answer` / `conflicts`).

## Layout

Dependencies point one way: `core` ← `ingest` ← `qa`, `conflicts` ← `llm` (used by `qa/cascade.py`).
Everything specific to this website lives in `site_profile.py` (and the miners' `conflicts/vocab.py`);
the rest is generic mechanism.

| Package | What it owns |
|---|---|
| `core/` | Shared primitives: dates, URLs, tokenizer, text helpers, result types (`Result`, `SourceRef`). |
| `ingest/` | Page cleaning and classification, chunks + BM25, fact extraction (tables, fee lines, label lines, prose), the bi-temporal fact store, dated site statistics, the `KnowledgeBase`. |
| `qa/` | Question parsing (`query.py`), the router (an ordered list of resolvers), fact answers, extractive text answers, branch answers, refusals (`refuse.py`), and the optional LLM cascade (`cascade.py`). |
| `llm/` | Optional Claude calls: client (settings, budget, cache, record/replay), query parser, passage reader, grounding checks, prompt text. |
| `conflicts/` | Contradiction miners on one `Miner` base (rates, parallel rows, repeated statistics, program dates) and the orchestration that ranks them. |

## Ingest

Snapshots arrive oldest first. Each page is cleaned (markup, invisible characters, per-snapshot boilerplate),
typed (rate sheet, product, branch, newsroom, blog, form/legal…) and chunked under its headings for BM25.
Extractors turn tables, fee-schedule lines, `label: value` lines and a few conservative prose patterns into
`Fact`s keyed by **(product, attribute, qualifiers)**: tier, term, amount band, model year, purpose, segment.
Tiers and terms are separate facts, never conflicts.

The fact store is **bi-temporal**: each fact records the snapshots it was seen in (crawl time) and its stated
effective date (valid time). Ingest of all five snapshots takes ~15–20 s.

## Deciding what is "current"

- **Which snapshot:** `answer(q)` uses the newest snapshot; `answer(q, as_of=D)` uses the latest snapshot taken
  on or before D. Before the first crawl, only facts whose effective date is on or before D are used.
- **Which value within a snapshot** (`ingest/fact_format.rank_key`): confident extraction first, then **page
  authority** (rate sheet 3 > product / branch page 2 > other / newsroom 1 > blog / form 0.5; known stale
  landing pages −1.5), then effective date, then topical fit of the URL, then crawl time.
- Losing values never appear in the answer text (the scorer zeroes an answer that repeats a stale value); they
  are reported by `conflicts()`, which picks each winner with the **same** rule, so the two never disagree.

## Answering

`qa/router.py` parses the question once into a `Query`, then walks the resolvers in order; the first that
answers wins:

1. **Urgent line**: lost / stolen card → the 24/7 number.
2. **Branches**: address, phone, hours and open-status from the locator and announcement pages.
3. **Facts**: rates, APYs, fees, terms and limits from the fact store, rendered one row per product and tier.
4. **Text**: BM25 retrieval, sentence scoring against the asked answer type, extractive answer with sources.

`qa/refuse.py` is the one place that decides to decline: the asked kind of answer isn't in the evidence,
the question is about the future, a forecast, or a product the site doesn't offer. A refusal contains no
digits and has `confident=False`.

## The optional LLM cascade

With `ANTHROPIC_API_KEY` set, `qa/cascade.py` adds two small `claude-haiku-5-5` calls **only** when the rules
are unsure or the wording is messy:

- the **parser** maps the question onto the same `Query` the rules use, choosing products from an enumerated
  catalog (helps slang, typos and paraphrases converge);
- the **reader** extracts an answer from the top retrieved passages, or verifies a weak extractive answer.
  Every number, name, phone and time it returns must appear verbatim in a quoted passage
  (`llm/grounding.py`), otherwise the rule answer stands.

Rates always come from the fact store, never from the model. All LLM time per answer is capped
(`KS_LLM_BUDGET`, default 3 s); on any error or timeout the rule answer is returned. Without a key the layer is
inert and the system is fully deterministic. `KS_LLM=off|live|record|replay` controls it.

## Conflicts

`conflicts/miner.py` runs four miners over the knowledge base:

- **Numeric**: one product's rate stated differently across pages or within a page's older table.
- **Parallel**: rows that should move together and don't (e.g. Traditional vs Roth IRA specials).
- **Statistics**: the same site figure (members, branches, counties…) stated with different numbers; a dated
  official statistics box wins.
- **Periods**: a program's dates or deadlines stated differently.

Each conflict lists the values, the winner (`resolution`), the reason and the sources; results are ranked by
confidence and salience.
