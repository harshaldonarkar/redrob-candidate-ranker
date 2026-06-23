# Redrob Candidate Ranker

Ranks the top 100 candidates from `candidates.jsonl` for the **Senior AI Engineer —
Founding Team** job description (NLP/IR/retrieval, product-company, 5–9 yrs).

## Reproduce the submission

The ranking step uses the **Python 3 standard library only** — no dependencies, CPU-only,
no network.

```bash
# place the released candidates.jsonl in the repo root first (it is gitignored: 487 MB)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv          # format check
```

`--in` is accepted as an alias for `--candidates`. On the full 100K pool the run takes
**~26 s** and peaks at **~1.8 GB RAM** (well inside the 5 min / 16 GB budget).
Output is **deterministic** — the same input produces a byte-identical CSV.

> **Data note:** `candidates.jsonl` is not committed (it exceeds GitHub's 100 MB limit).
> Drop the released file into the repo root before running.

## Sandbox: self-contained Docker recipe

Satisfies the spec's §10.5 sandbox requirement without a hosted account. The image is
just Python + the scripts (the ranker has no third-party dependencies); it **builds and
runs unmodified**.

```bash
# 1. Build
docker build -t redrob-ranker .

# 2. Quick <=100-row sample run (the spec's sandbox check). Make a sample without
#    committing any data, mount the current dir, rank it:
head -100 candidates.jsonl > sample.jsonl
docker run --rm -v "$PWD:/data" redrob-ranker \
    --candidates /data/sample.jsonl --out /data/submission.csv

# 3. Full 100K pool (same image)
docker run --rm -v "$PWD:/data" redrob-ranker \
    --candidates /data/candidates.jsonl --out /data/submission.csv

# 4. Validate the output (same image, different entrypoint).
#    NOTE: validate_submission.py requires exactly 100 rows, so validate the FULL
#    run (step 3). A <=100-row sample legitimately yields fewer than 100 ranked rows.
docker run --rm -v "$PWD:/data" --entrypoint python redrob-ranker \
    validate_submission.py /data/submission.csv
```

The mounted `/data` directory must contain the input JSONL; the ranked CSV is written
back to it. Runs CPU-only, no network, well within the 5 min / 16 GB budget.

## Approach (unsupervised, hand-encoded)

There are no relevance labels in the data — the ground-truth tiers sit with the graders.
`rank.py` is therefore **not a trained model**; it is a deliberate encoding of the JD's
definition of fit, validated by inspecting the ranking. Each candidate gets a score:

```
score = ( 0.40·career + 0.22·role + 0.16·ai + 0.14·exp + 0.08·behavioral )
        × hard_negative_gates
```

**Positive signals** (normalized to [0,1], blended additively so each weight is tunable):

| Signal | Wt | What it reads |
|---|----|---|
| **career-history relevance** | 0.40 | real IR/recsys/ranking/NLP work in `career_history[].description`, scaled by build-vs-mention language + eval rigor, recency-decayed, half-life aggregated. Saturating `x/(x+13)` normalization. The JD's decisive, hard-to-game signal. |
| **role-relevance tier** | 0.22 | `current_title` → core / adjacent / off-target (CV/speech/robotics titles → off-target). |
| **AI skill depth** | 0.16 | proficiency × years-used over **endorsed-and-used** skills only (not keyword count). Saturating `x/(x+7)` — the gameable axis, so down-weighted + corroborated. |
| **experience band** | 0.14 | soft preference flat across 5–9 yrs, gentle falloff outside (not a hard cut). |
| **behavioral** | 0.08 | recency + recruiter response + open-to-work. `-1`/missing sentinels → neutral, never 0. |

**Hard-negative gates** (multiplicative, applied *outside* the additive blend so strong
positives can't average them away; they stack):

| Gate | × | Fires on |
|---|----|---|
| keyword-stuffer | 0.05 | non-technical title + dense AI keywords + no corroborated depth + no career evidence |
| services-only | 0.10 | every role at a services firm / "IT Services" (any product stint exempts) |
| CV/speech-primary | 0.05 | vision/speech terms present **and** no IR/NLP evidence anywhere |
| recent-LLM-only | 0.25 | technical title, only <12-mo LLM-era skill tags, no classic ML (JD hedges → softer) |

*pure-research-only is intentionally not gated: tightened to academic-lab markers it
matches 0 candidates in this job-platform pool; the loose version false-positived real
product engineers.*

## Pipeline

```
load → compute features → score → sort (score desc, candidate_id asc)
     → honeypot end-gate over the top ranks → top 100 → reasoning → CSV
```

The **honeypot gate** runs high-precision impossibility checks (experience vs. history,
career/education date integrity, expert-skills-with-zero-duration) on the top of the
ranking and drops any impossible profile before taking 100. Current run: **0 honeypots,
0 gated candidates in the top 100** (nearest gated candidate sits 0.61 below the cutoff).

The **reasoning column** is generated per-candidate from the matched features (role tier,
years, the quoted relevant career line, endorsed skills, honest concerns) — deterministic,
no LLM, every claim grounded in the candidate's record.

## Files

| File | What |
|---|---|
| `rank.py` | the ranker (stdlib only) |
| `validate_submission.py` | format validator (organizer-provided) |
| `candidate.ipynb` | EDA notebook (distributions, sentinel handling, honeypot design) |
| `candidate_schema.json` | candidate record schema (reference) |
| `requirements.txt` | ranker deps (none) |
| `requirements-eda.txt` | EDA-notebook deps (pandas/numpy/matplotlib) |
| `submission_metadata.yaml` | submission metadata mirroring the portal |

## Determinism & compute

Time-based math is anchored to a data-derived `REF_DATE` (max `last_active_date`), never
the wall clock. Sort ties break on `candidate_id`. Two runs produce byte-identical output.
Verified: ~26 s, ~1.8 GB, CPU-only, no network on the full 100K pool.
