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

## Validation and limitations

There are no ground-truth relevance labels in the bundle, and the hidden score
(NDCG@10 / NDCG@50 / MAP / P@10) is computed by the organizers after close. So
ranking quality is **not measured** here. What follows is what was checked and what
was not.

**What was checked** (full 100K pool, `REF_DATE = 2026-05-27`):

- **Format:** `validate_submission.py` passes on the output.
- **Determinism:** repeated runs produce a byte-identical CSV (compared by SHA-256).
- **Honeypot gate:** the impossibility checks flag 66 profiles in the pool. One of
  them scores into the top 150 (rank 112 before gating) and is dropped; none are in
  the final top 100.
- **Hard-negative gates:** they fire on 22 (keyword-stuffer), 9,745 (services-only),
  54 (CV/speech-primary) and 262 (recent-LLM-only) candidates. No gated candidate is
  in the top 100: the best gated score is 0.117 against a rank-100 score of 0.731.
- **Inspection:** the top of the ranking and the candidates caught by each gate were
  read by hand in `candidate.ipynb` while the signals and thresholds were being set.

- **Unit tests:** `python -m unittest test_rank -v` runs 10 standard-library tests on
  synthetic candidates (keyword-stuffer gets the ×0.05 gate, services-only gets ×0.10,
  honeypot checks fire, `-1` sentinels stay neutral, two runs are byte-identical).

**Sensitivity of the top 100** (`python sensitivity.py`, full pool). Each weight is
changed by ±20% with the others rescaled to sum to 1, and each gate is switched off,
one at a time. The table counts how many of the baseline top 100 and top 10 survive.
This measures stability, not accuracy.

| Variant | Top 100 kept | Top 10 kept | Rank 1 unchanged |
|---|---|---|---|
| career weight −20% | 96/100 | 9/10 | yes |
| career weight +20% | 98/100 | 10/10 | yes |
| role weight −20% | 99/100 | 10/10 | yes |
| role weight +20% | 99/100 | 10/10 | yes |
| ai weight −20% | 97/100 | 9/10 | yes |
| ai weight +20% | 97/100 | 10/10 | yes |
| exp weight −20% | 99/100 | 9/10 | yes |
| exp weight +20% | 99/100 | 10/10 | yes |
| behavioral weight −20% | 100/100 | 10/10 | yes |
| behavioral weight +20% | 98/100 | 9/10 | yes |
| each hard-negative gate off (4 runs) | 100/100 | 10/10 | yes |
| all four gates off | 100/100 | 10/10 | yes |
| honeypot gate off | 100/100 | 10/10 | yes |

Two readings. The top 100 is stable to the weights: a ±20% change to any one weight
moves at most 4 of 100 candidates and never changes rank 1. And the gates are **not
what decides the top 100 on this pool**: switching any or all of them off leaves it
unchanged. The positive signals already keep those candidates out (a keyword-stuffer
has an off-target title and no career evidence; services roles count at 0.3× inside
the career signal). The gates act lower in the ranking and as a safeguard, so this
run gives no evidence that they improve the submitted list.

**What could not be measured:**

- NDCG, MAP and precision, or any other accuracy figure, since there are no labels.
- Honeypot recall. The organizers describe about 80 honeypots and the detector flags
  66, so some are probably missed. The checks were built for precision, and what
  matters for the submission is that none reach the top 100.
- False-positive rates of the gates beyond manual spot checks. The services-only gate
  is the coarsest: it covers 9.7% of the pool on company name and industry alone.

**Limitations:**

- **Hand-tuned weights.** The five weights, the gate multipliers and the saturation
  constants were set by judgment and inspection, not fitted. The sensitivity check
  above covers the weights and gate on/off only, not the saturation constants, the
  gate thresholds or the term lists.
- **Keyword and regex matching.** Career relevance comes from fixed term lists. Work
  described in other words is under-credited, and a description that uses the right
  terms without the substance is over-credited. There are no embeddings or semantic
  matching.
- **English only.** The patterns assume English titles and descriptions.
- **No learned component.** Nothing is trained, so the ranker cannot pick up signals
  that were not written into it by hand.
- **Tuned to one job description.** The term lists, title tiers and the services-firm
  list are specific to this role and this candidate pool.

## Files

| File | What |
|---|---|
| `rank.py` | the ranker (stdlib only) |
| `sensitivity.py` | weight ±20% and gate-off sweep over the top 100 (stdlib only) |
| `test_rank.py` | unit tests on synthetic candidates (stdlib `unittest`) |
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
