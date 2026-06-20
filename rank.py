#!/usr/bin/env python3
"""
rank.py — Redrob "Intelligent Candidate Discovery" ranker.

Baseline milestone: load candidates -> score -> honeypot gate -> top 100 ->
write a validator-clean CSV. The score here is a THROWAWAY placeholder so the
pipeline runs end to end. Replace `score_candidate` with the real JD-grounded
model; that is the graded work and it stays yours.

Compute contract (hard): CPU-only, no network, < 5 min, < 16 GB.
Determinism (hard): same input -> byte-identical CSV. Everything time-based is
anchored to REF_DATE (data-derived), never to "today".

Usage:
    python rank.py --in candidates.jsonl --out team_xxx.csv
    python validate_submission.py team_xxx.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import date, datetime

TOP_N = 100
GATE_BUFFER = 50          # score top (100 + buffer), gate, then keep 100
SCORE_DECIMALS = 6        # round score; sort on the SAME rounded value


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def load_candidates(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    records = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_ref_date(records: list[dict]) -> date:
    """Deterministic 'now' = latest last_active_date in the pool."""
    latest = None
    for c in records:
        s = (c.get("redrob_signals") or {}).get("last_active_date")
        if not s:
            continue
        try:
            d = datetime.fromisoformat(s).date()
        except ValueError:
            continue
        if latest is None or d > latest:
            latest = d
    return latest or date(2026, 1, 1)


# --------------------------------------------------------------------------
# Scoring  <-- THIS is the part you build. Everything below is plumbing.
# --------------------------------------------------------------------------
def score_candidate(c: dict, ref_date: date) -> float:
    """
    Return a relevance score (higher = better fit for the Redrob JD).

    Build this out per the JD model. Suggested structure, in rough weight order:

      1. HARD NEGATIVES (floor the score):
         - services-only career (TCS/Infosys/Wipro/Accenture/Cognizant/
           Capgemini) across the WHOLE career_history, not just current_company
         - non-technical current_title + dense AI skills (your is_stuffer)
         - CV/speech/robotics primary with no NLP/IR
         - pure-research-only, no production
         - recent-LLM-only (<12mo LangChain/OpenAI) with no pre-LLM ML

      2. CORE FIT (heaviest positive weight):
         - retrieval / vector-search / ranking evidence IN career_history
           descriptions (production), not just skill tags
         - eval rigor (NDCG/MRR/MAP/A-B) in descriptions
         - role-relevance tier of titles (core ML/AI/Applied Sci/Recsys/Search/NLP
           > Data Scientist/Data Eng/SWE-with-ML > off-target)
         - ai_depth / trusted_ai (endorsed AND used), from your notebook

      3. EXPERIENCE BAND: soft peak at 6-8 yrs, fine across 5-9, flexible beyond
         if core signals are strong. Do not hard-cut.

      4. PRODUCT-COMPANY: reward applied-ML time at product companies; penalize
         services. (The bullseye: 4-5 yrs applied ML at product cos.)

      5. NICE-TO-HAVE BOOSTS (small): LoRA/QLoRA/PEFT, learning-to-rank,
         HR-tech/marketplace, distributed systems, OSS/papers (external validation).

      6. BEHAVIORAL (weak tiebreakers): down-weight inactive (days_since_active)
         + low recruiter_response_rate. -1 sentinels -> NEUTRAL, never 0.

    Keep it deterministic and cheap (no per-candidate model calls).
    """
    # ---- THROWAWAY PLACEHOLDER: proximity to the 7-yr midpoint of the band. ----
    # Produces varied output so the pipeline/sort/CSV are exercised on real
    # variation. Delete this and implement the model above.
    yoe = (c.get("profile") or {}).get("years_of_experience") or 0
    return -abs(yoe - 7.0)


# --------------------------------------------------------------------------
# Honeypot gate  (paste your notebook's honeypot_reasons here)
# --------------------------------------------------------------------------
def honeypot_reasons(c: dict, ref_date: date) -> list[str]:
    """
    Return reasons a profile is impossible; empty list = passes the gate.
    Drop in the version from your notebook (Section 6). Stubbed to pass-through
    so the baseline runs; wire the real one in as its own commit.
    """
    return []


# --------------------------------------------------------------------------
# Reasoning  (Stage 4 grades this by hand; build from real matched features)
# --------------------------------------------------------------------------
def build_reasoning(c: dict, ref_date: date) -> str:
    """
    One line of concrete, per-candidate evidence. Replace the placeholder with
    the actual signals that drove the score (role tier, yrs, the endorsed+used
    AI skills, the relevant career line, recency). Avoid an identical template
    across rows. NOTE: commas/newlines are fine here — the CSV writer quotes them.
    """
    p = c.get("profile") or {}
    return f"placeholder: {p.get('current_title', '?')}, {p.get('years_of_experience', '?')} yrs"


# --------------------------------------------------------------------------
# Pipeline + validator-safe CSV
# --------------------------------------------------------------------------
def rank(records: list[dict], ref_date: date) -> list[dict]:
    scored = []
    for c in records:
        cid = c["candidate_id"]
        raw = score_candidate(c, ref_date)
        scored.append((round(float(raw), SCORE_DECIMALS), cid, c))

    # Deterministic order: score DESC, then candidate_id ASC.
    # This single sort key satisfies BOTH validator rules at once
    # (non-increasing score, and ties broken by candidate_id ascending).
    scored.sort(key=lambda t: (-t[0], t[1]))

    # Gate the top (100 + buffer), drop honeypots, keep 100.
    pool = scored[: TOP_N + GATE_BUFFER]
    clean = [(s, cid, c) for (s, cid, c) in pool if not honeypot_reasons(c, ref_date)]
    final = clean[:TOP_N]

    if len(final) < TOP_N:
        raise SystemExit(
            f"Only {len(final)} clean candidates in top {TOP_N + GATE_BUFFER}; "
            f"raise GATE_BUFFER."
        )

    return [
        {"candidate_id": cid, "rank": i + 1, "score": s,
         "reasoning": build_reasoning(c, ref_date)}
        for i, (s, cid, c) in enumerate(final)
    ]


def write_csv(rows: list[dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            # score formatted with fixed decimals so the written value == sort value
            w.writerow([r["candidate_id"], r["rank"],
                        f"{r['score']:.{SCORE_DECIMALS}f}", r["reasoning"]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="candidates.jsonl")
    ap.add_argument("--out", dest="out", default="submission.csv")
    args = ap.parse_args()

    records = load_candidates(args.inp)
    ref_date = compute_ref_date(records)
    print(f"Loaded {len(records):,} candidates. REF_DATE = {ref_date}")

    rows = rank(records, ref_date)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows -> {args.out}")
    print("Now run: python validate_submission.py", args.out)


if __name__ == "__main__":
    main()
