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
import re
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
# Career-history relevance — the JD's decisive, hard-to-game signal
# --------------------------------------------------------------------------
# Reads what each role actually DID (title + description) rather than trusting
# title/skill tags. Distinct IR/recsys/ranking/NLP concepts, scaled by build-vs-
# mention language and eval rigor, with a span-localized vision guard that tanks
# only profiles whose every domain hit is hugging a CV/speech word. Per-role
# scores are recency- and services-weighted, then aggregated with a half-life
# discount so the single strongest relevant role dominates.
DOMAIN_CONCEPTS = {
    "search": re.compile(r"\b(search|retrieval|ir|information retrieval)\b"),
    "recsys": re.compile(r"\b(recommendation|recsys|recommender)\b"),
    "ranking": re.compile(r"\b(ranking|learning-to-rank|ltr)\b"),
    "embedding": re.compile(r"\b(embedding|embeddings|vector search|rag)\b"),
    "nlp": re.compile(r"\b(nlp)\b"),
}
ACTION_RE = re.compile(r"\b(built|designed|architected|shipped|deployed|owned|led|scaled|migrated|rebuilt|trained|implemented)\b")
SCALE_RE = re.compile(r"\b(production|users|qps|latency|throughput|rolled out|scale|millions|billion)\b")
MENTION_RE = re.compile(r"\b(familiar with|exposure to|assisted with|learning|interested in|course|courses|tutorial|poc|side project|experimented|playing with)\b")
EVAL_RE = re.compile(r"\b(ndcg|mrr|mean average precision|a/b test|ab test|offline-online|offline to online)\b")
VISION_RE = re.compile(r"\b(image|images|visual|vision|cv|object detection|segmentation|ocr|speech|audio|gan|diffusion|computer vision)\b")
WORD_RE = re.compile(r"\b[\w-]+\b")

SERVICES_FIRMS = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tech mahindra", "mindtree", "hcl"}


def get_window_text(text: str, start: int, end: int, window_size: int = 3) -> str:
    """±window_size words around a character span (for the localized vision guard)."""
    words_before = WORD_RE.findall(text[:start])
    words_after = WORD_RE.findall(text[end:])
    return " ".join(words_before[-window_size:] + words_after[:window_size])


def score_career_history(career_history: list[dict], ref_date: date) -> float:
    """Span-window concept aggregation across roles; higher = more real IR/NLP work."""
    entry_scores = []
    for entry in career_history:
        text = f"{entry.get('title', '')} {entry.get('description', '')}".lower()

        has_domain_hits = False
        distinct_domain_hits = 0
        all_hits_vision_qualified = True  # assume pure-CV until a clean IR hit proves otherwise
        for pattern in DOMAIN_CONCEPTS.values():
            matches = list(pattern.finditer(text))
            if matches:
                has_domain_hits = True
                distinct_domain_hits += 1
                for match in matches:
                    window_text = get_window_text(text, match.start(), match.end(), 3)
                    if not VISION_RE.search(window_text):
                        all_hits_vision_qualified = False
        if not has_domain_hits:
            continue

        # Trapdoor: if EVERY domain hit hugged a vision/speech word, tank it.
        cv_penalty = 0.1 if all_hits_vision_qualified else 1.0

        action_hits = len(ACTION_RE.findall(text))
        scale_hits = len(SCALE_RE.findall(text))
        mention_hits = len(MENTION_RE.findall(text))
        eval_hits = len(EVAL_RE.findall(text))

        if action_hits > 0 and scale_hits > 0:
            base_multiplier = 3.0
        elif action_hits > 0:
            base_multiplier = 1.5
        else:
            base_multiplier = 0.5
        if mention_hits > 0:
            base_multiplier *= max(0.1, 1.0 - (0.4 * mention_hits))
        eval_bonus = eval_hits * 2.0

        company = str(entry.get("company", "")).lower()
        industry = str(entry.get("industry", "")).lower()
        is_services = industry == "it services" or any(f in company for f in SERVICES_FIRMS)
        company_weight = 0.3 if is_services else 1.0

        end_date_str = entry.get("end_date")
        try:
            end_d = date.fromisoformat(end_date_str[:10]) if end_date_str else ref_date
        except (ValueError, TypeError):
            end_d = ref_date
        duration_yrs = max(entry.get("duration_months", 0) / 12.0, 0.1)
        years_ago = max((ref_date - end_d).days / 365.25, 0)
        recency_decay = max(0.6, 0.8 ** years_ago)

        time_weight = duration_yrs * recency_decay
        base_score = distinct_domain_hits * base_multiplier * cv_penalty
        job_score = (base_score * company_weight * time_weight) + (eval_bonus * company_weight * recency_decay * cv_penalty)
        entry_scores.append(job_score)

    if not entry_scores:
        return 0.0
    entry_scores.sort(reverse=True)
    return sum(score / (2 ** i) for i, score in enumerate(entry_scores))


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
    # Driven by career-history relevance (component 2 above). Role tier, AI depth,
    # experience band, product-vs-services, and behavioral tiebreakers layer on next.
    career_history = c.get("career_history") or []
    return score_career_history(career_history, ref_date)


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
