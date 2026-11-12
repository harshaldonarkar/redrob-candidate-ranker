#!/usr/bin/env python3
"""
sensitivity.py — how much does the top 100 move when the hand-set knobs move?

There are no relevance labels, so this does not measure accuracy. It measures
stability: each blend weight is changed by +/-20% (the weights are then rescaled
to sum to 1), and each gate is switched off, one at a time. For every variant it
reports how many of the baseline top 100 and top 10 are still there.

The per-candidate signals are computed once with rank.py's own functions and
re-blended per variant, so the whole sweep costs about one ranking run. The
baseline is asserted equal to rank.rank() before anything is reported.

Usage:
    python sensitivity.py --candidates candidates.jsonl
"""

from __future__ import annotations

import argparse

import rank as R

GATES = {
    "keyword-stuffer": R.STUFFER_GATE,
    "services-only": R.SERVICES_GATE,
    "CV/speech-primary": R.CV_PRIMARY_GATE,
    "recent-LLM-only": R.RECENT_LLM_GATE,
}


def extract(records: list[dict], ref_date) -> list[tuple]:
    """(candidate_id, signals, gate flags, is_honeypot) per candidate."""
    rows = []
    for c in records:
        profile = c.get("profile") or {}
        career_raw = R.score_career_history(c.get("career_history") or [], ref_date)
        ai_hits, trusted_ai, ai_depth = R.ai_features(c)
        signals = {
            "career": R.normalize_career(career_raw),
            "role": R.role_tier_score(c),
            "ai": R.normalize_ai(ai_depth),
            "exp": R.exp_band(profile.get("years_of_experience")),
            "behavioral": R.behavioral_score(c, ref_date),
        }
        flags = {
            "keyword-stuffer": R.is_stuffer(c, career_raw, ai_hits, trusted_ai),
            "services-only": R.is_services_only(c),
            "CV/speech-primary": R.is_cv_primary(c),
            "recent-LLM-only": R.is_recent_llm_only(c),
        }
        rows.append((c["candidate_id"], signals, flags, bool(R.honeypot_reasons(c, ref_date))))
    return rows


def top_ids(rows: list[tuple], weights: dict, gates: dict, honeypot_gate: bool = True) -> list[str]:
    """Same ordering and gating as rank.rank(), with the given weights and gates."""
    scored = []
    for cid, signals, flags, is_honeypot in rows:
        positive = (
            weights["career"] * signals["career"]
            + weights["role"] * signals["role"]
            + weights["ai"] * signals["ai"]
            + weights["exp"] * signals["exp"]
            + weights["behavioral"] * signals["behavioral"]
        )
        m = 1.0
        for name, fired in flags.items():
            if fired:
                m *= gates[name]
        scored.append((round(positive * m, R.SCORE_DECIMALS), cid, is_honeypot))
    scored.sort(key=lambda t: (-t[0], t[1]))
    pool = scored[: R.TOP_N + R.GATE_BUFFER]
    if honeypot_gate:
        pool = [t for t in pool if not t[2]]
    return [cid for _, cid, _ in pool[: R.TOP_N]]


def scaled_weights(name: str, factor: float) -> dict:
    w = dict(R.WEIGHTS)
    w[name] *= factor
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", "--candidates", dest="inp", default="candidates.jsonl")
    args = ap.parse_args()

    records = R.load_candidates(args.inp)
    ref_date = R.compute_ref_date(records)
    rows = extract(records, ref_date)

    base = top_ids(rows, R.WEIGHTS, GATES)
    assert base == [r["candidate_id"] for r in R.rank(records, ref_date)], \
        "re-blended baseline differs from rank.rank()"

    variants = []
    for name in R.WEIGHTS:
        for factor, label in ((0.8, "-20%"), (1.2, "+20%")):
            variants.append((f"{name} weight {label}", top_ids(rows, scaled_weights(name, factor), GATES)))
    for name in GATES:
        variants.append((f"{name} gate off", top_ids(rows, R.WEIGHTS, {**GATES, name: 1.0})))
    variants.append(("all four gates off", top_ids(rows, R.WEIGHTS, {g: 1.0 for g in GATES})))
    variants.append(("honeypot gate off", top_ids(rows, R.WEIGHTS, GATES, honeypot_gate=False)))

    base100, base10 = set(base), set(base[:10])
    print(f"Pool: {len(records):,} candidates. REF_DATE = {ref_date}\n")
    print("| Variant | Top 100 kept | Top 10 kept | Rank 1 unchanged |")
    print("|---|---|---|---|")
    for label, ids in variants:
        kept100 = len(base100 & set(ids))
        kept10 = len(base10 & set(ids[:10]))
        print(f"| {label} | {kept100}/100 | {kept10}/10 | {'yes' if ids[0] == base[0] else 'no'} |")


if __name__ == "__main__":
    main()
