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
# Blend architecture: additive positives, multiplicative hard-negative gates
# --------------------------------------------------------------------------
# Positive signals are normalized to [0,1] and blended with explicit weights
# (transparent + tunable, career highest but not dominant-by-construction).
# Hard negatives apply as a MULTIPLICATIVE gate so they can't be averaged away
# by strong positives.
CAREER_K = 13.0  # saturating constant ~ the "clearly relevant" single-builder threshold

# Weights are set once with intent and sum to 1. All five positive signals are
# fully wired into `positive` in score_candidate; hard negatives apply separately
# as a multiplicative gate (see hard_negative_multiplier).
WEIGHTS = {"career": 0.40, "role": 0.22, "ai": 0.16, "exp": 0.14, "behavioral": 0.08}


def normalize_career(raw: float, k: float = CAREER_K) -> float:
    """Saturating transform career/(career+k): 0->0, 16->0.55, 26->0.67, 40->0.76.
    Compresses the long tail of star builders while keeping spread in the 0-26 band
    where most ranking decisions sit. Chosen over min-max, which collapses the
    heavily-skewed middle to ~0."""
    return raw / (raw + k) if raw > 0 else 0.0


def exp_band(yoe) -> float:
    """Soft preference: flat 1.0 across the JD's 5-9 years, gentle falloff outside.
    Not a hard cut — strong core signals can still carry someone beyond the band."""
    if not isinstance(yoe, (int, float)):
        return 0.5
    if 5 <= yoe <= 9:
        return 1.0
    if yoe < 5:
        return max(0.0, 1.0 - (5 - yoe) * 0.18)   # 3.1y -> 0.66
    return max(0.0, 1.0 - (yoe - 9) * 0.12)        # 13y  -> 0.52


# AI skill depth — corroborated (endorsed AND used) and saturating-normalized.
# This is the gameable axis (the skill-stuffing the JD planted traps around), so it
# is down-weighted vs career and fed the trusted (not raw) version.
AI_KEYWORDS = (
    "machine learning", "deep learning", "nlp", "natural language", "embeddings",
    "retrieval", "rag", "vector search", "vector database", "information retrieval",
    "ranking", "recommender", "recommendation", "transformers", "llm", "fine-tuning",
    "lora", "pytorch", "tensorflow", "sentence-transformers", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch", "bm25",
    "learning to rank", "xgboost", "semantic search", "bert", "gpt",
)
PROF_W = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}
AI_DEPTH_K = 7.0  # half-saturation at the ~p95 "genuinely deep" knee of the real distribution


def is_ai_skill(name) -> bool:
    n = (name or "").lower()
    return any(kw in n for kw in AI_KEYWORDS)


def ai_features(c: dict) -> tuple[int, int, float]:
    """Return (ai_hits, trusted_ai, trusted_depth). Depth is weighted over skills
    that are BOTH endorsed (>=1) and used (>=6 months), by proficiency x years (capped)."""
    skills = c.get("skills") or []
    ai = [s for s in skills if is_ai_skill(s.get("name"))]
    trusted = [s for s in ai if (s.get("duration_months") or 0) >= 6 and (s.get("endorsements") or 0) >= 1]
    depth = sum(
        PROF_W.get(s.get("proficiency"), 0) * min((s.get("duration_months") or 0) / 12.0, 5.0)
        for s in trusted
    )
    return len(ai), len(trusted), depth


def normalize_ai(depth: float, k: float = AI_DEPTH_K) -> float:
    """Saturating, same family as career but K=7 read off ai_depth's own knee."""
    return depth / (depth + k) if depth > 0 else 0.0


def behavioral_score(c: dict, ref_date: date) -> float:
    """Weak availability tiebreaker in [0,1]: recency + recruiter response + open-to-work.
    -1 / missing sentinels map to NEUTRAL (pool median), never to 0."""
    s = c.get("redrob_signals") or {}
    la = s.get("last_active_date")
    try:
        days = (ref_date - date.fromisoformat(la[:10])).days if la else 365
    except (ValueError, TypeError):
        days = 365
    recency = max(0.0, 1.0 - days / 365.0)
    rr = s.get("recruiter_response_rate")
    rr = rr if isinstance(rr, (int, float)) and rr >= 0 else 0.44  # neutral default
    otw = 1.0 if s.get("open_to_work_flag") else 0.0
    return 0.5 * recency + 0.3 * rr + 0.2 * otw


# Role-relevance tier from current_title. Not a tech/non-tech binary: an NLP/IR
# role makes Frontend/QA/Cloud/Mobile engineers off-target, and CV/speech/robotics
# titles off-target too (JD explicitly does not want those without NLP/IR).
OFFTARGET_PRIMARY = re.compile(r"\b(computer vision|cv engineer|vision engineer|speech|robotics|\bimage\b)\b", re.I)
CORE_TITLE = re.compile(
    r"\b(ai engineer|ml engineer|machine learning|applied scientist|applied ml|"
    r"research (scientist|engineer)|ai research|recommendation|recommender|recsys|"
    r"search engineer|information retrieval|nlp|ranking engineer|relevance engineer)\b",
    re.I,
)
ADJACENT_TITLE = re.compile(
    r"\b(data scientist|data engineer|analytics engineer|data analyst|"
    r"software engineer|backend engineer|full[- ]?stack|platform engineer)\b",
    re.I,
)
ROLE_TIER_SCORE = {"core": 1.0, "adjacent": 0.55, "off_target": 0.10}


def role_tier(title: str | None) -> str:
    title = title or ""
    if OFFTARGET_PRIMARY.search(title):   # CV/speech/robotics-primary -> off-target
        return "off_target"
    if CORE_TITLE.search(title):
        return "core"
    if ADJACENT_TITLE.search(title):
        return "adjacent"
    return "off_target"


def role_tier_score(c: dict) -> float:
    return ROLE_TIER_SCORE[role_tier((c.get("profile") or {}).get("current_title"))]


# Broad "any technical title" test (engineer/dev/scientist/...). A keyword stuffer
# is someone whose title is NOT even broadly technical (Marketing/Sales/HR/Accountant)
# yet whose skills are AI-dense — distinct from a real engineer in the wrong specialty.
BROAD_TECH = re.compile(
    r"engineer|scientist|\bml\b|machine learning|\bai\b|data scien|applied scien|"
    r"research|developer|architect|\bnlp\b",
    re.I,
)
STUFFER_GATE = 0.05    # confirmed-precise detector -> bury, but floor (not 0) as insurance
SERVICES_GATE = 0.10   # JD hard-no, but coarser/high-volume detector -> slightly higher floor
CV_PRIMARY_GATE = 0.05 # precise (no IR/NLP anywhere) -> bury like the stuffer
RECENT_LLM_GATE = 0.25 # JD hedges ("probably not, unless pre-LLM ML") -> softer penalty

# recent-LLM-only: AI experience is only recent LangChain/OpenAI-style work with no
# pre-LLM ML foundation. LLM-era skill tags vs classic-ML skill tags.
LLM_SKILL_RE = re.compile(r"langchain|openai|\bllm\b|llms|\bgpt\b|prompt eng|\brag\b|llamaindex", re.I)
CLASSIC_ML_RE = re.compile(
    r"pytorch|tensorflow|scikit|xgboost|regression|random forest|\bsvm\b|clustering|"
    r"recommender|recommendation|ranking|embedding|\bnlp\b|computer vision|"
    r"machine learning|deep learning|information retrieval|learning to rank",
    re.I,
)

# CV/speech/robotics surface terms, and any IR/NLP evidence. Used together: a
# CV-primary candidate is one with vision/speech words and NO retrieval/NLP signal
# anywhere in their history — the JD downweights "CV/speech/robotics without
# significant NLP/IR exposure", not anyone who has ever touched an image.
CV_PRIMARY_RE = re.compile(
    r"computer vision|\bcv\b|\bvision\b|speech|robotics|object detection|segmentation|\bocr\b|\bimage\b",
    re.I,
)
IR_ANY_RE = re.compile(
    r"search|retrieval|recommend|recsys|ranking|learning-to-rank|embedding|vector|"
    r"\bnlp\b|\brag\b|semantic|information retrieval",
    re.I,
)


def is_stuffer(c: dict, career_raw: float, ai_hits: int, trusted_ai: int) -> bool:
    """Non-technical title + dense AI keywords + ~no corroborated depth + no career
    build evidence. The JD: skills don't redeem a Marketing Manager 'no matter how
    perfect their skill list looks.' High precision on inspection (22 hits, all clean)."""
    title = (c.get("profile") or {}).get("current_title") or ""
    non_technical = not BROAD_TECH.search(title)
    return non_technical and ai_hits >= 5 and trusted_ai <= 1 and career_raw < 2.0


def is_services_only(c: dict) -> bool:
    """EVERY role is a services firm or 'IT Services' industry. JD: consulting-only
    careers are 'not a fit' regardless of skills — but 'prior product-company
    experience is fine', so requiring ALL entries to be services is the conservative
    reading (any single product stint exempts the candidate). ~9.7% of the pool."""
    hist = c.get("career_history") or []
    if not hist:
        return False
    for h in hist:
        company = (h.get("company") or "").lower()
        industry = (h.get("industry") or "").lower()
        if not (industry == "it services" or any(f in company for f in SERVICES_FIRMS)):
            return False
    return True


def is_cv_primary(c: dict) -> bool:
    """CV/speech/robotics terms present (title or any description) AND no IR/NLP
    evidence anywhere in the career history. The 'no IR anywhere' clause keeps this
    precise (~54 hits): a genuine IR person who also did some vision is NOT caught."""
    title = (c.get("profile") or {}).get("current_title") or ""
    blob = " ".join(
        f"{h.get('title', '')} {h.get('description', '')}"
        for h in (c.get("career_history") or [])
    )
    has_cv = bool(CV_PRIMARY_RE.search(title) or CV_PRIMARY_RE.search(blob))
    has_ir = bool(IR_ANY_RE.search(blob))
    return has_cv and not has_ir


def is_recent_llm_only(c: dict) -> bool:
    """Technical title whose AI skills are ONLY recent (<12mo) LLM-era tags, with no
    classic-ML skill at all. JD: '<12mo LangChain/OpenAI ... unless substantial
    pre-LLM ML' -> the absence of any classic-ML skill is that 'unless' check.
    Non-technical titles are excluded (those are stuffer territory)."""
    title = (c.get("profile") or {}).get("current_title") or ""
    if not BROAD_TECH.search(title):
        return False
    skills = c.get("skills") or []
    llm = [s for s in skills if LLM_SKILL_RE.search(s.get("name") or "")]
    if not llm:
        return False
    has_classic = any(
        CLASSIC_ML_RE.search(s.get("name") or "") and not LLM_SKILL_RE.search(s.get("name") or "")
        for s in skills
    )
    all_llm_short = all((s.get("duration_months") or 0) < 12 for s in llm)
    return all_llm_short and not has_classic


def hard_negative_multiplier(c: dict, career_raw: float, ai_hits: int, trusted_ai: int) -> float:
    """Multiplicative gate for HARD NEGATIVES — applied OUTSIDE the additive blend so
    they can't be averaged away by strong positives. Gates stack multiplicatively.
    Live: keyword-stuffer, whole-career services-only, CV/speech-primary-with-no-IR,
    recent-LLM-only. Omitted: pure-research-only — after tightening to academic-lab
    markers (vs the loose version that false-positived product 'Research Engineers'),
    it matched 0 candidates; this job-platform pool has no academic-only population."""
    m = 1.0
    if is_stuffer(c, career_raw, ai_hits, trusted_ai):
        m *= STUFFER_GATE
    if is_services_only(c):
        m *= SERVICES_GATE
    if is_cv_primary(c):
        m *= CV_PRIMARY_GATE
    if is_recent_llm_only(c):
        m *= RECENT_LLM_GATE
    return m


# --------------------------------------------------------------------------
# Scoring  <-- THIS is the part you build. Everything below is plumbing.
# --------------------------------------------------------------------------
def score_candidate(c: dict, ref_date: date) -> float:
    """
    Return a relevance score (higher = better fit for the Redrob JD).

    Build this out per the JD model. Suggested structure, in rough weight order:

      1. HARD NEGATIVES — multiplicative gate in hard_negative_multiplier():
         [GATED]   non-technical current_title + dense AI skills (is_stuffer, x0.05)
         [GATED]   services-only career across the WHOLE career_history (x0.10)
         [GATED]   CV/speech/robotics primary with no NLP/IR anywhere (x0.05)
         [GATED]   recent-LLM-only (<12mo LLM tags, no classic ML) (x0.25, JD hedges)
         [OMITTED] pure-research-only: 0 candidates in this pool once tightened to
                   academic-lab markers; loose version false-positived product roles

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
    profile = c.get("profile") or {}

    career_raw = score_career_history(c.get("career_history") or [], ref_date)
    career_norm = normalize_career(career_raw)
    ai_hits, trusted_ai, ai_depth = ai_features(c)
    ai_norm = normalize_ai(ai_depth)
    exp = exp_band(profile.get("years_of_experience"))

    # Additive positive blend (all five signals live).
    positive = (
        WEIGHTS["career"] * career_norm
        + WEIGHTS["role"] * role_tier_score(c)
        + WEIGHTS["ai"] * ai_norm
        + WEIGHTS["exp"] * exp
        + WEIGHTS["behavioral"] * behavioral_score(c, ref_date)
    )

    return positive * hard_negative_multiplier(c, career_raw, ai_hits, trusted_ai)


# --------------------------------------------------------------------------
# Honeypot gate  (paste your notebook's honeypot_reasons here)
# --------------------------------------------------------------------------
def honeypot_reasons(c: dict, ref_date: date) -> list[str]:
    """High-precision impossibility checks (notebook §6). Empty list = passes the
    gate. These are 'this cannot be true', not weak-candidate signals."""
    reasons = []
    p = c.get("profile") or {}
    hist = c.get("career_history") or []
    skills = c.get("skills") or []
    edu = c.get("education") or []
    yoe = p.get("years_of_experience") or 0
    total_months = sum((h.get("duration_months") or 0) for h in hist)
    max_tenure = max((h.get("duration_months") or 0) for h in hist) if hist else 0

    if yoe - total_months / 12.0 > 3:
        reasons.append("yoe_exceeds_summed_history")
    if yoe and max_tenure / 12.0 > yoe + 1:
        reasons.append("single_tenure_exceeds_yoe")

    starts = []
    for h in hist:
        sd, ed = h.get("start_date"), h.get("end_date")
        try:
            d0 = date.fromisoformat(sd[:10]) if sd else None
            d1 = date.fromisoformat(ed[:10]) if ed else None
        except (ValueError, TypeError):
            d0 = d1 = None
        if d0:
            starts.append(d0)
            if d0 > ref_date:
                reasons.append("job_starts_in_future")
        if d0 and d1:
            if d1 < d0:
                reasons.append("job_end_before_start")
            if abs((d1 - d0).days / 30.44 - (h.get("duration_months") or 0)) > 24:
                reasons.append("duration_contradicts_dates")
    if starts and yoe - (ref_date - min(starts)).days / 365.25 > 4:
        reasons.append("yoe_predates_first_job")

    expert_zero = sum(1 for s in skills if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0)
    if expert_zero >= 3:
        reasons.append("expert_skills_zero_duration")

    for e in edu:
        sy, ey = e.get("start_year"), e.get("end_year")
        if sy and ey and ey < sy:
            reasons.append("education_end_before_start")
        if ey and ey > ref_date.year + 1:
            reasons.append("education_graduates_in_future")

    return sorted(set(reasons))


# --------------------------------------------------------------------------
# Reasoning  (Stage 4 grades this by hand; build from real matched features)
# --------------------------------------------------------------------------
def _top_trusted_skills(c: dict, n: int = 3) -> list[str]:
    skills = c.get("skills") or []
    trusted = [
        s for s in skills
        if is_ai_skill(s.get("name")) and (s.get("duration_months") or 0) >= 6 and (s.get("endorsements") or 0) >= 1
    ]
    trusted.sort(key=lambda s: s.get("endorsements") or 0, reverse=True)
    return [s.get("name") for s in trusted[:n]]


def _best_career_line(c: dict) -> str:
    """The single most domain-relevant role description (for evidence in the reasoning)."""
    best_line, best_hits = "", 0
    for h in c.get("career_history") or []:
        text = f"{h.get('title', '')} {h.get('description', '')}".lower()
        hits = sum(1 for pat in DOMAIN_CONCEPTS.values() if pat.search(text))
        if hits > best_hits:
            best_hits, best_line = hits, (h.get("description") or "").strip()
    return best_line


def build_reasoning(c: dict, ref_date: date) -> str:
    """Per-candidate, deterministic, no LLM. Every claim is grounded in the record;
    conditional concern clauses keep the 100 rows substantively different and
    rank-consistent for the Stage-4 checks."""
    p = c.get("profile") or {}
    title = p.get("current_title") or "Unknown role"
    yoe = p.get("years_of_experience")
    yoe_s = f"{yoe:.1f}y" if isinstance(yoe, (int, float)) else "n/a exp"
    tier_phrase = {"core": "core-fit", "adjacent": "adjacent", "off_target": "off-target"}[role_tier(title)]
    parts = [f"{title} ({yoe_s}), {tier_phrase} title."]

    line = _best_career_line(c)
    if line:
        line = " ".join(line.split())
        if len(line) > 120:
            line = line[:117].rstrip() + "..."
        parts.append(f'Relevant experience: "{line}"')

    skills = _top_trusted_skills(c)
    if skills:
        parts.append(f"Endorsed+used AI skills: {', '.join(skills)}.")

    concerns = []
    hist = c.get("career_history") or []
    if hist:
        svc = sum(
            1 for h in hist
            if (h.get("industry") or "").lower() == "it services"
            or any(f in (h.get("company") or "").lower() for f in SERVICES_FIRMS)
        )
        if svc / len(hist) >= 0.5:
            concerns.append("services-heavy background")
    if isinstance(yoe, (int, float)) and not (5 <= yoe <= 9):
        concerns.append(f"{yoe:.1f}y outside the 5-9 band")
    la = (c.get("redrob_signals") or {}).get("last_active_date")
    try:
        days = (ref_date - date.fromisoformat(la[:10])).days if la else None
    except (ValueError, TypeError):
        days = None
    if days and days > 120:
        concerns.append(f"inactive {days}d")
    if concerns:
        parts.append("Concerns: " + "; ".join(concerns) + ".")

    return " ".join(parts)


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
    # --candidates is an alias for --in so the spec's canonical reproduce command works.
    ap.add_argument("--in", "--candidates", dest="inp", default="candidates.jsonl")
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
