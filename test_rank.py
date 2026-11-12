#!/usr/bin/env python3
"""Unit tests for rank.py on synthetic candidates. Standard library only.

    python -m unittest test_rank -v
"""

import os
import tempfile
import unittest
from datetime import date

import rank as R

REF = date(2026, 5, 27)


def candidate(i, title, description, company="Acme Search", industry="Software",
              skills=None, yoe=6.0):
    return {
        "candidate_id": f"CAND_{i:07d}",
        "profile": {"current_title": title, "years_of_experience": yoe},
        "career_history": [{
            "title": title, "company": company, "industry": industry,
            "description": description,
            "start_date": "2020-05-01", "end_date": None, "duration_months": 72,
        }],
        "education": [{"start_year": 2014, "end_year": 2018}],
        "skills": skills or [],
        "redrob_signals": {"last_active_date": "2026-05-20",
                           "recruiter_response_rate": 0.6, "open_to_work_flag": True},
    }


def skill(name, months=36, endorsements=5, proficiency="advanced"):
    return {"name": name, "duration_months": months,
            "endorsements": endorsements, "proficiency": proficiency}


STUFFED_SKILLS = [skill(n, months=0, endorsements=0, proficiency="intermediate")
                  for n in ("Machine Learning", "NLP", "RAG", "PyTorch", "Vector Search", "LLM")]


def stuffer(i=1):
    return candidate(i, "Marketing Manager", "Ran brand campaigns and events.",
                     industry="Retail", skills=STUFFED_SKILLS)


def builder(i=2):
    return candidate(
        i, "Machine Learning Engineer",
        "Built and deployed a semantic search and ranking system in production, "
        "evaluated with NDCG and A/B tests.",
        skills=[skill("Information Retrieval"), skill("PyTorch")],
    )


class GateTests(unittest.TestCase):
    def multiplier(self, c):
        career_raw = R.score_career_history(c["career_history"], REF)
        ai_hits, trusted_ai, _ = R.ai_features(c)
        return R.hard_negative_multiplier(c, career_raw, ai_hits, trusted_ai)

    def test_keyword_stuffer_gets_stuffer_multiplier(self):
        self.assertAlmostEqual(self.multiplier(stuffer()), 0.05)

    def test_real_builder_is_not_gated(self):
        self.assertEqual(self.multiplier(builder()), 1.0)

    def test_builder_outscores_stuffer(self):
        self.assertGreater(R.score_candidate(builder(), REF), R.score_candidate(stuffer(), REF))

    def test_services_only_career_is_gated(self):
        c = candidate(3, "Machine Learning Engineer", "Built a search system in production.",
                      company="Infosys", industry="IT Services")
        self.assertAlmostEqual(self.multiplier(c), 0.10)


class HoneypotTests(unittest.TestCase):
    def test_consistent_profile_passes(self):
        self.assertEqual(R.honeypot_reasons(builder(), REF), [])

    def test_expert_skills_with_zero_duration_flagged(self):
        c = builder()
        c["skills"] = [skill(n, months=0, proficiency="expert") for n in ("NLP", "RAG", "PyTorch")]
        self.assertIn("expert_skills_zero_duration", R.honeypot_reasons(c, REF))

    def test_experience_exceeding_history_flagged(self):
        c = builder()
        c["profile"]["years_of_experience"] = 15.0
        self.assertIn("yoe_exceeds_summed_history", R.honeypot_reasons(c, REF))


class SentinelTests(unittest.TestCase):
    def test_missing_response_rate_is_neutral_not_zero(self):
        missing, zero = builder(), builder()
        missing["redrob_signals"]["recruiter_response_rate"] = -1
        zero["redrob_signals"]["recruiter_response_rate"] = 0.0
        self.assertGreater(R.behavioral_score(missing, REF), R.behavioral_score(zero, REF))


class DeterminismTests(unittest.TestCase):
    def run_once(self, records, path):
        R.write_csv(R.rank(records, R.compute_ref_date(records)), path)
        with open(path, "rb") as f:
            return f.read()

    def test_two_runs_are_byte_identical(self):
        records = [stuffer(1), builder(2), builder(3), stuffer(4)]
        with tempfile.TemporaryDirectory() as d:
            first = self.run_once(records, os.path.join(d, "a.csv"))
            second = self.run_once(list(reversed(records)), os.path.join(d, "b.csv"))
        self.assertEqual(first, second)

    def test_ties_break_on_candidate_id(self):
        rows = R.rank([builder(9), builder(2)], REF)
        self.assertEqual([r["candidate_id"] for r in rows], ["CAND_0000002", "CAND_0000009"])


if __name__ == "__main__":
    unittest.main()
