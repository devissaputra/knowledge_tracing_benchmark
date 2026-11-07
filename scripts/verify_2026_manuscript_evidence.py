#!/usr/bin/env python3
"""Read-only check of reported 2026 matched-history summary versus archived JSON.
This checks transcription and scope; it does not rerun model training.
"""
import json
from pathlib import Path
r = json.loads((Path(__file__).resolve().parents[1] / "results/matched_history_budget_2026.json").read_text())
assert r["status"] == "passed_all_computational_gates"
assert r["source"]["source_sha256"] == "75e1f46131d5897a388c82411bc3032279899cb1f1e66a9043354d41234407a0"
assert r["source"]["stage3_learners"] == 4000
assert r["source"]["cohort_overlap_previous"] == 0
for k in ("1","20"):
    group = r["comparisons"][k]
    assert all(group["models"][m]["n"] == 20592 for m in ("window_count","ordered_logistic","window_gru"))
    x = group["paired_GRU_minus"]["ordered_logistic"]
    assert x["n_learners"] == 582 and x["n_bootstrap"] == 2000
def close(actual, expected, tolerance=0.000005):
    assert abs(actual - expected) <= tolerance, (actual, expected)
a = r["comparisons"]["20"]
close(a["models"]["window_count"]["brier"], 0.17667)
close(a["models"]["ordered_logistic"]["brier"], 0.17389)
close(a["models"]["window_gru"]["brier"], 0.17514)
d = a["paired_GRU_minus"]["ordered_logistic"]
close(d["difference_candidate_minus_reference"], 0.001255, tolerance=0.0000006)
lo, hi = d["bootstrap_95_interval"]
close(lo, 0.000075, tolerance=0.0000006)
close(hi, 0.002426, tolerance=0.0000006)
print("PASS: 2026 matched-history reported metrics agree with archived JSON at manuscript precision.")
print("LIMITATION: this does not establish that all figures, citations, source data or full PDF text agree.")
