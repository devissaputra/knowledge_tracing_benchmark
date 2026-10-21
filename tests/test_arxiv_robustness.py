"""Unit tests for manuscript robustness extension (no empirical datasets or synthetic study claims)."""
import importlib.util
import unittest
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("arxiv_robustness",ROOT/"scripts"/"robustness_extension.py")
MODULE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

class RobustnessTests(unittest.TestCase):
    def test_cluster_bootstrap_preserves_interaction_weighted_target(self):
        reference=pd.DataFrame({"learner":["a","a","a","b"],"y":[0,0,0,0],
                                "skill":["x","x","x","x"],"p":[0.,0.,0.,0.]})
        candidate=reference.copy()
        candidate["p"]=[1.,1.,1.,0.]
        result=MODULE.paired_cluster_bootstrap(reference,candidate,repetitions=300,seed=2)
        self.assertAlmostEqual(result["difference_candidate_minus_reference"],.75)
        self.assertEqual(result["n_learners"],2)
        self.assertEqual(result["n_interactions"],4)
        candidate.loc[0,"learner"]="different"
        with self.assertRaises(ValueError):
            MODULE.paired_cluster_bootstrap(reference,candidate,repetitions=300)
    def test_cross_skill_history_uses_only_previous_outcomes(self):
        rows=[{"user_id":"one","skill_ids":["x","x","y"],"grades":[1,0,1]}]
        x,y,users,skills,first=MODULE.pfa_plus_all_skill_history_frame(rows)
        self.assertEqual(y.tolist(),[1,0,1])
        self.assertEqual(x["all_success"].tolist(),[0,1,1])
        self.assertEqual(x["all_failure"].tolist(),[0,0,1])
        self.assertEqual(x["skill_success"].tolist(),[0,1,0])
        self.assertEqual(first,[True,False,True])

if __name__=="__main__":
    unittest.main()
