import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from knowledge_tracing_benchmark import core

class CoreTests(unittest.TestCase):
    def test_predict_before_update(self):
        params=core.BKTParams()
        expected=params.p_init*(1-params.p_slip)+(1-params.p_init)*params.p_guess
        self.assertAlmostEqual(core.predictive_trace([1])[0],expected)

    def test_mastery_trace_updates(self):
        values=core.trace([1,0,1])
        self.assertEqual(len(values),3)
        self.assertGreater(core.trace([1,1])[-1],core.BKTParams().p_init)

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            core.BKTParams(p_guess=1.2)
        with self.assertRaises(ValueError):
            core.predict_correct_probability(-.1)

    def test_non_binary_rejected(self):
        with self.assertRaises(ValueError):
            core.predictive_trace([1,2,0])

    def test_research_bundle_files(self):
        root=Path(__file__).resolve().parents[1]
        for p in ["RESEARCH_BUNDLE.md","docs/dataset_card.md","docs/research_protocol.md","scripts/run_research.py"]:
            self.assertTrue((root/p).exists(),p)

if __name__=="__main__":
    unittest.main()
