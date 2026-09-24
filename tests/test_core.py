import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from knowledge_tracing_benchmark import core


class CoreTests(unittest.TestCase):
    def test_mastery_trace_updates(self):
        values = core.trace([1, 0, 1])
        self.assertEqual(len(values), 3)
        self.assertGreater(core.trace([1, 1])[-1], core.BKTParams().p_init)

    def test_default_first_correct_update_is_expected(self):
        value = core.trace([1])[0]
        self.assertAlmostEqual(value, 0.6, places=7)

    def test_incorrect_response_can_reduce_mastery_before_learning(self):
        params = core.BKTParams(p_init=0.8, p_learn=0.0, p_guess=0.2, p_slip=0.1)
        self.assertLess(core.trace([0], params)[0], params.p_init)

    def test_invalid_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            core.BKTParams(p_guess=1.2)
        with self.assertRaises(ValueError):
            core.update_mastery(-0.1, True)

    def test_non_binary_response_is_rejected(self):
        with self.assertRaises(ValueError):
            core.trace([1, 2, 0])

    def test_empty_sequence_returns_empty_trace(self):
        self.assertEqual(core.trace([]), [])


if __name__ == "__main__":
    unittest.main()
