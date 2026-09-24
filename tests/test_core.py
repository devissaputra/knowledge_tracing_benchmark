import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from knowledge_tracing_benchmark import core


class CoreTests(unittest.TestCase):
    def test_mastery_trace_updates(self):
        trace = core.trace([1, 0, 1])
        self.assertEqual(len(trace), 3)
        self.assertGreater(core.trace([1, 1])[-1], core.BKTParams().p_init)

    def test_invalid_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            core.BKTParams(p_guess=1.2)


if __name__ == "__main__":
    unittest.main()
