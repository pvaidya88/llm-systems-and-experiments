import unittest

from bench.gates import evaluate, DEFAULT_THRESHOLDS


class GatesTest(unittest.TestCase):
    def test_pass(self):
        summary = {
            "baseline": {"accuracy": 0.8, "recall_at_10": 0.7, "p95_latency": 10, "cost_per_query": 100, "hallucination_rate": 0.1},
            "candidate": {"accuracy": 0.79, "recall_at_10": 0.69, "p95_latency": 19, "cost_per_query": 140, "hallucination_rate": 0.1},
        }
        result = evaluate(summary, DEFAULT_THRESHOLDS)
        self.assertTrue(result["passed"])

    def test_fail_accuracy(self):
        summary = {
            "baseline": {"accuracy": 0.8, "recall_at_10": 0.7, "p95_latency": 10, "cost_per_query": 100, "hallucination_rate": 0.1},
            "candidate": {"accuracy": 0.6, "recall_at_10": 0.7, "p95_latency": 10, "cost_per_query": 100, "hallucination_rate": 0.1},
        }
        result = evaluate(summary, DEFAULT_THRESHOLDS)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
