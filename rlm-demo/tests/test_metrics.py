import unittest

from bench.metrics import ndcg_at_k


class NDCGTest(unittest.TestCase):
    def test_ndcg_hit(self):
        hits = [
            {"doc_id": "a"},
            {"doc_id": "b"},
        ]
        score = ndcg_at_k(hits, ["a"], 10)
        self.assertAlmostEqual(score, 1.0)

    def test_ndcg_miss(self):
        hits = [{"doc_id": "x"}]
        score = ndcg_at_k(hits, ["a"], 10)
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
