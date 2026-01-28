import unittest

from bench.metrics import exact_match, f1_score, recall_at_k, parse_citations


class MetricsTest(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(exact_match("Answer", "answer"), 1.0)
        self.assertEqual(exact_match("Answer", "other"), 0.0)

    def test_f1(self):
        self.assertAlmostEqual(f1_score("foo bar", "foo"), 2/3)
        self.assertEqual(f1_score("", "foo"), 0.0)

    def test_recall_at_k(self):
        hits = [{"doc_id": "a"}, {"doc_id": "b"}]
        self.assertEqual(recall_at_k(hits, ["b"], 1), 0.0)
        self.assertEqual(recall_at_k(hits, ["b"], 2), 1.0)

    def test_parse_citations(self):
        text = "Answer CITE(doc_id=1, chunk_id=2, start=10, end=20)"
        cites = parse_citations(text)
        self.assertEqual(len(cites), 1)
        self.assertEqual(cites[0]["chunk_id"], "2")


if __name__ == "__main__":
    unittest.main()
