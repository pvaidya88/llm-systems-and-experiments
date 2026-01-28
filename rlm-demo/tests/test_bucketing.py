import unittest

from bench.bucketing import infer_buckets


class BucketingTest(unittest.TestCase):
    def test_low_overlap(self):
        buckets = infer_buckets("alpha beta", "gamma delta")
        self.assertIn("low_overlap", buckets)

    def test_acronym_alias(self):
        buckets = infer_buckets("Explain CPU usage", "Central processing unit usage")
        self.assertIn("acronym_alias", buckets)

    def test_multihop(self):
        buckets = infer_buckets("question", "gold", gold_doc_ids=["a", "b"])
        self.assertIn("multihop", buckets)


if __name__ == "__main__":
    unittest.main()
