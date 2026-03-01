import unittest

from limiter.token_bucket import TokenBucket


class TokenBucketTests(unittest.TestCase):
    def test_exact_boundary_can_consume(self) -> None:
        bucket = TokenBucket(capacity=5, refill_per_second=1, tokens=1, last_refill_ts=0.0)
        ok = bucket.consume(now_ts=0.0, amount=1)
        self.assertTrue(ok)
        self.assertEqual(bucket.tokens, 0)

    def test_fractional_refill_is_preserved(self) -> None:
        bucket = TokenBucket(capacity=2, refill_per_second=2.0, tokens=0.0, last_refill_ts=0.0)
        bucket.consume(now_ts=0.25, amount=0)  # refill side effect only
        bucket.consume(now_ts=0.50, amount=0)
        # At 2 tokens/sec, after 0.5s we should have gained 1.0 token.
        self.assertAlmostEqual(bucket.tokens, 1.0, places=6)

    def test_available_at_returns_now_when_already_available(self) -> None:
        bucket = TokenBucket(capacity=3, refill_per_second=1.0, tokens=2.0, last_refill_ts=10.0)
        ts = bucket.available_at(now_ts=12.0, amount=1.0)
        self.assertEqual(ts, 12.0)


if __name__ == "__main__":
    unittest.main()
