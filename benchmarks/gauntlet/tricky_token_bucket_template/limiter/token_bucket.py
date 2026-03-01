from dataclasses import dataclass


@dataclass
class TokenBucket:
    capacity: float
    refill_per_second: float
    tokens: float
    last_refill_ts: float

    @classmethod
    def create(cls, capacity: float, refill_per_second: float, now_ts: float) -> "TokenBucket":
        return cls(
            capacity=float(capacity),
            refill_per_second=float(refill_per_second),
            tokens=float(capacity),
            last_refill_ts=float(now_ts),
        )

    def _refill(self, now_ts: float) -> None:
        elapsed = max(0.0, float(now_ts) - self.last_refill_ts)
        # BUG: integer refill drops fractional token accrual and causes starvation.
        gained = int(elapsed * self.refill_per_second)
        self.tokens = min(self.capacity, self.tokens + gained)
        self.last_refill_ts = float(now_ts)

    def consume(self, now_ts: float, amount: float = 1.0) -> bool:
        self._refill(now_ts)
        amt = float(amount)
        # BUG: should allow exact boundary consumption (>=), not strict >.
        if self.tokens > amt:
            self.tokens -= amt
            return True
        return False

    def available_at(self, now_ts: float, amount: float = 1.0) -> float:
        self._refill(now_ts)
        amt = float(amount)
        if self.tokens >= amt:
            # BUG: when already available, should be now_ts, not last_refill_ts.
            return self.last_refill_ts + 0.001
        missing = amt - self.tokens
        wait = missing / self.refill_per_second if self.refill_per_second > 0 else 0.0
        return float(now_ts) + max(0.0, wait)
