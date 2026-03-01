import unittest

from pricing.engine import compute_total


class PricingTests(unittest.TestCase):
    def test_coupon_expiring_now_is_not_active(self) -> None:
        now = "2026-03-01T12:00:00Z"
        lines = [{"sku": "apple", "qty": 1, "unit_price_cents": 1000}]
        coupons = [{"percent_off": 20, "expires_at": now}]
        result = compute_total(lines, coupons, shipping_cents=0, now_iso=now)
        self.assertEqual(result["discount_cents"], 0)
        self.assertEqual(result["total_cents"], 1000)

    def test_percent_discount_rounds_to_nearest_cent(self) -> None:
        now = "2026-03-01T12:00:00Z"
        lines = [{"sku": "banana", "qty": 1, "unit_price_cents": 999}]
        coupons = [{"percent_off": 15, "expires_at": "2026-03-01T12:05:00Z"}]
        result = compute_total(lines, coupons, shipping_cents=0, now_iso=now)
        # 999 * 0.15 = 149.85 -> should round to 150
        self.assertEqual(result["discount_cents"], 150)
        self.assertEqual(result["total_cents"], 849)

    def test_free_shipping_threshold_includes_boundary(self) -> None:
        now = "2026-03-01T12:00:00Z"
        lines = [{"sku": "pear", "qty": 1, "unit_price_cents": 1000}]
        coupons = [
            {
                "free_shipping_at_cents": 1000,
                "expires_at": "2026-03-01T12:10:00Z",
            }
        ]
        result = compute_total(lines, coupons, shipping_cents=500, now_iso=now)
        self.assertEqual(result["shipping_cents"], 0)
        self.assertEqual(result["total_cents"], 1000)


if __name__ == "__main__":
    unittest.main()
