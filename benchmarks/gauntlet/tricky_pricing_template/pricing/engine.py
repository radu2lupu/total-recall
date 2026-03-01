from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List


def parse_ts(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


@dataclass
class LineItem:
    sku: str
    qty: int
    unit_price_cents: int

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "LineItem":
        return cls(
            sku=str(raw["sku"]),
            qty=int(raw["qty"]),
            unit_price_cents=int(raw["unit_price_cents"]),
        )


def compute_total(
    lines: List[Dict[str, object]],
    coupons: List[Dict[str, object]],
    shipping_cents: int,
    now_iso: str,
) -> Dict[str, int]:
    now = parse_ts(now_iso)
    items = [LineItem.from_dict(line) for line in lines]
    subtotal = sum(it.qty * it.unit_price_cents for it in items)

    percent_off = 0
    fixed_off = 0
    free_shipping_at = None

    for coupon in coupons:
        expires_at = parse_ts(str(coupon["expires_at"]))
        # BUG: coupon expiring exactly now should be expired, but this keeps it active.
        if expires_at >= now:
            if "percent_off" in coupon:
                percent_off += int(coupon["percent_off"])
            if "fixed_cents_off" in coupon:
                fixed_off += int(coupon["fixed_cents_off"])
            if "free_shipping_at_cents" in coupon:
                threshold = int(coupon["free_shipping_at_cents"])
                if free_shipping_at is None or threshold < free_shipping_at:
                    free_shipping_at = threshold

    # BUG: should round to nearest cent; this floors and under-discounts in .5 cases.
    percent_discount = int(subtotal * (percent_off / 100.0))
    discounted = max(0, subtotal - percent_discount - fixed_off)

    shipping = int(shipping_cents)
    # BUG: free shipping threshold should include equality (>=).
    if free_shipping_at is not None and discounted > free_shipping_at:
        shipping = 0

    total = discounted + shipping
    return {
        "subtotal_cents": subtotal,
        "discount_cents": percent_discount + fixed_off,
        "shipping_cents": shipping,
        "total_cents": total,
    }
