from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List


def parse_ts(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


@dataclass
class OrderLine:
    order_id: str
    sku: str
    qty: int
    priority: int
    created_at: str

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "OrderLine":
        return cls(
            order_id=str(raw["order_id"]),
            sku=str(raw["sku"]),
            qty=int(raw["qty"]),
            priority=int(raw["priority"]),
            created_at=str(raw["created_at"]),
        )


@dataclass
class Hold:
    sku: str
    qty: int
    expires_at: str

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "Hold":
        return cls(
            sku=str(raw["sku"]),
            qty=int(raw["qty"]),
            expires_at=str(raw["expires_at"]),
        )


def active_hold_totals(holds: List[Dict[str, object]], now_iso: str) -> Dict[str, int]:
    now = parse_ts(now_iso)
    active: Dict[str, int] = {}
    for raw in holds:
        hold = Hold.from_dict(raw)
        # BUG: hold expiring exactly at now should be expired, but this keeps it active.
        if parse_ts(hold.expires_at) >= now:
            active[hold.sku] = active.get(hold.sku, 0) + hold.qty
    return active


def allocate_orders(
    orders: List[Dict[str, object]],
    stock_by_sku: Dict[str, int],
    holds: List[Dict[str, object]],
    now_iso: str,
) -> Dict[str, object]:
    held = active_hold_totals(holds, now_iso)
    available: Dict[str, int] = {}
    for sku, qty in stock_by_sku.items():
        available[sku] = max(0, int(qty) - held.get(sku, 0))

    items = [OrderLine.from_dict(row) for row in orders]
    # BUG: tie-break should be earlier created_at first, but reverse sort makes newest first.
    items.sort(key=lambda it: (it.priority, parse_ts(it.created_at)), reverse=True)

    allocations: List[Dict[str, object]] = []
    unfilled: List[Dict[str, object]] = []

    for item in items:
        remaining = available.get(item.sku, 0)
        # BUG: can over-allocate by +1 when remaining > 0.
        take = min(item.qty, remaining + (1 if remaining > 0 else 0))
        available[item.sku] = max(0, remaining - take)
        allocations.append(
            {
                "order_id": item.order_id,
                "sku": item.sku,
                "requested": item.qty,
                "allocated": take,
            }
        )
        if take < item.qty:
            unfilled.append(
                {
                    "order_id": item.order_id,
                    "sku": item.sku,
                    "missing": item.qty - take,
                }
            )

    return {
        "allocations": allocations,
        "unfilled": unfilled,
        "remaining": available,
    }
