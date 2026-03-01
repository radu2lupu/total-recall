import unittest

from inventory.allocator import allocate_orders


class AllocationTests(unittest.TestCase):
    def test_hold_expiring_now_is_not_active(self) -> None:
        now = "2026-03-01T12:00:00Z"
        orders = [
            {
                "order_id": "A-100",
                "sku": "apple",
                "qty": 10,
                "priority": 5,
                "created_at": "2026-03-01T10:00:00Z",
            }
        ]
        stock = {"apple": 10}
        holds = [{"sku": "apple", "qty": 3, "expires_at": now}]
        result = allocate_orders(orders, stock, holds, now)
        self.assertEqual(result["allocations"][0]["allocated"], 10)
        self.assertEqual(result["remaining"]["apple"], 0)

    def test_priority_tie_uses_earliest_created_at(self) -> None:
        now = "2026-03-01T12:00:00Z"
        orders = [
            {
                "order_id": "EARLY",
                "sku": "banana",
                "qty": 5,
                "priority": 3,
                "created_at": "2026-03-01T08:00:00Z",
            },
            {
                "order_id": "LATE",
                "sku": "banana",
                "qty": 5,
                "priority": 3,
                "created_at": "2026-03-01T09:00:00Z",
            },
        ]
        stock = {"banana": 5}
        result = allocate_orders(orders, stock, [], now)
        first = result["allocations"][0]
        second = result["allocations"][1]
        self.assertEqual(first["order_id"], "EARLY")
        self.assertEqual(first["allocated"], 5)
        self.assertEqual(second["allocated"], 0)

    def test_never_over_allocates_last_unit(self) -> None:
        now = "2026-03-01T12:00:00Z"
        orders = [
            {
                "order_id": "ONE",
                "sku": "pear",
                "qty": 2,
                "priority": 9,
                "created_at": "2026-03-01T10:00:00Z",
            }
        ]
        stock = {"pear": 1}
        result = allocate_orders(orders, stock, [], now)
        alloc = result["allocations"][0]
        self.assertEqual(alloc["allocated"], 1)
        self.assertEqual(result["remaining"]["pear"], 0)
        self.assertEqual(result["unfilled"][0]["missing"], 1)


if __name__ == "__main__":
    unittest.main()
