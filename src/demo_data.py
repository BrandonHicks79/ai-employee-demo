"""Demo customer accounts — SIMULATED data for the sales demo.

This module stands in for a real CRM / billing integration. In production
it gets replaced by: caller-ID lookup + account API (CRM, QuickBooks,
Stripe, etc.). The dialogue manager only calls find_customer() — so this
module is the entire seam. Swap it, keep everything else.
"""

DEMO_CUSTOMERS = [
    {
        "name": "Brandon",
        "current_invoice": {
            "id": "INV-2041",
            "period": "October 2026",
            "amount_due": 189.50,
            "due_date": "October 20, 2026",
            "line_items": [
                ("Service visit", 149.00),
                ("Parts", 40.50),
            ],
        },
        "past_due": 0.00,
        "payments": [
            {"date": "Oct 2, 2026", "amount": 149.00, "method": "Card"},
        ],
    },
    {
        "name": "Sarah",
        "current_invoice": {
            "id": "INV-2038",
            "period": "October 2026",
            "amount_due": 85.00,
            "due_date": "November 1, 2026",
            "line_items": [
                ("Monthly plan", 85.00),
            ],
        },
        "past_due": 45.00,
        "payments": [
            {"date": "Aug 12, 2026", "amount": 85.00, "method": "Card"},
        ],
    },
    {
        "name": "Mike",
        "current_invoice": None,  # all paid up
        "past_due": 0.00,
        "payments": [
            {"date": "Oct 7, 2026", "amount": 75.00, "method": "Card"},
            {"date": "Oct 7, 2026", "amount": 75.00, "method": "Card"},
        ],
    },
]


def find_customer(name):
    """Case-insensitive first-name match against demo accounts."""
    key = name.strip().lower()
    for c in DEMO_CUSTOMERS:
        if c["name"].lower() in key or key in c["name"].lower():
            return c
    return None
