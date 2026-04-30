from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def create_sample_dataset(path: Path, customers: int = 700, seed: int = 42) -> Path:
    """Create Kaggle Online Retail-like transaction data for local smoke tests."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    countries = np.array(["United Kingdom", "Germany", "France", "Spain", "Netherlands", "Australia"])
    products = np.array(
        [
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "REGENCY CAKESTAND 3 TIER",
            "JUMBO BAG RED RETROSPOT",
            "PARTY BUNTING",
            "LUNCH BAG BLACK SKULL",
            "SET OF 3 CAKE TINS",
            "POSTAGE",
            "ASSORTED COLOUR BIRD ORNAMENT",
        ]
    )
    rows = []
    invoice_number = 536000
    end_date = pd.Timestamp.today().normalize()

    for customer_id in range(10000, 10000 + customers):
        segment_seed = rng.choice(["champion", "loyal", "spender", "occasional", "at_risk"])
        if segment_seed == "champion":
            invoices = rng.integers(12, 28)
            recency_max = 35
            unit_price_range = (5, 55)
        elif segment_seed == "loyal":
            invoices = rng.integers(8, 18)
            recency_max = 80
            unit_price_range = (3, 35)
        elif segment_seed == "spender":
            invoices = rng.integers(3, 10)
            recency_max = 120
            unit_price_range = (25, 120)
        elif segment_seed == "occasional":
            invoices = rng.integers(1, 6)
            recency_max = 180
            unit_price_range = (2, 30)
        else:
            invoices = rng.integers(2, 8)
            recency_max = 365
            unit_price_range = (2, 45)

        country = rng.choice(countries)
        latest_days_ago = int(rng.integers(1, recency_max + 1))
        for invoice_offset in range(invoices):
            invoice_date = end_date - pd.Timedelta(
                days=int(latest_days_ago + rng.integers(0, 260))
            )
            invoice_number += 1
            lines = rng.integers(1, 7)
            for _ in range(lines):
                quantity = int(rng.integers(1, 18))
                unit_price = float(rng.uniform(*unit_price_range))
                rows.append(
                    {
                        "InvoiceNo": str(invoice_number),
                        "StockCode": str(rng.integers(10000, 99999)),
                        "Description": rng.choice(products),
                        "Quantity": quantity,
                        "InvoiceDate": invoice_date,
                        "UnitPrice": round(unit_price, 2),
                        "CustomerID": customer_id,
                        "Country": country,
                    }
                )

    df = pd.DataFrame(rows).sort_values("InvoiceDate")
    df.to_csv(path, index=False)
    return path
