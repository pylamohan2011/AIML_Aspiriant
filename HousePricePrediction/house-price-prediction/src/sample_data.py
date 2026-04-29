from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def create_sample_dataset(path: Path, rows: int = 500, seed: int = 42) -> Path:
    """Create a Kaggle-like house price dataset for local smoke tests."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    neighborhoods = np.array(["CollgCr", "Veenker", "Crawfor", "NoRidge", "Somerst", "OldTown"])
    house_styles = np.array(["1Story", "2Story", "1.5Fin", "SLvl"])
    qualities = rng.integers(3, 10, size=rows)
    living_area = rng.normal(1500, 520, size=rows).clip(500, 4200).round()
    garage_cars = rng.integers(0, 4, size=rows)
    total_bsmt = rng.normal(950, 360, size=rows).clip(0, 2500).round()
    first_flr = (living_area * rng.uniform(0.45, 0.95, size=rows)).clip(400, 2500).round()
    full_bath = rng.integers(1, 4, size=rows)
    bedrooms = rng.integers(1, 6, size=rows)
    year_built = rng.integers(1950, 2023, size=rows)
    year_remod = np.maximum(year_built, rng.integers(1970, 2024, size=rows))
    lot_area = rng.normal(9500, 3600, size=rows).clip(2500, 25000).round()
    neighborhood = rng.choice(neighborhoods, size=rows)
    style = rng.choice(house_styles, size=rows)

    neighborhood_effect = {
        "CollgCr": 10000,
        "Veenker": 35000,
        "Crawfor": 25000,
        "NoRidge": 65000,
        "Somerst": 30000,
        "OldTown": -15000,
    }
    style_effect = {"1Story": 8000, "2Story": 15000, "1.5Fin": -2000, "SLvl": 4000}

    sale_price = (
        45000
        + living_area * 82
        + qualities * 18500
        + garage_cars * 14500
        + total_bsmt * 32
        + full_bath * 9000
        + bedrooms * 4500
        + (year_built - 1950) * 850
        + lot_area * 1.6
        + np.vectorize(neighborhood_effect.get)(neighborhood)
        + np.vectorize(style_effect.get)(style)
        + rng.normal(0, 18000, size=rows)
    ).clip(55000, 650000).round()

    df = pd.DataFrame(
        {
            "Id": np.arange(1, rows + 1),
            "OverallQual": qualities,
            "GrLivArea": living_area.astype(int),
            "GarageCars": garage_cars,
            "TotalBsmtSF": total_bsmt.astype(int),
            "1stFlrSF": first_flr.astype(int),
            "FullBath": full_bath,
            "BedroomAbvGr": bedrooms,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod,
            "LotArea": lot_area.astype(int),
            "Neighborhood": neighborhood,
            "HouseStyle": style,
            "SalePrice": sale_price.astype(int),
        }
    )
    df.to_csv(path, index=False)
    return path
