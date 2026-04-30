from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def create_sample_dataset(path: Path, rows: int = 900, seed: int = 42) -> Path:
    """Create realistic-enough OHLCV stock data for local smoke tests."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=rows)
    daily_returns = rng.normal(0.0006, 0.018, size=rows)
    close = 150 * np.exp(np.cumsum(daily_returns))
    open_price = close * (1 + rng.normal(0, 0.004, size=rows))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0.001, 0.02, size=rows))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0.001, 0.02, size=rows))
    volume = rng.integers(25_000_000, 120_000_000, size=rows)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": open_price.round(2),
            "High": high.round(2),
            "Low": low.round(2),
            "Close": close.round(2),
            "Volume": volume,
        }
    )
    df.to_csv(path, index=False)
    return path
