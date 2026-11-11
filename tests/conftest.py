from __future__ import annotations
import pandas as pd
import numpy as np
import pytest
from house_purchase.config import CATEGORICAL, NUMERICAL

@pytest.fixture
def sample_data():
    rng=np.random.default_rng(7)
    n=100
    records={name:rng.normal(size=n) for name in NUMERICAL}
    records.update({
        "country": ["Mexico", "Canada"] * 50,
        "city": ["Puebla", "Toronto"] * 50,
        "property_type": ["Apartment", "Studio"] * 50,
        "furnishing_status": ["Furnished", "Unfurnished"] * 50,
        "property_id": np.arange(n),
        "decision": [0, 1] * 50,
    })
    return pd.DataFrame(records)
