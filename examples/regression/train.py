from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# Synthetic house-price data so the example needs no downloads.
# Features (in this order): sqft, bedrooms, age_years.
rng = np.random.default_rng(0)
n = 3000
sqft = rng.uniform(500, 4000, n)
bedrooms = rng.integers(1, 6, n)
age = rng.uniform(0, 100, n)
price = 150 * sqft + 10_000 * bedrooms - 500 * age + 50_000 + rng.normal(0, 15_000, n)

X = np.column_stack([sqft, bedrooms, age])
model = GradientBoostingRegressor(random_state=0).fit(X, price)

out = Path(__file__).resolve().parent / "model.joblib"
# Write to a temp file then rename — an atomic swap, so a --watch reloader never
# sees a half-written model.
tmp = out.with_suffix(".joblib.tmp")
joblib.dump(model, tmp)
tmp.replace(out)
print(f"saved {out}  (train R^2 = {model.score(X, price):.3f})")
