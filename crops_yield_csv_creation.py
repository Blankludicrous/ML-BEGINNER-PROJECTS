import pandas as pd
import numpy as np

np.random.seed(42)

n = 120

rainfall_mm = np.random.randint(50, 300, n)
fertilizer_kg = np.random.randint(10, 100, n)
temperature_c = np.random.randint(18, 40, n)
soil_quality = np.random.randint(1, 10, n)
noise_feature = np.random.randn(n)

# crop yield formula
yield_value = (
    rainfall_mm * 0.05 +
    fertilizer_kg * 0.4 +
    soil_quality * 2 -
    temperature_c * 0.3 +
    np.random.randn(n) * 2
)

df = pd.DataFrame({
    "rainfall_mm": rainfall_mm,
    "fertilizer_kg": fertilizer_kg,
    "temperature_c": temperature_c,
    "soil_quality": soil_quality,
    "noise_feature": noise_feature,
    "crop_yield": yield_value
})
df.to_csv("crop_yield_dataset.csv", index=False)