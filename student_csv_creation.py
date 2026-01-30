import numpy as np
import pandas as pd

np.random.seed(42)

n_rows = 200

study_hours = np.random.uniform(1, 10, n_rows)
attendance = np.random.uniform(50, 100, n_rows)
sleep_hours = np.random.uniform(4, 9, n_rows)
previous_score = np.random.uniform(35, 95, n_rows)

# Pure noise (no relation to pass)
noise_feature = np.random.normal(0, 1, n_rows)

# Create probability of passing
# Strong dependence: study_hours, previous_score
# Weak dependence: attendance
logit = (
    0.6 * study_hours +
    0.08 * previous_score +
    0.02 * attendance -
    8
)

prob_pass = 1 / (1 + np.exp(-logit))

pass_label = (np.random.rand(n_rows) < prob_pass).astype(int)

df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "sleep_hours": sleep_hours,
    "previous_score": previous_score,
    "noise_feature": noise_feature,
    "pass": pass_label
})

print(df.head())
df.to_csv("student_marks_new.csv",index=False)