import numpy as np
import pandas as pd

np.random.seed(7)

n = 150

data = {
    "experience_years": np.random.randint(0, 21, n),
    "monthly_hours": np.random.randint(120, 260, n),
    "projects_handled": np.random.randint(1, 10, n),
    "salary": np.random.randint(20000, 120000, n),
    "work_life_balance": np.random.randint(1, 6, n),
    "remote_work": np.random.randint(0, 2, n),
}

df = pd.DataFrame(data)

# Linear-style performance score
df["performance_score"] = (
    2.5 * df["experience_years"]
    + 0.05 * df["monthly_hours"]
    + 3 * df["projects_handled"]
    + 0.0004 * df["salary"]
    + 4 * df["work_life_balance"]
    + 3 * df["remote_work"]
    + np.random.normal(0, 5, n)
).round(1)

df["performance_score"] = df["performance_score"].clip(0, 100)

# Attrition logic (non-linear)
df["left_company"] = (
    (df["work_life_balance"] <= 2) &
    (df["monthly_hours"] > 220)
).astype(int)

# Save CSV
df.to_csv("employee_productivity_ml.csv", index=False)

df.head()
