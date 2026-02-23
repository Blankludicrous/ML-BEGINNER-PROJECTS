import numpy as np
import pandas as pd

np.random.seed(42)

# Cluster 1 - High Performers
c1 = pd.DataFrame({
    "Study_Hours_Per_Day": np.random.normal(6, 0.8, 34),
    "Sleep_Hours": np.random.normal(7, 0.5, 34),
    "Attendance_Percentage": np.random.normal(90, 5, 34),
    "Exam_Score": np.random.normal(85, 5, 34)
})

# Cluster 2 - Average Performers
c2 = pd.DataFrame({
    "Study_Hours_Per_Day": np.random.normal(4, 0.7, 33),
    "Sleep_Hours": np.random.normal(6.5, 0.6, 33),
    "Attendance_Percentage": np.random.normal(75, 6, 33),
    "Exam_Score": np.random.normal(65, 6, 33)
})

# Cluster 3 - Low Performers
c3 = pd.DataFrame({
    "Study_Hours_Per_Day": np.random.normal(2, 0.5, 33),
    "Sleep_Hours": np.random.normal(5.5, 0.7, 33),
    "Attendance_Percentage": np.random.normal(60, 8, 33),
    "Exam_Score": np.random.normal(45, 7, 33)
})

df = pd.concat([c1, c2, c3]).reset_index(drop=True)

df.to_csv("students.csv", index=False)

print(df.head())
print("Shape:", df.shape)