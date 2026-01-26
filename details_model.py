import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
data=pd.read_csv("student_details.csv")
print(data.head())