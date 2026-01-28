import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
#load dataset 
df=pd.read_csv("student_performance.csv")
#
df=df.dropna(subset=["marks"])
#load x and y
X=df[["age","hours_studied","attendance"]]
y=df["marks"]
#train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#pipeline definition
pipeline=Pipeline([
    ("imputation",SimpleImputer(strategy="mean")),
    ("scaling",StandardScaler()),
    ("model",LinearRegression())
])
#fit and predict data
pipeline.fit(X_train,y_train)
pipeline.predict(X_test)
print("model R2 score",pipeline.score(X_test,y_test))