import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
#load dataset
df=pd.read_csv("student_details.csv")
#load x and y
X=df[["hours_studied","attendance"]]
y=df["marks"]
#train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#define pipline
pipeline=Pipeline([
    ("imputation",SimpleImputer(strategy="mean")),
    ("scaling",StandardScaler()),
    ("model",LinearRegression())
])
#data fit and predict
pipeline.fit(X_train,y_train)
pipeline.predict(X_test)
#printing R2 score of the model
print("model score is",pipeline.score(X_test,y_test))