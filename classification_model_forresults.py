import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
#load data
df=pd.read_csv("student_marks.csv")
#defining result column
df["result"]=(df["marks"]>=35).astype(int)
#checking for impurity
assert set(df["result"].unique()).issubset({0,1})
#load x and y
X=df[["marks"]]
y=df["result"]
#test_train split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#pipeline definition
pipeline=Pipeline([
    ("imputation",SimpleImputer()),
    ("scaling",StandardScaler()),
    ("model",LogisticRegression())
])
#fitting data into the model
pipeline.fit(X_train,y_train)
pipeline.predict(X_test)
print("train_model_score",pipeline.score(X_train,y_train))
print("test_model_score",pipeline.score(X_test,y_test))