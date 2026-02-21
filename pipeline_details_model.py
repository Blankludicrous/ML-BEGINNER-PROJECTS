import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#load dataset
data=pd.read_csv("student_details.csv")
#load feature and label
X=data[["age"]]
y=data["marks"]
#train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#pipeline creation
pipeline=Pipeline([
    ("imputation",SimpleImputer(strategy="mean")),
     ("model",LinearRegression())
])
pipeline.fit(X_train,y_train)
pipeline.predict(X_test)
print("R2 score",pipeline.score(X_test,y_test))