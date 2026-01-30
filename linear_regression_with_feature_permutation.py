import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
#load dataset
df=pd.read_csv("student_marks_new.csv")
df=df.dropna(subset="pass")
#load x and y
X=df[["study_hours","attendance","sleep_hours","previous_score","noise_feature"]]
y=df["pass"]
#train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#Pipeline definition
pipeline=Pipeline([
    ("imputation",SimpleImputer()),
    ("scaling",StandardScaler()),
    ("model",LogisticRegression())
])
#fit and predict data
pipeline.fit(X_train,y_train)
pred=pipeline.predict(X_test)
comparsion=pd.DataFrame(
    {
        "original values":y_test.values,
        "predicted values":pred
    }
)
print(comparsion)
print("train model accuracy",pipeline.score(X_train,y_train))
print("test model accuracy",pipeline.score(X_test,y_test))
result=permutation_importance(
    pipeline,
    X_test,
    y_test,
    scoring="r2",
    n_repeats=10,
    random_state=42
)
importance=pd.Series(
    result.importances_mean,
    index=X_test.columns).sort_values(ascending=False)
print(importance)