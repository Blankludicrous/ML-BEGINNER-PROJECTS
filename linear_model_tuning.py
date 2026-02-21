import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
#load csv file into dataframe
df=pd.read_csv("student_marks.csv")
#drop rows with nan
df=df.dropna(subset=["marks"])
#load x and y
X=df[["hours_studied","attendance"]]
y=df["marks"]
#train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#pipeline definition
pipeline=Pipeline([
    ("imputation",SimpleImputer()),
    ("scaling",StandardScaler()),
    ("model",Ridge(alpha=0.1))
])
pipeline.fit(X_train,y_train)
pred=pipeline.predict(X_test)
comparsion=pd.DataFrame({
    "predicted values":pred,
    "real_values":y_test.values
})
print(comparsion)
print("train_acc",pipeline.score(X_train,y_train))
print("test_acc",pipeline.score(X_test,y_test))
result=permutation_importance(
    pipeline,
    X_test,
    y_test,
    scoring="r2",
    n_repeats=20,
    random_state=42
)
importance=pd.Series(
    result.importances_mean,
    index=X_test.columns).sort_values(ascending=False)
print(importance)