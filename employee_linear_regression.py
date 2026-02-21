import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
#Load the dataset
df=pd.read_csv("employee_productivity_ml.csv")
#drop na rows from label
df=df.dropna(subset=["performance_score"])
#load X and y
X=df[["experience_years","monthly_hours","projects_handled","salary","work_life_balance","remote_work"]]
y=df["performance_score"]
#train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#define pipeline
pipeline=Pipeline([
    ("imputation",SimpleImputer(strategy="mean")),
    ("scaling",StandardScaler()),
    ("model",Ridge(alpha=0.1))
])
pipeline.fit(X_train,y_train)
pred=pipeline.predict(X_test)
comparsion=pd.DataFrame({
    "predicted_values":pred,
    "real_values":y_test.values
})
print(comparsion)
result=permutation_importance(
    pipeline,
    X_test,
    y_test,
    scoring="r2",
    random_state=42
)
importance=pd.Series(
    result.importances_mean,
    index=X_test.columns).sort_values(ascending=False)
print(importance)