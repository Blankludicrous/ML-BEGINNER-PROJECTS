import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
#load dataset
df=pd.read_csv("employee_productivity_ml.csv")
#remove nan rows from label
df=df.dropna(subset=["left_company"])
#define X and y
X=df[["experience_years","monthly_hours","projects_handled","salary","work_life_balance","remote_work","performance_score"]]
y=df["left_company"]
#train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
#define pipeline
pipeline=Pipeline(
    [("imputation",SimpleImputer(strategy="mean")),
     ("model",GradientBoostingClassifier(random_state=42))]
)
#define grid params
params_grid={
    "model__n_estimators":[50,100,200,300],
    "model__max_depth":[1,2,5],
    "model__learning_rate":[0.01,0.05,0.001]
}
#define grid
grid=GridSearchCV(
    estimator=pipeline,
    param_grid=params_grid,
    cv=5,
    scoring="accuracy"
)
#fit and predict data
grid.fit(X_train,y_train)
pred=grid.predict(X_test)
comparsion=pd.DataFrame(
    {
        "predicted_Values":pred,
        "real_values":y_test.values
    }
)
#print accuracy
print("train_Acc",grid.score(X_train,y_train))
print("test_acc",grid.score(X_test,y_test))
print("best parameters",grid.best_params_)
print("best score",grid.best_score_)
#visulization
ConfusionMatrixDisplay.from_estimator(grid,X_test,y_test)
plt.show()
RocCurveDisplay.from_estimator(grid,X_test,y_test)
plt.show()
#print permutation importance
result=permutation_importance(
    grid,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)
#importance
importance=pd.Series(
    result.importances_mean,
    index=X_test.columns).sort_values(ascending=False)
print(importance)