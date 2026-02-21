import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay
import matplotlib.pyplot as plt
#load dataset
df=pd.read_csv("crop_yield_dataset.csv")
#define label
df["high_yield"]=(df["crop_yield"]>df["crop_yield"].median()).astype("int")
#remove nan rows from label
df=df.dropna(subset=["high_yield"])
#define X and y
X=df[["rainfall_mm","fertilizer_kg","temperature_c","soil_quality","noise_feature"]]
y=df["high_yield"]
#train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)
#define pipeline
pipeline=Pipeline([
    ("imputation",SimpleImputer(strategy="mean")),
    ("model",GradientBoostingClassifier(random_state=42))
])
#define grid_params
params_grid={
    "model__n_estimators":[50,100,200,300],
    "model__max_depth":[2,3,4],
    "model__learning_rate":[0.01,0.1,1,10]
}
grid=GridSearchCV(
    estimator=pipeline,
    param_grid=params_grid,
    cv=5,
    scoring="accuracy"
)
#fit and predict data
grid.fit(X_train,y_train)
pred=grid.predict(X_test)
print("train_Acc",grid.score(X_train,y_train))
print("test_Acc",grid.score(X_test,y_test))
print("best params",grid.best_params_)
print("best score",grid.best_score_)
comparsion=pd.DataFrame(
    {
        "predicted_values":pred,
        "Real_values":y_test.values
    }
)
print(comparsion)
#display metric
ConfusionMatrixDisplay.from_estimator(grid,X_test,y_test)
plt.show()
RocCurveDisplay.from_estimator(grid,X_test,y_test)
plt.show()
#permutation importance
result=permutation_importance(
    grid,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)
importance=pd.Series(result.importances_mean,
                     index=X_test.columns).sort_values(ascending=False)
print(importance)