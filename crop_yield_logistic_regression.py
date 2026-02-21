import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay,classification_report
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
#load dataset
df=pd.read_csv("crop_yield_dataset.csv")
#define the classification label
df["high_yield"]=(df["crop_yield"]>df["crop_yield"].median()).astype("int")
#drop nan rows in label
df=df.dropna(subset=["high_yield"])
#define X and y
X=df[["rainfall_mm","fertilizer_kg","temperature_c","soil_quality","noise_feature"]]
y=df["high_yield"]
#train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#define pipleine
pipeline=Pipeline(
    [
        ("imputation",SimpleImputer(strategy="mean")),
        ("scaling",StandardScaler()),
        ("logreg",LogisticRegression(max_iter=1000,random_state=42))]
)
#define params_grid
params_grid={
    "logreg__C":[0.01,0.1,1,10,100],
    "logreg__penalty":["l1","l2"],
    "logreg__class_weight":["balanced"],
    "logreg__solver":["liblinear"]
}
#define grid
grid=GridSearchCV(
    estimator=pipeline,
    param_grid=params_grid,
    cv=5,
    scoring="accuracy"
)
grid.fit(X_train,y_train)
print("train_Acc",grid.score(X_train,y_train))
print("test_Acc",grid.score(X_test,y_test))
print("best params",grid.best_params_)
pred=grid.predict(X_test)
comparison=pd.DataFrame(
    {
        "predicted_values":pred,
        "real_values":y_test.values
    }
)
print(comparison)
ConfusionMatrixDisplay.from_estimator(grid,X_test,y_test)
plt.show()
print("classification_report",classification_report(y_test,pred))
#print permutraion importance
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