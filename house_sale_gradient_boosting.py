import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
#load dataset
df=pd.read_csv("house_sales_ml.csv")
#remove nan rows from label
df=df.dropna(subset=["sold"])
#define X and y
X=df[["size_sqft","bedrooms","bathrooms","age_years","distance_city_km","floor","price_lakhs"]]
y=df["sold"]
#train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)
#define pipeline
pipline=Pipeline(
    [
        ("imputation",SimpleImputer(strategy="mean")),
        ("model",GradientBoostingClassifier(random_state=42))
    ]
)
#define grid params
params_grid={
    "model__n_estimators":[50,100,150,200],
    "model__max_depth":[1,2,5],
    "model__learning_rate":[0.001,0.005,0.01,0.05,0.1]
    }
#define grid
grid=GridSearchCV(
    estimator=pipline,
    param_grid=params_grid,
    cv=5,
    scoring="accuracy"
)
#fit and predict data
grid.fit(X_train,y_train)
pred=grid.predict(X_test)
comparsion=pd.DataFrame({
    "predicted_values":pred,
    "real_values":y_test.values
})
print(comparsion)
print("train_acc",grid.score(X_train,y_train))
print("test_acc",grid.score(X_test,y_test))
print("best parameters",grid.best_params_)
print("best score",grid.best_score_)
ConfusionMatrixDisplay.from_estimator(grid,X_test,y_test)
plt.show()
RocCurveDisplay.from_estimator(grid,X_test,y_test)
plt.show()
#get result and importance
result=permutation_importance(
    grid,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)
importance=pd.Series(
    result.importances_mean,
    index=X_test.columns
).sort_values(ascending=False)