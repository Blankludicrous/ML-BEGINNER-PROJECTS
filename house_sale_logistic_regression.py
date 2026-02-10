import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
#load dataset
df=pd.read_csv("house_sales_ml.csv")
#drop nan rows of label
df=df.dropna(subset=["sold"])
#define X and y
X=df[["size_sqft","bedrooms","bathrooms","age_years","distance_city_km","floor","price_lakhs"]]
y=df["sold"]
#train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42,stratify=y)
#define pipeline
pipeline=Pipeline([
    ("imputation",SimpleImputer(strategy="mean")),
    ("scaling",StandardScaler()),
    ("logreg",LogisticRegression())
])
#define grid_params
param_grid={
    "logreg__C":[0.01,0.1,1,10,100],
    "logreg__penalty":["l1","l2"],
    "logreg__solver":["liblinear"],
    "logreg__class_weight":["balanced"]
}
#define the gridsearchCV
grid=GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="accuracy"
)
grid.fit(X_train, y_train)
print("best parmaters",grid.best_params_)
print("best score",grid.best_score_)
#get the predicted values
pred=grid.predict(X_test)
print("classification report",classification_report(y_test,pred))
ConfusionMatrixDisplay.from_estimator(grid,X_test,y_test)
plt.show()
result=permutation_importance(
    grid,
    X_test,
    y_test,
    scoring="accuracy",
    n_repeats=10,
    random_state=42
)
importance=pd.Series(
    result.importances_mean,
    index=X_test.columns).sort_values(ascending=False)
print(importance)