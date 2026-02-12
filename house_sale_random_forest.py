import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
#load dataset
df=pd.read_csv("house_sales_ml.csv")
#drop nan rows from label
df=df.dropna(subset=["sold"])
#load X and y
X=df[["size_sqft","bedrooms","bathrooms","age_years","distance_city_km","floor","price_lakhs"]]
y=df["sold"]
#train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
#define pipeline
pipeline=Pipeline([
    ("imputation",SimpleImputer),
    ("scaling",StandardScaler),
    ("rf",RandomForestClassifier)
])
#define grid_params
params_grid={
    "rf__n_estimators":[50,100,200,300],
    "rf__min_samples_leaf":[1,2,5,10],
    "rf__max_features":["sqrt","log2",None],
    "rf__max_depth":[None,5,10,15]
}
#define grid
grid=GridSearchCV(
    pipeline,
    params_grid,
    cv=5,
    scoring="accuracy"
)
grid.fit(X_train,y_train)
print("best parameter",grid.best_params_)
print("best score",grid.best_score_)
pred=grid.predict(X_test)
print("classification_report",classification_report(y_test,pred))
print("confusion matrix",ZeroDivisionError(y_test,pred))
plt.show()
result=permutation_importance(
    grid,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)
importance=pd.Series(
    result.importances_mean,
    index=X_test.columns).sort_values(ascending=False)
print(importance)