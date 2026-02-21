import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay,classification_report
import matplotlib.pyplot as plt
#load dataset
df=pd.read_csv("crop_yield_dataset.csv")
#define high yield 
df["high_yield"]=(df["crop_yield"]>df["crop_yield"].median()).astype("int")
#remove nan rows from label
df=df.dropna(subset=["high_yield"])
#define X and y
X=df[["rainfall_mm","fertilizer_kg","temperature_c","soil_quality","noise_feature"]]
y=df["high_yield"]
#train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
#define pipeline
pipeline=Pipeline(
    [
        ("imputation",SimpleImputer(strategy="mean")),
        ("rf",RandomForestClassifier())
    ]
)
#define grid_params
params_grid={
    "rf__max_depth":[None,5,10,15],
    "rf__n_estimators":[50,100,200,300],
    "rf__min_samples_leaf":[1,2,5,10],
    "rf__max_features":["log2","sqrt",None]
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
comparsion=pd.DataFrame(
    {
        "predicted_values":pred,
        "real_values":y_test.values
    }
)
print("train_Acc",grid.score(X_train,y_train))
print("test_Acc",grid.score(X_test,y_test))
print("best parameter",grid.best_params_)
print("best score",grid.best_score_)
print(comparsion)
ConfusionMatrixDisplay.from_estimator(grid,X_test,y_test)
plt.show()
print("classification report",classification_report(y_test,pred))
#find feature importance
result=permutation_importance(
    grid,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)
#print importance
importance=pd.Series(result.importances_mean,
                     index=X_test.columns).sort_values(ascending=False)
print(importance)