import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
#load dataset csv
df=pd.read_csv("crop_yield_dataset.csv")
#remove nan rows from label
df=df.dropna(subset=["crop_yield"])
#visualize data
features=["rainfall_mm","fertilizer_kg","temperature_c","soil_quality","noise_feature"]
label="crop_yield"
for feature in features:
    plt.scatter(df[feature],df[label])
    plt.xlabel(feature)
    plt.ylabel(label)
    plt.show()
#define X and y
X=df[features]
y=df[label]
#train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#define_pipeline
pipeline=Pipeline([
    ("imputation",SimpleImputer(strategy="mean")),
    ("scaling",StandardScaler()),
    ("ridge",Ridge())
])
#define grid params
params_grid={
    "ridge__alpha":[0.01,0.1,1]
}
#define grid
grid=GridSearchCV(
    estimator=pipeline,
    param_grid=params_grid,
    cv=5,
    scoring="r2"
)
#fit and predict data
grid.fit(X_train,y_train)
pred=grid.predict(y_test)
print("train_Acc",grid.score(X_train,y_train))
print("test_Acc",grid.score(X_test,y_test))
comparsion=pd.DataFrame(
    {
        "predicted_values":pred,
        "real_values":y_test.values
    }
)
print(comparsion)
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