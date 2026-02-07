import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.inspection import permutation_importance
#load dataset into dataframe
df=pd.read_csv("employee_productivity_ml.csv")
#remove nan rows from label
df=df.dropna(subset=["left_company"])
#define X and y
X=df[["experience_years","monthly_hours","projects_handled","salary","work_life_balance","remote_work","performance_score"]]
y=df["left_company"]
#train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#define pipeline
pipeline=Pipeline([
    ("imputation",SimpleImputer()),
    ("scaling",StandardScaler()),
    ("rf",RandomForestClassifier())
])
#fit and predict data
pipeline.fit(X_train,y_train)
pred=pipeline.predict(X_test)
comparsion=pd.DataFrame(
    {
        "predicted_values":pred,
        "real_values":y_test.values
    }
)
print(comparsion)
train_acc=pipeline.score(X_train,y_train)
test_acc=pipeline.score(X_test,y_test)
result=permutation_importance(
    pipeline,
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
#find best depth
depth=5
n=50
leaf=5
mfs=["log2","sqrt",None]
for mf in mfs:
    final_pipeline=pipeline.set_params(
        rf__min_samples_leaf=leaf,
        rf__n_estimators=n,
        rf__max_depth=depth,
        rf__max_features=mf)
    final_pipeline.fit(X_train,y_train)
    train_acc=final_pipeline.score(X_train,y_train)
    test_acc=final_pipeline.score(X_test,y_test)
    print(f"mf={mf} | train_acc={train_acc} | test_acc={test_acc}")
