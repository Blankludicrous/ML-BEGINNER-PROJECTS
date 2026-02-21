import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
#load dataset
df=pd.read_csv("employee_productivity_ml.csv")
#remove nan rows from label
df=df.dropna(subset=["left_company"])
#load X and y
X=df[["experience_years","monthly_hours","projects_handled","salary","work_life_balance","remote_work","performance_score"]]
y=df["left_company"]
#train_test_Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#pipeline definition
pipeline=Pipeline([
    ("imputation",SimpleImputer()),
    ("scaling",StandardScaler()),
    ("logreg",LogisticRegression(max_iter=1000,random_state=42))
])
#fit and predict using pipeline
pipeline.fit(X_train,y_train)
pred=pipeline.predict(X_test)
comparsion=pd.DataFrame({
    "predicted values":pred,
    "real_values":y_test.values
})
print(comparsion)
result=permutation_importance(
    pipeline,
    X_test,
    y_test,
    scoring="accuracy",
    n_repeats=100
)
importance=pd.Series(
    result.importances_mean,
    index=X_test.columns).sort_values(ascending=False)
print(importance)
C_values=[0.01,0.1,1,10,100]
best_C=None
best_score=0
for C in C_values:
    pipeline.set_params(logreg__C=C)
    pipeline.fit(X_train,y_train)

    train_acc=pipeline.score(X_train,y_train)
    test_acc=pipeline.score(X_test,y_test)
    if test_acc>best_score:
        best_score=test_acc
        best_C=C
penalties=["l1","l2"]
for penalty in penalties:
    final_pipeline=pipeline.set_params(
    logreg__C=best_C,
    logreg__penalty=penalty,
    logreg__class_weight="balanced",
    logreg__solver="liblinear")
    final_pipeline.fit(X_train,y_train)
    train_acc=final_pipeline.score(X_train,y_train)
    test_acc=final_pipeline.score(X_test,y_test)
    print(f"train_acc={train_acc} | test_acc={test_acc}")
print("confusion_matrix",confusion_matrix(y_test,pred))
print("classification_report",classification_report(y_test,pred))