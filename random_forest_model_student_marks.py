import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix
#load dataset
df=pd.read_csv("student_marks_new.csv")
df=df.dropna(subset="pass")
#load X and y
X=df[["study_hours","attendance","sleep_hours","previous_score","noise_feature"]]
y=df["pass"]
#train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#pipeline definition
pipeline=Pipeline([
    ("imputation",SimpleImputer()),
    ("model",RandomForestClassifier())
])
pipeline.fit(X_train,y_train)
pred=pipeline.predict(X_test)
comparsion=pd.DataFrame(
    {
        "predicted_values":pred,
        "real values":y_test.values
    }
)
print(comparsion)
print("train model accuracy",pipeline.score(X_train,y_train))
print("Test model accuracy",pipeline.score(X_test,y_test))
print("confusion_matrix of the model",confusion_matrix(y_test,pred))
print("classification report of the model",classification_report(y_test,pred))
result=permutation_importance(
    pipeline,
    X_test,
    y_test,
    scoring="accuracy",
    n_repeats=20,
    random_state=100,
)
importance=pd.Series(
    result.importances_mean,
        index=X_test.columns).sort_values(ascending=False)
print(importance)