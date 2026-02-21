import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix,classification_report
#load data
df=pd.read_csv("student_performance.csv")
df=df.dropna(subset="marks")
#define result column
df["result"]=(df["marks"]>=35).astype(int)
#define x and y
X=df[["age","hours_studied","attendance"]]
y=df["result"]
#train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#pipeline definition
pipeline=Pipeline([
    ("imputation",SimpleImputer(strategy="mean")),
    ("scaling",StandardScaler()),
    ("model",LogisticRegression())
])
#fit and predict
pipeline.fit(X_train,y_train)
pred=pipeline.predict(X_test)
comparsion=pd.DataFrame(
    {
    "predicted values":pred,
    "real values": y_test.values
    }
)
print("train acuuracy",pipeline.score(X_train,y_train))
print("test accuracy",pipeline.score(X_test,y_test))
print("confusion matrix",confusion_matrix(y_test,pred))
print("classification report",classification_report(y_test,pred))