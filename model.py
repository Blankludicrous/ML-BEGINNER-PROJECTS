import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
data=pd.read_csv("students_marks.csv")
x=data[["hours_studied","attendance"]]
y=data["marks"]
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
error_mean=mean_absolute_error(y_test,y_pred)
print("Mean Absolute Error",error_mean)
print("Model Coefficient",model.coef_)
print("Model intercept",model.intercept_)
print("✅ THIS IS THE FIXED FILE")
