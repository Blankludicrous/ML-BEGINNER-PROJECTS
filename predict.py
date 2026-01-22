import pandas as pd
from sklearn.linear_model import LinearRegression
data=pd.read_csv("student_marks.csv")
X=data[["hours_studied","attendance"]]
y=data["marks"]
model=LinearRegression
model.fit(X,y)
new_student=pd.DataFrame({
    "hours_studied":[7],
    "attendance":[78]
}) 
marks_predicted=model.predict(new_student)
print("Predicted marks",marks_predicted)