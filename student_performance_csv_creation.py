import pandas as pd
import numpy as np
students_data=[[18,2,60,35],
[19,4,75,55],
[18,1,50,30],
[20,6,85,70],
[21,5,80,65],
[19,3,70,45],
[22,7,90,80],
[20,2,65,40],
[18,8,95,88],
[21,6,88,75],
[19,np.nan,72,50],
[20,4,np.nan,60],
[22,5,78,np.nan],
[18,3,68,48],
[21,7,92,82]] 
df=pd.DataFrame(students_data,columns=["age","hours_studied","attendance","marks"])
df.to_csv("student_performance.csv",index=False)
