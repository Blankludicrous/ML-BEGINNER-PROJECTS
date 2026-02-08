import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
#load dataset
df=pd.read_csv("house_sales_ml.csv")
df=df.dropna(subset=["price_lakhs"])  
#visualization
features=["size_sqft","bedrooms","bathrooms","age_years","distance_city_km","floor"]
label="price_lakhs"
for feature in features:
    plt.scatter(df[feature],df[label])
    plt.xlabel(feature)
    plt.ylabel(label)
    plt.title(f"{feature}-x vs {label}-y")
    plt.show()
#remove nan rows from label  
#load x and y
X=df[features]
y=df[label]
#train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)
#define pipeline
pipeline=Pipeline([
    ("imputation",SimpleImputer(strategy="mean")),
    ("Scaling",StandardScaler()),
    ("model",Ridge(alpha=0.1))
]
)
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
result=permutation_importance(
    pipeline,
    X_test,
    y_test,
    scoring="r2",
    n_repeats=10,
    random_state=42
)
importance=pd.Series(
    result.importances_mean,
    index=X_test.columns).sort_values(ascending=False)
print(importance)